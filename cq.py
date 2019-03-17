from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.cluster.vq import vq, kmeans2
import tqdm

class CQ(object):

    def __init__(self, depth=4, Ks=256):
        self.depth = depth
        self.Ks = Ks
        self.code_dtype = np.uint8 if Ks <= 2 ** 8 \
            else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
        self.codewords = None
        self.D = None
        self.mu = 8
        self.epsilons = np.zeros((self.depth, self.Ks), dtype=np.float32)

    def class_message(self):
        return "CQ with {} residual layers and {} codebook each layer"\
            .format(self.depth, self.Ks)

    def fit(self, X, iter=20, lr=0.01):
        N, D = X.shape

        self.D = D
        residuals = np.empty(shape=X.shape, dtype=X.dtype)
        codes = np.zeros((N, self.depth), dtype=self.code_dtype)

        # the last dimension of codewords would be used to approximate norm
        self.codewords = np.zeros((self.depth, self.Ks, D + 1), dtype=np.float32)
        residuals[:, :] = X[:, :]

        assert self.depth == 2

        for layer in range(self.depth):
            # initialize codewords and assign code with RQ
            self.codewords[layer, :, :D], _ = kmeans2(residuals, self.Ks, iter=iter, minit='points')
            codes[:, layer], _ = vq(residuals, self.codewords[layer, :, :D])
            if layer > 0:
                # loop
                for _ in tqdm.tqdm(range(10)):
                    error = np.zeros((N, self.Ks), dtype=np.float32)
                    # update epsilon and  codeword by codeword
                    for i in range(self.Ks):
                        # c: the i'th code
                        c = self.codewords[layer, i, :D]
                        # index to the items reside in codewords[layer, i, :]
                        idx = np.where(codes[:, layer] == i)[0]
                        if  len(idx) != 0:
                            pre_layer = self.codewords[layer-1, tuple(codes[idx, layer-1]), :D].reshape(-1, D)
                            # 1. update \epsilon
                            self.epsilons[layer, i] = 2.0 * np.dot(np.mean(pre_layer, axis=0), c)
                            # 2. update codeword
                            residual = residuals[idx, :]
                            # update codeword with sgd
                            # grad = first + second - third, 1D-array[D]
                            for _ in range(100):
                                first = 2 * np.mean(c - residual, axis=0)
                                a = np.dot(pre_layer / len(idx), c)
                                second = 8 * self.mu * np.sum(a.reshape(-1, 1) * pre_layer, axis=0)
                                third = 4 * self.mu * self.epsilons[layer, i] * np.mean(pre_layer, axis=0)

                                grad =  np.reshape(first + second - third, -1)
                                c = c - grad  * lr

                            self.codewords[layer, i, :D] = c
                            self.epsilons[layer, i] = 2.0 * np.dot(np.mean(pre_layer, axis=0), c)
                        else:
                            self.epsilons[layer, i] = 0

                        # 3. assign code according error to centers
                        # error = first  +  second  1D-array[N]
                        first = np.linalg.norm(residuals - c, axis=1) # N * D -> N
                        pre_layer_all = self.codewords[layer - 1, codes[:, layer-1], :D]
                        second = 2 * np.dot(pre_layer_all, c) - self.epsilons[layer, i] #
                        error[:, i] = first ** 2 + self.mu * second**2
                    # choose code
                    codes[:, layer] = np.argmin(error, axis=1)
                    min_err = np.min(error, axis=1)
                    print("Errors {} {} {}".format(np.mean(min_err), np.max(min_err), np.min(min_err)))

            self.codewords[layer, :, -1] = np.linalg.norm(self.codewords[layer, :, :D], axis=1) ** 2 + self.epsilons[layer, :]
            compressed = self.codewords[layer, codes[:, layer], :D]
            residuals[:, :] = residuals - compressed
        return self

    def encode(self, X):
        '''
        :param X:
        :return: (N * depth)
        '''
        N, D = np.shape(X)

        residuals = np.zeros((N, D), dtype=X.dtype)
        residuals[:, :] = X[:, :]
        codes = np.zeros((N, self.depth), dtype=self.code_dtype)  # N * deep * M

        for layer in range(self.depth):
            if layer == 0:
                codes[:, layer], _ = vq(residuals, self.codewords[layer, :, :D])
                compressed = self.codewords[layer, codes[:, layer], :D]
                residuals[:, :] = residuals - compressed
            else:
                error = np.zeros((N, self.Ks), dtype=np.float32)
                for i in range(self.Ks):
                    c = self.codewords[layer, i, :D]
                    # 3. error to centers
                    first = np.linalg.norm(residuals - c, axis=1)  # N * D -> N
                    second = 2 * np.dot(self.codewords[layer-1, codes[:, layer-1], :D], c) - self.epsilons[layer, i]  #
                    error[:, i] = first ** 2 + self.mu * second ** 2
                # 3'. choose code
                codes[:, layer] = np.argmin(error, axis=1)
                min_err = np.min(error, axis=1)
                print("Errors {} {} {}".format(np.mean(min_err), np.max(min_err), np.min(min_err)))

        return codes

    def decode(self, codes):
        vecss = [self.codewords[i, codes[:, i], :] for i in range(self.depth)]
        return np.sum(vecss, axis=0)


    def compress(self, X):
        return self.decode(self.encode(X))
