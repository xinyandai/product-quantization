from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.cluster.vq import vq, kmeans2


class INRQ(object):

    def __init__(self, depth=4, Ks=256):
        self.depth = depth
        self.Ks = Ks
        self.code_dtype = np.uint8 if Ks <= 2 ** 8 \
            else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
        self.codewords = None
        self.D = -1

    def fit(self, X, iter=20):
        N, D = X.shape
        self.D = D
        vecs = np.empty(shape=X.shape, dtype=X.dtype)
        vecs[:, :] = X[:, :]
        eXList = np.zeros((self.depth, N), dtype=np.int32)
        cXList = np.zeros((N, D), dtype=np.float32)
        self.codewords = np.zeros((self.depth, self.Ks, D + 1), dtype=np.float32)
        cX = np.zeros((N, D), dtype=np.float32)
        for layer in range(self.depth):
            self.codewords[layer, :, :D], _ = kmeans2(vecs, self.Ks, iter=iter, minit='points')
            # eX: N * 1
            eX, _ = vq(vecs, self.codewords[layer, :, :D])
            eXList[layer, :] = eX

            cX = self.codewords[layer, eX, :D]
            if layer == 0:
                self.codewords[0, :, -1] = np.linalg.norm(self.codewords[0, :, :-1], axis=1) ** 2
            else:
                for i in range(self.Ks):
                    idx = np.where(eX == i)
                    NSum = np.zeros((1, len(idx[0])), dtype=np.float32)
                    for j in range(layer):
                        eidx = tuple(eXList[j, idx])
                        NSum[:] += self.codewords[j, eidx, -1]
                    V = np.linalg.norm(cXList[idx] + cX[idx], axis=1) ** 2 - NSum
                    self.codewords[layer, i, -1] = np.average(V)

            cXList += cX
            vecs = vecs - cX
            del cX
        return self


    def encode(self, X):
        '''
        :param X:
        :return: (N * depth)
        '''
        N, D = np.shape(X)

        vecs = np.zeros((N, D), dtype=X.dtype)
        vecs[:, :] = X[:, :]

        codes = np.zeros((N, self.depth), dtype=self.code_dtype)  # N * deep * M
        for layer in range(self.depth):
            codes[layer, :], _ = vq(vecs, self.codewords[layer, :, :D])
            residual = self.codewords[layer, codes, :D]
            vecs -= residual[:, :D]

        return codes

    def decode(self, codes):
        vecss = [self.codewords[i, codes[i, :], :] for i in range(self.depth)]
        return np.sum(vecss, axis=0)


    def compress(self, X):
        N, D = np.shape(X)

        sum_residual = np.zeros((N, D + 1), dtype=X.dtype)
        vecs = np.zeros((N, D), dtype=X.dtype)
        vecs[:, :] = X[:, :]

        for layer in range(self.depth):
            codes, _ = vq(vecs, self.codewords[layer, :, :D])
            residual = self.codewords[layer, codes, :]
            sum_residual += residual

            vecs -= residual[:, :D]

        return sum_residual
