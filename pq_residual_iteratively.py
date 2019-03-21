from __future__ import division
from __future__ import print_function
from pq_norm import *
import tqdm
from scipy.cluster.vq import vq, _vq, kmeans2

def increase_kmeans(data, code_book, iter):
    nc = len(code_book)
    for i in range(iter):
        # Compute the nearest neighbor for each obs using the current code book
        label = vq(data, code_book)[0]
        # Update the code book by computing centroids
        new_code_book, has_members = _vq.update_cluster_means(data, label, nc)
        if not has_members.all():
            # Set the empty clusters to their previous positions
            new_code_book[~has_members] = code_book[~has_members]
        code_book = new_code_book

    return code_book, label


class IterativelyResidualPQ(object):
    def __init__(self, deep, Ks=256, verbose=True):
        self.verbose = verbose
        self.deep = deep
        self.Ks = Ks
        self.codewords = None
        self.code_dtype = np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)

    def class_message(self):
        return "IterativelyResidualPQ layers: {}".format(self.deep)

    @property
    def num_codebooks(self):
        return self.M * self.deep

    def fit(self, T, iter, save_codebook=False, save_decoded=[], save_residue_norms=[], save_results_T=False, dataset_name=None, save_dir=None, D=None):
        assert T.dtype == np.float32
        assert T.ndim == 2
        N, D = np.shape(T)
        residual = np.empty(shape=(N, D), dtype=T.dtype)
        compressed = np.empty(shape=(self.deep, N, D), dtype=T.dtype)
        self.codewords = np.empty(shape=(self.deep, self.Ks, D))

        residual[:, :] = T[:, :]
        for m in tqdm.tqdm(range(self.deep)):

            self.codewords[m, :, :], codes = kmeans2(residual, self.Ks, iter=iter, minit='points')
            compressed[m, :, :] = self.codewords[m, codes, :]
            residual = residual - compressed[m, :, :]

        for _ in tqdm.tqdm(range(iter)):
            for m in range(self.deep):
                layer_residual = T[:, :] - (np.sum(compressed, axis=0) - compressed[m])

                self.codewords[m, :, :], codes_m = increase_kmeans(layer_residual, code_book=self.codewords[m, :, :], iter=iter)
                compressed[m, :, :] = self.codewords[m, codes_m, :]


        return self

    def encode(self, vecs):
        """
        :param vecs:
        :return: (N, deep * M)
        """
        print("# iteratively ecoding")
        N, D = np.shape(vecs)
        codes = np.zeros((N, self.deep), dtype=self.code_dtype)  # N * deep * M
        residual = vecs.copy()
        for m in range(self.deep):
            codes[:, m] = vq(residual, self.codewords[m])[0]
            residual[:, :] = residual - self.codewords[m, codes[:, m], :]
        print("# iteratively ecoding")
        compressed = np.empty(shape=(self.deep, N, D), dtype=vecs.dtype)
        for _ in tqdm.tqdm(range(20)):
            for m in range(self.deep):
                layer_residual = vecs[:, :] - (np.sum(compressed, axis=0) - compressed[m])
                codes_m = vq(layer_residual, code_book=self.codewords[m, :, :])[0]
                compressed[m, :, :] = self.codewords[m, codes_m, :]
                codes[:, m] = codes_m
        return codes

    def decode(self, codes, left=0, right=None):

        vecss = [self.codewords[m, codes[:, m]] for m in range(self.deep)]
        return np.sum(vecss, axis=0)

    def compress(self, X):
        return self.decode(self.encode(X))
