from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.cluster.vq import vq, kmeans2


class PQ(object):
    def __init__(self, M, Ks, verbose=True):
        assert 0 < Ks <= 2 ** 32
        self.M, self.Ks, self.verbose = M, Ks, verbose
        self.code_dtype = np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
        self.codewords = None
        self.Ds = None
        self.Dim = -1

    def class_message(self):
        return "Subspace PQ, M: {}, Ks : {}, code_dtype: {}".format(self.M, self.Ks, self.code_dtype)

    def fit(self, vecs, iter):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"
        self.Dim = D

        reminder = D % self.M
        quotient = int(D / self.M)
        dims_width = [quotient + 1 if i < reminder else quotient for i in range(self.M)]
        self.Ds = np.cumsum(dims_width)     # prefix sum
        self.Ds = np.insert(self.Ds, 0, 0)  # insert zero at beginning

        # [m][ks][ds]: m-th subspace, ks-the codeword, ds-th dim
        self.codewords = np.zeros((self.M, self.Ks, np.max(dims_width)), dtype=np.float32)
        for m in range(self.M):
            if self.verbose:
                print("#    Training the subspace: {} / {}, {} -> {}".format(m, self.M, self.Ds[m], self.Ds[m+1]))
            vecs_sub = vecs[:, self.Ds[m]:self.Ds[m+1]]
            self.codewords[m, :, :self.Ds[m+1] - self.Ds[m]], _ = kmeans2(
                vecs_sub, self.Ks, iter=iter, minit='points')

        return self

    def encode(self, vecs):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape

        # codes[n][m] : code of n-th vec, m-th subspace
        codes = np.empty((N, self.M), dtype=self.code_dtype)
        for m in range(self.M):
            vecs_sub = vecs[:, self.Ds[m]: self.Ds[m+1]]
            codes[:, m], _ = vq(vecs_sub,
                                self.codewords[m, :, :self.Ds[m+1] - self.Ds[m]])

        return codes

    def decode(self, codes):
        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.M
        assert codes.dtype == self.code_dtype

        vecs = np.empty((N, self.Dim), dtype=np.float32)
        for m in range(self.M):
            vecs[:, self.Ds[m]: self.Ds[m+1]] = self.codewords[m, codes[:, m], :self.Ds[m+1] - self.Ds[m]]

        return vecs

    def compress(self, vecs):
        return self.decode(self.encode(vecs))
