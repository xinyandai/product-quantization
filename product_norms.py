from __future__ import division
from __future__ import print_function
import numpy as np
from pq_norm import NormPQ
from pq import PQ


class ProductNorms(object):
    def __init__(self, M, n_percentile, Ks=256, verbose=True, true_norm=False):
        assert 0 < Ks <= 2 ** 32
        self.M, self.Ks, self.verbose, self.n_percentile = M, Ks, verbose, n_percentile
        self.code_dtype = np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
        self.codewords = None
        self.Ds = None
        self.Dim = -1
        self.norm_pqs = [
            NormPQ(n_percentile=n_percentile, quantize=PQ(1, self.Ks), true_norm=true_norm, verbose=False)
            for _ in range(self.M)]

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
        self.codewords = np.zeros((self.M, self.Ks, np.max(self.Ds)), dtype=np.float32)
        for m in range(self.M):
            if self.verbose:
                print("#    Training the subspace: {} / {}, {} -> {}".format(m, self.M, self.Ds[m], self.Ds[m+1]))
            vecs_sub = vecs[:, self.Ds[m]:self.Ds[m+1]]
            self.norm_pqs[m].fit(vecs_sub, iter=iter)

        return self

    def compress(self, vecs):
        compressed = np.zeros(vecs.shape, np.float32)
        for m in range(self.M):
            vecs_sub = vecs[:, self.Ds[m]:self.Ds[m+1]]
            compressed[:, self.Ds[m]:self.Ds[m+1]] = self.norm_pqs[m].compress(vecs_sub)
        return compressed
