from __future__ import division
from __future__ import print_function
import numpy as np
from aq import AQ


class PQX(object):
    def __init__(self, xpqs, verbose=True):
        """
        :param M: how many sub-Quantizer
        :param xpqs: sub-Quantizer
        :param verbose:
        """
        self.verbose = verbose
        self.xpqs = xpqs
        self.M = len(xpqs)
        self.Ds = None

    def class_message(self):
        return "XPQ PQ, M: {}, quantizer : {}".format(self.M, self.xpqs)

    def fit(self, vecs, iter):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape

        reminder = D % self.M
        quotient = int(D / self.M)
        dims_width = [quotient + 1 if i < reminder else quotient for i in range(self.M)]
        self.Ds = np.cumsum(dims_width)     # prefix sum
        self.Ds = np.insert(self.Ds, 0, 0)  # insert zero at beginning

        for m in range(self.M):
            if self.verbose:
                print("#    Training the XPQ subspace: {} / {}, {} -> {}".format(
                     m, self.M, self.Ds[m], self.Ds[m+1]))
            vecs_sub = vecs[:, self.Ds[m]:self.Ds[m+1]]
            self.xpqs[m].fit(vecs_sub, iter=iter)

        return self

    def compress(self, vecs):
        compressed = np.zeros(vecs.shape, np.float32)
        for m in range(self.M):
            vecs_sub = vecs[:, self.Ds[m]:self.Ds[m+1]]
            compressed[:, self.Ds[m]:self.Ds[m+1]] = self.xpqs[m].compress(vecs_sub)
        return compressed
