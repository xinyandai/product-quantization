from __future__ import division
from __future__ import print_function
import numpy as np
from aq import AQ


class XPQ(object):
    def __init__(self, M, sub_m, Ks, quantizer=AQ, verbose=True):
        """
        :param M: how many sub-Quantizer
        :param sub_m: how many codebook in each sub-Quantizer
        :param Ks: how many codeword in each codebook
        :param quantizer: the type of sub-Quantizer, PQ/OPQ/AQ
        :param verbose:
        """
        assert 0 < Ks <= 2 ** 32
        self.M, self.Ks, self.verbose, self.quantizer = M, Ks, verbose, quantizer
        self.code_dtype = np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
        self.Ds = None
        self.xpqs = [self.quantizer(M=sub_m, Ks=self.Ks) for _ in range(self.M)]

    def class_message(self):
        return "XPQ PQ, M: {}, Ks : {}, quantizer type: {}, code_dtype: {}".format(
            self.M, self.Ks, self.quantizer, self.code_dtype)

    def fit(self, vecs, iter):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"

        reminder = D % self.M
        quotient = int(D / self.M)
        dims_width = [quotient + 1 if i < reminder else quotient for i in range(self.M)]
        self.Ds = np.cumsum(dims_width)     # prefix sum
        self.Ds = np.insert(self.Ds, 0, 0)  # insert zero at beginning

        for m in range(self.M):
            if self.verbose:
                print("#    Training the {}_PQ subspace: {} / {}, {} -> {}".format(self.quantizer, m, self.M, self.Ds[m], self.Ds[m+1]))
            vecs_sub = vecs[:, self.Ds[m]:self.Ds[m+1]]
            self.xpqs[m].fit(vecs_sub, iter=iter)

        return self

    def compress(self, vecs):
        compressed = np.zeros(vecs.shape, np.float32)
        for m in range(self.M):
            vecs_sub = vecs[:, self.Ds[m]:self.Ds[m+1]]
            compressed[:, self.Ds[m]:self.Ds[m+1]] = self.xpqs[m].compress(vecs_sub)
        return compressed
