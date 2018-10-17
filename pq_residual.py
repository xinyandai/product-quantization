from __future__ import division
from __future__ import print_function
from pq import *


class ResidualPQ(object):
    def __init__(self, M, Ks=256, deep=0, verbose=True):
        assert 0 < Ks <= 2 ** 32
        self.M, self.Ks, self.deep, self.verbose = M, Ks, deep, verbose
        self.code_dtype = np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
        self.codewords = None
        self.Ds = None
        self.pqs = [PQ(M, Ks, verbose) for _ in range(deep)]

        if verbose:
            print("M: {}, Ks: {}, residual deep : {}, code_dtype: {}".format(M, Ks, deep, self.code_dtype))

    def fit(self, vecs, iter=20, seed=123):
        for pq in self.pqs:
            pq.fit(vecs=vecs, iter=iter, seed=seed)
            # N * D
            compressed = pq.compress(vecs)
            vecs = vecs - compressed
            print("=====================")
            print('norm error:', np.linalg.norm(vecs))
            print("=====================")

        return self

    def encode(self, vecs):
        # deep * N * M
        code = np.zeros((self.deep, len(vecs), self.M), dtype=self.code_dtype)
        for i, pq in enumerate(self.pqs):
            code[i][:][:] = pq.encode(vecs)
            vecs = vecs - pq.decode(code[i][:][:])
        # N * deep * M
        return np.swapaxes(code, 0, 1)

    def decode(self, codes):
        # N * deep * M
        # deep * N  * M
        codes = np.swapaxes(codes, 0, 1)
        vecss = [self.pqs[i].decode(codes[i]) for i in range(self.deep)]
        return np.sum(vecss, axis=0)

    def compress(self, vecs):
        return self.decode(self.encode(vecs))

