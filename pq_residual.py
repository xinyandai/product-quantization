from __future__ import division
from __future__ import print_function
from pq_norm import *


class ResidualPQ(object):
    def __init__(self, M, Ks=256, deep=1, n_percentile=0, verbose=True):
        assert 0 < Ks <= 2 ** 32
        if n_percentile != 0:
            assert M == 2
        self.M, self.Ks, self.deep, self.n_percentile, self.verbose = M, Ks, deep, n_percentile, verbose
        self.code_dtype = np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
        self.codewords = None
        self.Ds = None
        if n_percentile == 0:
            self.pqs = [PQ(M, Ks, verbose) for _ in range(deep)]
        else:
            self.pqs = [NormPQ(n_percentile, Ks, verbose) for _ in range(deep)]

        if verbose:
            print("M: {}, Ks: {}, residual deep : {}, code_dtype: {}".format(M, Ks, deep, self.code_dtype))

    def fit(self, vecs, iter=20, seed=123):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
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
        """
        :param vecs:
        :return: (N * deep * M)
        """
        code = np.zeros((self.deep, len(vecs), self.M), dtype=self.code_dtype)
        for i, pq in enumerate(self.pqs):
            code[i][:][:] = pq.encode(vecs)
            vecs = vecs - pq.decode(code[i][:][:])
        # deep * N  * M -> N * deep * M
        return np.swapaxes(code, 0, 1)

    def decode(self, codes):
        # N * deep * M -> deep * N  * M
        codes = np.swapaxes(codes, 0, 1)
        vecss = [self.pqs[i].decode(codes[i]) for i in range(self.deep)]
        return np.sum(vecss, axis=0)

    def compress(self, vecs):
        return self.decode(self.encode(vecs))

