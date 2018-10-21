from __future__ import division
from __future__ import print_function
from pq_norm import *


class ResidualPQ(object):
    def __init__(self, M=1, Ks=256, deep=1, pqs=None, verbose=True):
        if pqs is None:
            assert 0 < Ks <= 2 ** 32
            assert deep > 0

            self.M, self.Ks, self.deep, self.verbose = M, Ks, deep, verbose
            self.code_dtype = np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
            self.codewords = None
            self.Ds = None
            self.pqs = [PQ(M, Ks, verbose) for _ in range(deep)]

            if verbose:
                print("M: {}, Ks: {}, residual layer : {}, code_dtype: {}".format(M, Ks, deep, self.code_dtype))
        else:
            assert len(pqs) > 0
            self.verbose = verbose
            self.deep = len(pqs)
            self.code_dtype = pqs[0].code_dtype
            self.M = max([pq.M for pq in pqs])
            self.pqs = pqs

            if self.verbose:
                print("maximum M: {}, residual layer : {}, code_dtype: {}".format(self.M, deep, self.code_dtype))
        for pq in self.pqs:
            if isinstance(pq, NormPQ):
                print("---type: {}, M: {}, Ks : {}, n_percentile: {}, true_norm: {}, code_dtype: {}".format(
                    type(pq), self.M, pq.Ks, pq.n_percentile, pq.true_norm, pq.code_dtype))
            else:
                print("---type: {}, M: {}, Ks : {}, code_dtype: {}".format(
                    type(pq), self.M, pq.Ks, pq.code_dtype))
            assert pq.code_dtype == self.code_dtype
        print('-------------------------------------------------------------------------------------------------')

    def fit(self, vecs, iter=20, seed=123):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        for layer, pq in enumerate(self.pqs):
            if self.verbose:
                print("------------------------------------------------------------\nlayer: {}".format(layer))

            pq.fit(vecs=vecs, iter=iter, seed=seed)
            compressed = pq.compress(vecs)
            vecs = vecs - compressed

            if self.verbose:
                norms = np.linalg.norm(vecs, axis=1)
                print("layer: {},  residual average norm : {} max norm: {} min norm: {}"
                      .format(layer, np.mean(norms), np.max(norms), np.min(norms)))

        return self

    def encode(self, vecs):
        """
        :param vecs:
        :return: (N * deep * M)
        """
        code = np.zeros((self.deep, len(vecs), self.M), dtype=self.code_dtype)  # deep * N  * M
        for i, pq in enumerate(self.pqs):
            code[i][:][:pq.M] = pq.encode(vecs)
            vecs = vecs - pq.decode(code[i][:][:pq.M])
        return np.swapaxes(code, 0, 1)  # deep * N  * M -> N * deep * M

    def decode(self, codes):
        codes = np.swapaxes(codes, 0, 1)  # N * deep * M -> deep * N  * M
        vecss = [pq.decode(codes[i][:][:pq.M]) for i, pq in enumerate(self.pqs)]
        return np.sum(vecss, axis=0)

    def compress(self, vecs):
        N, D = np.shape(vecs)
        compressed = np.zeros((self.deep, N, D), dtype=vecs.dtype)
        for i, pq in enumerate(self.pqs):
            compressed[i][:][:] = pq.compress(vecs)
            vecs = vecs - compressed[i][:][:]
        return np.sum(compressed, axis=0)
