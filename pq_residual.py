from __future__ import division
from __future__ import print_function
from pq_norm import *


class ResidualPQ(object):
    def __init__(self, pqs=None, verbose=True):

        assert len(pqs) > 0
        self.verbose = verbose
        self.deep = len(pqs)
        self.code_dtype = pqs[0].code_dtype
        self.M = max([pq.M for pq in pqs])
        self.pqs = pqs

        for pq in self.pqs:
            assert pq.code_dtype == self.code_dtype

    def class_message(self):
        messages = ""
        for i, pq in enumerate(self.pqs):
            messages += pq.class_message()
        return messages

    def fit(self, vecs, iter=20, seed=123):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        for layer, pq in enumerate(self.pqs):

            pq.fit(vecs, iter, seed=seed)
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
            code[i, :, :pq.M] = pq.encode(vecs)
            vecs = vecs - pq.decode(code[i, :, :pq.M])
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
