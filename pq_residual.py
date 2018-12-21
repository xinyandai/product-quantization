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

    def fit(self, vecs, iter):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        for layer, pq in enumerate(self.pqs):

            pq.fit(vecs, iter)
            compressed = pq.compress(vecs)
            vecs = vecs - compressed

            if self.verbose:
                norms = np.linalg.norm(vecs, axis=1)
                print("# layer: {},  residual average norm : {} max norm: {} min norm: {}"
                      .format(layer, np.mean(norms), np.max(norms), np.min(norms)))

        return self

    def encode(self, vecs):
        """
        :param vecs:
        :return: (N * deep * M)
        """
        codes = np.zeros((len(vecs), self.deep, self.M), dtype=self.code_dtype)  # N * deep * M
        for i, pq in enumerate(self.pqs):
            codes[:, i, :pq.M] = pq.encode(vecs)
            vecs = vecs - pq.decode(codes[:, i, :pq.M])
        return codes  # N * deep * M

    def decode(self, codes):
        vecss = [pq.decode(codes[:, i, :pq.M]) for i, pq in enumerate(self.pqs)]
        return np.sum(vecss, axis=0)

    def compress(self, vecs):
        N, D = np.shape(vecs)
        sum_residual = np.zeros((N, D), dtype=vecs.dtype)
        for i, pq in enumerate(self.pqs):
            compressed = pq.compress(vecs)
            vecs = vecs - compressed
            sum_residual = sum_residual + compressed
        return sum_residual
