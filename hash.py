from __future__ import division
from __future__ import print_function
import numpy as np


class RandomProjection(object):
    def __init__(self, bit, verbose=True):
        self.projector = None
        self.L = bit

    def class_message(self):
        return "RandomProjection , bit length: {}".format(self.L)

    def fit(self, vecs, niter, seed):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape

        self.projector = np.random.normal(size=(D, self.L))
        for i in range(self.L):
            self.projector[:, i] = np.random.normal(size=D)
        return self

    def encode(self, vecs):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape

        codes = vecs @ self.projector  # (N, D) (D, L) -> (N, L)
        assert codes.shape == (N, self.L)

        return codes

    def decode(self, codes):
        pass

    def compress(self, vecs):
        return self.encode(vecs)
