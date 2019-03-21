import numpy as np
import numba as nb

class RQIterativeNorm(object):
    def __init__(self, rq):
        self.rq = rq
        self.deep = rq.deep
        self.code_dtype = rq.code_dtype
        self.M = rq.M

    def class_message(self):
        return 'RQIterativeNorm: ' + self.rq.class_message()

    @property
    def num_codebooks(self):
        return self.rq.num_codebooks

    def update_norm_sqr(self, m, c, compressed, codes):
        items_inside = np.where(codes[:, m] == c)[0]
        if len(items_inside) == 0:
            return

        compressed_inside = compressed[items_inside]

        others_norm_sqr_mean = np.sum([self.norms_sqr[m_, codes[items_inside, m_]] for m_ in range(self.rq.num_codebooks) if m_ != m]) / len(items_inside)

        self.norms_sqr[m, c] = np.mean(np.sum(compressed_inside ** 2, axis=1)) - others_norm_sqr_mean

    def fit(self, T, iter=20):
        self.rq.fit(T, iter)

        self.dim = T.shape[1]
        self.data_dtype = T.dtype

        codes = self.rq.encode(T)
        compressed = self.rq.decode(codes)

        self.norms_sqr = np.zeros((self.rq.num_codebooks, 256), dtype=T.dtype)

        for m in range(self.rq.num_codebooks):
            for c in range(256):
                self.update_norm_sqr(m, c, compressed, codes)

        #for i in range(5000):
        for i in range(100000):
            m = np.random.randint(0, self.rq.num_codebooks)
            c = np.random.randint(0, 256)
            self.update_norm_sqr(m, c, compressed, codes)

    def encode(self, vecs):
        return self.rq.encode(vecs)

    def decode(self, codes):
        decoded = np.empty((codes.shape[0], self.dim + 1), dtype=self.data_dtype)

        decoded[:, :-1] = self.rq.decode(codes)
        decoded[:, -1] = np.sum([self.norms_sqr[m, codes[:, m]] for m in range(self.rq.num_codebooks)], axis=0)

        return decoded

    def compress(self, vecs):
        return self.decode(self.encode(vecs))
