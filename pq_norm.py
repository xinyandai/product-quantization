from pq import *


class NormPQ(object):
    def __init__(self, n_percentile=256, Ks=256, verbose=True):
        assert 0 < Ks <= 2 ** 32
        self.n_percentile, self.Ks, self.verbose = n_percentile, Ks, verbose
        self.code_dtype = np.uint8 if max(Ks, n_percentile) <= 2 ** 8 \
            else (np.uint16 if max(Ks, n_percentile) <= 2 ** 16 else np.uint32)
        self.centers = None
        self.percentiles = None

        if verbose:
            print("percentiles: {}, Ks: {}, code_dtype: {}".format(n_percentile, Ks, self.code_dtype))

    def normalize(self, vecs):
        norms = np.linalg.norm(vecs, axis=1)
        normalized_vecs = vecs / norms[:, np.newaxis]
        return norms, normalized_vecs

    def fit(self, vecs, iter=20, seed=123):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"
        assert self.n_percentile < N, "the number of norm intervals should be more than Ks"

        np.random.seed(seed)
        if self.verbose:
            print("iter: {}, seed: {}".format(iter, seed))

        norms, normalized_vecs = self.normalize(vecs)
        self.percentiles = np.percentile(norms, np.linspace(0, 100, self.n_percentile + 1)[:])

        self.centers = np.zeros((self.Ks, D), dtype=np.float32)
        self.centers[:], _ = kmeans2(normalized_vecs, self.Ks, iter=iter, minit='points')

        return self

    def encode(self, vecs):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2

        N, D = vecs.shape
        norms, normalized_vecs = self.normalize(vecs)
        codes = np.empty((N, 2), dtype=self.code_dtype)
        codes[:, 1], _ = vq(normalized_vecs, self.centers)

        norm_index = [np.argmax(self.percentiles[1:] > n) for n in norms]
        norm_index = np.clip(norm_index, 1, self.n_percentile)

        codes[:, 0] = norm_index

        return codes

    def decode(self, codes):
        assert codes.ndim == 2
        assert codes.dtype == self.code_dtype

        vecs = self.centers[codes[:, 1], :]
        _, vecs = self.normalize(vecs)
        norm_index = codes[:, 0]

        norms = (self.percentiles[norm_index]+self.percentiles[norm_index-1]) / 2.0

        return np.multiply(vecs, np.tile(norms, (len(vecs[0]), 1)).transpose())

    def compress(self, vecs):
        return self.decode(self.encode(vecs))
