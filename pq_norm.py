from pq import *


class NormPQ(object):
    def __init__(self, n_percentile=256, Ks=256, true_norm=False, verbose=True, mahalanobis_matrix=None):
        assert 0 < Ks <= 2 ** 32
        self.M = 2
        self.n_percentile, self.Ks, self.true_norm, self.verbose, self.mahalanobis_matrix = \
            n_percentile, Ks, true_norm, verbose, mahalanobis_matrix
        self.code_dtype = np.uint8 if max(Ks, n_percentile) <= 2 ** 8 \
            else (np.uint16 if max(Ks, n_percentile) <= 2 ** 16 else np.uint32)
        self.centers = None
        self.percentiles = None

    def normalize(self, vecs):
        norms = np.linalg.norm(vecs, axis=1)[:, np.newaxis]
        # normalized_vecs = vecs / norms
        # divide by zero problem:
        normalized_vecs = np.divide(vecs, norms, out=np.zeros_like(vecs), where=norms != 0)
        return norms, normalized_vecs

    def fit(self, vecs, iter=20, seed=123):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"
        assert self.n_percentile < N, "the number of norm intervals should be more than Ks"

        np.random.seed(seed)

        norms, normalized_vecs = self.normalize(vecs)
        # float64 here -> float64
        self.percentiles = np.percentile(norms, np.linspace(0, 100, self.n_percentile + 1)[:])
        self.percentiles = np.array(self.percentiles, dtype=np.float32)

        self.centers = np.zeros((self.Ks, D), dtype=np.float32)
        self.centers[:], _ = kmeans2(normalized_vecs, self.Ks, iter=iter, minit='points', matrix=self.mahalanobis_matrix)

        return self

    def encode(self, vecs):
        """
        :param vecs:
        :return: (N * 2)
        """
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2

        N, D = vecs.shape
        norms, normalized_vecs = self.normalize(vecs)
        codes = np.empty((N, 2), dtype=self.code_dtype)
        codes[:, 1], _ = vq(normalized_vecs, self.centers, matrix=self.mahalanobis_matrix)

        norm_index = [np.argmax(self.percentiles[1:] > n) for n in norms]
        norm_index = np.clip(norm_index, 1, self.n_percentile)

        codes[:, 0] = norm_index

        return codes

    def decode(self, codes, norms=None):
        assert codes.ndim == 2
        assert codes.dtype == self.code_dtype

        vecs = self.centers[codes[:, 1], :]
        _, vecs = self.normalize(vecs)
        norm_index = codes[:, 0]

        if not self.true_norm:
            norms = (self.percentiles[norm_index]+self.percentiles[norm_index-1]) / 2.0
        assert norms is not None
        return (vecs.transpose() * norms).transpose()  # can only apply broadcast on columns, so transpose is needed

    def compress(self, vecs):
        norms = np.linalg.norm(vecs, axis=1) if self.true_norm else None
        return self.decode(self.encode(vecs), norms)
