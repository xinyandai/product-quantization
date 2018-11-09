from pq import *


def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1)
    norms_matrix = norms[:, np.newaxis]
    normalized_vecs = np.divide(vecs, norms_matrix, out=np.zeros_like(vecs), where=norms_matrix != 0)  # divide by zero problem
    return norms, normalized_vecs


class NormPQ(object):
    def __init__(self, n_percentile, quantize, true_norm=False, verbose=True, mahalanobis_matrix=None):

        self.M = 2
        self.n_percentile, self.true_norm, self.verbose = n_percentile, true_norm, verbose
        self.code_dtype = np.uint8 if n_percentile <= 2 ** 8 \
            else (np.uint16 if n_percentile <= 2 ** 16 else np.uint32)

        self.percentiles = None
        self.quantize = quantize

    def class_message(self):
        return "NormPQ, percentiles: {}, quantize: {}".format(self.n_percentile, self.quantize.class_message())

    def fit(self, vecs, iter=20, seed=123):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.n_percentile < N, "the number of norm intervals should be more than Ks"

        np.random.seed(seed)

        norms, normalized_vecs = normalize(vecs)
        # float64 here -> float64
        self.percentiles = np.percentile(norms, np.linspace(0, 100, self.n_percentile + 1)[:])
        self.percentiles = np.array(self.percentiles, dtype=np.float32)

        self.quantize.fit(normalized_vecs, iter, seed)

        return self

    def encode(self, vecs):
        """
        :param vecs:
        :return: (N * 2)
        """
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2

        norms, normalized_vecs = normalize(vecs)
        codes = self.quantize.encode(normalized_vecs)

        norm_index = [np.argmax(self.percentiles[1:] > n) for n in norms]
        norm_index = np.clip(norm_index, 1, self.n_percentile)

        return codes, norm_index

    def decode(self, codes, norm_index, norms=None):
        assert codes.dtype == self.code_dtype

        vecs = self.quantize.decode(codes)
        _, vecs = normalize(vecs)

        if not self.true_norm:
            norms = (self.percentiles[norm_index]+self.percentiles[norm_index-1]) / 2.0
        assert norms is not None
        return (vecs.transpose() * norms).transpose()  # can only apply broadcast on columns, so transpose is needed

    def compress(self, vecs):
        norms, normalized_vecs = normalize(vecs)
        if self.true_norm :
            norms = np.linalg.norm(vecs, axis=1)

        compressed_vecs = self.quantize.compress(normalized_vecs)

        return (compressed_vecs.transpose() * norms).transpose()
