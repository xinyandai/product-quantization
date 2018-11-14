from pq import *
from transformer import normalize


class NormPQ(object):
    def __init__(self, n_percentile, quantize, true_norm=False, verbose=True, method='kmeans'):

        self.M = 2
        self.n_percentile, self.true_norm, self.verbose = n_percentile, true_norm, verbose
        self.method = method
        self.code_dtype = np.uint8 if n_percentile <= 2 ** 8 \
            else (np.uint16 if n_percentile <= 2 ** 16 else np.uint32)

        self.percentiles = None
        self.quantize = quantize

    def class_message(self):
        return "NormPQ, percentiles: {}, quantize: {}".format(self.n_percentile, self.quantize.class_message())

    def fit(self, vecs, iter):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.n_percentile < N, "the number of norm intervals should be more than Ks"

        norms, normalized_vecs = normalize(vecs)

        if self.method == 'kmeans':
            self.percentiles, _ = kmeans2(norms[:, np.newaxis], self.n_percentile, iter=iter, minit='points')
        elif self.method == 'percentile':
            self.percentiles = np.percentile(norms, np.linspace(0, 100, self.n_percentile + 1)[:])
            self.percentiles = np.array(self.percentiles, dtype=np.float32)
        elif self.method == 'uniform':
            self.percentiles = np.linspace(np.min(norms), np.max(norms), self.n_percentile + 1)
            self.percentiles = np.array(self.percentiles, dtype=np.float32)
        else:
            assert False

        self.quantize.fit(normalized_vecs, iter)

        return self

    def encode_norm(self, norms):

        if self.method == 'kmeans':
            norm_index, _ = vq(norms[:, np.newaxis], self.percentiles)
        else:
            norm_index = [np.argmax(self.percentiles[1:] > n) for n in norms]
            norm_index = np.clip(norm_index, 1, self.n_percentile)
        return norm_index

    def decode_norm(self, norm_index):
        if self.method == 'kmeans':
            return self.percentiles[norm_index, 0]
        else:
            return (self.percentiles[norm_index]+self.percentiles[norm_index-1]) / 2.0

    def compress(self, vecs):
        norms, normalized_vecs = normalize(vecs)

        if not self.true_norm:
            norms = self.decode_norm(self.encode_norm(norms))

        compressed_vecs = self.quantize.compress(normalized_vecs)
        _, compressed_vecs = normalize(compressed_vecs)

        return (compressed_vecs.transpose() * norms).transpose()
