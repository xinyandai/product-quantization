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

    def fit(self, T, iter, save_codebook=False, save_decoded=[], save_residue_norms=[], save_results_T=False, dataset_name=None, save_dir=None, D=None):
        assert T.dtype == np.float32
        assert T.ndim == 2

        if save_dir is None:
            save_dir = './results'

        vecs = np.empty(shape=T.shape, dtype=T.dtype)
        vecs[:, :] = T[:, :]
        if D is not None:
            vecs_d = np.empty(shape=D.shape, dtype=D.dtype)
            vecs_d[:, :] = D[:, :]
        if save_codebook:
            codebook_f = open(save_dir + '/' + dataset_name + '_rq_' + str(self.deep) + '_' + str(self.pqs[0].Ks) + '_codebook', 'wb')

        for layer, pq in enumerate(self.pqs):
            pq.fit(vecs, iter)
            compressed = pq.compress(vecs)
            vecs = vecs - compressed
            del compressed

            if D is not None:
                compressed_d = pq.compress(vecs_d)
                vecs_d -= compressed_d

            if self.verbose:
                norms = np.linalg.norm(vecs, axis=1)
                print("# layer: {},  residual average norm : {} max norm: {} min norm: {}"
                      .format(layer, np.mean(norms), np.max(norms), np.min(norms)))

            if (layer + 1) in save_residue_norms:
                with open(save_dir + '/' + dataset_name + '_rq_' + str(layer + 1) + '_' + str(self.pqs[0].Ks) + '_residue_norms', 'wb') as f:
                    if save_results_T:
                        np.linalg.norm(vecs, axis=1).tofile(f)
                    if D is not None:
                        np.linalg.norm(vecs_d, axis=1).tofile(f)

            if (layer + 1) in save_decoded:
                with open(save_dir + '/' + dataset_name + '_rq_' + str(layer + 1) + '_' + str(self.pqs[0].Ks) + '_decoded', 'wb') as f:
                    if save_results_T:
                        (T - vecs).tofile(f)
                    if D is not None:
                        (D - vecs_d).tofile(f)

            if save_codebook:
                pq.codewords.tofile(codebook_f)
                codebook_f.flush()

        if save_codebook:
            codebook_f.close()

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

    def compress(self, X):
        N, D = np.shape(X)

        sum_residual = np.zeros((N, D), dtype=X.dtype)

        vecs = np.zeros((N, D), dtype=X.dtype)
        vecs[:, :] = X[:, :]

        for i, pq in enumerate(self.pqs):
            compressed = pq.compress(vecs)
            vecs[:, :] = vecs - compressed
            sum_residual[:, :] = sum_residual + compressed
            del compressed

        return sum_residual

