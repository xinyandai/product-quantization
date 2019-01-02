import numpy as np
from scipy.cluster.vq import kmeans2
from pq import PQ

class RQGraph(object):
    def __init__(self, Ks=256, depth=2):
        self.Ks = Ks
        self.depth = depth
        self.pqs = [PQ(1, Ks) for i in range((1 + self.Ks) * (depth // 2) + depth % 2)]
        self.code_dtype = np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)

    def class_message(self):
        return "RQGraph, depth: {}, Ks : {}, code_dtype: {}".format(self.depth, self.Ks, self.code_dtype)

    def fit(self, T, iter=20, save_codebook=False, save_decoded=[], save_residue_norms=[], save_results_T=False, dataset_name=None, save_dir=None, D=None):
        if save_dir is None:
            save_dir = './results'

        N, _ = T.shape

        vecs = np.empty(shape=T.shape, dtype=T.dtype)
        vecs[:, :] = T[:, :]

        codes = np.empty((N, self.depth), dtype=self.code_dtype)

        if D is not None:
            vecs_d = np.empty(shape=D.shape, dtype=D.dtype)
            vecs_d[:, :] = D[:, :]

            codes_d = np.empty((vecs_d.shape[0], self.depth), dtype=self.code_dtype)

        if save_codebook:
            codebook_f = open(save_dir + '/' + dataset_name + '_rqgraph_' + str(self.depth) + '_' + str(self.Ks) + '_codebook', 'wb')

        for i in range(self.depth // 2):
            pq = self.pqs[i * (1 + self.Ks)]
            pq.fit(vecs, iter)

            codes[:, i * 2] = pq.encode(vecs).reshape(-1)
            vecs -= pq.decode(codes[:, i * 2].reshape((-1, 1)))

            if D is not None:
                codes_d[:, i * 2] = pq.encode(vecs_d).reshape(-1)
                vecs_d -= pq.decode(codes_d[:, i * 2].reshape((-1, 1)))

            if (i + 1) in save_residue_norms:
                with open(save_dir + '/' + dataset_name + '_rqgraph_' + str(i + 1) + '_' + str(self.Ks) + '_residue_norms', 'wb') as f:
                    if save_results_T:
                        np.linalg.norm(vecs, axis=1).tofile(f)
                    if D is not None:
                        np.linalg.norm(vecs_d, axis=1).tofile(f)

            if (i + 1) in save_decoded:
                with open(save_dir + '/' + dataset_name + '_rqgraph_' + str(i + 1) + '_' + str(self.Ks) + '_decoded', 'wb') as f:
                    if save_results_T:
                        (T - vecs).tofile(f)
                    if D is not None:
                        (D - vecs_d).tofile(f)

            if save_codebook:
                pq.codewords.tofile(codebook_f)
                codebook_f.flush()

            sub_pqs = self.pqs[i * (1 + self.Ks) + 1 : (i + 1) * (1 + self.Ks)]
            for k, sub_pq in enumerate(sub_pqs):
                sub_mask = (codes[:, i * 2] == k)
                sub_num = np.count_nonzero(sub_mask)
                if sub_num == 0:
                    # TODO: Handle this case correctly. This may cause problem if the training set is not the same as the set used for encoding
                    continue
                elif sub_num <= self.Ks:
                    sub_pq.codewords = vecs[sub_mask]
                    codes[sub_mask, i * 2 + 1] = range(sub_num)
                    vecs[sub_mask] = 0
                    continue
                
                sub_pq.fit(vecs[sub_mask], iter)

                codes[sub_mask, i * 2 + 1] = sub_pq.encode(vecs[sub_mask]).reshape(-1)
                vecs[sub_mask] -= sub_pq.decode(codes[sub_mask, i * 2 + 1].reshape((-1, 1)))

                if D is not None:
                    sub_mask_d = (codes_d[:, i * 2] == k)
                    codes_d[sub_mask_d, i * 2 + 1] = sub_pq.encode(vecs_d[sub_mask_d]).reshape(-1)
                    vecs_d[sub_mask_d] -= sub_pq.decode(codes_d[sub_mask_d, i * 2 + 1].reshape((-1, 1)))

                if save_codebook:
                    sub_pq.codewords.tofile(codebook_f)
                    codebook_f.flush()

            if (i + 2) in save_residue_norms:
                with open(save_dir + '/' + dataset_name + '_rqgraph_' + str(i + 2) + '_' + str(self.Ks) + '_residue_norms', 'wb') as f:
                    if save_results_T:
                        np.linalg.norm(vecs, axis=1).tofile(f)
                    if D is not None:
                        np.linalg.norm(vecs_d, axis=1).tofile(f)

            if (i + 2) in save_decoded:
                with open(save_dir + '/' + dataset_name + '_rqgraph_' + str(i + 2) + '_' + str(self.Ks) + '_decoded', 'wb') as f:
                    if save_results_T:
                        (T - vecs).tofile(f)
                    if D is not None:
                        (D - vecs_d).tofile(f)

        if self.depth % 2 == 1:
            pq = self.pqs[-1]
            pq.fit(vecs, iter)

            codes[:, -1] = pq.encode(vecs).reshape(-1)
            vecs -= pq.decode(codes[:, -1].reshape((-1, 1)))

            if D is not None:
                codes_d[:, -1] = pq.encode(vecs_d).reshape(-1)
                vecs_d -= pq.decode(codes_d[:, -1].reshape((-1, 1)))

            if self.depth in save_residue_norms:
                with open(save_dir + '/' + dataset_name + '_rqgraph_' + str(self.depth) + '_' + str(self.Ks) + '_residue_norms', 'wb') as f:
                    if save_results_T:
                        np.linalg.norm(vecs, axis=1).tofile(save_dir + '/' + dataset_name + '_rqgraph_' + str(self.depth) + '_' + str(self.Ks) + '_residue_norms')
                    if D is not None:
                        np.linalg.norm(vecs_d, axis=1).tofile(save_dir + '/' + dataset_name + '_rqgraph_' + str(self.depth) + '_' + str(self.Ks) + '_residue_norms')

            if self.depth in save_decoded:
                with open(save_dir + '/' + dataset_name + '_rqgraph_' + str(self.depth) + '_' + str(self.Ks) + '_decoded', 'wb') as f:
                    if save_results_T:
                        (T - vecs).tofile(f)
                    if D is not None:
                        (D - vecs_d).tofile(f)

            if save_codebook:
                pq.codewords.tofile(codebook_f)
                codebook_f.flush()

        if save_codebook:
            codebook_f.close()

        if D is not None:
            return codes, vecs, codes_d, vecs_d
        else:
            return codes, vecs

    def encode(self, X):
        N, D = X.shape

        vecs = np.empty(shape=X.shape, dtype=X.dtype)
        vecs[:, :] = X[:, :]

        codes = np.empty((N, self.depth), dtype=self.code_dtype)

        for i in range(self.depth // 2):
            pq = self.pqs[i * (1 + self.Ks)]

            codes[:, i * 2] = pq.encode(vecs).reshape(N)
            vecs -= pq.decode(codes[:, i * 2].reshape((N, 1)))

            sub_pqs = self.pqs[i * (1 + self.Ks) + 1 : (i + 1) * (1 + self.Ks)]
            for k, sub_pq in enumerate(sub_pqs):
                sub_mask = (codes[:, i * 2] == k)
                if np.count_nonzero(sub_mask) == 0:
                    continue

                codes[sub_mask, i * 2 + 1] = sub_pq.encode(vecs[sub_mask]).reshape(-1)
                vecs[sub_mask] -= sub_pq.decode(codes[sub_mask, i * 2 + 1].reshape((-1, 1)))

        if self.depth % 2 == 1:
            pq = self.pqs[-1]

            codes[:, -1] = pq.encode(vecs).reshape(-1)
            vecs -= pq.decode(codes[:, -1].reshape((-1, 1)))

        return codes, vecs


    def compress(self, X):
        _, vecs = self.encode(X)
        return X - vecs

