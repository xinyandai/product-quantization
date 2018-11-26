from pq import PQ
import numpy as np
from pq_residual import ResidualPQ


class OPQ(object):
    """Pure python implementation of Optimized Product Quantization (OPQ) [Ge14]_.

    OPQ is a simple extension of PQ.
    The best rotation matrix `R` is prepared using training vectors.
    Each input vector is rotated via `R`, then quantized into PQ-codes
    in the same manner as the original PQ.

    .. [Ge14] T. Ge et al., "Optimized Product Quantization", IEEE TPAMI 2014

    Args:
        M (int): The number of sub-spaces
        Ks (int): The number of codewords for each subspace (typically 256, so that each sub-vector is quantized
            into 256 bits = 1 byte = uint8)
        verbose (bool): Verbose flag

    Attributes:
        R (np.ndarray): Rotation matrix with the shape=(D, D) and dtype=np.float32


    """
    def __init__(self, M, Ks, verbose=True, layer=1):

        self.pq = ResidualPQ([PQ(M, Ks, verbose) for _ in range(layer)])
        self.layer = layer
        self.M = M
        self.Ks = Ks
        self.code_dtype = self.pq.code_dtype
        self.verbose = verbose

        self.R = None

    def class_message(self):
        return "ORQ, RQ : [{}],  M: {}, Ks : {}, code_dtype: {}".format(
            self.pq.class_message(), self.M, self.Ks, self.code_dtype)

    def fit(self, vecs, iter):

        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        _, D = vecs.shape
        self.R = np.eye(D, dtype=np.float32)

        rotation_iter = iter
        pq_iter = iter

        from tqdm import tqdm
        iterator = tqdm(range(rotation_iter)) if self.verbose else range(rotation_iter)
        for i in iterator:
            X = vecs @ self.R

            # (a) Train codewords

            if i == rotation_iter - 1:
                # stop iterator display; show the pq process bar
                if type(iterator) is tqdm:
                    iterator.close()
                # In the final loop, run the full training
                pq_tmp = ResidualPQ([PQ(self.M, self.Ks, self.verbose) for _ in range(self.layer)], verbose=self.verbose)
                pq_tmp.fit(X, iter=pq_iter)
            else:
                # During the training for OPQ, just run one-pass (iter=1) PQ training
                pq_tmp = ResidualPQ([PQ(self.M, self.Ks, False) for _ in range(self.layer)],  verbose=False)
                pq_tmp.fit(X, iter=1)

            # (b) Update a rotation matrix R
            X_ = pq_tmp.compress(X)
            U, s, V = np.linalg.svd(vecs.T @ X_)

            if i == rotation_iter - 1:
                self.pq = pq_tmp
                break
            else:
                self.R = U @ V

        return self

    def rotate(self, vecs):

        assert vecs.dtype == np.float32
        assert vecs.ndim in [1, 2]

        if vecs.ndim == 2:
            return vecs @ self.R
        elif vecs.ndim == 1:
            return (vecs.reshape(1, -1) @ self.R).reshape(-1)

    def encode(self, vecs):

        return self.pq.encode(self.rotate(vecs))

    def decode(self, codes):
        # Because R is a rotation matrix (R^t * R = I), R^-1 should be R^t
        return self.pq.decode(codes) @ self.R.T

    def compress(self, vecs):
        return self.decode(self.encode(vecs))









