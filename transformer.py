from scipy.stats import ortho_group
import numpy as np
import numba as nb
from scipy.cluster.vq import kmeans2, vq


def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1)
    norms_matrix = norms[:, np.newaxis]
    normalized_vecs = np.divide(vecs, norms_matrix, out=np.zeros_like(vecs), where=norms_matrix != 0)  # divide by zero problem
    return norms, normalized_vecs


def zero_mean(X, Q):
    mean = X.mean(axis=0, keepdims=True)
    X = X - mean
    Q = Q - mean
    return X, Q


def random_rotate(X, Q):
    R = ortho_group.rvs(dim=len(X[0]))
    R = np.array(R, dtype=np.float32)
    X = R.dot(X.transpose()).transpose()
    Q = R.dot(Q.transpose()).transpose()
    return X, Q


def scale(X, Q):
    scale = np.max(np.linalg.norm(X, axis=1))
    X /= scale
    Q /= scale
    return X, Q


def one_half_coeff_scale(X, Q):
    mean = np.mean(np.absolute(X))
    X /= (mean * 2);
    Q /= (mean * 2);
    return X, Q

def coeff_scale(X, Q, scale):
    mean = np.mean(np.absolute(X))
    X /= (mean / scale);
    Q /= (mean / scale);
    return X, Q

def inverse_d_coeff_scale(X, Q):
    mean = np.mean(np.absolute(X))
    X /= (mean * X.shape[1]);
    Q /= (mean * X.shape[1])
    return X, Q

@nb.jit(nopython=True)
def norm_range(norm_sqrs, num_intervals):
    num_intervals_minus_1 = float(num_intervals - 1)

    norm_sqr_max = np.amax(norm_sqrs)
    norm_sqr_min = np.amin(norm_sqrs)

    means = np.empty((norm_sqrs.shape[0]), dtype=np.float32)

    for i in range(norm_sqrs.shape[0]):
        bucket = int((norm_sqrs[i] - norm_sqr_min) / (norm_sqr_max - norm_sqr_min) * num_intervals_minus_1)
        left = bucket / num_intervals_minus_1 * (norm_sqr_max - norm_sqr_min) + norm_sqr_min
        right = (bucket + 1) / num_intervals_minus_1 * (norm_sqr_max - norm_sqr_min) + norm_sqr_min
        mean = (left + right) / 2.0

        means[i] = mean

    return means

def norm_range_non_uniform(norm_sqrs):
    code_book, code = kmeans2(norm_sqrs[:, np.newaxis], 256, iter=20, minit='points')
    return code_book[code, 0]

def e2m_transform(X, Q):
    M = np.max(np.linalg.norm(X, axis=1))
    X_plus = np.zeros((len(X), 4), dtype=np.float32)
    Q_plus = np.zeros((len(Q), 4), dtype=np.float32)

    X_plus[:, 0] = M - np.linalg.norm(X, axis=1) ** 2
    Q_plus[:, 0] = 0.5
    X = np.append(X, X_plus, axis=1)
    Q = np.append(Q, Q_plus, axis=1)
    return X, Q


def e2m_mahalanobis(X):
    X_plus = np.full((len(X), 1), fill_value=-0.5, dtype=np.float32)
    X = np.append(X, X_plus, axis=1)
    return np.dot(X.transpose(), X) / float(len(X))
