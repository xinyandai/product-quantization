from pq import *
from multiprocessing import cpu_count
import numba as nb
import math


@nb.jit
def arg_sort(distances):
    top_k = min(131072, len(distances)-1)
    indices = np.argpartition(distances, top_k)[:top_k]
    return indices[np.argsort(distances[indices])]


@nb.jit
def product_arg_sort(q, compressed, norms_sqr, distances):
    np.dot(compressed, q, out=distances)
    distances[:] = - distances
    return arg_sort(distances)


@nb.jit
def angular_arg_sort(q, compressed, norms_sqr, distances):
    norm_q = np.linalg.norm(q)
    for i in range(compressed.shape[0]):
        distances[i] = -np.dot(q, compressed[i]) / (norm_q * norms_sqr[i])
    return arg_sort(distances)


@nb.jit
def euclidean_arg_sort(q, compressed, norms_sqr, distances):
    distances = [np.linalg.norm(q - center) for center in compressed]
    for i in range(len(compressed)):
        distances[i] = np.linalg.norm(q - compressed[i])
    return arg_sort(distances)


@nb.jit
def sign_arg_sort(q, compressed, norms_sqr, distances):
    for i in range(len(compressed)):
        distances[i] = np.count_nonzero((q > 0) != (compressed[i] > 0))
    return arg_sort(distances)


@nb.jit
def euclidean_norm_arg_sort(q, compressed, norms_sqr, distances):
    for i in range(len(compressed)):
        distances[i] = norms_sqr[i] - 2.0 * np.dot(compressed[i], q)
    return arg_sort(distances)


@nb.jit
def parallel_sort(metric, compressed, Q, X):
    """
    for each q in 'Q', sort the compressed items in 'compressed' by their distance,
    where distance is determined by 'metric'
    :param metric: euclid product
    :param compressed: compressed items, same dimension as origin data, shape(N * D)
    :param Q: queries, shape(len(Q) * D)
    :return:
    """

    norms_sqr = np.linalg.norm(X, axis=1) ** 2
    rank = np.empty((Q.shape[0], min(131072, compressed.shape[0]-1)), dtype=np.int32)
    tmp_distance = np.empty(shape=(compressed.shape[0]), dtype=X.dtype)

    if metric == 'product':
        for i in nb.prange(Q.shape[0]):
            rank[i, :] = product_arg_sort(Q[i], compressed, norms_sqr, tmp_distance)
    elif metric == 'angular':
        for i in nb.prange(Q.shape[0]):
            rank[i, :] = angular_arg_sort(Q[i], compressed, norms_sqr, tmp_distance)
    elif metric == 'euclid_norm':
        for i in nb.prange(Q.shape[0]):
            rank[i, :] = euclidean_norm_arg_sort(Q[i], compressed, norms_sqr, tmp_distance)
    else:
        for i in nb.prange(Q.shape[0]):
            rank[i, :] = euclidean_arg_sort(Q[i], compressed, norms_sqr, tmp_distance)

    return rank


class Sorter(object):
    def __init__(self, compressed, Q, X, metric):
        self.Q = Q
        self.X = X

        self.topK = parallel_sort(metric, compressed, Q, X)

    def recall(self, G, T):
        t = min(T, len(self.topK[0]))
        return t, self.sum_recall(G, T) / len(self.Q)

    def sum_recall(self, G, T):
        assert len(self.Q) == len(self.topK), "number of query not equals"
        assert len(self.topK) <= len(G), "number of queries should not exceed the number of queries in ground truth"
        true_positive = [len(np.intersect1d(G[i], self.topK[i][:T])) for i in range(len(self.Q))]
        return np.sum(true_positive) / len(G[0])  # TP / K


class BatchSorter(object):
    def __init__(self, compressed, Q, X, G, Ts, metric, batch_size):
        self.Q = Q
        self.X = X
        self.recalls = np.zeros(shape=(len(Ts)))
        for i in range(math.ceil(len(Q) / float(batch_size))):
            q = Q[i*batch_size: (i + 1) * batch_size, :]
            g = G[i*batch_size: (i + 1) * batch_size, :]
            sorter = Sorter(compressed, q, X, metric=metric)
            self.recalls[:] = self.recalls[:] + [sorter.sum_recall(g, t) for t in Ts]
        self.recalls = self.recalls / len(self.Q)

    def recall(self):
        return self.recalls
