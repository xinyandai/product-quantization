from pq import *
import numba as nb

@nb.njit
def arg_sort(metric, compressed, q):
    """
    for q, sort the compressed items in 'compressed' by their distance,
    where distance is determined by 'metric'
    :param arguments: a tuple of (metric, compressed, q)
    :return:
    """

    if metric == 0:
        tmp = -np.dot(compressed, q)
    elif metric == 1:
        tmp = -np.dot(compressed[:, :-1], q) + compressed[:, -1]
    elif metric == 2:
        tmp = np.empty((compressed.shape[0]), dtype=np.float32)
        norm_q = np.linalg.norm(q)
        for i in range(compressed.shape[0]):
            tmp[i] = -np.dot(q, compressed[i]) / (norm_q * np.linalg.norm(compressed[i]))
    else:
        tmp2 = compressed - q
        tmp = np.empty((compressed.shape[0]), dtype=np.float32)
        for i in range(compressed.shape[0]):
            tmp[i] = np.dot(tmp2[i], tmp2[i])

    return np.argsort(tmp).astype(np.int32)

@nb.njit(parallel=True)
def parallel_sort(metric, compressed, Q):
    """
    for each q in 'Q', sort the compressed items in 'compressed' by their distance,
    where distance is determined by 'metric'
    :param metric: euclid product
    :param compressed: compressed items, same dimension as origin data, shape(N * D)
    :param Q: queries, shape(len(Q) * D)
    :return:
    """
    rank = np.zeros((Q.shape[0], compressed.shape[0]), dtype=np.int32)
    for i in nb.prange(Q.shape[0]):
        rank[i] = arg_sort(metric, compressed, Q[i])

    return rank


class Sorter(object):
    def __init__(self, compressed, Q, X, metric='euclid'):
        self.Q = Q
        if metric == 'product':
            metric_int = 0
        elif metric == 'product_plus_half_mean_sqr':
            metric_int = 1
        elif metric == 'angular':
            metric_int = 2
        else:
            metric_int = 3
        self.topK = parallel_sort(metric_int, compressed, Q)

    def recall(self, G, T):
        t = min(T, len(self.topK[0]))
        top_k = [len(np.intersect1d(G[i], self.topK[i][:T])) for i in range(self.Q.shape[0])]
        return t, np.mean(top_k) / len(G[0])
