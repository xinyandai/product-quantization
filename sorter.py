from pq import *
import warnings
from multiprocessing import Pool, cpu_count, sharedctypes


def _init(arr_to_populate):
    """Each pool process calls this initializer. Load the array to be populated into that process's global namespace"""
    global arr
    arr = arr_to_populate


def arg_sort(arguments):
    """
    for q, sort the compressed items in 'compressed' by their distance,
    where distance is determined by 'metric'
    :param arguments: a tuple of (metric, compressed, q)
    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        compressed = np.ctypeslib.as_array(arr)
    metric, q = arguments
    if metric == 'product':
        return np.argsort([- np.dot(np.array(q).flatten(), np.array(center).flatten()) for center in compressed])
    else:
        return np.argsort([np.linalg.norm(q - center) for center in compressed])


def parallel_sort(metric, compressed, Q):
    """
    for each q in 'Q', sort the compressed items in 'compressed' by their distance,
    where distance is determined by 'metric'
    :param metric: euclid product
    :param compressed: compressed items, same dimension as origin data, shape(N * D)
    :param Q: queries, shape(len(Q) * D)
    :return:
    """
    tmp = np.ctypeslib.as_ctypes(compressed)
    shared_arr = sharedctypes.Array(tmp._type_, tmp, lock=False)

    pool = Pool(processes=cpu_count(), initializer=_init, initargs=(shared_arr, ))

    rank = pool.map(arg_sort, zip([metric for _ in Q], Q))
    pool.close()
    pool.join()
    return rank


class Sorter(object):
    def __init__(self, compressed, Q, metric='euclid'):
        self.Q = Q
        self.topK = parallel_sort(metric, compressed, Q)

    def recall(self, G, T):
        return min(T, len(self.topK[0])), np.mean([len(np.intersect1d(G[i], self.topK[i][:T])) for i in range(len(G))]) / len(G[0])
