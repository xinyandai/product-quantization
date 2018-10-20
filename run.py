import matplotlib.pyplot as plt
from numpy.linalg import norm
from vecs_io import *
import math
from pq_residual import *
from sorter import *
from transformer import *


def draw():
    bins = 200
    data_set = 'netflix'
    M = 1
    Ks = 128
    deep = 2
    folder_path= '../data/%s' % data_set
    file_path = folder_path + '/%s_base.fvecs' % data_set
    query_path = folder_path + '/%s_query.fvecs' % data_set
    X = fvecs_read(file_path)
    Q = fvecs_read(query_path)

    residual_pq = ResidualPQ(M=1, Ks=Ks, deep=deep)
    residual_pq.fit(X, iter=20, seed=808)
    residual_compressed = residual_pq.compress(X)

    pq = PQ(M=M, Ks=Ks)
    pq.fit(X, iter=20, seed=808)
    compressed = pq.compress(X)

    mse_eorrs = [np.linalg.norm(X[i] - compressed[i]) for i in range(len(X))]
    norms = [np.linalg.norm(x) for x in X]
    norm_errors = [np.linalg.norm(X[i]) - np.linalg.norm(compressed[i]) for i in range(len(X))]
    relative_errors = [1.0 - np.linalg.norm(compressed[i]) / np.linalg.norm(X[i]) for i in range(len(X))]

    inner_erros = [norm(1 - residual_compressed[i] / norm(X[i])) for i in range(len(X))]
    angular_errors = [norm(1 - residual_compressed[i]/norm(residual_compressed[i])) for i in range(len(X))]

    plt.hist(mse_eorrs, bins=bins)
    plt.title('%s with %s residual deep %s code book with %s centers errors' % (data_set, deep, M, Ks))
    plt.show()


def loader(metric='euclid', ground_metric=None, data_set='audio', top_k=20):
    folder_path = '../data/%s' % data_set
    base_file = folder_path + '/%s_base.fvecs' % data_set
    query_file = folder_path + '/%s_query.fvecs' % data_set
    ground_truth = folder_path + '/%s_%s_%s_groundtruth.ivecs' % \
                   (top_k, data_set, metric if ground_metric is None else ground_metric)

    print("load the base data {}, \nload the queries {}, \nload the ground truth {}".format(base_file, query_file,
                                                                                            ground_truth))
    X = fvecs_read(base_file)
    Q = fvecs_read(query_file)
    G = ivecs_read(ground_truth)
    return X, Q, G


def execute(pq, metric='euclid', ground_metric=None, data_set='audio', top_k=20, transformer=None):

    X, Q, G = loader(metric, ground_metric, data_set, top_k)

    if transformer is not None:
        X, Q = transformer(X, Q)

    pq.fit(X, iter=20, seed=808)
    print('compress items')
    compressed = pq.compress(X)
    print("sorting items")
    queries = Sorter(compressed, Q, metric=metric)
    print("searching!")

    for item in range(0, 4000, 200):
        actual_items, recall = queries.recall(G, item)
        print("items {}, actual items {}, recall {}".format(item, actual_items, recall))
    print('-------------------------------------------------------------------------------------')

    for item in [2 ** i for i in range(2+int(math.log2(len(X))))]:
        actual_items, recall = queries.recall(G, item)
        print("items {}, actual items {}, recall {}".format(item, actual_items, recall))


if __name__ == '__main__':
    metric = 'product'
    ground_metric = metric
    top_k = 10
    Ks = 256
    data_set = 'sift1m'
    M = 1
    deep = 8
    n_percentile = 0
    norm_pq_layer = 0
    pq = ResidualPQ(M=M, Ks=Ks, deep=deep, n_percentile=n_percentile, norm_pq_layer=norm_pq_layer, true_norm=False)

    # X, Q, G = loader(metric, ground_metric, data_set, top_k)
    # pq.fit(X, iter=20, seed=808)
    #
    # vecs = X
    # compressed = np.zeros((pq.deep, len(X), len(X[0])), dtype=X.dtype)
    # for i, pq in enumerate(pq.pqs):
    #     compressed[i][:][:] = pq.compress(vecs)
    #     vecs = vecs - compressed[i][:][:]
    # norms = np.linalg.norm(compressed, axis=2)
    # for i in range(deep-1):
    #     print(np.count_nonzero(norms[i] < norms[i+1]))
    execute(pq, metric=metric, data_set=data_set, top_k=top_k)
