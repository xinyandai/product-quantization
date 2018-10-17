import matplotlib.pyplot as plt
from numpy.linalg import norm
from vecs_io import *
from pq_norm import *
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


def execute(pq, metric='euclid', ground_metric=None, data_set='audio', top_k=20, transformer=None):
    folder_path = '../data/%s' % data_set
    file_path = folder_path + '/%s_base.fvecs' % data_set
    query_path = folder_path + '/%s_query.fvecs' % data_set
    ground_truth = folder_path + '/%s_%s_%s_groundtruth.ivecs' % \
                   (top_k, data_set, metric if ground_metric is None else ground_metric)

    print("load the base data {}, the queries {}, and the ground truth {}".format(file_path, query_path, ground_truth))
    X = fvecs_read(file_path)
    Q = fvecs_read(query_path)
    G = ivecs_read(ground_truth)

    if transformer is not None:
        X, Q = transformer(X, Q)

    pq.fit(X, iter=20, seed=808)
    print("sorting items")
    queries = Sorter(pq, Q, metric=metric, X=X)
    print("searching!")

    print("items {}, actual items {}, recall {}".format(0, 0, queries.recall(G)[1]))
    item = 1
    while item < len(X):
        queries.probe(item)
        actual_items, recall = queries.recall(G)
        print("items {}, actual items {}, recall {}".format(item, actual_items, recall))
        item = item * 2

    queries.probe(len(X))
    print("items {}, actual items {}, recall {}".format(len(X), len(X), queries.recall(G)[1]))


def run_pq():
    M = 4
    Ks = 32
    deep = 1

    print("train the residual PQ with {} code book and {} layer of residual, each kmeans has {} centers"
          .format(M, deep, Ks))
    residual_pq = ResidualPQ(M=M, Ks=Ks, deep=deep)
    execute(residual_pq, data_set='audio', top_k=20)


def run_aq():
    M = 1
    Ks = 256
    deep = 4

    print("train the residual PQ with {} code book and {} layer of residual, each kmeans has {} centers"
          .format(M, deep, Ks))
    residual_pq = ResidualPQ(M=M, Ks=Ks, deep=deep)
    execute(residual_pq, metric='product', ground_metric='euclid', data_set='audio', top_k=20, transformer=e2m_transform)


def run_norm_pq():
    n_percentile = 32
    Ks = 32
    deep = 1

    print("train the NormPQ with {} percentile and {} layer of residual, each kmeans has {} centers"
          .format(n_percentile, deep, Ks))
    norm_pq = NormPQ(n_percentile=n_percentile, Ks=Ks)
    execute(norm_pq, metric='product', data_set='netflix', top_k=20)


if __name__ == '__main__':
    run_norm_pq()
