import matplotlib.pyplot as plt
from numpy.linalg import norm
from vecs_io import loader
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
    X, Q, G = loader(data_set, top_k, metric)

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

    X, Q, G = loader(data_set, top_k, metric if ground_metric is None else ground_metric)

    if transformer is not None:
        X, Q = transformer(X, Q)

    pq.fit(X, iter=20, seed=1007)
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
    top_k = 20
    Ks = 32
    data_set = 'netflix'
    M = 1

    pqs = [PQ(M, Ks), PQ(M, Ks)]
    pq = ResidualPQ(pqs=pqs)

    execute(pq, metric=metric, data_set=data_set, top_k=top_k)
