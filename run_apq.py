from vecs_io import *
import math
from pq_residual import *
from sorter import *
from transformer import *
from aq import AQ
from opq import OPQ
from xpq import XPQ
import argparse


def execute(pq, X, Q, G, metric='euclid', train_size=100000):
    np.random.seed(123)
    print("# ranking metric {}".format(metric))
    print("# "+pq.class_message())
    pq.fit(X[:train_size], iter=20)
    print('# compress items')
    compressed = pq.compress(X)
    print("# sorting items")
    queries = Sorter(compressed, Q, X, metric=metric)
    print("# searching!")

    print("expected items, overall time, avg recall, avg precision, avg error, avg items")
    for item in [2 ** i for i in range(2+int(math.log2(len(X))))]:
        actual_items, recall = queries.recall(G, item)
        print("{}, {}, {}, {}, {}, {}".format(
            item, 0, recall, recall * len(G[0]) / actual_items, 0, actual_items))


if __name__ == '__main__':
    import sys
    dataset = sys.argv[1]
    topk = int(sys.argv[2])
    codebook = int(sys.argv[3])
    Ks = int(sys.argv[4])

    X, Q, G = loader(dataset, topk, 'product')
    # pq, rq, or component of norm-pq
    quantizer = XPQ([AQ(codebook//2, Ks) for _ in range(2)])
    execute(quantizer, X, Q, G, 'product')
