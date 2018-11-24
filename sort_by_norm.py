import math
from transformer import *
from vecs_io import loader
from pq_norm import NormPQ
from pq import PQ

"""
Only sort by norms, without considering others.
Result Report:
- with Netflix, slightly better than KMeans.
- with YahooMusic, Slightly worse than KMeans.
"""


class NormSorter(object):
    def __init__(self, norms):
        self.norms = norms
        self.topK = np.argsort(-self.norms)

    def recall(self, G, T):
        t = min(T, len(self.topK))
        top_k = [len(np.intersect1d(G[i], self.topK[:T])) for i in range(len(G))]
        return t, np.mean(top_k) / len(G[0])


def execute():
    X, Q, G = loader('yahoomusic', 20, 'product')
    pq = NormPQ(256, PQ(1, 256))
    np.random.seed(123)
    pq.fit(X, 20)
    queries = NormSorter(pq.decode_norm(pq.encode_norm(np.linalg.norm(X, axis=1))))

    print("expected items, overall time, avg recall, avg precision, avg error, avg items")
    for item in [2 ** i for i in range(2+int(math.log2(len(X))))]:
        actual_items, recall = queries.recall(G, item)
        print("{}, {}, {}, {}, {}, {}".format(
            item, 0, recall, recall * len(G[0]) / actual_items, 0, actual_items))


if __name__ == '__main__':
    execute()