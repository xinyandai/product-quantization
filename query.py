from pq import *


class Index(object):
    def __init__(self, pq):
        self.pq = pq
        # key : (deep, M)
        # value : list of compare
        self.container = [[] for _ in range(pq.num_code_book())]
        self.centers = [self.pq.decode(pq.index_to_code(i)) for i in range(len(self.container))]
        self.size = 0

    def add(self, vecs):
        # shape=(deep, N, M)
        codes = self.pq.encode(vecs)
        # shape=(N, deep, M)
        [self.container[self.pq.code_to_index(c)].append(i) for i, c in enumerate(codes)]
        self.size = len(codes)


class Query(object):
    def __init__(self, pq, Q, X=None, metric=None):
        def euclid(x, y):
            return np.linalg.norm(x - y)

        def inner_product(x, y):
            return - np.dot(np.array(x).flatten(), np.array(y).flatten())

        metric = inner_product if metric == 'product' else euclid

        self.index = Index(pq)
        self.index.add(X)

        # bottleneck
        self.bucket_dist = [np.argsort([metric(q, center) for center in self.index.centers]) for q in Q]
        self.Q = Q
        self.topK = np.full((len(Q), self.index.size), fill_value=-1, dtype=np.int32)
        self.filled = [0 for _ in Q]
        self.bucket_id = [0 for _ in Q]

    def probe(self, T):
        for i, (candidate, q, bucket) in enumerate(zip(
                self.topK, self.Q, self.bucket_dist)):
            while self.filled[i] < T:
                items = self.index.container[bucket[self.bucket_id[i]]]
                candidate[self.filled[i]:self.filled[i]+len(items)] = items
                self.filled[i] += len(items)
                self.bucket_id[i] += 1

    def recall(self, G):
        return np.mean(self.filled), np.mean([len(np.intersect1d(G[i], self.topK[i])) for i in range(len(G))]) / len(G[0])
