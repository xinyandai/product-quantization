from pq import *


class Sorter(object):
    def __init__(self, pq, Q, metric='euclid', X=None):
        def euclid(x, y):
            return np.linalg.norm(x - y)

        def inner_product(x, y):
            return - np.dot(np.array(x).flatten(), np.array(y).flatten())

        metric = inner_product if metric == 'product' else euclid

        compressed = pq.compress(X)
        self.topK = [np.argsort([metric(q, center) for center in compressed]) for q in Q]
        self.Q = Q
        self.T = 0

    def probe(self, T):
        self.T = T

    def recall(self, G):
        return self.T, np.mean([len(np.intersect1d(G[i], self.topK[i][:self.T])) for i in range(len(G))]) / len(G[0])
