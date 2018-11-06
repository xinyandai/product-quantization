from vecs_io import *
import math
from pq_residual import *
from sorter import *
from transformer import *
from aq import AQ
from opq import OPQ


def execute(pq, X, Q, G, metric='euclid'):

    print("ranking metric {}".format(metric))
    print(pq.class_message())
    pq.fit(X[:100000], 20, seed=1007)
    print('compress items')
    compressed = pq.compress(X)
    print("sorting items")
    queries = Sorter(compressed, Q, X, metric=metric)
    print("searching!")

    for item in [2 ** i for i in range(2+int(math.log2(len(X))))]:
        actual_items, recall = queries.recall(G, item)
        print("items {}, actual items {}, recall {}".format(item, actual_items, recall))


if __name__ == '__main__':

    top_k = 20
    Ks = 256
    data_set = 'yahoomusic'
    metric = "product"

    def transform_based(cal_matrix=False):
        X, Q, G = loader(data_set, top_k, 'euclid')
        X, Q = zero_mean(X, Q)
        X, Q = scale(X, Q)
        # matrix = e2m_mahalanobis(X) if cal_matrix else None
        X, Q = e2m_transform(X, Q)
        X, _ = zero_mean(X, Q)

        matrix = np.dot(Q.transpose(), Q) / float(len(Q)) if cal_matrix else None

        pqs = [PQ(M=4, Ks=Ks) for _ in range(1)]
        pq = ResidualPQ(pqs=pqs)
        # pq = AQ(4, Ks)

        execute(pq, X, Q, G, 'product')

    def raw():
        X, Q, G = loader(data_set, top_k, 'product')
        X, Q = scale(X, Q)

        pqs = [PQ(M=1, Ks=Ks) for _ in range(4)]
        pq = ResidualPQ(pqs=pqs)
        execute(pq, X, Q, G, 'product')

        # opq = [OPQ(M=4, Ks=Ks) for _ in range(1)]
        # pq = ResidualPQ(pqs=opq)
        # execute(pq, X, Q, G, 'euclid')
        #
        # pqs = [PQ(M=1, Ks=Ks) for _ in range(4)]
        # pq = ResidualPQ(pqs=pqs)
        # execute(pq, X, Q, G, 'euclid')

    def aq():
        X, Q, G = loader(data_set, top_k, 'product', folder='../data/')
        # X, Q = scale(X, Q)
        quantizer = AQ(4, Ks)
        execute(quantizer, X, Q, G, 'product')

    def norm_pq():
        X, Q, G = loader(data_set, top_k, 'product', folder='../data/')
        # X, Q = scale(X, Q)
        pqs = [PQ(M=1, Ks=Ks) for _ in range(3)]
        rq = ResidualPQ(pqs=pqs)
        quantize = NormPQ(Ks, rq)
        execute(quantize, X, Q, G, 'product')

    norm_pq()
