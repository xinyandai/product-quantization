from vecs_io import *
import math
from pq_residual import *
from sorter import *
from transformer import *
from aq import AQ
from opq import OPQ
import argparse


def execute(pq, X, Q, G, metric='euclid', train_size=100000):
    np.random.seed(123)
    print("# ranking metric {}".format(metric))
    print("# "+pq.class_message())
    pq.fit(X[:train_size], iter=20)
    print('# compress items')
    compressed = pq.compress(X)
    print("# sorting items")
    Ts = [2 ** i for i in range(2+int(math.log2(len(X))))]
    recalls = BatchSorter(compressed, Q, X, G, Ts, metric=metric, batch_size=200).recall()
    print("# searching!")

    print("expected items, overall time, avg recall, avg precision, avg error, avg items")
    for i, (t, recall) in enumerate(zip(Ts, recalls)):
        print("{}, {}, {}, {}, {}, {}".format(
            2**i, 0, recall, recall * len(G[0]) / t, 0, t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input method and parameters.')
    parser.add_argument('-q', '--quantizer', type=str.lower,
                        help='choose the quantizer to use, RQ, PQ, NormPQ, AQ, ...')
    parser.add_argument('--sup_quantizer', type=str.lower,
                        help='choose the sup_quantizer: NormPQ')
    parser.add_argument('--dataset', type=str, help='choose data set name')
    parser.add_argument('--topk', type=int, help='topk of ground truth')
    parser.add_argument('--metric', type=str, help='metric of ground truth, euclid by default')
    parser.add_argument('--ranker', type=str, help='metric of ranker, euclid by default')

    parser.add_argument('--num_codebook', type=int, help='number of codebooks')
    parser.add_argument('--Ks', type=int, help='number of centroids in each sub-dimension/sub-quantizer')
    parser.add_argument('--layer', type=int, help='number of layers for residual PQ')
    parser.add_argument('--norm_centroid', type=int, help='number of norm centroids for NormPQ')
    parser.add_argument('--true_norm', type=bool, help='use true norm for NormPQ', default=False)

    args = parser.parse_args()

    X, Q, G = loader(args.dataset, args.topk, args.metric)

    # pq, rq, or component of norm-pq
    if args.quantizer in ['PQ'.lower(), 'RQ'.lower()]:
        pqs = [PQ(M=args.num_codebook, Ks=args.Ks) for _ in range(args.layer)]
        quantizer = ResidualPQ(pqs=pqs)
    elif args.quantizer in ['OPQ'.lower()]:
        pqs = [OPQ(M=args.num_codebook, Ks=args.Ks) for _ in range(args.layer)]
        quantizer = ResidualPQ(pqs=pqs)
    elif args.quantizer == 'AQ'.lower():
        quantizer = AQ(M=args.num_codebook, Ks=args.Ks)
    else:
        assert False
    if args.sup_quantizer == 'NormPQ'.lower():
        quantizer = NormPQ(args.norm_centroid, quantizer, true_norm=args.true_norm)

    execute(quantizer, X, Q, G, args.ranker)

    # TODO(Kelvin): The following code is not used, but is left there as a record. When they are needed, I will merge them into the current framework
    #def aq_fast_norm():
    #    X, Q, G = loader(data_set, top_k, 'euclid', folder='../data/')
    #    Q = Q[:200]
    #    X, Q = scale(X, Q)
    #    norm_sqrs = np.linalg.norm(X, axis=1) ** 2
    #    means = norm_range(norm_sqrs, 128)
    #    means /= 2.0
    #    quantizer = AQ(4, 256)

    #    print('fitting')
    #    quantizer.fit(X[:100000], 20, seed=1007)
    #    print('compress items')
    #    compressed = quantizer.compress(X)
    #    print("sorting items")
    #    queries = Sorter(np.column_stack((compressed, means)), Q, metric='product_plus_half_mean_sqr')
    #    print("searching!")

    #    for item in [2 ** i for i in range(2+int(math.log2(len(X))))]:
    #        actual_items, recall = queries.recall(G, item)
    #        print("items {}, actual items {}, recall {}".format(item, actual_items, recall))

    #def aq_euclid_transform():
    #    X, Q, G = loader(data_set, top_k, 'euclid', folder='../data/')
    #    X, Q = scale(X, Q)
    #    X, Q = zero_mean(X, Q)
    #    X, Q = e2m_transform(X, Q)
    #    quantizer = AQ(4, 256)
    #    execute(quantizer, X, Q, G, 'product')

    #def aq_euclid_transform_inverse_d_coeff_scale():
    #    X, Q, G = loader(data_set, top_k, 'euclid', folder='../data/')
    #    X, Q = zero_mean(X, Q)
    #    X, Q = inverse_d_coeff_scale(X, Q);
    #    X, Q = e2m_transform(X, Q)
    #    quantizer = AQ(4, 256)
    #    execute(quantizer, X, Q, G, 'product')

    #def aq_euclid_transform_norm_range():
    #    X, Q, G = loader(data_set, top_k, 'euclid', folder='../data/')
    #    X, Q = zero_mean(X, Q)
    #    #X, Q = one_half_coeff_scale(X, Q)
    #    X, Q = coeff_scale(X, Q, 1.0/25.0)
    #    X, Q = e2m_transform(X, Q)
    #    means = norm_range(X[:, -1], 256)
    #    X[:, -1] -= means
    #    means /= 2.0
    #    quantizer = AQ(4, 256)

    #    print('fitting')
    #    quantizer.fit(X[:100000], 20, seed=1007)
    #    print('compress items')
    #    compressed = quantizer.compress(X)
    #    print("sorting items")
    #    queries = Sorter(np.column_stack((compressed, means)), Q, metric='product_plus_half_mean_sqr')
    #    print("searching!")

    #    for item in [2 ** i for i in range(2+int(math.log2(len(X))))]:
    #        actual_items, recall = queries.recall(G, item)
    #        print("items {}, actual items {}, recall {}".format(item, actual_items, recall))
