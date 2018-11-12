from vecs_io import *
import math
from pq_residual import *
from sorter import *
from transformer import *
from aq import AQ
from opq import OPQ
import argparse


def execute(pq, X, Q, G, metric='euclid', train_size=100000):

    print("# ranking metric {}".format(metric))
    print("# "+pq.class_message())
    pq.fit(X[:train_size])
    print('# compress items')
    compressed = pq.compress(X)
    print("# sorting items")
    queries = Sorter(compressed, Q, X, metric=metric)
    print("# searching!")

    print("expected items, overall time, avg recall, avg precision, avg error, avg items")
    for item in [2 ** i for i in range(2+int(math.log2(len(X))))]:
        actual_items, recall = queries.recall(G, item)
        print("{}, {}, {}, {}, {}".format(
            item, 0, recall, recall * len(G[0]) / actual_items, 0, actual_items))


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

    parser.add_argument('-M', '--num_codebook', type=int, help='number of codebooks')
    parser.add_argument('--Ks', type=int, help='number of centroids in each sub-dimension/sub-quantizer')
    parser.add_argument('--layer', type=int, help='number of layers for residual PQ')
    parser.add_argument('--norm_centroid', type=int, help='number of norm centroids for NormPQ')

    args = parser.parse_args()

    X, Q, G = loader(args.dataset, args.topk, args.metric)
    X, Q = scale(X, Q)

    # pq, rq, or component of norm-pq
    if args.quantizer in ['PQ'.lower(), 'RQ'.lower()]:
        pqs = [PQ(M=args.num_codebook, Ks=args.Ks) for _ in range(args.layer)]
        quantizer = ResidualPQ(pqs=pqs)
    elif args.quantizer == 'AQ'.lower():
        quantizer = AQ(M=args.num_codebook, Ks=args.Ks)
    else:
        assert False
    if args.quantizer == 'sup_quantizer'.lower():
        quantizer = NormPQ(args.norm_centroid, quantizer)

    execute(quantizer, X, Q, G, args.ranker)
