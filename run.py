from vecs_io import *
import math
from pq_residual import *
from sorter import *
from transformer import *
from aq import AQ
from opq import OPQ
from rq_graph import RQGraph
import argparse


def chunk_compress(pq, vecs):
    chunk_size = 1000000
    compressed_vecs = np.empty(shape=vecs.shape, dtype=np.float32)
    for i in tqdm.tqdm(range(math.ceil(len(vecs) / chunk_size))):
        compressed_vecs[i * chunk_size: (i + 1) * chunk_size, :] \
            = pq.compress(vecs[i * chunk_size: (i + 1) * chunk_size, :].astype(dtype=np.float32))
    return compressed_vecs


def rank(pq, X, Q, G, metric='euclid'):
    print('# compress items')
    compressed = chunk_compress(pq, X)

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
    parser.add_argument('--data_type', type=str, default='fvecs', help='data type of base and queries')
    parser.add_argument('--topk', type=int, help='topk of ground truth')
    parser.add_argument('--metric', type=str, help='metric of ground truth, euclid by default')
    parser.add_argument('--ranker', type=str, help='metric of ranker, euclid by default')
    parser.add_argument('--rank', type=bool, help='should rank', default=True)
    parser.add_argument('--train_size', type=int, help='train size', default=100000)

    parser.add_argument('--save_codebook', type=bool, help='should save codebook', default=False)
    parser.add_argument('--save_results_X', type=bool, help='should save results of X', default=False)
    parser.add_argument('--save_results_T', type=bool, help='should save results of T', default=False)
    parser.add_argument('--save_results_Q', type=bool, help='should save results of Q', default=False)
    parser.add_argument('--save_decoded', type=bool, help='should save decoded vectors', default=False)
    parser.add_argument('--save_residue_norms', type=bool, help='should save residue norms', default=False)
    parser.add_argument('--save_decoded_stages', type=str, help='stages to save decoded vectors', default='')
    parser.add_argument('--save_residue_norms_stages', type=str, help='stages to save residue norms', default='')
    parser.add_argument('--save_dir', type=str, help='dir to save results', default='./results')

    parser.add_argument('--num_codebook', type=int, help='number of codebooks')
    parser.add_argument('--Ks', type=int, help='number of centroids in each sub-dimension/sub-quantizer')
    parser.add_argument('--layer', type=int, help='number of layers for residual PQ', default=1)
    parser.add_argument('--norm_centroid', type=int, help='number of norm centroids for NormPQ')
    parser.add_argument('--true_norm', type=bool, help='use true norm for NormPQ', default=False)

    args = parser.parse_args()

    X, T, Q, G = loader(args.dataset, args.topk, args.metric, data_type=args.data_type)
    if T is None:
        T = X[:args.train_size]

    # pq, rq, or component of norm-pq
    if args.quantizer in ['PQ'.lower(), 'RQ'.lower()]:
        pqs = [PQ(M=args.num_codebook, Ks=args.Ks) for _ in range(args.layer)]
        quantizer = ResidualPQ(pqs=pqs)
    elif args.quantizer in ['OPQ'.lower()]:
        pqs = [OPQ(M=args.num_codebook, Ks=args.Ks) for _ in range(args.layer)]
        quantizer = ResidualPQ(pqs=pqs)
    elif args.quantizer == 'AQ'.lower():
        quantizer = AQ(M=args.num_codebook, Ks=args.Ks)
    elif args.quantizer == 'RQGraph'.lower():
        quantizer = RQGraph(Ks=args.Ks, depth=args.layer)
    else:
        assert False
    if args.sup_quantizer == 'NormPQ'.lower():
        quantizer = NormPQ(args.norm_centroid, quantizer, true_norm=args.true_norm)

    np.random.seed(123)
    print("# "+quantizer.class_message())

    if args.save_decoded or args.save_residue_norms:
        to_decode = []
        if args.save_results_X:
            to_decode.append(X)
        if args.save_results_Q:
            to_decode.append(Q)
        if len(to_decode) == 1:
            D = to_decode[0]
        else:
            D = np.concatenate(to_decode)
    else:
        D = None

    if args.quantizer in ['RQ'.lower(), 'RQGraph'.lower()]:
        if args.save_decoded_stages == 'all':
            save_decoded = list(range(1, args.layer + 1))
        else:
            save_decoded = list(map(int, args.save_decoded_stages.split(',')))
        if args.save_residue_norms_stages == 'all':
            save_residue_norms = list(range(1, args.layer + 1))
        else:
            save_residue_norms = list(map(int, args.save_residue_norms_stages.split(',')))

        quantizer.fit(T.astype(dtype=np.float32), iter=20, save_codebook=args.save_codebook, save_decoded=save_decoded, save_residue_norms=save_residue_norms, save_results_T=args.save_results_T, dataset_name=args.dataset, save_dir=args.save_dir, D=D)
    else:
        quantizer.fit(T.astype(dtype=np.float32), iter=20)

        if args.save_codebook:
            quantizer.codewords.tofile(save_dir + '/' + dataset_name + '_' + args.quantizer.lower() + '_' + str(args.num_codebook) + '_' + str(args.Ks) + '_codebook')
        if args.save_decoded or args.save_residue_norms:
            if args.save_results_T:
                T_d = quantizer.compress(T)
            D_d = quantizer.compress(D)
        if args.save_decoded:
            with open(save_dir + '/' + dataset_name + '_' + args.quantizer.lower() + '_' + str(args.num_codebook) + '_' + str(args.Ks) + '_decoded', 'wb'):
                if args.save_results_T:
                    T_d.tofile(f)
                if D is not None:
                    D_d.tofile(f)
        if args.save_residue_norms:
            with open(save_dir + '/' + dataset_name + '_' + args.quantizer.lower() + '_' + str(args.num_codebook) + '_' + str(args.Ks) + '_residue_norms', 'wb'):
                if args.save_results_T:
                    np.linalg.norm(T - T_d, axis=1).tofile(f)
                if D is not None:
                    np.linalg.norm(D - D_d, axis=1).tofile(f)

    if args.rank:
        rank(quantizer, X, Q, G, args.ranker)

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
