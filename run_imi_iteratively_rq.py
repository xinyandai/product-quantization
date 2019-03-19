from sorter import *
from transformer import *
from vecs_io import loader
from pq_residual import ResidualPQ

def chunk_compress(pq, vecs):
    chunk_size = 1000000
    compressed_vecs = np.empty(shape=vecs.shape, dtype=np.float32)
    for i in tqdm.tqdm(range(math.ceil(len(vecs) / chunk_size))):
        compressed_vecs[i * chunk_size: (i + 1) * chunk_size, :] \
            = pq.compress(vecs[i * chunk_size: (i + 1) * chunk_size, :].astype(dtype=np.float32))
    return compressed_vecs

def parse_args():
    # override default parameters with command line parameters
    import argparse
    parser = argparse.ArgumentParser(description='Process input method and parameters.')
    parser.add_argument('--dataset', type=str, help='choose data set name')
    parser.add_argument('--topk', type=int, help='required topk of ground truth')
    parser.add_argument('--metric', type=str, help='metric of ground truth')
    parser.add_argument('--num_codebook', type=int, help='number of codebooks')
    parser.add_argument('--Ks', type=int, help='number of centroids in each quantizer')
    args = parser.parse_args()
    return args.dataset, args.topk, args.num_codebook, args.Ks, args.metric


def run_ex(dataset, topk, codebook, Ks, metric):
    X, T, Q, G = loader(dataset, topk, metric, folder='data/')


    # pq, rq, or component of norm-pq
    imi = PQ(M=2, Ks=Ks)
    pq = ResidualPQ(pqs=[PQ(M=1, Ks=Ks) for _ in range(codebook // 2)])
    full_pq = ResidualPQ(pqs=[PQ(M=1, Ks=Ks) for _ in range(codebook)])

    if T is None:
        imi.fit(X[:100000].astype(dtype=np.float32), iter=20)
        pq.fit(X[:100000].astype(dtype=np.float32), iter=20)
        full_pq.fit(X[:100000].astype(dtype=np.float32), iter=20)
    else:
        imi.fit(T.astype(dtype=np.float32), iter=20)
        pq.fit(T.astype(dtype=np.float32), iter=20)
        full_pq.fit(T.astype(dtype=np.float32), iter=20)

    print('# compress items')
    x_candidate = chunk_compress(imi, X)
    x_compress = chunk_compress(pq, X)
    x_full_compress = chunk_compress(full_pq, X)

    norm_sqr = np.sum(x_full_compress * x_full_compress, axis=1, keepdims=True)
    norm_quantizer = PQ(M=1, Ks=256)
    norm_quantizer.fit(norm_sqr, iter=20)
    compressed_norm_sqr = norm_quantizer.compress(norm_sqr)

    print('# sort imi')
    sorted_candidate = parallel_sort(metric, x_candidate, Q, X, norms_sqr=None)

    for n_item, n_resort in [(65536, 65536//2)]:
        print('# probe {} items resort items '.format(n_item))

        I = np.empty((Q.shape[0], min(131072, n_item-1)), dtype=np.int32)
        for i in tqdm.tqdm(nb.prange(Q.shape[0])):
            top_k_candidate_id = sorted_candidate[i, :n_item]
            candidate_compressed = x_compress[top_k_candidate_id, :]
            candidate_norm_sqr = compressed_norm_sqr[top_k_candidate_id, 0]

            sort_by_pq = euclidean_norm_arg_sort(Q[i], candidate_compressed, candidate_norm_sqr)
            I[i, :] = sorted_candidate[i, sort_by_pq]

        sorted_candidate = I
        I = np.empty((Q.shape[0], min(131072, n_resort-1)), dtype=np.int32)
        for i in tqdm.tqdm(nb.prange(Q.shape[0])):
            top_k_candidate_id = sorted_candidate[i, :n_resort]
            candidate_compressed = x_full_compress[top_k_candidate_id, :]
            candidate_norm_sqr = compressed_norm_sqr[top_k_candidate_id, 0]
            sort_by_pq = euclidean_norm_arg_sort(Q[i], candidate_compressed, candidate_norm_sqr)
            I[i, :] = sorted_candidate[i, sort_by_pq]

        probe_size = n_item
        ranks = [1, 2, 16, 32, probe_size // 8, probe_size // 4, probe_size // 2, probe_size]
        print("probe_size,\t top  K \t:", end='')
        for rank in ranks:
            print("\t%4d" % (rank), end=' ')
        print()
        for topK in [1, 10, 20, 50, 100, 1000]:
            print("%d items,\t top  %d \t:" % (probe_size, topK), end=' ')
            for rank in ranks:
                n_ok = true_positives(I, Q, G[:, :topK], rank).sum()
                print("\t%.4f" % (n_ok / float(len(Q)) / topK), end=' ')
            print()


if __name__ == '__main__':
    dataset = 'imagenet'
    topk = 1000
    codebook = 16
    Ks = 256
    metric = 'euclid'

    # override default parameters with command line parameters
    import sys
    if len(sys.argv) > 3:
        dataset, topk, codebook, Ks, metric = parse_args()
    else:
        import warnings
        warnings.warn("Using  Default Parameters ")
    print("# Parameters: dataset = {}, topK = {}, codebook = {}, Ks = {}, metric = {}"
          .format(dataset, topk, codebook, Ks, metric))

    run_ex(dataset, topk, codebook, Ks, metric)

