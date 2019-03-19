from sorter import *
from transformer import *
from vecs_io import loader


def chunk_compress(pq, vecs):
    chunk_size = 1000000
    compressed_vecs = np.empty(shape=vecs.shape, dtype=np.float32)
    for i in tqdm.tqdm(range(math.ceil(len(vecs) / chunk_size))):
        compressed_vecs[i * chunk_size: (i + 1) * chunk_size, :] \
            = pq.compress(vecs[i * chunk_size: (i + 1) * chunk_size, :].astype(dtype=np.float32))
    return compressed_vecs

def print_recalls(probe_size, Q, G, I):
    ranks = [2 ** i for i in range(18)]
    print("Probe, top-K, ", end='')
    for rank in ranks:
        print("%05d., " % (rank), end=' ')
    print()
    for topK in [1, 10, 20, 50, 100, 1000]:
        print("%5d, %4d, " % (probe_size, topK), end=' ')
        for rank in ranks:
            n_ok = true_positives(I, Q, G[:, :topK], rank).sum()
            print("%.4f, " % (n_ok / float(len(Q)) / topK), end=' ')
        print()

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
    return args.dataset, args.topk, args.ls, args.Ks, args.metric


def run_ex(dataset, topk, codebook, Ks, metric, by_residual=True):
    X, T, Q, G = loader(dataset, topk, metric, folder='./data/')

    # pq, rq, or component of norm-pq
    imi = PQ(M=2, Ks=Ks)
    pq = PQ(M=codebook, Ks=Ks)
    Q = Q[:200, :]
    G = G[:200, :]

    imi.fit(X[:100000].astype(dtype=np.float32), iter=20)
    print('# compress items')
    x_candidate = chunk_compress(imi, X)
    residual = X - x_candidate

    pq.fit(residual[:100000].astype(dtype=np.float32), iter=20)
    residual_compress = chunk_compress(pq, residual)
    x_compress = residual_compress + x_candidate

    norm_sqr = np.sum(x_compress * x_compress, axis=1, keepdims=True)
    compressed_norm_sqr = norm_sqr

    print('# sort imi')
    sorted_candidate = parallel_sort(metric, x_candidate, Q, X, norms_sqr=None)

    probed_items = [2 ** i for i in [4, 8, 10, 12, 14, 16]]
    for n_item in probed_items:
        probe_size = min(131072, n_item)
        print()
        print('# probe {} items'.format(probe_size))
        I = np.empty((Q.shape[0], min(131072, probe_size - 1)), dtype=np.int32)
        for i in nb.prange(Q.shape[0]):
            top_k_candidate_id = sorted_candidate[i, :probe_size]
            candidate_compressed = x_compress[top_k_candidate_id, :]
            candidate_norm_sqr = compressed_norm_sqr[top_k_candidate_id, 0]
            sort_by_pq = euclidean_norm_arg_sort(Q[i], candidate_compressed, candidate_norm_sqr)
            I[i, :] = sorted_candidate[i, sort_by_pq]

        print_recalls(probe_size, Q, G, I)


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

