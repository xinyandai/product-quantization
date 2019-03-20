from vecs_io import *
from pq_residual import *
from sorter import *
from run_pq import chunk_compress
from run_pq import parse_args
import pickle



def execute(pq, X, T, Q, G, metric, train_size=100000):
    np.random.seed(123)
    print("# ranking metric {}".format(metric))
    print("# "+pq.class_message())
    if T is None:
        pq.fit(X[:train_size].astype(dtype=np.float32), iter=20)
    else:
        pq.fit(T.astype(dtype=np.float32), iter=20)

    print('# compress items')
    compressed = chunk_compress(pq, X)
    norm_sqr = np.sum(compressed * compressed, axis=1, keepdims=True)
    norm_quantizer = PQ(M=1, Ks=256)
    norm_quantizer.fit(norm_sqr, iter=20)
    compressed_norm_sqr = norm_quantizer.compress(norm_sqr).reshape(-1)

    print("# sorting items")
    Ts = [2 ** i for i in range(2+int(math.log2(len(X))))]
    recalls = BatchSorter(compressed, Q, X, G, Ts, metric='euclid_norm', batch_size=200, norms_sqr=compressed_norm_sqr).recall()
    print("# searching!")

    print("expected items, overall time, avg recall, avg precision, avg error, avg items")
    for i, (t, recall) in enumerate(zip(Ts, recalls)):
        print("{}, {}, {}, {}, {}, {}".format(
            2**i, 0, recall, recall * len(G[0]) / t, 0, t))


if __name__ == '__main__':
    dataset = 'netflix'
    topk = 20
    codebook = 4
    Ks = 256
    metric = 'euclid'

    # override default parameters with command line parameters
    import sys
    args = parse_args(dataset, topk, codebook, Ks, metric)
    print("# Parameters: dataset = {}, topK = {}, codebook = {}, Ks = {}, metric = {}"
          .format(args.dataset, args.topk, args.num_codebook, args.Ks, args.metric))

    X, T, Q, G = loader(args.dataset, args.topk, args.metric, folder=args.data_dir)
    if T is None:
        T = X[:args.train_size]
    else:
        T = T[:args.train_size]
    T = np.ascontiguousarray(T, np.float32)

    # pq, rq, or component of norm-pq
    pqs = [PQ(M=1, Ks=args.Ks) for _ in range(args.num_codebook - 1)]
    quantizer = ResidualPQ(pqs=pqs)
    if args.rank:
        execute(quantizer, X, T, Q, G, args.metric)
    if args.save_model:
        if not args.rank:
            quantizer.fit(T, iter=20)
        with open(args.save_dir + '/' + args.dataset + '_rq' + args.result_suffix + '.pickle', 'wb') as f:
            pickle.dump(quantizer, f)
