from vecs_io import *
from pq_residual import *
from rq_iterative_norm import *
from sorter import *
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
    compressed = pq.compress(X)
    compressed_norms_sqr = np.sum(compressed[:, :-1] ** 2, axis=1)
    mean_norms_sqr = np.mean(compressed_norms_sqr)
    mean_norms_sqr_err = np.mean(np.abs(compressed[:, -1] - compressed_norms_sqr))
    print('mean_norms_sqr_err:', mean_norms_sqr_err)
    print('mean_norms_sqr:', mean_norms_sqr)
    print("# sorting items")
    Ts = [2 ** i for i in range(2+int(math.log2(len(X))))]
    recalls = BatchSorter(compressed[:, :-1], Q, X, G, Ts, metric='euclid_norm', batch_size=200, norms_sqr=compressed[:, -1]).recall()
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
    metric = 'product'

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
    pqs = [PQ(M=1, Ks=args.Ks) for _ in range(args.num_codebook)]
    quantizer = ResidualPQ(pqs=pqs)
    quantizer = RQIterativeNorm(quantizer)
    if args.rank:
        execute(quantizer, X, T, Q, G, args.metric)
    if args.save_model:
        if not args.rank:
            quantizer.fit(T, iter=20)
        with open(args.save_dir + '/' + args.dataset + '_rq_iterative_norm' + args.result_suffix + '.pickle', 'wb') as f:
            pickle.dump(quantizer, f)
