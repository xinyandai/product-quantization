from sorter import *
from vecs_io import *
import math
import argparse
import numpy as np


def gen_chunk(X, chunk_size=1000000):
    for i in range(math.ceil(len(X) / chunk_size)):
        yield X[i * chunk_size : (i + 1) * chunk_size]


def rank(compressed, Q, G, metric):
    print("# sorting items")
    Ts = [2 ** i for i in range(2+int(math.log2(len(compressed))))]
    X = None
    compressed_norms_sqr = np.sum(compressed[:, :-1] ** 2, axis=1)
    mean_norms_sqr = np.mean(compressed_norms_sqr)
    mean_norms_sqr_err = np.mean(np.abs(compressed[:, -1] - compressed_norms_sqr))
    print('mean_norms_sqr_err:', mean_norms_sqr_err)
    print('mean_norms_sqr:', mean_norms_sqr)
    recalls = BatchSorter(compressed[:, :-1], Q, X, G, Ts, metric='euclid_norm', batch_size=200, norms_sqr=compressed[:, -1]).recall()
    print("# searching!")

    print("expected items, overall time, avg recall, avg precision, avg error, avg items")
    for i, (t, recall) in enumerate(zip(Ts, recalls)):
        print("{}, {}, {}, {}, {}, {}".format(
            2**i, 0, recall, recall * len(G[0]) / t, 0, t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input method and parameters.')
    parser.add_argument('--data_dir', type=str, help='directory storing the data', default='./data/')
    parser.add_argument('--dataset', type=str, help='choose data set name')
    parser.add_argument('--data_type', type=str, default='fvecs', help='data type of base and queries')
    parser.add_argument('--topk', type=int, help='topk of ground truth')
    parser.add_argument('--metric', type=str, help='metric of ground truth, euclid by default')

    parser.add_argument('--save_dir', type=str, help='dir to save results', default='./results')
    parser.add_argument('--result_suffix', type=str, help='suffix to be added to the file names of the results', default='')

    parser.add_argument('--chunk_size', type=int, help='chunk size', default=1000000)

    args = parser.parse_args()

    X, _, Q, G = mmap_loader(args.dataset, args.topk, args.metric, folder=args.data_dir, data_type=args.data_type)

    np.random.seed(123)

    with open(args.save_dir + '/' + args.dataset + args.result_suffix + '_decoded', 'rb') as f:
        compressed = np.fromfile(f, dtype=np.float32).reshape((X.shape[0], -1))
    del X

    rank(compressed, Q, G, args.metric)
