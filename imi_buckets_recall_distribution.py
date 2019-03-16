from vecs_io import *
import math
import argparse
import pickle
import numpy as np
import numba as nb


def gen_chunk(X, chunk_size=1000000):
    for i in range(math.ceil(len(X) / chunk_size)):
        yield X[i * chunk_size : (i + 1) * chunk_size]


@nb.njit(parallel=True)
def parallel_argsort(distances):
    res = np.empty(distances.shape, dtype=np.uint64)
    for i in nb.prange(len(distances)):
        res[i] = np.argsort(distances[i])
    return res


@nb.njit
def norm_sqr_axis_1(V):
    return np.sum(V ** 2, axis=1)


@nb.njit(parallel=True)
def pairwise_euclidean_distances(Q, X):
    distances = np.empty((Q.shape[0], X.shape[0]), dtype=np.float32)

    for i in nb.prange(Q.shape[0]):
        q = Q[i]
        distances[i] = norm_sqr_axis_1(X - q)

    return distances


@nb.njit
def intersect_count(X, Y):
    count = 0
    for x in X:
        for y in Y:
            if x == y:
                count += 1
                break
    return count


@nb.njit(parallel=True)
def compute_bucket_recalls(Q, G, imiCenters, itemIds, bucketOffsets, bucketLengths, bucketIdsSorted):
    bucketRecallsSep = np.empty((len(Q), bucketIdsSorted.shape[1]), dtype=np.uint32)

    for qid in nb.prange(len(Q)):
        for bucketRank in range(bucketIdsSorted.shape[1]):
            bid = bucketIdsSorted[qid, bucketRank]
            itemsInBucket = itemIds[bucketOffsets[bid] : bucketOffsets[bid]+bucketLengths[bid]]
            bucketRecallsSep[qid, bucketRank] = intersect_count(itemsInBucket, G[qid])

    return np.sum(bucketRecallsSep, axis=0).astype(np.float32) / (len(Q) * G.shape[1])


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

    _, _, Q, G = mmap_loader(args.dataset, args.topk, args.metric, folder=args.data_dir, data_type=args.data_type)

    np.random.seed(123)

    with np.load(args.save_dir + '/' + args.dataset + args.result_suffix + '_imi.npz') as f:
        itemIds = f['itemIds']
        bucketLengths = f['bucketLengths']
        bucketOffsets = f['bucketOffsets']
        imiCenters = f['imiCenters']

    if args.metric == 'euclid':
        distances = pairwise_euclidean_distances(Q, imiCenters)
    elif args.metric == 'product':
        distances = -(Q @ imiCenters.T)
    else:
        raise ValueError('Metric ' + args.metric + ' is not supported')

    bucketIdsSorted = parallel_argsort(distances)

    bucketRecalls = compute_bucket_recalls(Q, G[:, :args.topk], imiCenters, itemIds, bucketOffsets, bucketLengths, bucketIdsSorted)

    for i in range(len(bucketRecalls)):
        print(bucketRecalls[i])
