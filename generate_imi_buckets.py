import argparse
import pickle
import numpy as np
import numba as nb


@nb.njit(parallel=True)
def generateIMICodes(imiSize):
    codes = np.empty((imiSize * imiSize, 2), dtype=np.uint8)

    for i in nb.prange(imiSize * imiSize):
        codes[i, 0] = i / imiSize
        codes[i, 1] = i % imiSize

    return codes


@nb.njit(parallel=True)
def imiId2BucketId(imiIds, imiSize):
    bucketIds = np.empty(imiIds.shape[0], dtype=np.uint64)

    for i in nb.prange(imiIds.shape[0]):
        bucketIds[i] = imiIds[i][0] * imiSize + imiIds[i][1]

    return bucketIds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input method and parameters.')
    parser.add_argument('--dataset', type=str, help='choose data set name')

    parser.add_argument('--save_dir', type=str, help='dir to save results', default='./results')
    parser.add_argument('--result_suffix', type=str, help='suffix to be added to the file names of the results', default='')

    args = parser.parse_args()

    np.random.seed(123)

    with open(args.save_dir + '/' + args.dataset + args.result_suffix + '.pickle', 'rb') as f:
        quantizer = pickle.load(f)

    with open(args.save_dir + '/' + args.dataset + args.result_suffix + '_encoded', 'rb') as f:
        codes = np.fromfile(f, dtype=quantizer.code_dtype).reshape(-1, quantizer.num_codebooks)

    codesIMI = codes[:, :2]
    imiSize = 256
    bucketIds = imiId2BucketId(codesIMI, imiSize)

    itemIds = np.argsort(bucketIds).astype(np.uint64)
    bucketIdsSorted = bucketIds[itemIds]

    bucketLengths = np.bincount(bucketIdsSorted.astype(np.int64)).astype(np.uint32) # TODO: bucketIdsSorted should not be converted to int64, seems to be a bug of numpy

    bucketOffsets = np.concatenate(([0], np.cumsum(bucketLengths)[:-1]))

    imiCenters = quantizer.decode(generateIMICodes(imiSize))

    np.savez(args.save_dir + '/' + args.dataset + args.result_suffix + '_imi.npz', 
             itemIds=itemIds,
             bucketLengths=bucketLengths,
             bucketOffsets=bucketOffsets,
             imiCenters=imiCenters)
