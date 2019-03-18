import argparse
import pickle
import numpy as np
import numba as nb


@nb.njit(parallel=True)
def generateIMICodes(imiSize, imiDim, code_dtype):
    codes = np.empty((imiSize ** imiDim, imiDim), dtype=code_dtype)

    for i in nb.prange(codes.shape[0]):
        cur_code = i + 0
        for j in range(imiDim - 1, -1, -1):
            codes[i, j] = cur_code % imiSize
            cur_code //= imiSize

    return codes


@nb.njit(parallel=True)
def imiId2BucketId(imiIds, imiSize, imiDim):
    bucketIds = np.zeros(imiIds.shape[0], dtype=np.uint64)

    for i in nb.prange(imiIds.shape[0]):
        for j in range(imiDim):
            bucketIds[i] *= imiSize
            bucketIds[i] += imiIds[i, j]

    return bucketIds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input method and parameters.')
    parser.add_argument('--dataset', type=str, help='choose data set name')

    parser.add_argument('--imiSize', type=int, help='number of codes per IMI dimension', default=256)
    parser.add_argument('--imiDim', type=int, help='IMI dimension (1 means IVFADC)', default=2)

    parser.add_argument('--save_dir', type=str, help='dir to save results', default='./results')
    parser.add_argument('--result_suffix', type=str, help='suffix to be added to the file names of the results', default='')

    args = parser.parse_args()

    np.random.seed(123)

    with open(args.save_dir + '/' + args.dataset + args.result_suffix + '.pickle', 'rb') as f:
        quantizer = pickle.load(f)

    with open(args.save_dir + '/' + args.dataset + args.result_suffix + '_encoded', 'rb') as f:
        codes = np.fromfile(f, dtype=quantizer.code_dtype).reshape(-1, quantizer.num_codebooks)

    codesIMI = codes[:, :args.imiDim]
    bucketIds = imiId2BucketId(codesIMI, args.imiSize, args.imiDim)

    itemIds = np.argsort(bucketIds).astype(np.uint64)
    bucketIdsSorted = bucketIds[itemIds]

    bucketLengths = np.bincount(bucketIdsSorted.astype(np.int64), minlength=args.imiSize ** args.imiDim).astype(np.uint32) # TODO: bucketIdsSorted should not be converted to int64, seems to be a bug of numpy

    bucketOffsets = np.concatenate(([0], np.cumsum(bucketLengths)[:-1]))

    imiCenters = quantizer.decode(generateIMICodes(args.imiSize, args.imiDim, codesIMI.dtype))

    np.savez(args.save_dir + '/' + args.dataset + args.result_suffix + '_imi.npz', 
             itemIds=itemIds,
             bucketLengths=bucketLengths,
             bucketOffsets=bucketOffsets,
             imiCenters=imiCenters)
