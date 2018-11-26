from __future__ import division
from __future__ import print_function
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr
from multiprocessing import Pool, cpu_count
from functools import partial


def solveDimensionLeastSquares(startDim, dimCount, data, indices, indptr, trainPoints, codebookSize, M):
    A = sparse.csr_matrix((data, indices, indptr), shape=(trainPoints.shape[0], M*codebookSize), copy=False)
    discrepancy = 0
    dimCount = min(dimCount, trainPoints.shape[1] - startDim)
    codebooksComponents = np.zeros((M, codebookSize, dimCount), dtype='float32')
    for dim in range(startDim, startDim+dimCount):
        b = trainPoints[:, dim].flatten()
        solution = lsmr(A, b, show=False, maxiter=250)
        codebooksComponents[:, :, dim-startDim] = np.reshape(solution[0], (M, codebookSize))
        discrepancy += solution[3] ** 2
    return (codebooksComponents, discrepancy)


def encodePointsBeamSearch(startPid, pointsCount, pointCodebookProducts, codebooksProducts, codebooksNorms, branch):
    M = codebooksProducts.shape[0]
    K = codebooksProducts.shape[1]
    hashArray = np.array([13 ** i for i in range(M)])
    pointsCount = min(pointsCount, pointCodebookProducts.shape[0] - startPid)
    assigns = np.zeros((pointsCount, M), dtype='int32')
    errors = np.zeros((pointsCount), dtype='float32')
    for pid in range(startPid, startPid+pointsCount):
        distances = - pointCodebookProducts[pid,:] + codebooksNorms
        bestIdx = distances.argsort()[0:branch]
        vocIds = bestIdx // K
        wordIds = bestIdx % K
        bestSums = -1 * np.ones((branch, M), dtype='int32')
        for candidateIdx in range(branch):
            bestSums[candidateIdx,vocIds[candidateIdx]] = wordIds[candidateIdx]
        bestSumScores = distances[bestIdx]
        for m in range(1, M):
            candidatesScores = np.array([bestSumScores[i].repeat(M * K) for i in range(branch)]).flatten()
            candidatesScores += np.tile(distances, branch)
            globalHashTable = np.zeros(115249, dtype='int8')
            for candidateIdx in range(branch):
                for m in range(M):
                      if bestSums[candidateIdx,m] < 0:
                          continue
                      candidatesScores[candidateIdx*M*K:(candidateIdx+1)*M*K] += \
                          codebooksProducts[m, bestSums[candidateIdx,m], :]
                      candidatesScores[candidateIdx*M*K + m*K:candidateIdx*M*K+(m+1)*K] += 999999
            bestIndices = candidatesScores.argsort()
            found = 0
            currentBestIndex = 0
            newBestSums = -1 * np.ones((branch, M), dtype='int32')
            newBestSumsScores = -1 * np.ones((branch), dtype='float32')
            while found < branch:
                bestIndex = bestIndices[currentBestIndex]
                candidateId = bestIndex // (M * K)
                codebookId = (bestIndex % (M * K)) // K
                wordId = (bestIndex % (M * K)) % K
                bestSums[candidateId,codebookId] = wordId
                hashIdx = np.dot(bestSums[candidateId,:], hashArray) % 115249
                if globalHashTable[hashIdx] == 1:
                    bestSums[candidateId,codebookId] = -1
                    currentBestIndex += 1
                    continue
                else:
                    bestSums[candidateId,codebookId] = -1
                    globalHashTable[hashIdx] = 1
                    newBestSums[found,:] = bestSums[candidateId,:]
                    newBestSums[found,codebookId] = wordId
                    newBestSumsScores[found] = candidatesScores[bestIndex]
                    found += 1
                    currentBestIndex += 1
            bestSums = newBestSums.copy()
            bestSumScores = newBestSumsScores.copy()
        assigns[pid-startPid,:] = bestSums[0,:]
        errors[pid-startPid] = bestSumScores[0]
    return (assigns, errors)


def encodePointsAQ(points, codebooks, branch):
    pointsCount = points.shape[0]
    M = codebooks.shape[0]
    K = codebooks.shape[1]
    codebooksProducts = np.zeros((M,K,M*K), dtype='float32')
    fullProducts = np.zeros((M,K,M,K), dtype='float32')
    codebooksNorms = np.zeros((M*K), dtype='float32')
    for m1 in range(M):
        for m2 in range(M):
            fullProducts[m1,:,m2,:] = 2 * np.dot(codebooks[m1,:,:], codebooks[m2,:,:].T)
        codebooksNorms[m1*K:(m1+1)*K] = fullProducts[m1,:,m1,:].diagonal() / 2
        codebooksProducts[m1,:,:] = np.reshape(fullProducts[m1,:,:,:], (K,M*K))
    assigns = np.zeros((pointsCount, M), dtype='int32')
    pidChunkSize = min(pointsCount, 5030)
    errors = np.zeros(pointsCount, dtype='float32')
    for startPid in range(0, pointsCount, pidChunkSize):
        realChunkSize = min(pidChunkSize, pointsCount - startPid)
        chunkPoints = points[startPid:startPid+realChunkSize,:]
        queryProducts = np.zeros((realChunkSize, M * K), dtype=np.float32)
        for pid in range(realChunkSize):
            errors[pid+startPid] += np.dot(chunkPoints[pid,:], chunkPoints[pid,:].T)
        for m in range(M):
            queryProducts[:,m*K:(m+1)*K] = 2 * np.dot(chunkPoints, codebooks[m,:,:].T)
        poolSize = 8
        chunkSize = realChunkSize // poolSize

        pool = Pool(processes=poolSize+1)
        ans = pool.map_async(partial(encodePointsBeamSearch, \
                               pointsCount=chunkSize, \
                               pointCodebookProducts=queryProducts, \
                               codebooksProducts=codebooksProducts, \
                               codebooksNorms=codebooksNorms, \
                               branch=branch), range(0, realChunkSize, chunkSize)).get()
        pool.close()
        pool.join()
        for startChunkPid in range(0, realChunkSize, chunkSize):
            pidsCount = min(chunkSize, realChunkSize - startChunkPid)
            assigns[startPid+startChunkPid:startPid+startChunkPid+pidsCount,:] = ans[startChunkPid//chunkSize][0]
            errors[startPid+startChunkPid:startPid+startChunkPid+pidsCount] += ans[startChunkPid//chunkSize][1]
    return (assigns, errors)


def learnCodebooksAQ(points, dim, M, K, pointsCount, branch, threadsCount=8, itsCount=10, codebooks=None):
    if M < 1:
        raise Exception('M is not positive!')
    threadsCount = threadsCount if threadsCount <= dim else dim

    assigns = np.zeros((pointsCount, M), dtype='int32')

    # random initialization of assignment variables
    # (initializations from (O)PQ should be used for better results)
    for m in range(M):
        assigns[:,m] = np.random.randint(0, K, pointsCount)

    data = np.ones(M * pointsCount, dtype='float32')
    indices = np.zeros(M * pointsCount, dtype='int32')
    indptr = np.array(range(0, pointsCount + 1)) * M
    from tqdm import tqdm 
    for it in tqdm(range(itsCount)):
        for i in range(pointsCount * M):
            indices[i] = 0
        for pid in range(pointsCount):
            for m in range(M):
                indices[pid * M + m] = m * K + assigns[pid,m]
        dimChunkSize = dim // threadsCount
        pool = Pool(threadsCount)
        ans = pool.map(partial(solveDimensionLeastSquares, \
                               dimCount=dimChunkSize, \
                               data=data, \
                               indices=indices, \
                               indptr=indptr, \
                               trainPoints=points, \
                               codebookSize=K, M=M), range(0, dim, dimChunkSize))
        pool.close()
        pool.join()
        for d in range(0, dim, dimChunkSize):
            dimCount = min(dimChunkSize, dim - d)
            codebooks[:, :, d:d+dimCount] = ans[d // dimChunkSize][0]

        (assigns, errors) = encodePointsAQ(points, codebooks, branch)

    return codebooks, assigns


class AQ(object):
    def __init__(self, M, Ks=256, verbose=True):
        assert 0 < Ks <= 2 ** 32
        self.M, self.Ks, self.verbose = M, Ks, verbose
        self.code_dtype = np.int32
        self.codewords = None
        self.branch = 64

    def class_message(self):
        return "AQ, M: {}, Ks : {}, code_dtype: {}".format(self.M, self.Ks, self.code_dtype)

    def fit(self, points, iter):
        assert points.dtype == np.float32
        assert points.ndim == 2
        pointsCount, dim = points.shape
        assert self.Ks < pointsCount, "the number of training vector should be more than Ks"
        self.codewords = np.zeros((self.M, self.Ks, dim), dtype='float32')
        self.codewords, codes = learnCodebooksAQ(
            points, dim, self.M, self.Ks, pointsCount, self.branch, cpu_count(), iter, self.codewords)

    def encode(self, vecs):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.codewords.shape == (self.M, self.Ks, D)

        (codes, errors) = encodePointsAQ(vecs, self.codewords, self.branch)
        print("# Mean AQ quantization error: %f" % (np.mean(errors)))
        return codes

    def decode(self, codes):
        assert codes.ndim == 2
        _, Ks, D = self.codewords.shape
        N, M = codes.shape
        assert M == self.M
        assert codes.dtype == self.code_dtype

        vecs = np.empty((N, D, self.M), dtype=np.float32)
        for m in range(self.M):
            vecs[:, :, m] = self.codewords[m][codes[:, m], :]
        return np.sum(vecs, axis=2)

    def compress(self, vecs):
        return self.decode(self.encode(vecs))
