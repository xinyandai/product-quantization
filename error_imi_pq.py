from sorter import *
from transformer import *
from vecs_io import loader


if __name__ == '__main__':
    dataset = 'sift1m'
    topk = 20
    Ks = 256
    metric = 'euclid'

    X, T, Q, G = loader(dataset, topk, metric, folder='./data/')

    # pq, rq, or component of norm-pq
    imi = PQ(M=2, Ks=Ks)

    imi.fit(X[:100000].astype(dtype=np.float32), iter=20)
    print('# compress items')
    x_candidate = imi.compress(X)
    residual = X - x_candidate

    for codebook in 2 * np.arange(1, 33):
        residual_pq = PQ(M=codebook, Ks=Ks, verbose=False)

        residual_pq.fit(residual[:100000].astype(dtype=np.float32), iter=20)
        residual_compress = residual_pq.compress(residual)
        residual_error = np.linalg.norm(residual - residual_compress, axis=1)
        del residual_pq
        del residual_compress

        x_pq = PQ(M=codebook, Ks=Ks, verbose=False)
        x_pq.fit(X[:100000].astype(dtype=np.float32), iter=20)
        X_compress = x_pq.compress(X)
        X_error = np.linalg.norm(X - X_compress, axis=1)
        del x_pq
        del X_compress

        print("{}, {}, {}".format(codebook, np.mean(residual_error), np.mean(X_error)))

