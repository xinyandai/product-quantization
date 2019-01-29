from vecs_io import *
from pq_residual import *
from sorter import *


if __name__ == '__main__':
    np.random.seed(808)
    dataset = 'imagenet'
    topk = 20

    bench_mark = 4
    small_norm_codebook = 3
    big_norm_codebook = 8
    big_norm_percent = float(bench_mark - small_norm_codebook) / (big_norm_codebook - small_norm_codebook)

    print("# number of codebook for all {} number of codebook for {} items with big norm percent {}".format(
        small_norm_codebook, big_norm_codebook, big_norm_percent))

    Ks = 256
    metric = 'product'
    train_size = 100000

    X, Q, G = loader(dataset, topk, 'product')
    sorted_norm_index = np.argsort(np.linalg.norm(X, axis=1))

    big_norm_count = int(big_norm_percent * len(X))
    small_norm_index = sorted_norm_index[:-big_norm_count]
    big_norm_index = sorted_norm_index[-big_norm_count:]

    big_norm_items = X[big_norm_index, :]
    small_norm_items = X[small_norm_index, :]

    compressed = np.zeros_like(X)
    compressed[small_norm_index, :] = small_norm_items
    compressed[big_norm_index, :] = big_norm_items

    print(np.linalg.norm(X - compressed) ** 2)

    # pq, rq, or component of norm-pq
    small_norm_quantizer = NormPQ(
        n_percentile=Ks,
        quantize=ResidualPQ(
            pqs=[PQ(M=1, Ks=Ks) for _ in range(small_norm_codebook - 1)])
    )
    big_norm_quantizer = NormPQ(
        n_percentile=Ks,
        quantize=ResidualPQ(
            pqs=[PQ(M=1, Ks=Ks) for _ in range(big_norm_codebook - 1)])
    )

    print("# ranking metric {}".format(metric))
    print('# training')
    print()
    small_norm_quantizer.fit(X[:train_size].astype(dtype=np.float32), iter=20)
    big_norm_quantizer.fit(X[:train_size].astype(dtype=np.float32), iter=20)

    print('# compress items')
    small_norm_compressed = small_norm_quantizer.compress(small_norm_items)
    big_norm_compressed = big_norm_quantizer.compress(big_norm_items)

    compressed[small_norm_index, :] = small_norm_compressed
    compressed[big_norm_index, :] = big_norm_compressed

    print(np.linalg.norm(small_norm_compressed - small_norm_items)**2)
    print(np.linalg.norm(big_norm_compressed - big_norm_items)**2)
    print(np.linalg.norm(X - compressed)**2)

    print("# sorting items")
    Ts = [2 ** i for i in range(2+int(math.log2(len(X))))]
    recalls = BatchSorter(compressed, Q, X, G, Ts, metric=metric, batch_size=200).recall()
    print("# searching!")

    print("expected items, overall time, avg recall, avg precision, avg error, avg items")
    for i, (t, recall) in enumerate(zip(Ts, recalls)):
        print("{}, {}, {}, {}, {}, {}".format(
            2**i, 0, recall, recall * len(G[0]) / t, 0, t))
