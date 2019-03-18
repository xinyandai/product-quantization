from vecs_io import *
from pq_residual import ResidualPQ
from sorter import *


if __name__ == '__main__':
    np.random.seed(808)
    dataset = 'music100'
    topk = 20

    bench_mark = 4

    norm_codebook = [3, 4, 8]
    norm_percent =  [0.2, 0.4, 0.4]

    print("# norm_codebook {} \n"
          "# norm_percent {}".format(norm_codebook, norm_percent))

    Ks = 256
    metric = 'product'
    train_size = 100000

    X, Q, G = loader(dataset, topk, 'product', folder='./data/')
    sorted_norm_index = np.argsort(np.linalg.norm(X, axis=1))

    norm_count = [int(i * len(X))+1 for i in norm_percent]
    norm_index = []
    norm_items = []
    start = 0
    for i in norm_count:
        index = sorted_norm_index[start:start+i]
        norm_index.append(index)
        norm_items.append(X[index, :])
        start += i

    print("# norm_count {}".format(norm_count))
    compressed = np.zeros_like(X)
    # pq, rq, or component of norm-pq
    for index, items, codebook in zip(norm_index, norm_items, norm_codebook):
        print("# ranking metric {}".format(metric))
        print('# training')
        rq = ResidualPQ(pqs=[PQ(M=1, Ks=Ks) for _ in range(codebook)])
        rq.fit(X[:train_size].astype(dtype=np.float32), iter=20)
        compressed[index, :] = rq.compress(items)

    print("# sorting items")
    Ts = [2 ** i for i in range(2+int(math.log2(len(X))))]
    recalls = BatchSorter(compressed, Q, X, G, Ts, metric=metric, batch_size=200).recall()
    print("# searching!")

    print("expected items, overall time, avg recall, avg precision, avg error, avg items")
    for i, (t, recall) in enumerate(zip(Ts, recalls)):
        print("{}, {}, {}, {}, {}, {}".format(
            2**i, 0, recall, recall * len(G[0]) / t, 0, t))
