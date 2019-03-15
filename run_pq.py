from sorter import *
from transformer import *
from vecs_io import loader
import pickle

def chunk_compress(pq, vecs):
    chunk_size = 1000000
    compressed_vecs = np.empty(shape=vecs.shape, dtype=np.float32)
    for i in tqdm.tqdm(range(math.ceil(len(vecs) / chunk_size))):
        compressed_vecs[i * chunk_size: (i + 1) * chunk_size, :] \
            = pq.compress(vecs[i * chunk_size: (i + 1) * chunk_size, :].astype(dtype=np.float32))
    return compressed_vecs


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
    print("# sorting items")
    Ts = [2 ** i for i in range(2+int(math.log2(len(X))))]
    recalls = BatchSorter(compressed, Q, X, G, Ts, metric=metric, batch_size=200).recall()
    print("# searching!")

    print("expected items, overall time, avg recall, avg precision, avg error, avg items")
    for i, (t, recall) in enumerate(zip(Ts, recalls)):
        print("{}, {}, {}, {}, {}, {}".format(
            2**i, 0, recall, recall * len(G[0]) / t, 0, t))


def parse_args(dataset=None, topk=None, codebook=None, Ks=None, metric=None):
    # override default parameters with command line parameters
    import argparse
    parser = argparse.ArgumentParser(description='Process input method and parameters.')
    parser.add_argument('--dataset', type=str, help='choose data set name', default=dataset)
    parser.add_argument('--topk', type=int, help='required topk of ground truth', default=topk)
    parser.add_argument('--metric', type=str, help='metric of ground truth', default=metric)
    parser.add_argument('--num_codebook', type=int, help='number of codebooks', default=codebook)
    parser.add_argument('--Ks', type=int, help='number of centroids in each quantizer', default=Ks)
    parser.add_argument('--rank', type=int, help='should rank', default=1)
    parser.add_argument('--save_model', type=int, help='should save the model', default=0)
    parser.add_argument('--save_dir', type=str, help='dir to save results', default='./results')
    parser.add_argument('--result_suffix', type=str, help='suffix to be added to the file names of the results', default='')
    parser.add_argument('--train_size', type=int, help='train size', default=100000)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    dataset = 'netflix'
    topk = 20
    codebook = 4
    Ks = 256
    metric = 'product'

    # override default parameters with command line parameters
    args = parse_args(dataset, topk, codebook, Ks, metric)
    print("# Parameters: dataset = {}, topK = {}, codebook = {}, Ks = {}, metric = {}"
          .format(args.dataset, args.topk, args.num_codebook, args.Ks, args.metric))

    X, T, Q, G = loader(args.dataset, args.topk, args.metric, folder='data/')
    if T is None:
        T = X[:args.train_size]
    else:
        T = T[:args.train_size]
    T = np.ascontiguousarray(T, np.float32)

    # pq, rq, or component of norm-pq
    quantizer = PQ(M=args.num_codebook, Ks=args.Ks)
    if args.rank:
        execute(quantizer, X, T, Q, G, args.metric)
    if args.save_model:
        if not args.rank:
            quantizer.fit(T, iter=20)
        with open(args.save_dir + '/' + args.dataset + '_pq' + args.result_suffix + '.pickle', 'wb') as f:
            pickle.dump(quantizer, f)
