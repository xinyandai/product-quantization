from sorter import *
from transformer import *
from vecs_io import loader
from rptree import RPTree

topK = [1, 5, 10, 20, 50, 100, 1000]

def probe_same_number_items(X, T, Q, G, Ks, num_tables, metric, train_size=100000, iters=20, average_items=800):
    np.random.seed(123)
    assert G is not None
    if T is None:
        T = X[:train_size]

    codes, _ = kmeans2(T, Ks * num_tables, iter=iters, minit='points')

    pq = PQ(1, num_tables *  Ks, verbose=False)
    pq.fit(T.astype(dtype=np.float32), iter=20)
    compressed = pq.compress(X)
    Ts = [average_items]

    for K in topK:
        recall = BatchSorter(compressed, Q, X, G[:, :K], Ts, metric=metric, batch_size=200, verbose=False).recall()[0]
        print("{}, {}".format(K, recall))

def probe_same_number_of_bucket(X, T, Q, G, Ks, num_tables, metric, train_size=100000, iters=20):
    np.random.seed(123)
    assert G is not None
    if T is None:
        T = X[:train_size]

    centers, _ = kmeans2(T, Ks * num_tables, iter=iters, minit='points')
    codes = vq(X, centers)[0]

    items = np.empty(shape=(len(Q)))
    recalls = np.empty(shape=(len(Q), len(topK)))
    ranked = parallel_sort(metric, centers, Q, X)

    for i, q in tqdm.tqdm(enumerate(Q)):
        result = [np.where(codes == ranked[i, t])[0] for t in range(num_tables)]
        result = np.unique(np.concatenate(result))
        items[i] = len(result)
        recalls[i, :] = np.array([len(np.intersect1d(result, G[i, :AtT])) / float(AtT) for AtT in topK])
    recalls = np.mean(recalls, axis=0)
    print('topK, Recall, Items {}'.format(np.mean(items)))

    for T, R in zip(topK, recalls):
        print('{}, {}'.format(T, R))

    return np.mean(items)


def joint_inverted_table(X, T, Q, G, Ks, num_tables, metric, train_size=100000, iters=20):
    np.random.seed(123)
    assert G is not None
    if T is None:
        T = X[:train_size]

    KxL_centers, _ = kmeans2(T, Ks * num_tables, iter=iters, minit='points')


    buckets = np.zeros(shape=(Ks, num_tables), dtype=np.int)
    tree = RPTree(max_size=num_tables)
    tree.make_tree(KxL_centers, np.arange(len(KxL_centers)), buckets, tree)

    grouped_centers = [KxL_centers[buckets[:, i]] for i in range(num_tables)]

    x_codes = np.array([vq(X, grouped_centers[i])[0] for i in range(num_tables)])
    q_codes = np.array([vq(Q, grouped_centers[i])[0] for i in range(num_tables)])

    items = np.empty(shape=(len(Q)))
    recalls = np.empty(shape=(len(Q), len(topK)))
    for i, q in tqdm.tqdm(enumerate(Q)):
        result = [np.where(x_codes[t] == q_codes[t, i])[0] for t in range(num_tables)]
        result = np.unique(np.concatenate(result))
        items[i] = len(result)
        recalls[i, :] = np.array([len(np.intersect1d(result, G[i, :T])) / float(T) for T in topK])
    recalls = np.mean(recalls, axis=0)
    print('topK, Recall, Items {}'.format(np.mean(items)))

    for T, R in zip(topK, recalls):
        print('{}, {}'.format(T, R))

    return np.mean(items)

def parse_args(dataset, Ks, metric, tables):
    # override default parameters with command line parameters
    import argparse
    parser = argparse.ArgumentParser(description='Process input method and parameters.')
    parser.add_argument('--dataset', type=str, default=dataset, help='choose data set name')
    parser.add_argument('--metric', type=str, default=metric, help='metric of ground truth')
    parser.add_argument('--Ks', type=int, default=Ks, help='number of centroids in each quantizer')
    parser.add_argument('--tables', type=int, default=tables, help='number of tables')
    args = parser.parse_args()
    return args.dataset, args.Ks, args.metric, args.tables


if __name__ == '__main__':
    dataset = 'netflix'
    Ks = 256
    tables = 16
    metric = 'euclid'

    # override default parameters with command line parameters
    dataset, Ks, metric, tables = parse_args(dataset, Ks, metric, tables)

    print("# Parameters: dataset = {}, Ks = {}, metric = {}".format(dataset, Ks, metric))
    X, T, Q, G = loader(dataset, 1000, metric, folder='data/')


    # single table:  probe same number of bucket
    probe_same_number_of_bucket(X, T, Q, G, Ks, tables, metric)
    # joint inverted index
    average_items = joint_inverted_table(X, T, Q, G, Ks, tables, metric)
    # single table:  probe same number of item
    probe_same_number_items(X, T, Q, G, Ks, tables, metric, average_items=int(average_items))

"""
# Parameters: dataset = netflix, Ks = 256, metric = euclid
# load the base data data/netflix/netflix_base.fvecs, 
# load the queries data/netflix/netflix_query.fvecs, 
# load the ground truth data/netflix/1000_netflix_euclid_groundtruth.ivecs
100%|██████████| 1000/1000 [00:02<00:00, 393.08it/s]
1000it [00:00, 2646.54it/s]
topK, Recall, Items 79.663
1, 0.957
5, 0.9215999999999974
10, 0.8926999999999939
20, 0.8356500000000001
50, 0.6962600000000001
100, 0.5237399999999997
1000, 0.07900699999999986
1000it [00:00, 1975.51it/s]
topK, Recall, Items 828.82
1, 0.995
5, 0.9949999999999998
10, 0.9936999999999995
20, 0.991050000000001
50, 0.9798000000000047
100, 0.9571300000000076
1000, 0.5543130000000002
1, 1.0
5, 0.9997999999999999
10, 0.9995
20, 0.9988
50, 0.99802
100, 0.9944000000000001
1000, 0.6874750000000001

// single table 
ImageNet
topK, Recall, Items 17690.297
1, 1.0
5, 0.8971999999999966
10, 0.8702999999999959
20, 0.8414000000000004
50, 0.8037600000000007
100, 0.770210000000001
1000, 0.6178089999999999
1000it [02:39,  6.55it/s]

// multi table
topK, Recall, Items 185877.764
1, 1.0
5, 0.9977999999999998
10, 0.9975999999999996
20, 0.9970500000000008
50, 0.9944200000000019
100, 0.9916700000000029
1000, 0.9705990000000024

// probed item wrt 185877
1, 1.0
5, 0.9984
10, 0.9963
20, 0.9954
50, 0.99376
100, 0.99175
1000, 0.974345
"""