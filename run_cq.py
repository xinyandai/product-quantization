from vecs_io import *
from cq import *
from sorter import *
from run_pq import parse_args
from transformer import *

def execute(pq, X, T, Q, G, metric, train_size=100000):
    np.random.seed(123)
    print("# ranking metric {}".format(metric))
    print("# "+pq.class_message())
    if T is None:
        pq.fit(X[:train_size].astype(dtype=np.float32), iter=20)
    else:
        pq.fit(T.astype(dtype=np.float32), iter=20)

    print('# compress items')
    compressed = pq.compress(X)
    # assert not np.any(compressed - pq.decode(pq.encode(X)))
    print("# sorting items")

    Q = np.append(Q, [[-0.5]]*len(Q), axis=1)
    Ts = [2 ** i for i in range(2+int(math.log2(len(X))))]
    recalls = BatchSorter(compressed, Q, X, G, Ts, metric='product', batch_size=200).recall()
    print("# searching!")

    print("expected items, overall time, avg recall, avg precision, avg error, avg items")
    for i, (t, recall) in enumerate(zip(Ts, recalls)):
        print("{}, {}, {}, {}, {}, {}".format(
            2**i, 0, recall, recall * len(G[0]) / t, 0, t))

if __name__ == '__main__':
    dataset = 'netflix'
    topk = 20
    codebook = 2
    Ks = 256
    metric = 'euclid'

    # override default parameters with command line parameters
    args = parse_args(dataset, topk, codebook, Ks, metric)
    print("# Parameters: dataset = {}, topK = {}, codebook = {}, Ks = {}, metric = {}"
          .format(args.dataset, args.topk, args.num_codebook, args.Ks, args.metric))

    X, T, Q, G = loader(args.dataset, args.topk, args.metric, folder='data/')

    # pq, rq, or component of norm-pq
    quantizer = CQ(depth=codebook)
    execute(quantizer,  X, T, Q, G, metric)
'''
# Parameters: dataset = netflix, topK = 20, codebook = 2, Ks = 256, metric = euclid
# load the base data data/netflix/netflix_base.fvecs, 
# load the queries data/netflix/netflix_query.fvecs, 
# load the ground truth data/netflix/20_netflix_euclid_groundtruth.ivecs
# ranking metric euclid
# CQ with 2 residual layers and 256 codebook each layer
/home/xinyan/.conda/envs/py3/lib/python3.6/site-packages/numpy/linalg/linalg.py:2480: RuntimeWarning: overflow encountered in multiply
  s = (x.conj() * x).real
/home/xinyan/program/product-quantization/cq.py:66: RuntimeWarning: overflow encountered in square
  error[:, i] = first ** 2 + self.mu * second**2
# compress items
# sorting items
100%|██████████| 200/200 [00:04<00:00, 48.94it/s]
100%|██████████| 200/200 [00:04<00:00, 49.92it/s]
100%|██████████| 200/200 [00:04<00:00, 49.46it/s]
100%|██████████| 200/200 [00:03<00:00, 51.08it/s]
100%|██████████| 200/200 [00:03<00:00, 50.54it/s]
# searching!
expected items, overall time, avg recall, avg precision, avg error, avg items
1, 0, 0.0169, 0.33799999999999997, 0, 1
2, 0, 0.0338, 0.33799999999999997, 0, 2
4, 0, 0.06265000000000001, 0.31325000000000003, 0, 4
8, 0, 0.1154, 0.2885, 0, 8
16, 0, 0.1992, 0.249, 0, 16
32, 0, 0.31930000000000003, 0.19956250000000003, 0, 32
64, 0, 0.47645, 0.148890625, 0, 64
128, 0, 0.6436499999999999, 0.1005703125, 0, 128
256, 0, 0.7945000000000001, 0.06207031250000001, 0, 256
512, 0, 0.9115, 0.03560546875, 0, 512
1024, 0, 0.9741500000000001, 0.0190263671875, 0, 1024
2048, 0, 0.99605, 0.00972705078125, 0, 2048
4096, 0, 0.9998499999999999, 0.004882080078125, 0, 4096
8192, 0, 1.0, 0.00244140625, 0, 8192
16384, 0, 1.0, 0.001220703125, 0, 16384
32768, 0, 1.0, 0.0006103515625, 0, 32768



# Parameters: dataset = netflix, topK = 20, codebook = 2, Ks = 256, metric = euclid
# load the base data data/netflix/netflix_base.fvecs, 
# load the queries data/netflix/netflix_query.fvecs, 
# load the ground truth data/netflix/20_netflix_euclid_groundtruth.ivecs
# ranking metric euclid
# ORQ, RQ : [Subspace PQ, M: 2, Ks : 256, code_dtype: <class 'numpy.uint8'>],  M: 2, Ks : 256, code_dtype: <class 'numpy.uint8'>
 95%|█████████▌| 19/20 [00:03<00:00,  5.04it/s]
#    Training the subspace: 0 / 2, 0 -> 150
#    Training the subspace: 1 / 2, 150 -> 300
# layer: 0,  residual average norm : 0.325784295797348 max norm: 1.178204894065857 min norm: 0.08761157840490341
# compress items
  0%|          | 0/1 [00:00<?, ?it/s]# sorting items
100%|██████████| 1/1 [00:00<00:00,  8.75it/s]
100%|██████████| 200/200 [00:02<00:00, 73.61it/s]
100%|██████████| 200/200 [00:02<00:00, 75.23it/s]
100%|██████████| 200/200 [00:02<00:00, 72.71it/s]
100%|██████████| 200/200 [00:03<00:00, 65.10it/s]
100%|██████████| 200/200 [00:02<00:00, 80.62it/s]
# searching!
expected items, overall time, avg recall, avg precision, avg error, avg items
1, 0, 0.029799999999999997, 0.596, 0, 1
2, 0, 0.05735, 0.5735, 0, 2
4, 0, 0.10535, 0.52675, 0, 4
8, 0, 0.1886, 0.4715, 0, 8
16, 0, 0.31560000000000005, 0.3945000000000001, 0, 16
32, 0, 0.48965, 0.30603125, 0, 32
64, 0, 0.6814, 0.2129375, 0, 64
128, 0, 0.8558499999999999, 0.13372656249999998, 0, 128
256, 0, 0.9562, 0.07470312500000001, 0, 256
512, 0, 0.9906999999999999, 0.03869921875, 0, 512
1024, 0, 0.9980999999999999, 0.019494140624999996, 0, 1024
2048, 0, 0.9995999999999999, 0.009761718749999999, 0, 2048
4096, 0, 0.99995, 0.004882568359375, 0, 4096
8192, 0, 1.0, 0.00244140625, 0, 8192
16384, 0, 1.0, 0.001220703125, 0, 16384
32768, 0, 1.0, 0.0006103515625, 0, 32768
'''