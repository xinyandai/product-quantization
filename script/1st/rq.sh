# hpc9.cse.cuhk.edu.hk:/research/jcheng2/xinyan/github/product-quantization> python run_rq.py --dataset sift1m --topk 20 --metric euclid --num_codebook 8 --Ks 256
# Parameters: dataset = sift1m, topK = 20, codebook = 8, Ks = 256, metric = euclid
# load the base data ./data/sift1m/sift1m_base.fvecs,
# load the queries ./data/sift1m/sift1m_query.fvecs,
# load the ground truth ./data/sift1m/20_sift1m_euclid_groundtruth.ivecs
# ranking metric euclid
# Subspace PQ, M: 1, Ks : 256, code_dtype: <class 'numpy.uint8'>Subspace PQ, M: 1, Ks : 256, code_dtype: <class 'numpy.uint8'>Subspace PQ, M: 1, Ks : 256, code_dtype: <class 'numpy.uint8'>Subspace PQ, M: 1, Ks : 256, code_dtype: <class 'numpy.uint8'>Subspace PQ, M: 1, Ks : 256, code_dtype: <class 'numpy.uint8'>Subspace PQ, M: 1, Ks : 256, code_dtype: <class 'numpy.uint8'>Subspace PQ, M: 1, Ks : 256, code_dtype: <class 'numpy.uint8'>Subspace PQ, M: 1, Ks : 256, code_dtype: <class 'numpy.uint8'>
#    Training the subspace: 0 / 1, 0 -> 128
# layer: 0,  residual average norm : 233.3867645263672 max norm: 415.0407409667969 min norm: 65.19554138183594
#    Training the subspace: 0 / 1, 0 -> 128
# layer: 1,  residual average norm : 205.8367156982422 max norm: 381.01849365234375 min norm: 0.0
#    Training the subspace: 0 / 1, 0 -> 128
#/research/jcheng2/xinyan/anaconda3/lib/python3.6/site-packages/numpy/linalg/linalg.py:2390: RuntimeWarning: invalid value encountered in sqrt
#  return sqrt(add.reduce(s, axis=axis, keepdims=keepdims))
# layer: 2,  residual average norm : 187.71099853515625 max norm: 362.5182800292969 min norm: 0.0
#    Training the subspace: 0 / 1, 0 -> 128
# layer: 3,  residual average norm : 174.05514526367188 max norm: 344.7825012207031 min norm: 0.0
#    Training the subspace: 0 / 1, 0 -> 128
# layer: 4,  residual average norm : 163.07882690429688 max norm: 325.1580505371094 min norm: 0.0
#    Training the subspace: 0 / 1, 0 -> 128
# layer: 5,  residual average norm : 153.7753143310547 max norm: 310.92034912109375 min norm: 0.0
#    Training the subspace: 0 / 1, 0 -> 128
# layer: 6,  residual average norm : 146.46560668945312 max norm: 295.88134765625 min norm: 0.0
#    Training the subspace: 0 / 1, 0 -> 128
# layer: 7,  residual average norm : 139.4024658203125 max norm: 279.1372985839844 min norm: 0.0
# compress items
# sorting items
# searching!
expected items, overall time, avg recall, avg precision, avg error, avg items
1, 0, 0.04984999999999999, 0.9969999999999999, 0, 1
2, 0, 0.09255000000000001, 0.9255000000000001, 0, 2
4, 0, 0.1599, 0.7994999999999999, 0, 4
8, 0, 0.2587, 0.6467499999999999, 0, 8
16, 0, 0.39130000000000004, 0.48912500000000003, 0, 16
32, 0, 0.54525, 0.34078125000000004, 0, 32
64, 0, 0.70125, 0.219140625, 0, 64
128, 0, 0.82985, 0.1296640625, 0, 128
256, 0, 0.9223500000000001, 0.07205859375000001, 0, 256
512, 0, 0.971, 0.037929687499999996, 0, 512
1024, 0, 0.9905499999999999, 0.0193466796875, 0, 1024
2048, 0, 0.9977999999999999, 0.009744140625, 0, 2048
4096, 0, 0.9998500000000001, 0.004882080078125001, 0, 4096
8192, 0, 1.0, 0.00244140625, 0, 8192
16384, 0, 1.0, 0.001220703125, 0, 16384
32768, 0, 1.0, 0.0006103515625, 0, 32768
65536, 0, 1.0, 0.00030517578125, 0, 65536
131072, 0, 1.0, 0.000152587890625, 0, 131072
262144, 0, 1.0, 7.62939453125e-05, 0, 262144
524288, 0, 1.0, 3.814697265625e-05, 0, 524288
1048576, 0, 1.0, 1.9073486328125e-05, 0, 1048576
