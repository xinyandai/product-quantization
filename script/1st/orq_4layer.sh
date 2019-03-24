# hpc10.cse.cuhk.edu.hk:/research/jcheng2/xinyan/github/product-quantization> python run_rpq.py --dataset sift1m --topk 20 --metric euclid --num_codebook 8 --Ks 256
# Parameters: dataset = sift1m, topK = 20, codebook = 8, Ks = 256, metric = euclid
# load the base data ./data/sift1m/sift1m_base.fvecs,
# load the queries ./data/sift1m/sift1m_query.fvecs,
# load the ground truth ./data/sift1m/20_sift1m_euclid_groundtruth.ivecs
# ranking metric euclid
# ORQ, RQ : [Subspace PQ, M: 2, Ks : 256, code_dtype: <class 'numpy.uint8'>],  M: 2, Ks : 256, code_dtype: <class 'numpy.uint8'>ORQ, RQ : [Subspace PQ, M: 2, Ks : 256, code_dtype: <class 'numpy.uint8'>],  M: 2, Ks : 256, code_dtype: <class 'numpy.uint8'>ORQ, RQ : [Subspace PQ, M: 2, Ks : 256, code_dtype: <class 'numpy.uint8'>],  M: 2, Ks : 256, code_dtype: <class 'numpy.uint8'>ORQ, RQ : [Subspace PQ, M: 2, Ks : 256, code_dtype: <class 'numpy.uint8'>],  M: 2, Ks : 256, code_dtype: <class 'numpy.uint8'>
# 95%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎           | 19/20 [00:26<00:01,  1.37s/it]
#    Training the subspace: 0 / 2, 0 -> 64
#    Training the subspace: 1 / 2, 64 -> 128
# layer: 0,  residual average norm : 211.19927978515625 max norm: 368.34381103515625 min norm: 69.70755767822266
# layer: 0,  residual average norm : 211.19927978515625 max norm: 368.34381103515625 min norm: 69.70755004882812
# 95%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎           | 19/20 [00:32<00:01,  1.63s/it]
#    Training the subspace: 0 / 2, 0 -> 64
#    Training the subspace: 1 / 2, 64 -> 128
# layer: 0,  residual average norm : 176.78555297851562 max norm: 318.4595642089844 min norm: 52.446922302246094
# layer: 1,  residual average norm : 176.78558349609375 max norm: 318.4595947265625 min norm: 52.446922302246094
# 95%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎           | 19/20 [00:24<00:01,  1.31s/it]
#    Training the subspace: 0 / 2, 0 -> 64
#    Training the subspace: 1 / 2, 64 -> 128
# layer: 0,  residual average norm : 155.1817626953125 max norm: 293.48138427734375 min norm: 52.87228012084961
# layer: 2,  residual average norm : 155.1817626953125 max norm: 293.4813537597656 min norm: 52.87228012084961
# 95%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎           | 19/20 [00:26<00:01,  1.56s/it]
#    Training the subspace: 0 / 2, 0 -> 64
#    Training the subspace: 1 / 2, 64 -> 128
# layer: 0,  residual average norm : 139.305908203125 max norm: 263.7986755371094 min norm: 56.19874572753906
# layer: 3,  residual average norm : 139.305908203125 max norm: 263.79864501953125 min norm: 56.19874572753906
# compress items
# sorting items
# searching!
expected items, overall time, avg recall, avg precision, avg error, avg items
1, 0, 0.0499, 0.998, 0, 1
2, 0, 0.093, 0.9299999999999999, 0, 2
4, 0, 0.16254999999999997, 0.8127499999999999, 0, 4
8, 0, 0.26225, 0.6556249999999999, 0, 8
16, 0, 0.39565, 0.4945625, 0, 16
32, 0, 0.5528, 0.3455, 0, 32
64, 0, 0.70575, 0.220546875, 0, 64
128, 0, 0.8365000000000001, 0.13070312500000003, 0, 128
256, 0, 0.92325, 0.07212890625, 0, 256
512, 0, 0.9711000000000002, 0.03793359375000001, 0, 512
1024, 0, 0.9924000000000001, 0.019382812500000002, 0, 1024
2048, 0, 0.99865, 0.00975244140625, 0, 2048
4096, 0, 0.9995, 0.0048803710937500005, 0, 4096
8192, 0, 0.99995, 0.0024412841796875, 0, 8192
16384, 0, 1.0, 0.001220703125, 0, 16384
32768, 0, 1.0, 0.0006103515625, 0, 32768
65536, 0, 1.0, 0.00030517578125, 0, 65536
131072, 0, 1.0, 0.000152587890625, 0, 131072
262144, 0, 1.0, 7.62939453125e-05, 0, 262144
524288, 0, 1.0, 3.814697265625e-05, 0, 524288
1048576, 0, 1.0, 1.9073486328125e-05, 0, 1048576
