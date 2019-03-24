# hpc9.cse.cuhk.edu.hk:/research/jcheng2/xinyan/github/product-quantization> python run_rpq.py --dataset sift1m --topk 20 --metric euclid --num_codebook 8 --Ks 256
# Parameters: dataset = sift1m, topK = 20, codebook = 8, Ks = 256, metric = euclid
# load the base data ./data/sift1m/sift1m_base.fvecs,
# load the queries ./data/sift1m/sift1m_query.fvecs,
# load the ground truth ./data/sift1m/20_sift1m_euclid_groundtruth.ivecs
# ranking metric euclid
# ORQ, RQ : [Subspace PQ, M: 4, Ks : 256, code_dtype: <class 'numpy.uint8'>],  M: 4, Ks : 256, code_dtype: <class 'numpy.uint8'>ORQ, RQ : [Subspace PQ, M: 4, Ks : 256, code_dtype: <class 'numpy.uint8'>],  M: 4, Ks : 256, code_dtype: <class 'numpy.uint8'>
# 95%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎           | 19/20 [00:37<00:01,  1.92s/it]
#    Training the subspace: 0 / 4, 0 -> 32
#    Training the subspace: 1 / 4, 32 -> 64
#    Training the subspace: 2 / 4, 64 -> 96
#    Training the subspace: 3 / 4, 96 -> 128
# layer: 0,  residual average norm : 183.70541381835938 max norm: 336.2265625 min norm: 61.80232238769531
# layer: 0,  residual average norm : 183.70541381835938 max norm: 336.2265930175781 min norm: 61.80230712890625
# 95%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎           | 19/20 [00:35<00:01,  1.92s/it]
#    Training the subspace: 0 / 4, 0 -> 32
#    Training the subspace: 1 / 4, 32 -> 64
#    Training the subspace: 2 / 4, 64 -> 96
#    Training the subspace: 3 / 4, 96 -> 128
# layer: 0,  residual average norm : 143.2368927001953 max norm: 270.3423767089844 min norm: 50.18209457397461
# layer: 1,  residual average norm : 143.2368927001953 max norm: 270.3423767089844 min norm: 50.18209457397461
# compress items
# sorting items
# searching!
expected items, overall time, avg recall, avg precision, avg error, avg items
1, 0, 0.04985, 0.997, 0, 1
2, 0, 0.09319999999999999, 0.9319999999999999, 0, 2
4, 0, 0.15875, 0.79375, 0, 4
8, 0, 0.256, 0.64, 0, 8
16, 0, 0.38375000000000004, 0.47968750000000004, 0, 16
32, 0, 0.5350499999999999, 0.33440624999999996, 0, 32
64, 0, 0.6907, 0.21584375, 0, 64
128, 0, 0.825, 0.12890625, 0, 128
256, 0, 0.9171500000000001, 0.07165234375000001, 0, 256
512, 0, 0.9686, 0.0378359375, 0, 512
1024, 0, 0.9896, 0.019328125, 0, 1024
2048, 0, 0.9971500000000001, 0.00973779296875, 0, 2048
4096, 0, 0.9994500000000001, 0.004880126953125, 0, 4096
8192, 0, 0.99995, 0.0024412841796875, 0, 8192
16384, 0, 1.0, 0.001220703125, 0, 16384
32768, 0, 1.0, 0.0006103515625, 0, 32768
65536, 0, 1.0, 0.00030517578125, 0, 65536
131072, 0, 1.0, 0.000152587890625, 0, 131072
262144, 0, 1.0, 7.62939453125e-05, 0, 262144
524288, 0, 1.0, 3.814697265625e-05, 0, 524288
1048576, 0, 1.0, 1.9073486328125e-05, 0, 1048576
