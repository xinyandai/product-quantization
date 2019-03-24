# hpc10.cse.cuhk.edu.hk:/research/jcheng2/xinyan/github/product-quantization> python run_opq.py --dataset sift1m --topk 20 --metric euclid --num_codebook 9 --Ks 256
# Parameters: dataset = sift1m, topK = 20, codebook = 9, Ks = 256, metric = euclid
# load the base data ./data/sift1m/sift1m_base.fvecs,
# load the queries ./data/sift1m/sift1m_query.fvecs,
# load the ground truth ./data/sift1m/20_sift1m_euclid_groundtruth.ivecs
# ranking metric euclid
# ORQ, RQ : [Subspace PQ, M: 9, Ks : 256, code_dtype: <class 'numpy.uint8'>],  M: 9, Ks : 256, code_dtype: <class 'numpy.uint8'>
# 95%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎           | 19/20 [01:22<00:04,  4.10s/it]
#    Training the subspace: 0 / 9, 0 -> 15
#    Training the subspace: 1 / 9, 15 -> 30
#    Training the subspace: 2 / 9, 30 -> 44
#    Training the subspace: 3 / 9, 44 -> 58
#    Training the subspace: 4 / 9, 58 -> 72
#    Training the subspace: 5 / 9, 72 -> 86
#    Training the subspace: 6 / 9, 86 -> 100
#    Training the subspace: 7 / 9, 100 -> 114
#    Training the subspace: 8 / 9, 114 -> 128
# layer: 0,  residual average norm : 137.1579132080078 max norm: 265.6442565917969 min norm: 54.73295974731445
# compress items
# sorting items
# searching!
expected items, overall time, avg recall, avg precision, avg error, avg items
1, 0, 0.04995, 0.999, 0, 1
2, 0, 0.09359999999999999, 0.9359999999999999, 0, 2
4, 0, 0.16469999999999999, 0.8234999999999999, 0, 4
8, 0, 0.268, 0.67, 0, 8
16, 0, 0.40395000000000003, 0.5049375, 0, 16
32, 0, 0.5611999999999999, 0.35074999999999995, 0, 32
64, 0, 0.71865, 0.22457812500000002, 0, 64
128, 0, 0.84745, 0.1324140625, 0, 128
256, 0, 0.9335, 0.0729296875, 0, 256
512, 0, 0.9738000000000001, 0.038039062500000005, 0, 512
1024, 0, 0.9931000000000001, 0.019396484375000002, 0, 1024
2048, 0, 0.9984500000000001, 0.00975048828125, 0, 2048
4096, 0, 0.9997999999999999, 0.0048818359375, 0, 4096
8192, 0, 1.0, 0.00244140625, 0, 8192
16384, 0, 1.0, 0.001220703125, 0, 16384
32768, 0, 1.0, 0.0006103515625, 0, 32768
65536, 0, 1.0, 0.00030517578125, 0, 65536
131072, 0, 1.0, 0.000152587890625, 0, 131072
262144, 0, 1.0, 7.62939453125e-05, 0, 262144
524288, 0, 1.0, 3.814697265625e-05, 0, 524288
1048576, 0, 1.0, 1.9073486328125e-05, 0, 1048576
