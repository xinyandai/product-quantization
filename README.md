# product-quantization
A general framework of vector quantization with python.

# [NEQ](https://arxiv.org/abs/1911.04654), AAAI 2020, Oral
Norm-Explicit Quantization: Improving Vector Quantization for Maximum Inner Product Search.
* Abstract

  Vector quantization (VQ) techniques are widely used in similarity search for
  data compression, fast metric computation and etc. Originally designed for
  Euclidean distance, existing VQ techniques (e.g., PQ, AQ) explicitly or
  implicitly minimize the quantization error. In this paper, we present a new
  angle to analyze the quantization error, which decomposes the quantization
  error into norm error and direction error. We show that quantization errors in
  norm have much higher influence on inner products than quantization errors in
  direction, and small quantization error does not necessarily lead to good
  performance in maximum inner product search (MIPS). Based on this observation,
  we propose norm-explicit quantization (NEQ) --- a general paradigm that
  improves existing VQ techniques for MIPS. NEQ quantizes the norms of items in a
  dataset explicitly to reduce errors in norm, which is crucial for MIPS. For the
  direction vectors, NEQ can simply reuse an existing VQ technique to quantize
  them without modification. We conducted extensive experiments on a variety of
  datasets and parameter configurations. The experimental results show that NEQ
  improves the performance of various VQ techniques for MIPS, including PQ, OPQ,
  RQ and AQ.

# Datasets
The netflix dataset is contained in this repository, you can download more datasets from
 [here](https://xinyandai.github.io/#Datasets), 
 then you can calculate the ground truth with the script

    python run_ground_truth.py  --dataset netflix --topk 50 --metric product

# Run examples

    python run_pq.py --dataset netflix --topk 20 --metric product --num_codebook 4 --Ks 256
    python run_opq.py --dataset netflix --topk 20 --metric product --num_codebook 4 --Ks 256
    python run_rq.py --dataset netflix --topk 20 --metric product --num_codebook 4 --Ks 256
    python run_aq.py --dataset netflix --topk 20 --metric product --num_codebook 4 --Ks 256 # very slow

# Reproduce results of NEQ

    python run_norm_pq.py --dataset netflix --topk 20 --metric product --num_codebook 4 --Ks 256
    python run_norm_opq.py --dataset netflix --topk 20 --metric product --num_codebook 4 --Ks 256
    python run_norm_rq.py --dataset netflix --topk 20 --metric product --num_codebook 4 --Ks 256
    python run_norm_aq.py --dataset netflix --topk 20 --metric product --num_codebook 4 --Ks 256 # very slow

# Reference
If you use this code, please cite the following [paper](https://arxiv.org/abs/1911.04654)

    @article{xinyandai,
      title={Norm-Explicit Quantization: Improving Vector Quantization for Maximum Inner Product Search},
      author={Dai, Xinyan and Yan, Xiao and Ng, Kelvin KW and Liu, Jie and Cheng, James},
      journal={arXiv preprint arXiv:1911.04654},
      year={2019}
    }
