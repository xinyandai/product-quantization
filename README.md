product-quantization
==================
A general framework of product quantization with python.


# Datasets
The netflix dataset is contained in this repository, you can download more datasets from
 [here](https://xinyandai.github.io/#Datasets), 
 then you can calculate the ground truth with the script

    python run_ground_truth.py  --dataset netflix --topk 50 --metric product

# run examples

    python run_pq.py --dataset netflix --topk 20 --metric product --num_codebook 4 --Ks 256
    python run_opq.py --dataset netflix --topk 20 --metric product --num_codebook 4 --Ks 256
    python run_rq.py --dataset netflix --topk 20 --metric product --num_codebook 4 --Ks 256
    python run_aq.py --dataset netflix --topk 20 --metric product --num_codebook 4 --Ks 256 # very slow

# reproduce result of NEQ

    python run_norm_pq.py --dataset netflix --topk 20 --metric product --num_codebook 4 --Ks 256
    python run_norm_opq.py --dataset netflix --topk 20 --metric product --num_codebook 4 --Ks 256
    python run_norm_rq.py --dataset netflix --topk 20 --metric product --num_codebook 4 --Ks 256
    python run_norm_aq.py --dataset netflix --topk 20 --metric product --num_codebook 4 --Ks 256 # very slow
