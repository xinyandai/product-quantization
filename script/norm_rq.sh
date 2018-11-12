#!/usr/bin/env bash
cd ..

codebook=`expr $1 - 1`
norm_centroid=$2
dataset=$3

log=script/$1_codebook_$(basename "$0").log
echo witing into file : $log
python3 ./run.py \
    -q rq \
    --sup_quantizer NormPQ \
    --dataset ${dataset} \
    --topk 20 \
    --metric product \
    --ranker product \
    --num_codebook 1 \
    --layer ${codebook} \
    --Ks 256 \
    --norm_centroid ${norm_centroid} \
    > $log
