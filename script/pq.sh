#!/usr/bin/env bash
cd ..

codebook=$1
dataset=$3

mkdir script/$3
log=script/$3/$1_$2_$(basename "$0").log
echo witing into file : $log

python3 ./run.py \
    -q rq \
    --dataset ${dataset} \
    --topk 20 \
    --metric product \
    --ranker product \
    --num_codebook ${codebook} \
    --layer 1 \
    --Ks 256 \
    > $log
