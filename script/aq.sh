#!/usr/bin/env bash
cd ..

codebook=$1
dataset=$3

mkdir script/$3/$4
log=script/$3/$4/$1_$2_$(basename "$0").log
echo witing into file : $log

python3 ./run.py \
    -q aq \
    --dataset ${dataset} \
    --topk $4 \
    --metric product \
    --ranker product \
    --num_codebook ${codebook} \
    --Ks 256 \
    > $log
