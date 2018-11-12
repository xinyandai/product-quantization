#!/usr/bin/env bash
cd ..

codebook=2

python3 ./run.py \
    -q rq \
    --dataset yahoomusic \
    --topk 20 \
    --metric product \
    --ranker product \
    --num_codebook ${codebook} \
    --layer 1 \
    --Ks 256
