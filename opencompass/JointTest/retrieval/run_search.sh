#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python queries2retrievers.py \
    --search_method "hybrid" \
    --top_k 10