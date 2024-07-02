python evaluate_passage_ranker.py --split dev \
                                                --method t5 \
                                                --model models/T5-base_zh/checkpoint-45000 \
                                                --dataset ../data/msmarco/passage/cn \
                                                --model-type t5-base \
                                                --task msmarco \
                                                --index-dir ../indexes/index-msmarco-passage-20191117-0ed488 \
                                                --batch-size 32 \
                                                --output-file runs/run.monot5.zh-45000.dev.tsv