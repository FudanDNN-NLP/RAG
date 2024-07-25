python evaluate_passage_ranker.py --split dev \
                                                --method bge_reranker \
                                                --model BAAI/bge-reranker-v2-m3 \
                                                --dataset ../data/msmarco_ans_small \
                                                --task msmarco \
                                                --index-dir ../indexes/index-msmarco-passage-20191117-0ed488 \
                                                --batch-size 32 \
                                                --output-file runs/run.bge.ans_small.dev.tsv