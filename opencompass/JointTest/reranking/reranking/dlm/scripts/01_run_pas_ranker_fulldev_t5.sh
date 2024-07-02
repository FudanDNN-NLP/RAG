nohup python evaluate_passage_ranker.py --split dev \
                                                --method t5 \
                                                --model castorini/monot5-base-msmarco \
                                                --dataset ../data/msmarco/passage/dev-6980 \
                                                --model-type t5-base \
                                                --task msmarco \
                                                --index-dir ../indexes/index-msmarco-passage-20191117-0ed488 \
                                                --batch-size 32 \
                                                --output-file runs/run.monot5.ans_full.dev.tsv > test-t5full.log 2>&1 &