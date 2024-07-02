nohup python evaluate_passage_ranker.py --split dev \
                                                --method seq_class_transformer \
                                                --model castorini/monobert-large-msmarco \
                                                --dataset ../data/msmarco/passage/dev-6980 \
                                                --model-type bert-large-uncased \
                                                --task msmarco \
                                                --index-dir ../indexes/index-msmarco-passage-20191117-0ed488 \
                                                --output-file runs/run.monobert.ans_full.dev.tsv > test-bertfull.log 2>&1 &