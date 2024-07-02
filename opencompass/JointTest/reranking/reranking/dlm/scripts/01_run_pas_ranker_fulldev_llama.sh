nohup python evaluate_passage_ranker.py --split dev \
                                                --method llama \
                                                --model castorini/rankllama-v1-7b-lora-passage \
                                                --dataset ../data/msmarco/passage/dev-6980 \
                                                --model-type Llama-2-7b-hf \
                                                --task msmarco \
                                                --index-dir ../indexes/index-msmarco-passage-20191117-0ed488 \
                                                --batch-size 32 \
                                                --output-file runs/run.rankllama.ans_full.dev.tsv > test-llamafull.log 2>&1 &