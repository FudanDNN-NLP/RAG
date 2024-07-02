python evaluate_document_ranker.py --split dev \
                                                --method seq_class_transformer \
                                                --model castorini/monobert-large-msmarco \
                                                --dataset ../data/msmarco_doc_ans_small/fh \
                                                --model-type bert-large-uncased \
                                                --task msmarco \
                                                --index-dir ../indexes/index-msmarco-doc-20201117-f87c94 \
                                                --batch-size 32 \
                                                --output-file runs/run.monobert.doc_fh.dev.tsv