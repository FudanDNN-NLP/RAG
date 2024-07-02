nohup python3 TILDE/expansion.py \
--corpus_path ../data/msmarco/passage/collection.tsv \
--output_dir TILDE/data/collection/expanded \
--topk 200 > test.log 2>&1 &
python3 TILDE/expansion.py \
--corpus_path ../data/msmarco/passage/collection.tsv \
--output_dir TILDE/data/collection/expanded \
--topk 200