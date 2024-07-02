nohup python3 TILDE/indexingv2.py \
--ckpt_path_or_name ielab/TILDEv2-TILDE200-exp \
--collection_path TILDE/data/collection/expanded/ \
--output_path TILDE/data/index/TILDEv2 > test.log 2>&1 &
python3 TILDE/indexingv2.py \
--ckpt_path_or_name ielab/TILDEv2-TILDE200-exp \
--collection_path TILDE/data/collection/expanded/ \
--output_path TILDE/data/index/TILDEv2