python3 testRecomp.py \
--model_name_or_path fangyuan/nq_abstractive_compressor \
--do_predict \
--test_file  test.json \
--max_target_length 512 \
--output_dir outputs/ \
--per_device_train_batch_size=4 \
--per_device_eval_batch_size=16 \
--predict_with_generate
