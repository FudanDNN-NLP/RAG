export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

export OPENAI_API_KEY='your-api-key-here'

# test_list = {
#     "with_retrieval_classification": [True, False],
#     "search_method": ["hyde", "original", "hybrid","hyde_with_hybrid"],  
#     "rerank_model": ["MonoT5", "TILDE", "MonoBERT", "RankLLaMA"],  
#     "compression_method": ["recomp", "longllmlingua"],  
#     "repack_method": ["sides", "compact", "compact_reverse"],  
# }



classification=True \
search_method=hyde_with_hybrid \
rerank_model=MonoT5 \
compression_method=recomp \
repack_method=compact_reverse \
milvus=1 \
nohup python run.py ./configs/myeval/eval_all.py -w outputs/t_hyde_mt_r_s/eval_all > best_practice.out 2>&1 &

classification=True \
search_method=hybrid \
rerank_model=TILDE \
compression_method=recomp \
repack_method=compact_reverse \
milvus=1 \
nohup python run.py ./configs/myeval/eval_all.py -w outputs/t_hyde_mt_r_s/eval_all > efficiency_practice.out 2>&1 &



