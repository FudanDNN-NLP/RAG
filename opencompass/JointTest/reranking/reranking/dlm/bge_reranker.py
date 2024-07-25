from FlagEmbedding import FlagReranker


def get_model():
    return FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation


def bge_rerank(model, query, docs):
    # reranker = get_model()
    reranker = model

    qd_list = [[query, doc] for doc in docs]
    scores = reranker.compute_score(qd_list)

    doc_score_pairs = list(zip(docs, scores))
    sorted_doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    reranked_docs = [pair[0] for pair in sorted_doc_score_pairs]
    reranked_scores = [pair[1] for pair in sorted_doc_score_pairs]

    return reranked_docs, reranked_scores
