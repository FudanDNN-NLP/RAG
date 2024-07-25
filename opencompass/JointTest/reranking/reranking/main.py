from typing import List, Tuple

from .dlm.pygaggle.pygaggle.rerank.base import Query, Text

from .tilde.run_rerank import tilde_rerank
from .dlm.rankllama import rankllama_rerank
from .dlm.bge_reranker import bge_rerank


def rerank(model, expansion_model, tokenizer, bert_tokenizer, mode, query: str, docs: List[str], top_k: int = None) -> Tuple[List[str], List[float]]:
    if top_k is None:
        top_k = len(docs)

    reranked_docs, similarity_scores = run_rerank(model, expansion_model, tokenizer, bert_tokenizer, mode, query, docs, top_k)
    return reranked_docs, similarity_scores


def run_rerank(model, expansion_model, tokenizer, bert_tokenizer, mode, query, docs, top_k=None):

    if top_k is None or top_k > len(docs):
        top_k = len(docs)

    if mode == "MonoT5" or mode == "MonoBERT":
        model = model
        query = Query(query)
        texts = [Text(doc) for doc in docs]

        reranked = model.rerank(query, texts)

        print("=============== After Reranking ===============")
        reranked_docs, scores = [], []
        for i in range(top_k):
            reranked_docs.append(reranked[i].text)
            scores.append(float(f"{reranked[i].score:f}"))
            print(f"{i + 1:2} {reranked[i].score:.5f} {reranked[i].text}")

        return reranked_docs, scores

    elif mode == "RankLLaMA":
        query = Query(query)
        reranked_docs, scores = rankllama_rerank(model, tokenizer, query, docs)

        print("=============== After Reranking ===============")
        for i in range(top_k):
            print(f"{i + 1:2} {scores[i]:.5f} {reranked_docs[i]}")
        return reranked_docs[:top_k], scores[:top_k]

    elif mode == "TILDE":
        reranked_docs, scores = tilde_rerank(model, expansion_model, tokenizer, bert_tokenizer, query, docs)

        print("=============== After Reranking ===============")
        for i in range(top_k):
            print(f"{i + 1:2} {scores[i]:.5f} {reranked_docs[i]}")
        return reranked_docs[:top_k], scores[:top_k]

    elif mode == "BGE":
        reranked_docs, scores = bge_rerank(model, query, docs)

        print("=============== After Reranking ===============")
        for i in range(top_k):
            print(f'{i + 1:2} {scores[i]:.5f} {reranked_docs[i]}')
        return reranked_docs[:top_k], scores[:top_k]
