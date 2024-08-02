import re
import time
import json
import pandas as pd
from tqdm import tqdm
# from .nodes2retrievers import (
from nodes2retrievers import (
    get_text_retriever,
    getnodes,
)
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
# from . import data2nodes
import data2nodes
from llama_index.core import PromptTemplate
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
# from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from langchain.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from pyserini.search.lucene import LuceneSearcher


class SubQuery(BaseModel):
    """Search over a database of external knowledge."""

    sub_query: str = Field(
        ...,
        description="A very specific query against the database.",
    )


llm, embed_model, _ = data2nodes.setingAll()


def hyde_generate(query):
    hyde = HyDEQueryTransform(llm=llm, include_original=True)
    hydoc = hyde.run(query)
    print(hydoc.custom_embedding_strs)
    # return hydoc.custom_embedding_strs
    return hydoc


def rewrite_query(query: str, llm, num_queries: int = 3):

    query_gen_str = """\
    You are a helpful assistant that generates multiple search queries based on a \
    single input query. Generate {num_queries} search queries, one on each line, \
    related to the following input query:
    Query: {query}
    Queries:
    """
    query_gen_prompt = PromptTemplate(query_gen_str)
    response = llm.predict(query_gen_prompt, num_queries=num_queries, query=query)
    queries = response.split("\n")
    results = []
    for query in queries:
        query = re.sub(r"\d+\.", "", query)
        query = query.strip().replace('"', "")
        results.append(query)
    results = results[:num_queries]
    print(results)
    return results


def generate_sub_questions(query):
    system = """You are an expert at converting user questions into database queries. \
    You have access to a database of up-to-date external information. \

    Perform query decomposition. Given a user question, break it down into distinct sub questions that \
    you need to answer in order to answer the original question.

    If there are acronyms or words you are not familiar with, do not try to rephrase them."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    llm_with_tools = llm.bind_tools([SubQuery])
    parser = PydanticToolsParser(tools=[SubQuery])
    query_analyzer = prompt | llm_with_tools | parser
    sub_questions = query_analyzer.invoke({"question": {query}})
    results = []
    for sub_question in sub_questions:
        results.append(sub_question.sub_query)
    return results


def merge_nodes(nodes_list):
    id_score_dict = {}
    merged_nodes = []
    for node in nodes_list:
        if node.node_id in id_score_dict:
            id_score_dict[node.node_id].append(node.score)
        else:
            id_score_dict[node.node_id] = [node.score]
            merged_nodes.append(node)
    for id, score_list in id_score_dict.items():
        average_score = max(score_list)
        id_score_dict[id] = average_score
    for node in merged_nodes:
        node.score = id_score_dict[node.node_id]
    sorted_merged_nodes = sorted(merged_nodes, key=lambda x: x.score, reverse=True)
    return sorted_merged_nodes


def retrieve_nodes(queries, retriever, similarity_top_k: int):
    top_k_nodes = []
    # dense_hits = []
    for query in queries:
        nodes = retriever(similarity_top_k=similarity_top_k).retrieve(query)
        top_k_nodes.extend(nodes)
    merged_nodes = merge_nodes(top_k_nodes)[:similarity_top_k]
    rank = 1
    for node in merged_nodes:
        print("=" * 100)
        # dense_hits.append(node.node_id)
        print(rank, "  ", node.node_id, "\t", node.score, "\n", node.text)  #
        rank += 1
    return merged_nodes


def calculate_weighted_scores(nodes_list, weights):
    id_weighted_score_dict = {}
    weighted_nodes = []
    for nodes, weight in zip(nodes_list, weights):
        for node in nodes:
            if node.node_id in id_weighted_score_dict:
                id_weighted_score_dict[node.node_id] += node.score * weight
            else:
                id_weighted_score_dict[node.node_id] = node.score * weight
                weighted_nodes.append(node)
    for node in weighted_nodes:
        node.score = id_weighted_score_dict[node.node_id]
    sorted_weighted_nodes = sorted(weighted_nodes, key=lambda x: x.score, reverse=True)
    for node in sorted_weighted_nodes:
        print(node.node_id, "\t", node.score)
    return sorted_weighted_nodes


def hybrid_results(
    dense_results, sparse_results, alpha, k, normalization=True, weight_on_dense=False
):
    dense_hits = {node.node_id: node.score for node in dense_results}
    sparse_hits = {hit.docid: hit.score for hit in sparse_results}
    hybrid_result = []
    min_dense_score = min(dense_hits.values()) if len(dense_hits) > 0 else 0
    max_dense_score = max(dense_hits.values()) if len(dense_hits) > 0 else 1
    min_sparse_score = min(sparse_hits.values()) if len(sparse_hits) > 0 else 0
    max_sparse_score = max(sparse_hits.values()) if len(sparse_hits) > 0 else 1
    for doc in set(dense_hits.keys()) | set(sparse_hits.keys()):
        if doc not in dense_hits:
            sparse_score = sparse_hits[doc]
            dense_score = min_dense_score
        elif doc not in sparse_hits:
            sparse_score = min_sparse_score
            dense_score = dense_hits[doc]
        else:
            sparse_score = sparse_hits[doc]
            dense_score = dense_hits[doc]
        if normalization:
            sparse_score = (sparse_score - min_sparse_score) / (
                max_sparse_score - min_sparse_score
            )
            dense_score = (dense_score - min_dense_score) / (
                max_dense_score - min_dense_score
            )
            # sparse_score = (sparse_score - (min_sparse_score + max_sparse_score) / 2) \
            #                / (max_sparse_score - min_sparse_score)
            # dense_score = (dense_score - (min_dense_score + max_dense_score) / 2) \
            #               / (max_dense_score - min_dense_score)
        score = (
            alpha * sparse_score + dense_score
            if not weight_on_dense
            else sparse_score + alpha * dense_score
        )
        hybrid_result.append((doc, score))
        sorted_hybrid_result = sorted(hybrid_result, key=lambda x: x[1], reverse=True)[:k]
    return sorted_hybrid_result


# recall@k function
def recall(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = round(len(act_set & pred_set) / float(len(act_set)), 2)
    return result


def evaluate():
    dataset = EmbeddingQAFinetuneDataset.from_json("./data/validation.json")
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    eval_results = []
    for q_id, q_text in tqdm(queries.items()):
        retrieved_nodes = search(q_text, "hyde")
        retrieved_ids = [node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[q_id][0]
        top_5_nodes = retrieved_ids[:5]
        is_hit = expected_id in top_5_nodes
        eval_result = {
            "query": q_id,
            "is_hit": is_hit,
            "expected_id": expected_id,
            "retrieved_id": top_5_nodes,
        }
        eval_results.append(eval_result)
    with open("./results/eval_results_hyde.json", "w") as f:
        json.dump(eval_results, f)
    hit_rate = pd.DataFrame(eval_results)
    print(hit_rate)
    average_hit_rate = hit_rate["is_hit"].mean()
    print(average_hit_rate)
    return eval_results


def persist_bm25_index(nodes_path):
    start = time.time()
    nodes = getnodes(nodes_path)[:10]
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    index = VectorStoreIndex(
        nodes=nodes, storage_context=storage_context, show_progress=True
    )
    index.storage_context.persist(persist_dir="./index/test_index")
    end = time.time()
    total_time = end - start
    print(f"Store index: {total_time:.4f}s")


def search(query, search_method, top_k, milvus_id=1):
    if search_method == "original":
        nodes = get_text_retriever(
            similarity_top_k=top_k, milvus_id=milvus_id
        ).retrieve(query)
        top_k_nodes = [
            {"node_id": node.node_id, "score": node.score, "text": node.text}
            for node in nodes
        ]
    elif search_method == "hyde":
        nodes = get_text_retriever(
            similarity_top_k=top_k, milvus_id=milvus_id
        ).retrieve(hyde_generate(query))
        top_k_nodes = [
            {"node_id": node.node_id, "score": node.score, "text": node.text}
            for node in nodes
        ]
    elif search_method == "hybrid":
        searcher = LuceneSearcher(
            # "/data/zfr/finalTest/opencompass/JointTest/data/bm25_index"
            "/home/xgao/RAG/RAG/data/index/bm25_index"
        )  # BM25
        dense_results = get_text_retriever(
            similarity_top_k=top_k, milvus_id=milvus_id
        ).retrieve(query)
        sparse_results = searcher.search(query, k=top_k)
        nodes = hybrid_results(dense_results, sparse_results, alpha=0.3, k=top_k)
        top_k_nodes = []
        for node in nodes:
            json_doc = json.loads(searcher.doc(node[0]).raw())
            contents = json_doc["contents"]
            top_k_nodes.append({"node_id": node[0], "score": node[1], "text": contents})

    elif search_method == "hyde_with_hybrid":
        searcher = LuceneSearcher(
            # "/data/zfr/finalTest/opencompass/JointTest/data/bm25_index"
            "/home/xgao/RAG/RAG/data/index/bm25_index"
        )  # BM25
        pseudo_doc = hyde_generate(query).custom_embedding_strs[0]
        dense_results = get_text_retriever(
            similarity_top_k=top_k, milvus_id=milvus_id
        ).retrieve(pseudo_doc)
        sparse_results = searcher.search(pseudo_doc, k=top_k)
        nodes = hybrid_results(dense_results, sparse_results, alpha=0.3, k=top_k)
        top_k_nodes = []
        for node in nodes:
            json_doc = json.loads(searcher.doc(node[0]).raw())
            contents = json_doc["contents"]
            top_k_nodes.append({"node_id": node[0], "score": node[1], "text": contents})

    elif search_method == "bm25":
        searcher = LuceneSearcher("./data/bm25_index")  # BM25
        start = time.time()
        hits = searcher.search(query, k=top_k)
        rank = 1
        for hit in hits:
            print("=" * 100)
            json_doc = json.loads(searcher.doc(hit.docid).raw())
            contents = json_doc["contents"]
            print(f"{rank}  {hit.docid}  {hit.score}\n{contents}")  #
            rank += 1
        end = time.time()
        total_time = end - start
        print(f"BM25 Retriever: {total_time:.4f}s")
        # retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=top_k, verbose=True)
        # top_k_nodes = retriever.retrieve(query)
        # for node in top_k_nodes:
        #     print(rank, "  ", node.node_id, "\t", node.score, "\n", node.text)
        #     rank += 1
    elif search_method == "rewrite":
        top_k_nodes = retrieve_nodes(
            rewrite_query(query=query, llm=llm, num_queries=3),
            get_text_retriever,
            top_k,
        )
    elif search_method == "decomposition":
        top_k_nodes = retrieve_nodes(
            generate_sub_questions(query), get_text_retriever, top_k
        )
    else:
        raise ValueError(f"Unknown search method: {search_method}")
    rank = 1
    for node in top_k_nodes:
        print("=" * 100)
        print(rank, "  ", node["node_id"], "\t", node["score"], "\n", node["text"])  #
        rank += 1
    return top_k_nodes

search("Who is joe biden?", "hybrid", 10)