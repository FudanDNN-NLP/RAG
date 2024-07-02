# -*- coding: utf-8 -*-
import nest_asyncio
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import Settings
from llama_index.core.schema import MetadataMode, IndexNode

nest_asyncio.apply()
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    Document,
    load_index_from_storage,
)
from llama_index.core import Settings
from llama_index.core.schema import IndexNode
import re
from llama_index.core.schema import BaseNode, IndexNode
from .modify_files.modify_keywordRetriever import (
    KeywordTableGPTRetriever,
    KeywordTableRAKERetriever,
)
from llama_index.core import RAKEKeywordTableIndex
from llama_index.core.vector_stores.types import VectorStoreQueryMode


nest_asyncio.apply()


def getnodes(persist_path) -> list[BaseNode]:
    dc = SimpleDocumentStore.from_persist_path(persist_path)
    node_ids = [v for k, v in dc.get_all_document_hashes().items()]
    total_nodes = dc.get_nodes(node_ids=node_ids)
    return total_nodes


"""
document_title
excerpt_keywords
next_section_summary
questions_this_excerpt_can_answer
"""


def generate_index_nodes(total_nodes, div_num=4, mode="text"):
    text_len = 0
    metadata_len = 0
    effient_len = 0
    ori2index = {}
    if mode == "text" and div_num == 1:
        for node in total_nodes:
            ori2index[node.id_] = {"st": [IndexNode.from_text_node(node, node.node_id)]}
        return ori2index
    if mode == "text":
        for node in total_nodes:
            node.excluded_llm_metadata_keys = []
            text_len = len(node.text)
            metadata_len = len(node.get_metadata_str(mode=MetadataMode.LLM))
            effient_len = int(text_len / div_num + metadata_len) + int(text_len) / 5
            small_splitter = SentenceSplitter(chunk_size=effient_len, chunk_overlap=int(text_len) / 8)
            ori2index = {}
            sub_nodes = small_splitter.get_nodes_from_documents([node])
            st_nodes = [IndexNode.from_text_node(sn, node.node_id) for sn in sub_nodes]
            ori2index[node.id_] = {"st": st_nodes}
    elif mode == "sq":
        for node in total_nodes:
            pattern = re.compile(r"\d+\.\s*(.*?)\?")
            merge_que = node.metadata["questions_this_excerpt_can_answer"]
            sq_inodes = [IndexNode(text=que, index_id=node.node_id) for que in pattern.findall(merge_que)]
            ori2index[node.id_] = {"sq": sq_inodes}
    return ori2index


def mapfun(node, forbid_keys=["questions_this_excerpt_can_answer", "excerpt_keywords"]):
    node.excluded_llm_metadata_keys = forbid_keys
    node.excluded_embed_metadata_keys = forbid_keys

    return node


def get_text_retriever(similarity_top_k, milvus_id=1, div_num=1):
    # all_nodes = []
    # total_nodes = getnodes(doc_store_path)
    # ori2index = generate_index_nodes(total_nodes=total_nodes,div_num=div_num,mode="text")
    # for k, v in ori2index.items():
    #     all_nodes = all_nodes + v["st"]
    # for node in total_nodes:
    #     all_nodes.append(IndexNode.from_text_node(node,node.node_id))
    # all_nodes = list(map(mapfun,all_nodes))
    # all_nodes_dict = {node.node_id:node for node in all_nodes}
    # service_context = ServiceContext.from_defaults(embed_model=Settings.embed_model)
    # # st_vec = VectorStoreIndex(all_nodes, service_context=service_context)
    vector_store = MilvusVectorStore(
        uri=f"./opencompass/JointTest/data/wikipedia_milvus{milvus_id}.db",
        dim=768,
        # enable_sparse=True
    )
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )
    # load your index from stored vectors
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    vector_st_retri = index.as_retriever(similarity_top_k=similarity_top_k)
    #     hybrid_retriever = index.as_retriever(
    #     vector_store_query_mode=VectorStoreQueryMode.HYBRID
    # )
    #     nodes = hybrid_retriever.retrieve("Joe Biden")
    #     rank = 1
    #     for node in nodes:
    #         print("=" * 100)
    #         print(rank, "  ", node.node_id, "\t", node.score, "\n", node.text) #
    #         rank += 1
    return vector_st_retri


def get_question_retriever(doc_store_path, similarity_top_k=5):
    all_nodes = []
    total_nodes = getnodes(doc_store_path)
    total_nodes = list(map(mapfun, total_nodes))
    ori2index = generate_index_nodes(total_nodes=total_nodes, mode="sq")
    for k, v in ori2index.items():
        all_nodes = all_nodes + v["sq"]
    all_nodes_dict = {node.node_id: node for node in total_nodes}
    dim = 768
    vector_store = MilvusVectorStore(
        uri="./data/wikipedia_milvus_93GB.db",
        dim=dim,  # embedding size=768
        overwrite=True,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=all_nodes, storage_context=storage_context, show_progress=True)

    service_context = ServiceContext.from_defaults(embed_model=Settings.embed_model)
    sq_vec = VectorStoreIndex(all_nodes, service_context=service_context)
    vector_sq_retri = index.as_retriever(similarity_top_k=similarity_top_k)
    vector_sq_retri = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_sq_retri},
        node_dict=all_nodes_dict,
        verbose=True,
    )
    return vector_sq_retri


def get_keyword_retriever(doc_store_path, similarity_top_k=5):
    all_nodes = []
    total_nodes = getnodes(doc_store_path)
    keyword_index = RAKEKeywordTableIndex(total_nodes, max_keywords_per_chunk=5)
    key_2_id = keyword_index.index_struct
    for node in total_nodes:
        key_2_id.add_node(keywords=node.metadata["excerpt_keywords"].split(","), node=node)
    kw_retri = KeywordTableRAKERetriever(keyword_index, num_chunks_per_query=similarity_top_k, max_keywords_per_query=2)
    # kw_retri = KeywordTableGPTRetriever(keyword_index, num_chunks_per_query=similarity_top_k)
    return kw_retri

