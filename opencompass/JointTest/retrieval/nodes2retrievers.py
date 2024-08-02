import nest_asyncio
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.schema import BaseNode

nest_asyncio.apply()


def getnodes(persist_path) -> list[BaseNode]:
    dc = SimpleDocumentStore.from_persist_path(persist_path)
    node_ids = [v for k, v in dc.get_all_document_hashes().items()]
    total_nodes = dc.get_nodes(node_ids=node_ids)
    return total_nodes


def get_text_retriever(similarity_top_k, milvus_id=1):
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
        uri=f"/home/xgao/RAGGA/RAG/opencompass/JointTest/data/wikipedia_milvus{milvus_id}.db",  # ./opencompass/JointTest/data/wikipedia_milvus{milvus_id}.db
        dim=768,
    )
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )
    # load your index from stored vectors
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
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