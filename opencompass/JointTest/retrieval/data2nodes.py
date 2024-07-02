import nest_asyncio
import torch
from transformers import BitsAndBytesConfig
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Settings
#请求openai存在延迟，使用协程
nest_asyncio.apply()
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor
)
import openai
import os
from llama_index.core import Settings
from transformers import AutoTokenizer
import os
import uuid
import json
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core.schema import TextNode,IndexNode
import re
from llama_index.core import Settings
nest_asyncio.apply()


#相关的模型
llm = None
tokenizer = None
embed_model = None

def get_llm():
    global llm
    device = torch.device('cuda:0')
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    )
    if llm is None:
        llm = HuggingFaceLLM(
        model_name="HuggingFaceH4/zephyr-7b-alpha",
        tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
        # query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
        context_window=3900,
        max_new_tokens=300,
        model_kwargs={"quantization_config": quantization_config},
        # tokenizer_kwargs={},
        generate_kwargs={"temperature": 0.3, "top_k": 30, "top_p": 0.75},
        # messages_to_prompt=messages_to_prompt,
        # device_map="auto",
        device_map = "cuda:0"
        )
    return llm

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer =  AutoTokenizer.from_pretrained(
            "HuggingFaceH4/zephyr-7b-alpha"
        )
    return  tokenizer

def get_embedding_model():
    global embed_model
    if embed_model is None:
        embed_args = {'model_name': 'BAAI/llm-embedder', 'max_length': 512, 'embed_batch_size': 8, 'device': 'cuda:0'}
        embed_model = HuggingFaceEmbedding(**embed_args)
    return embed_model

def setingAll():
    Settings.llm = get_llm()
    Settings.embed_model = get_embedding_model()
    Settings.tokenizer = get_tokenizer()
    return  Settings.llm,Settings.embed_model,Settings.tokenizer

    
def nochunk_to_nodes(root_dir,llm,nodes_store_path,fintune_store_path):
    # root_dir = "./data/corpus/"
    # nodes_store_path = "./data/ragga/nodes_storage"
    # fintune_store_path = "./data/ragga/fintuneDataset.json"
    text_pipeline = IngestionPipeline(
    transformations=[
        KeywordExtractor(llm=llm, keywords=6, metadata_mode="none"),
        QuestionsAnsweredExtractor(llm=llm, questions=2,metadata_mode="none"),
        SummaryExtractor(llm=llm, summaries=['self'],metadata_mode="none"),
        TitleExtractor(llm=llm, nodes=1,metadata_mode="none")
    ]
    )
    qa_pipeline = IngestionPipeline(
    transformations=[
        KeywordExtractor(llm=llm, keywords=6, metadata_mode="none"),
        SummaryExtractor(llm=llm, summaries=['self'],metadata_mode="none"),
        TitleExtractor(llm=llm, nodes=1,metadata_mode="none")
    ]
    )

    text_total_nodes = []
    qa_total_nodes = []

    for filename in os.listdir(root_dir):
        if "qa" in filename.lower():
            with open(os.path.join(root_dir,filename),"rb") as f:
                qa_list = json.load(f)
                for qa in qa_list:
                    # print(qa[0])
                    node = TextNode(text = qa[1])
                    cur_que = qa[0]
                    pattern1 = re.compile(r'\d+\.')
                    pattern2 = re.compile(r'\?')
                    if pattern1.match(cur_que[0:2]) is None:
                        cur_que = "1."+cur_que
                    if pattern2.match(cur_que[-1:]) is None:
                        cur_que = cur_que+"?"
                    node.metadata["questions_this_excerpt_can_answer"] = cur_que
                    qa_total_nodes.append(node)

        elif "text" in filename.lower():
            with open(os.path.join(root_dir,filename),"rb") as f:
                text_list = json.load(f)
                for text in text_list:
                    node = TextNode(text = text)
                    text_total_nodes.append(node)

    if len(qa_total_nodes)!=0:
        qa_total_nodes = qa_pipeline.run(nodes = qa_total_nodes)
    if len(text_total_nodes)!=0:
        text_total_nodes = text_pipeline.run(nodes =text_total_nodes)


    total_nodes = qa_total_nodes + text_total_nodes

    storage_nodes(total_nodes,nodes_store_path)
    generate_fintune_dataset(total_nodes,fintune_store_path)


def chunk_to_nodes(root_dir, llm, chunk_size, chunk_overloop,nodes_storage_path,fintune_dataset_path):
    documents = SimpleDirectoryReader(root_dir).load_data()

    splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overloop
                )

    transformations=[
        KeywordExtractor(llm=llm, keywords=6, metadata_mode="none"),
        QuestionsAnsweredExtractor(llm=llm, questions=2,metadata_mode="none"),
        SummaryExtractor(llm=llm, summaries=['self',"prev","next"],metadata_mode="none"),
        TitleExtractor(llm=llm, nodes=3,metadata_mode="none")
    ]
    transformations.insert(0,splitter)
    pipeline = IngestionPipeline(transformations=transformations)
    total_nodes = pipeline.run(documents=documents)

    storage_nodes(total_nodes, nodes_storage_path)
    generate_fintune_dataset(total_nodes, fintune_dataset_path)



def generate_fintune_dataset(total_nodes,fintune_dataset_path):
    queries = {}
    corpus = {}
    relevant_docs = {}
    pattern = re.compile(r'\d+\.\s*(.*?)\?')
    for node in total_nodes:
        merge_que = node.metadata['questions_this_excerpt_can_answer']
        for que in pattern.findall(merge_que):
            question_id = str(uuid.uuid4())
            queries[question_id] = que
            corpus[node.id_] = node.get_content(metadata_mode="none")
            relevant_docs[question_id] = [node.id_]
    ftdataset = EmbeddingQAFinetuneDataset(
            queries=queries, corpus=corpus, relevant_docs=relevant_docs
        )
    ftdataset.save_json(fintune_dataset_path)


def storage_nodes(total_nodes, nodes_store_path):
    dc = SimpleDocumentStore()
    dc.add_documents(total_nodes)
    dc.persist(nodes_store_path)

