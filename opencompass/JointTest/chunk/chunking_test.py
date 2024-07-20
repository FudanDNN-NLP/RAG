#测试token，sentence，semnatic，agent，silding window，hirerarchical，query
import argparse
import nest_asyncio
import torch
from llama_index.core.retrievers import RecursiveRetriever
from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
import llama_index.core.storage.docstore 
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Settings
from llama_index.core import SimpleKeywordTableIndex
from llama_index.core.schema import MetadataMode,IndexNode,TextNode
from tqdm import tqdm
# from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.indices import keyword_table
#请求openai存在延迟，使用协程
nest_asyncio.apply()
from llama_index.core.node_parser import(
    TokenTextSplitter,
    SentenceSplitter,
    SentenceWindowNodeParser,
    SemanticSplitterNodeParser,
    TokenTextSplitter,
    HierarchicalNodeParser
)

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    
)
from llama_index.core.evaluation import (
    DatasetGenerator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
import pandas as pd
from llama_index.llms.openai import OpenAI
import time
from llama_index.core import Settings
from transformers import AutoTokenizer
import os
nest_asyncio.apply()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["OPENAI_API_KEY"]=""
parse = argparse.ArgumentParser(description="test different factors of chunk")
parse.add_argument("--chunk_sizes",type=list,default=[512],help="different chunk size")
parse.add_argument("--chunk_manners",type=list,default=["small2big"])
parse.add_argument("--num_page", type=int, default=10, help="number of pages for retrieval")
parse.add_argument("--return_retrieval_num",default=2,type=int,help="top_k of retrieval")
parse.add_argument("--retrieval_source_path", type=str, default="./data/10k", help="the source to retrieval")
parse.add_argument("--dataset_store_file_path", type=str, default="./data/qa_test.csv", help="the path of dataset")
parse.add_argument("--test_chunk_size_path", type=str, default="./data/chunksize_test.csv", help="the path for chunksize result")
parse.add_argument("--defaultchunksize",default=512,type=int,help="测试方式的chunksize")
parse.add_argument("--defaultoverlap",default=20,type=int,help="测试方式的chunksize")
group = parse.add_mutually_exclusive_group()
group.add_argument("--testsize",action='store_true', help="test size")
group.add_argument("--testmanners",action="store_true", help="test manners")
args = parse.parse_args()

gpt3 = OpenAI(model="gpt-3.5-turbo")
documents = SimpleDirectoryReader(args.retrieval_source_path).load_data()[:args.num_page]
eval_documents = documents
eval_questions = pd.read_csv(args.dataset_store_file_path)["query"].tolist()
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt3)
faithfulness_gpt4 = FaithfulnessEvaluator(service_context=service_context_gpt4)
relevancy_gpt4 = RelevancyEvaluator(service_context=service_context_gpt4)
class sensp:
    def __init__(self,chunksize,overlap):
        self.sp = SentenceSplitter(chunk_size=chunksize,chunk_overlap=overlap)
    def __call__(self,text):
        cur_node = TextNode(text = text)
        total_nodes = self.sp([cur_node])
        return [node.text for node in total_nodes]

sws = sensp(chunksize=180,overlap=0)


def evaluate_response_time_and_accuracy(chunk_size, eval_questions, query=None):

    total_response_time = 0
    total_faithfulness = 0
    total_relevancy = 0
    num_questions = len(eval_questions)


    # create vector index
    if query == None:
        service_context = ServiceContext.from_defaults(chunk_size=chunk_size, chunk_overlap=20)
        vector_index = VectorStoreIndex.from_documents(
            eval_documents, service_context=service_context,embed_model=Settings.embed_model
        )
        # build query engine
        query_engine = vector_index.as_query_engine(llm=Settings.llm)
    else:
        query_engine = query

    for question in tqdm(eval_questions):
        start_time = time.time()
        response_vector = query_engine.query(question)
        elapsed_time = time.time() - start_time
        
        faithfulness_result = faithfulness_gpt4.evaluate_response(
            response=response_vector
        ).passing
        
        relevancy_result = relevancy_gpt4.evaluate_response(
            query=question, response=response_vector
        ).passing

        total_response_time += elapsed_time
        total_faithfulness += faithfulness_result
        total_relevancy += relevancy_result

    average_response_time = total_response_time / num_questions
    average_faithfulness = total_faithfulness / num_questions
    average_relevancy = total_relevancy / num_questions

    return average_response_time, average_faithfulness, average_relevancy
import numpy as np
st_nodes = []
ori2index = {}

if args.testsize:
    df = pd.DataFrame(columns=["chunksize","Average_Response_time","Average_Faithfulness","Average_Relevancy"])
    for chunk_size in args.chunksize:
        avg_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(chunk_size,eval_questions)
        data=[chunk_size,avg_time,avg_faithfulness,avg_relevancy]
        data = np.array(data).reshape(1,4)
        df1 = pd.DataFrame(columns=["chunksize","Average_Response_time","Average_Faithfulness","Average_Relevancy"],data=data)
        df = pd.concat([df, df1], ignore_index=True) 
        df.to_csv(args.test_chunk_size_path)
else:
    df = pd.DataFrame(columns=["chunkmanner","Average_Response_time","Average_Faithfulness","Average_Relevancy"])
    for man in args.chunkmanners:
        print(f"正在处理{man}")
        splitter = None
        query_engine = None
        nodes = None
        if man == "tok":
            splitter = TokenTextSplitter(
                chunk_size=args.defaultchunksize,
                chunk_overlap=args.defaultoverlap,
                separator=" ",
            )
            nodes = splitter(documents)
        elif man == "sen":
            splitter = SentenceSplitter(
                chunk_size=1024,
                chunk_overlap=20,
            )
            nodes = splitter(documents)
        elif man == "sem":
            embed_model = OpenAIEmbedding()
            splitter = SemanticSplitterNodeParser(
                buffer_size=1, 
                breakpoint_percentile_threshold=95,
                embed_model=embed_model
            )
            nodes = splitter(documents)
        elif man == "small2big":

            splitter = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=20,
            )
            total_nodes = splitter(documents)

            for node in total_nodes:
                #delete metadata
                node.metadata = {}
                effient_len = 180
                small_splitter = SentenceSplitter(
                                chunk_size=effient_len,
                                chunk_overlap=20
                )
                sub_nodes = small_splitter.get_nodes_from_documents([node])
                st_nodes.extend([IndexNode.from_text_node(sn,node.node_id) for sn in sub_nodes])
                st_nodes.append(IndexNode.from_text_node(node,node.node_id))
            for node in total_nodes:
                ori2index[node.node_id] = node

        elif man == "hir":
            splitter = HierarchicalNodeParser.from_defaults(
                chunk_sizes=[512, 256, 128]
            )
            nodes = splitter(documents)

        elif  man == "siw":
            splitter = SentenceWindowNodeParser.from_defaults(
                sentence_splitter=sws,
                # how many sentences on either side to capture
                window_size=1,
                # the metadata key that holds the window of surrounding sentences
                window_metadata_key="window",
                # the metadata key that holds the original sentence
                original_text_metadata_key="original_sentence",
            )
            nodes = splitter(documents)
        elif man == "que":
            pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(chunk_size=512, chunk_overlap=0),
                    QuestionsAnsweredExtractor(
                        questions=2, llm=gpt3, metadata_mode=MetadataMode.EMBED
                    )
                ]
            )
            nodes = pipeline.run(documents=documents)    
        elif man == "sum":
            pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(chunk_size=512, chunk_overlap=32),
                    SummaryExtractor(summaries=["prev", "self", "next"], llm=gpt3,metadata_mode=MetadataMode.EMBED)
                ]
            )
            nodes = pipeline.run(documents=documents)   
        elif man =="tit":
            pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(chunk_size=512, chunk_overlap=32),
                    TitleExtractor(nodes=5, llm=gpt3,metadata_mode=MetadataMode.EMBED),
                ]
            )
            nodes = pipeline.run(documents=documents)   
        elif man =="ent":
            pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(chunk_size=512, chunk_overlap=32),
                    KeywordExtractor(
                        keywords=10,
                        metadata_mode=MetadataMode.EMBED
                    )
                ]
            )
            nodes = pipeline.run(documents=documents)   
        else:
            print("输出参数存在错误")
            break


        if  man == "siw":
            index = VectorStoreIndex(
                nodes=nodes
            )
            query_engine = index.as_query_engine(
                similarity_top_k=args.simsize,
                node_postprocessors=[
                    MetadataReplacementPostProcessor(target_metadata_key="window")
                ]
            )
        elif man == "hir":
            leaf_nodes = get_leaf_nodes(nodes)
            from llama_index.core.storage.docstore import SimpleDocumentStore
            docstore = SimpleDocumentStore()
            docstore.add_documents(nodes)
            storage_context = StorageContext.from_defaults(docstore=docstore)
            base_index = VectorStoreIndex(
                nodes = leaf_nodes,
                storage_context=storage_context
            )
            base_retriever = base_index.as_retriever(similarity_top_k=args.simsize)
            retriever = AutoMergingRetriever(base_retriever, verbose=True,storage_context=storage_context)
            query_engine = RetrieverQueryEngine.from_args(retriever)

        elif man == "small2big":
            st_vec = VectorStoreIndex(nodes = st_nodes)
            vector_st_retri = st_vec.as_retriever(similarity_top_k = args.simsize)
            vector_st_retri = RecursiveRetriever(
                "vector",
                retriever_dict={"vector":vector_st_retri},
                node_dict = ori2index,
                # verbose = True
            )
            query_engine = RetrieverQueryEngine.from_args(retriever=vector_st_retri)
        else:
            index = VectorStoreIndex(nodes=nodes)
            query_engine = index.as_query_engine(
                similarity_top_k=2,
            )
        avg_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(222,eval_questions,query_engine)
        print(f"Chunk man {man} - Average Response time: {avg_time:.2f}s, Average Faithfulness: {avg_faithfulness:.2f}, Average Relevancy: {avg_relevancy:.2f}")
        data=[man,avg_time,avg_faithfulness,avg_relevancy]
        data = np.array(data).reshape(1,4)
        df1 = pd.DataFrame(columns=["chunksize","Average_Response time","Average_Faithfulness","Average_Relevancy"],data=data)
        # print(f"Chunk size {chunk_size} - Average Response time: {avg_time:.2f}s, Average Faithfulness: {avg_faithfulness:.2f}, Average Relevancy: {avg_relevancy:.2f}")
        df = pd.concat([df, df1], ignore_index=True) 
        df.to_csv("./test_chunk_man.csv")