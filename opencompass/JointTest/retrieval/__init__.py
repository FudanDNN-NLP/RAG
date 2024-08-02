# package
# __init__.py
import argparse
import nest_asyncio
import torch
from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Settings
from llama_index.core.schema import MetadataMode
from tqdm import tqdm
# from progen import (getgenq , genqs)
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
    BaseExtractor,
)
from llama_index.llms.openai import OpenAI
import openai
import time
from llama_index.core import Settings
from transformers import AutoTokenizer

__all__ = ['os', 'sys', 're', 'urllib']
nest_asyncio.apply()
openai.api_key='aeEDQklGBuDJ8wYvB9E57d12D36b4c8995E7A8E30f2a5aDb' 
openai.base_url="https://api.pumpkinaigc.online/v1"