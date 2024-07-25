from JointTest.compressor.compressor.compressorRaga import compressor
from llama_index.core.schema import MetadataMode
from typing import List, Tuple
from JointTest.retrieval.queries2retrievers import search
from JointTest.reranking.reranking.main import rerank
from JointTest.reranking.reranking.text_chunker import repack
from JointTest.reranking.reranking.dlm.pygaggle.pygaggle.rerank.transformer import MonoT5, MonoBERT
from FlagEmbedding import FlagReranker
import time
from tqdm import tqdm
from llmlingua import PromptCompressor
import torch.nn as nn
import torch
from peft import PeftModel, PeftConfig
from transformers.generation.utils import GenerationConfig
from transformers import (
    BertTokenizer,
    BertLMHeadModel,
    BertForSequenceClassification,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    MBartTokenizer,
    MBartTokenizerFast,
    AutoModel,
)
from JointTest.classification.task_cls.task_cls_bert import task_cls
from JointTest.reranking.reranking.tilde.TILDE.modelingv2 import TILDEv2


class JointRetrieval:
    def __init__(
        self,
        retrieval_classification_model_path="google-bert/bert-base-multilingual-cased",
        activity_model_path="meta-llama/Llama-2-7b-chat-hf",
        compressor_model_path="fangyuan/nq_extractive_compressor",
        retrieval_model_path="",
        reranking_model_path="castorini/monot5-base-msmarco-10k",
        device="auto",
        retrieval_config=None,
        milvus_id=1,
    ):
        super().__init__()
        self.retrieval_classification_model_path = retrieval_classification_model_path
        self.compressor_model_path = compressor_model_path
        self.retrieval_model_path = retrieval_model_path
        self.reranking_model_path = reranking_model_path
        self.activity_model_path = activity_model_path
        self.device = device
        self.retrieval_config = retrieval_config
        self.milvus_id = milvus_id

        (
            self.retrieval_classification_model,
            self.retrieval_classification_tokenizer,
            self.rerank_model,
            self.expansion_model,
            self.rerank_tokenizer,
            self.bert_tokenizer,
            self.compressor_model,
            self.compressor_tokenizer,
        ) = self.init_models()

    def init_models(
        self,
    ):

        print("*****************")
        print("init_models")
        print("*****************")
        print(self.retrieval_config)
        print(f"milvus:{self.milvus_id}")

        def init_Compressor_model():
            if self.retrieval_config["compression_method"] == "recomp":
                tokenizer = AutoTokenizer.from_pretrained("fangyuan/nq_extractive_compressor")
                model = AutoModel.from_pretrained("fangyuan/nq_extractive_compressor").to("cuda:0")
                return model, tokenizer

            else:
                print("init llm_lingua")
                llm_lingua = PromptCompressor("NousResearch/Llama-2-7b-hf", device_map="cuda:0")
                return llm_lingua, None

        def init_Classification_model():

            retrieval_classification_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
            retrieval_classification_model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-multilingual-cased", device_map=self.device)
            retrieval_classification_model.load_state_dict(torch.load("./opencompass/JointTest/classification/task_cls/bert_best_model.pth"))
            return retrieval_classification_model, retrieval_classification_tokenizer

        def init_rerank_model():
            def get_model(peft_model_name, device):
                config = PeftConfig.from_pretrained(peft_model_name)
                base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=1)
                model = PeftModel.from_pretrained(base_model, peft_model_name)
                model = model.merge_and_unload()
                model.to("cuda:0").eval()
                return model

            model_name = self.retrieval_config["rerank_model"]
            if model_name == "MonoT5" or model_name == "MonoBERT":
                if model_name == "MonoBERT":
                    rerank_model = MonoBERT()
                else:
                    rerank_model = MonoT5()
                    pass
                tokenizer = None
                bert_tokenizer = None
                expansion_model = None
            elif model_name == "RankLLaMA":
                tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
                rerank_model = get_model("castorini/rankllama-v1-7b-lora-passage", self.device)
                bert_tokenizer = None
                expansion_model = None
            elif model_name == "BGE":
                rerank_model = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
                tokenizer = None
                bert_tokenizer = None
                expansion_model = None
            else:
                tokenizer = AutoTokenizer.from_pretrained("ielab/TILDEv2-TILDE200-exp", use_fast=False, cache_dir="./cache")
                rerank_model = TILDEv2.from_pretrained("ielab/TILDEv2-TILDE200-exp", cache_dir="./cache").eval().to(self.device)
                bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                expansion_model = BertLMHeadModel.from_pretrained("ielab/TILDE", cache_dir="./cache").eval().to(self.device)
            return rerank_model, expansion_model, tokenizer, bert_tokenizer

        rerank_model, expansion_model, rerank_tokenizer, bert_tokenizer = init_rerank_model()
        retrieval_classification_model, retrieval_classification_tokenizer = init_Classification_model()
        compressor_model, compressor_tokenizer = init_Compressor_model()

        return (
            retrieval_classification_model,
            retrieval_classification_tokenizer,
            rerank_model,
            expansion_model,
            rerank_tokenizer,
            bert_tokenizer,
            compressor_model,
            compressor_tokenizer,
        )

    def should_retrieval(self, query, retrieval_classification_model, retrieval_tokenizer):
        s_search = time.time()
        if self.retrieval_config["with_retrieval_classification"]:
            should_retrieval = task_cls(retrieval_classification_model, retrieval_tokenizer, query)
        else:
            should_retrieval = True
        e_search = time.time()

        return should_retrieval

    def _search(self, query, **kwargs):
        print("*****************")
        print("searching")
        print("*****************")
        s_search = time.time()
        docs = search(query, top_k=kwargs["retrieval_config"]["search_k"], search_method=kwargs["retrieval_config"]["search_method"], milvus_id=self.milvus_id)
        e_search = time.time()
        result = []
        for i in docs:
            words = i["text"].split()
            if len(words) > 500:  # We truncate here
                truncated = " ".join(words[:500])
            else:
                truncated = i["text"]
            result.append(truncated)

        # docs = [i["text"] for i in docs]

        return result, e_search - s_search

    def _repack(self, query, docs, **kwargs):

        if self.retrieval_config["repack_method"] == "compact":
            docs = repack("compact", query, docs=docs)
        elif self.retrieval_config["repack_method"] == "compact_reverse":
            docs = repack("compact", query, docs=docs, ordering="reverse")
        elif self.retrieval_config["repack_method"] == "sides":
            docs = repack("compact", query, docs=docs, ordering="sides")

        return docs

    def _rerank(self, query, model, expansion_model, tokenizer, bert_tokenizer, docs, **kwargs):
        print("*****************")
        print("reranking")
        print("*****************")
        s_search = time.time()

        docs = rerank(
            model, expansion_model, tokenizer, bert_tokenizer, mode=kwargs["retrieval_config"]["rerank_model"], query=query, docs=docs, top_k=kwargs["retrieval_config"]["top_k"]
        )

        e_search = time.time()

        return docs, e_search - s_search

    def _compress(self, query, model, tokenizer, docs, **kwargs):
        print("*****************")
        print("compressing")
        print("*****************")
        s_search = time.time()

        doc = ""

        compressed_docs = compressor(
            query,
            docs,
            compression_ratio=kwargs["retrieval_config"]["compression_ratio"],
            model=model,
            tokenizer=tokenizer,
        )
        if self.retrieval_config["compression_method"] == "longllmlingua":
            compressed_docs = compressed_docs.strip(query)
        e_search = time.time()

        return compressed_docs, e_search - s_search

    def retrieval(self, classification_query: str, query: str):
        print("*****************")
        print("retrieving")
        print("*****************")
        print(f"cq:{classification_query}")
        print(f"q:{query}")
        if self.should_retrieval(classification_query, self.retrieval_classification_model, self.retrieval_classification_tokenizer):
            # if True:
            search_docs, _ = self._search(query=query, retrieval_config=self.retrieval_config)
            reranked_docs, _ = self._rerank(
                query=query,
                model=self.rerank_model,
                expansion_model=self.expansion_model,
                tokenizer=self.rerank_tokenizer,
                bert_tokenizer=self.bert_tokenizer,
                docs=search_docs,
                retrieval_config=self.retrieval_config,
            )
            # print(reranked_docs[0])
            docs = self._repack(query, reranked_docs[0])
            compressed_docs, _ = self._compress(query=query, model=self.compressor_model, tokenizer=self.compressor_tokenizer, docs=docs, retrieval_config=self.retrieval_config)

            return compressed_docs
        else:
            return ""

    def retrieval_only(self, classification_query: str, query: str):
        print("*****************")
        print("only retrieving")
        print("*****************")

        if self.should_retrieval(classification_query, self.retrieval_classification_model, self.retrieval_classification_tokenizer):
            # if True:
            search_docs, _ = self._search(query=query, retrieval_config=self.retrieval_config)
            return search_docs
        else:
            return ""

    def retrieval_wo_compression(self, classification_query: str, query: str):
        print("*****************")
        print("retrieving without compression")
        print("*****************")

        if self.should_retrieval(classification_query, self.retrieval_classification_model, self.retrieval_classification_tokenizer):
            # if True:
            search_docs, _ = self._search(query=query, retrieval_config=self.retrieval_config)
            reranked_docs, _ = self._rerank(
                query=query,
                model=self.rerank_model,
                expansion_model=self.expansion_model,
                tokenizer=self.rerank_tokenizer,
                bert_tokenizer=self.bert_tokenizer,
                docs=search_docs,
                retrieval_config=self.retrieval_config,
            )
            # print(reranked_docs[0])
            docs = self._repack(query, reranked_docs[0])
            return docs
        else:
            return ""
        