import torch
import nest_asyncio
from transformers import BitsAndBytesConfig, AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# 请求openai存在延迟，使用协程
nest_asyncio.apply()


# 相关的模型
llm = None
tokenizer = None
embed_model = None


def get_llm():
    global llm
    device = torch.device("cuda:0")
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
            device_map="cuda:0",
        )
    return llm


def get_embedding_model():
    global embed_model
    if embed_model is None:
        embed_args = {
            "model_name": "BAAI/llm-embedder",
            "max_length": 512,
            "embed_batch_size": 8,
            "device": "cuda:0",
        }
        embed_model = HuggingFaceEmbedding(**embed_args)
    return embed_model


def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
    return tokenizer


def setingAll():
    Settings.llm = get_llm()
    Settings.embed_model = get_embedding_model()
    Settings.tokenizer = get_tokenizer()
    return Settings.llm, Settings.embed_model, Settings.tokenizer