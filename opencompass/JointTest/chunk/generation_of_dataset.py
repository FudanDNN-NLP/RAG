import argparse
import nest_asyncio
import openai
import os
from transformers import AutoTokenizer
from llama_index.core import Settings
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
nest_asyncio.apply()

"""

this is a simple demo to show how to generate the queries/questions from your documnets.

"""
llm = None
tokenizer = None

def get_llm(model_name_path):
    global llm
    if llm is None:
        llm = HuggingFaceLLM(
            model_name=model_name_path,
            tokenizer_name=model_name_path,
            max_new_tokens=1024,
            generate_kwargs={"do_sample": False,"pad_token_id":2},
            device_map="auto",
        )
    return llm

def get_tokenizer(tokenizer_name_path):
    global tokenizer
    if tokenizer is None:
        tokenizer =  AutoTokenizer.from_pretrained(
            tokenizer_name_path
        )
    return  tokenizer

def genqs(page_num, question_num, source_path, store_path):
    documents = SimpleDirectoryReader(source_path).load_data()[:page_num]
    data_generator = RagDatasetGenerator.from_documents(
        documents=documents,
        llm=Settings.llm,
        num_questions_per_chunk=question_num, 
        show_progress=True,
    )
    eval_questions = data_generator.generate_questions_from_nodes()
    eval_questions.to_pandas().to_csv(store_path)


if __name__ == '__main__':
    #set parameters
    paeser = argparse.ArgumentParser(description="produce qa using for test chunk")
    paeser.add_argument("--generation_model_name_path", type=str, default="HuggingFaceH4/zephyr-7b-alpha", help="the model of producing problems")
    paeser.add_argument("--page_num", type=int, default=10, help="the number of page")
    paeser.add_argument("--question_num_per_page", type=int, default=2, help="the number of questions per page")
    paeser.add_argument("--store_file_path", type=str, default="./data/qa_test.csv", help="the data to test")
    paeser.add_argument("--retrieval_source_path", type=str, default="./data/10k", help="the source to retrieval")
    args = paeser.parse_args()
    #set off-line model
    Settings.llm = get_llm(args.generation_model_name_path)
    Settings.tokenizer = get_tokenizer(args.generation_model_name_path)
    #extract questions from documents
    genqs(page_num=args.page_num, question_num=args.question_num_per_page, source_path=args.retrieval_source_path, store_path=args.store_file_path)


