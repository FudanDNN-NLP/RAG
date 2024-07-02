from llmlingua import PromptCompressor
import time
import json
import pandas as pd
from datasets import load_dataset
from llmlingua import PromptCompressor
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import math
import math
from nltk.tokenize import sent_tokenize
import nltk

def text_to_set(docs):
    
    sentences = sent_tokenize(docs)
    
    return sentences



def lingua_compress(llm_lingua,query,docs,compression_ratio = 0.6):
    
    
    contexts = docs
    question = query
    

    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    contexts_tokens = tokenizer(contexts, return_tensors="pt")
    num_contexts_tokens = len(contexts_tokens["input_ids"][0])
    # print(num_contexts_tokens)
    
    sentences = text_to_set(contexts)
    
    compressed_prompt = llm_lingua.compress_prompt(
        sentences,
        instruction="",
        question=question,
        target_token=math.ceil(num_contexts_tokens*compression_ratio),
        condition_compare=True,
        condition_in_question="after",
        rank_method="longllmlingua",
        use_sentence_level_filter=False,
        context_budget="+100",
        dynamic_context_compression_ratio = compression_ratio,  # enable dynamic_context_compression_ratio
        reorder_context="sort",
    )
    
    return compressed_prompt["compressed_prompt"]



def main():
    dataset = load_dataset("lytang/MeetingBank-transcript")["train"]
    contexts = dataset[1]["source"]
    question = "Question: How much did the crime rate increase last year?\nAnswer:"
    lingua_compress(question,contexts,0.4)
    return

if __name__=="__main__":
    
    main()