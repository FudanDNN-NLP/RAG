import torch
import transformers
from .testPegasus import pegasus_summary
from .testPegasusChinese import pegasus_summary_zh
from .testRecomp import recomp_summary
from .testRecompe import recomp_extractive
from .testLLMlingua import lingua_compress

import langid

lan_type = ["en","zh"]
def compressor(query, docs, compression_ratio, model, tokenizer=None, ):
    
    
    # query_lan_type = check_lan(query)
    # docs_lan_type = check_lan(docs)
    
    # if(query_lan_type == docs_lan_type):
    #    return query_based_abstractive_summarization(model, query,docs,compression_ratio)
    # else:
    #    return abstractive_summarization(model, tokenizer, docs,compression_ratio)
    
    if tokenizer==None:
       return query_based_abstractive_summarization(model, query,docs,compression_ratio)
    else:
       return query_based_exstractive_summarization(model, tokenizer, query,docs,compression_ratio)
    
def check_lan(text):
    
    return langid.classify(text)[0]

def abstractive_summarization(docs,compression_ratio):
    
    if(lan_type=="en"):
        
        return pegasus_summary(docs) # {'summary':[''],'gold':''}
    
    else:
        
        return pegasus_summary_zh(docs) # ""
        
    
def query_based_exstractive_summarization(model, tokenizer, query,docs,compression_ratio):
    
    return recomp_extractive(model, tokenizer, query, docs, compression_ratio)

def query_based_abstractive_summarization(llm_lingua, query,docs,compression_ratio):
    
    return lingua_compress(llm_lingua, query, docs, compression_ratio)


def seperate_lan(text):
    
    return 

def seperate_lan_mixed(text):
    
    return

def main():
    print("compressor")

if __name__=='__main__':
    
    main()
    



