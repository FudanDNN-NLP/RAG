# Retrieval-Augmented Generation Project

## Introduction

enter from : comprssorRaga.py
```
def compressor(query, docs, compression_ratio):
    
    # 判断语言的类型
    
    query_lan_type = check_lan(query)
    docs_lan_type = check_lan(docs)
    
    if(query_lan_type == docs_lan_type):
       return query_based_abstrctive_summarization(query,docs,query_lan_type,compression_ratio)
    else:
       return abstrctive_summarization(docs,compression_ratio)

```
