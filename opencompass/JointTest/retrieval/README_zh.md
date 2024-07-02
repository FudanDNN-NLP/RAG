# Retrieval-Augmented Generation Project

## Introduction
### data2nodes.py
#### 两种chunk方法：
- 针对于数据集：[参考数据集](https://huggingface.co/datasets/FreedomIntelligence/ApolloCorpus) 主要针对于数据集的qa形式以及text形式。
- 针对于文本。
- [llamaindex](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)
### retrievers2nodes.py
#### 三种index：
- 文本index。
- 问题index。
- 关键字index。
- [llamaindex](https://docs.llamaindex.ai/en/stable/module_guides/indexing/)
### finetuneEmbedding.py
#### embedding微调：
- 一层adapter。
- 两层adapter。
- 全参微调。
- [llamaindex](https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding/?h=fin)
### queries2retrievers.py
#### query改写：
- 查询改写。
- 伪文档生成。

### todolist
- 生成速度慢。
  - 了解推理加速
- 关键字模糊匹配以及中英匹配的问题。
  - es搜索引擎。
- 中文英文元数据提取问题。
  - 中文



