import torch
from transformers import AutoTokenizer, AutoModel

queries = ["Who is Joe Biden", "My name is WXH"]
keys = ["Joe Biden is the president of the USA", "My name is WXH"]

def get_similarity(query_list, key_list):
    # Define queries and keys
    # Load model
    tokenizer = AutoTokenizer.from_pretrained('BAAI/llm-embedder')
    model = AutoModel.from_pretrained('BAAI/llm-embedder')

    # Tokenize sentences
    query_inputs = tokenizer(query_list, padding=True, return_tensors='pt')
    key_inputs = tokenizer(key_list, padding=True, return_tensors='pt')

    # Encode
    with torch.no_grad():
        query_outputs = model(**query_inputs)
        key_outputs = model(**key_inputs)
        # CLS pooling
        query_embeddings = query_outputs.last_hidden_state[:, 0]
        key_embeddings = key_outputs.last_hidden_state[:, 0]
        # Normalize
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        key_embeddings = torch.nn.functional.normalize(key_embeddings, p=2, dim=1)

    similarity = query_embeddings @ key_embeddings.T
    #print(similarity)
    final_result = []
    for i in range(len(queries)):
        final_result.append(similarity[i][i])
    print(final_result)
    return final_result
    # [[0.8971, 0.8534]
    # [0.8462, 0.9091]]

get_similarity(queries, keys)
