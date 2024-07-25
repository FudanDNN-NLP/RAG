import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig


def get_model(peft_model_name, device):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=1)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.to(device).eval()
    return model


def rankllama_rerank(model, tokenizer, query, docs):
    # Load the tokenizer and model
    # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = get_model('castorini/rankllama-v1-7b-lora-passage', device)

    # Tokenize the query-passage pair
    scores = []
    for passage in docs:
        # inputs = tokenizer(f'query: {query}', f'document: {passage}', return_tensors='pt').to(model.device)
        inputs = tokenizer(f"query: {query}", f"document: {passage}", return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        # Run the model forward
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            score = logits[0][0]
            scores.append(score.item())

    doc_score_pairs = list(zip(docs, scores))
    sorted_doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    reranked_docs = [pair[0] for pair in sorted_doc_score_pairs]
    reranked_scores = [pair[1] for pair in sorted_doc_score_pairs]

    return reranked_docs, reranked_scores
