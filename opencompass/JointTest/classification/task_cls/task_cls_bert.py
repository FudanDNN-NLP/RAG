import torch
from transformers import BertTokenizer, BertForSequenceClassification


def task_cls(model, tokenizer, input):

    inputs = tokenizer(input, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(predictions, dim=1).item()

    return predicted_class
