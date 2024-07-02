from datasets import load_dataset

dataset = load_dataset("Stanford/web_questions")

for split, dataset in dataset.items():
    dataset.to_json(f"webquestions-{split}.jsonl")