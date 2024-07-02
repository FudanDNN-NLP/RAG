import json


with open(
    "test.jsonl",
    encoding="utf-8",
) as f:
    data = []
    raw_data = []
    for line in f:
        data.append(json.loads(line.strip()))
    for item in data:
        if "input" in item:
            item["question"]=item.pop('input')
            item["gold_context"]=item.pop('background')
            item["gold_answer"]=item.pop('output')
        else:
            item["gold_context"]=item.pop('gold documents')
            item["gold_answer"]=item.pop('gold answer')
with open("rag_test.json", 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)