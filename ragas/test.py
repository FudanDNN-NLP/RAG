
from datasets import Dataset,load_dataset
import os
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_relevancy,
    faithfulness,
    context_recall,
    context_precision,    
)
os.environ["OPENAI_API_KEY"] = "sk-wehmnSfGvOLVECJd328bBbA2E5284f168087F26d9784D110"
os.environ["OPENAI_API_BASE"] = "https://api.pumpkinaigc.online/v1"


# loading the V2 dataset
amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")
print(amnesty_qa)

score = evaluate(
    amnesty_qa['eval'],
    metrics=[
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        context_relevancy
    ],
)

print(score)
score.to_pandas()