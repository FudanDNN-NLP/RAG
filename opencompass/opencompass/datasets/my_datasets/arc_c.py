import json
import os.path as osp

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from opencompass.datasets.base import BaseDataset
from opencompass.utils.logging import get_logger


@LOAD_DATASET.register_module()
class arccDataset(BaseDataset):

    @staticmethod
    def load():
        with open("/data/zfr/finalTest/opencompass/data/my_datasets/arc_c/ARC-Challenge-Test.jsonl", "r", errors="ignore") as in_f:
            rows = []
            index = 0
            for line in in_f:
                if index >= 500:
                    break
                index += 1
                item = json.loads(line.strip())
                question = item["question"]
                if len(question["choices"]) != 4:
                    continue
                labels = [c["label"] for c in question["choices"]]
                answerKey = "ABCD"[labels.index(item["answerKey"])]
                rows.append(
                    {
                        "question": question["stem"],
                        "answerKey": answerKey,
                        "A": question["choices"][0]["text"],
                        "B": question["choices"][1]["text"],
                        "C": question["choices"][2]["text"],
                        "D": question["choices"][3]["text"],
                    }
                )
            return Dataset.from_list(rows)
        
