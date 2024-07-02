import csv
import json
import os.path as osp
from datasets import Dataset, DatasetDict
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils.text_postprocessors import general_postprocess

from opencompass.datasets.base import BaseDataset
from opencompass.utils.logging import get_logger


@LOAD_DATASET.register_module()
class omniDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        with open(f"{path}{name}/{name}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_data = []
            for item in data:
                classification_query = item["classification_query"]
                query = item["query"]
                doc = item["doc"]
                # print(doc)
                if name in ["arc", "obqa"]:
                    answer = item["answerKey"]
                elif name in ["fever", "pubhealth"]:
                    answer = item["label"]
                elif name in ["hotpotqa", "musique", "nq", "pubmedqa", "wiki"]:
                    answer = item["answer"]
                elif name == "mmlu":
                    answer = item["target"]
                elif name in ["webq", "tqa"]:
                    answer = item["answers"]
                raw_data.append({"classification_query": classification_query, "query": query, "doc": doc, "answer": answer})
        return Dataset.from_list(raw_data)






# datasets_list = ["mmlu", "obqa", "arc", "fever", "pubhealth", "nq", "webq", "tqa", "hotpotqa", "musique", "wiki", "pubmedqa"]
# for id in datasets_list:

#     a = nqDataset.load("/data/zfr/finalTest/opencompass/generate_docs/true_hh_mt_results/", id)
#     print(a)
