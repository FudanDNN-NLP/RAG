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
class mmluDataset(BaseDataset):

    @staticmethod
    def load(name: str):
        raw_data = []
        filename = osp.join("/data/zfr/finalTest/opencompass/data/my_datasets/mmlu/", f"{name}_test.csv")
        with open(filename, encoding="utf-8") as f:
            reader = csv.reader(f)
            data = list(reader)
            len1 = len(data)
            if len1 < 10:
                len1 = 10
            a = int(len1 / 10)
            if a > 20:
                a = 20
            for row in data[: a]:
            # for row in data[:1]:
                assert len(row) == 6
                raw_data.append(
                    {
                        "input": row[0],
                        "A": row[1],
                        "B": row[2],
                        "C": row[3],
                        "D": row[4],
                        "target": row[5],
                    }
                )
        dataset = Dataset.from_list(raw_data)
        return dataset
