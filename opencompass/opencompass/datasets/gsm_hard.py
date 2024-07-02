import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class GSMHardDataset(BaseDataset):

    @staticmethod
    def load(path):
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                dataset.append({
                    'question': line['input'],
                    'answer': str(line['target'])
                })
        dataset = Dataset.from_list(dataset)
        return dataset
