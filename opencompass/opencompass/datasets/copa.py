import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class COPADataset_V2(BaseDataset):

    @staticmethod
    def load(path):
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                line['label'] = 'AB'[line['label']]
                dataset.append(line)
        return Dataset.from_list(dataset)
