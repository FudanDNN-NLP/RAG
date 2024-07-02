from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class SafetyDataset(BaseDataset):

    @staticmethod
    def load(path):
        dataset = DatasetDict()

        data_list = list()
        idx = 0
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data_list.append({'idx': idx, 'prompt': line.strip()})
                    idx += 1

        dataset['test'] = Dataset.from_list(data_list)
