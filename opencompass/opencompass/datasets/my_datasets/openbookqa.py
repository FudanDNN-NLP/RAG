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
class openbookqaDataset(BaseDataset):

    @staticmethod
    def load():
        import os

        current_directory = os.path.dirname(__file__)
        parent_directory = os.path.dirname(current_directory)
        gp_directory = os.path.dirname(parent_directory)
        ggp_directory = os.path.dirname(gp_directory)
        file_path = os.path.join(ggp_directory, "data/my_datasets/openbookqa/test.jsonl")
        dataset_list = []
        with open(file_path, 'r') as f:
            for line in f:
                line = json.loads(line)
                item = {
                    'A': line['question']['choices'][0]['text'],
                    'B': line['question']['choices'][1]['text'],
                    'C': line['question']['choices'][2]['text'],
                    'D': line['question']['choices'][3]['text'],
                    'question_stem': line['question']['stem'],
                    'answerKey': line['answerKey'],
                }
                dataset_list.append(item)
        return Dataset.from_list(dataset_list[:500])