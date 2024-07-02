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
class pubmedQADataset(BaseDataset):
    @staticmethod
    def load():
        with open(
            "/data/zfr/finalTest/opencompass/data/my_datasets/pubmedqa/pubmedQA.json",
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)
            raw_data = []
            index = 0
            for id in data:
                if index >= 500:
                    break
                index += 1
                question = data[id]["QUESTION"]
                context = data[id]["CONTEXTS"]
                long_answer = data[id]["LONG_ANSWER"]
                answer = data[id]["final_decision"]
                raw_data.append({"question": question, "answer": answer, "context": context, "long_answer": long_answer})
        return Dataset.from_list(raw_data)
    
@ICL_EVALUATORS.register_module()
class pubmedQAEvaluator(BaseEvaluator):

    def score(self, predictions, references):

        if len(predictions) != len(references):
            return {"error": "predictions and references have different " "length"}
        processed_predictions = []
        for prediction in predictions:
            prediction = prediction.strip().split("\n")[0].lower()
            if prediction.strip().split(" ")[0].lower() in ["yes","no","maybe"]:
                processed_predictions.append(prediction.strip().split(" ")[0].lower())
                continue
            
            prediction = general_postprocess(prediction)
            if "answer is" in prediction:
                prediction = prediction.split("answer is")[-1]
            processed_predictions.append(prediction)
        processed_answers = [[general_postprocess(i).lower()] for i in references]

        details = []
        cnt = 0
        for pred, cand_ans in zip(processed_predictions, processed_answers):
            detail = {"pred": pred, "answer": cand_ans, "correct": False}
            # EM
            # is_correct = any([cand == pred for cand in cand_ans])
            # EM in
            is_correct = any([cand in pred for cand in cand_ans])
            cnt += int(is_correct)
            detail["correct"] = is_correct

            details.append(detail)

        EM_score = cnt / len(predictions)

        return {"Acc": EM_score, "details": details}
