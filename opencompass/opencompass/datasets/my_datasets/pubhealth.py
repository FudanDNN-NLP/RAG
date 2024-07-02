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
class pubhealthDataset(BaseDataset):
    @staticmethod
    def load():
        with open(
            "/data/zfr/finalTest/opencompass/data/my_datasets/pubhealth/health_claims.jsonl",
            "r",
            encoding="utf-8",
        ) as f:
            data = []
            raw_data = []
            for line in f:
                data.append(json.loads(line.strip()))
            for item in data[:500]:
                claim = item["claim"]
                label = item["label"]
                raw_data.append(
                    {
                        "claim": claim,
                        "label": label,
                    }
                )
        return Dataset.from_list(raw_data)


@ICL_EVALUATORS.register_module()
class pubhealthEvaluator(BaseEvaluator):

    def score(self, predictions, references):

        if len(predictions) != len(references):
            return {"error": "predictions and references have different " "length"}
        processed_predictions = []
        for prediction in predictions:
            prediction = prediction.strip().split("\n")[0].lower()
            if prediction.strip().split(" ")[0].lower() in ["support", "refute"]:
                processed_predictions.append(prediction.strip().split(" ")[0].lower() + "s")
                continue

            prediction = general_postprocess(prediction)
            if "answer is" in prediction:
                prediction = prediction.split("answer is")[-1]
            if prediction == "support" or prediction == "refute":
                prediction += "s"
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
