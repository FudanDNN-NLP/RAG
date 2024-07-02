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
class ragDataset(BaseDataset):

    @staticmethod
    def load():
        with open("/data/zfr/ragas/rag_test.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_data = []
            for item in data:
                question = item["question"]
                gold_context = item["gold_context"]
                if isinstance(gold_context, str):
                    gold_context = [gold_context]
                gold_answer = item["gold_answer"]
                if isinstance(gold_answer, str):
                    gold_answer = [gold_answer]
                raw_data.append({"question": question, "gold_answer": gold_answer, "gold_context": gold_context})

        return Dataset.from_list(raw_data)


@ICL_EVALUATORS.register_module()
class ragEvaluator(BaseEvaluator):

    def score(self, predictions, references):

        if len(predictions) != len(references):
            return {"error": "predictions and references have different " "length"}
        processed_predictions = []
        for prediction in predictions:
            prediction = prediction.strip().split("\n")[0].lower()
            if "answer is" in prediction:
                prediction = prediction.split("answer is")[-1]
            prediction = general_postprocess(prediction)
            processed_predictions.append(prediction)
        references = [list(item) if isinstance(item, str) else item for item in references]
        processed_answers = [[general_postprocess(j).lower() for j in i] for i in references]
        details = []
        cnt = 0
        F1_all = 0
        for pred, cand_ans in zip(processed_predictions, processed_answers):
            detail = {"pred": pred, "answer": cand_ans, "correct": False, "F1_score": 0}
            # EM
            # is_correct = any([cand == pred for cand in cand_ans])
            # EM in
            is_correct = any([cand in pred for cand in cand_ans])
            cnt += int(is_correct)
            detail["correct"] = is_correct

            if is_correct:
                F1_max = 1

            else:
                F1_max = 0
                for gold in cand_ans:
                    prediction_chars = pred.split()
                    reference_chars = gold.split()
                    common_tokens = set(prediction_chars) & set(reference_chars)
                    if not common_tokens:
                        F1_score = 0
                    else:
                        num_common_tokens = len(common_tokens)
                        precision = num_common_tokens / len(prediction_chars)
                        recall = num_common_tokens / len(reference_chars)
                        if precision + recall == 0:
                            F1_score = 0
                        else:
                            F1_score = 2 * (precision * recall) / (precision + recall)
                    if F1_score > F1_max:
                        F1_max = F1_score
            detail["F1_score"] = F1_max
            details.append(detail)
            F1_all += F1_max

        EM_score = cnt / len(predictions)
        F1_avg = F1_all / len(predictions)
        return {"EM score": EM_score, "F1 score": F1_avg, "details": details}
