from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import feverDataset, feverEvaluator


fever_reader_cfg = dict(input_columns=["claim"], output_column="label")

fever_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt="Determine whether the given background supports, refutes, or provides not enough information about the claim, and respond with 'SUPPORT', 'REFUTE', or 'NOT ENOUGH INFO'. Do not generate any other content.\nClaim: {claim}\nAnswer: ",
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

fever_eval_cfg = dict(evaluator=dict(type=feverEvaluator), pred_role="BOT")

fever_datasets = [
    dict(
        type=feverDataset,
        abbr="fever",
        # path="/data/zfr/finalTest/opencompass/data/my_datasets/fever",
        reader_cfg=fever_reader_cfg,
        infer_cfg=fever_infer_cfg,
        eval_cfg=fever_eval_cfg,
    )
]
