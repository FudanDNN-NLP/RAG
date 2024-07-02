from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import pubhealthDataset, pubhealthEvaluator


pubhealth_reader_cfg = dict(input_columns=["claim"], output_column="label")

pubhealth_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt="Determine whether the given background supports or refutes the claim, and respond with 'SUPPORT' or 'REFUTE'. Do not generate any other content.\nClaim: {claim}\nAnswer: ",
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

pubhealth_eval_cfg = dict(evaluator=dict(type=pubhealthEvaluator), pred_role="BOT")

pubhealth_datasets = [
    dict(
        type=pubhealthDataset,
        abbr="PubHealth",
        # path="/data/zfr/finalTest/opencompass/data/my_datasets/pubhealth",
        reader_cfg=pubhealth_reader_cfg,
        infer_cfg=pubhealth_infer_cfg,
        eval_cfg=pubhealth_eval_cfg,
    )
]
