from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import pubhealthDataset, pubhealthEvaluator
from opencompass.datasets import omniDataset


pubhealth_reader_cfg = dict(input_columns=["classification_query", "query"], output_column="answer")

pubhealth_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role="HUMAN", prompt="{classification_query}")],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

pubhealth_eval_cfg = dict(evaluator=dict(type=pubhealthEvaluator), pred_role="BOT")

pubhealth_datasets = [
    dict(
        abbr="PubHealth",
        type=omniDataset,
        path="/data/zfr/finalTest/opencompass/generate_docs/true_hh_results/",
        name="pubhealth",
        reader_cfg=pubhealth_reader_cfg,
        infer_cfg=pubhealth_infer_cfg,
        eval_cfg=pubhealth_eval_cfg,
    )
]
