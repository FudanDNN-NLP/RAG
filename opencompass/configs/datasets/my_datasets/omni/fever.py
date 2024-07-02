from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import feverDataset, feverEvaluator
from opencompass.datasets import omniDataset


fever_reader_cfg = dict(input_columns=["classification_query", "query"], output_column="answer")

fever_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role="HUMAN", prompt="{classification_query}")],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

fever_eval_cfg = dict(evaluator=dict(type=feverEvaluator), pred_role="BOT")

fever_datasets = [
    dict(
        type=omniDataset,
        abbr="fever",
        path="/data/zfr/finalTest/opencompass/generate_docs/true_hh_results/",
        name="fever",
        reader_cfg=fever_reader_cfg,
        infer_cfg=fever_infer_cfg,
        eval_cfg=fever_eval_cfg,
    )
]
