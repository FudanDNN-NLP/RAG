from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import pubmedQADataset, pubmedQAEvaluator
from opencompass.datasets import omniDataset

pubmedQA_reader_cfg = dict(input_columns=["classification_query", "query"], output_column="answer")

pubmedQA_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role="HUMAN", prompt="{classification_query}")],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

pubmedQA_eval_cfg = dict(evaluator=dict(type=pubmedQAEvaluator), pred_role="BOT")

pubmedQA_datasets = [
    dict(
        abbr="pubmedQA",
        type=omniDataset,
        path="/data/zfr/finalTest/opencompass/generate_docs/true_hh_results/",
        name="pubmedqa",
        reader_cfg=pubmedQA_reader_cfg,
        infer_cfg=pubmedQA_infer_cfg,
        eval_cfg=pubmedQA_eval_cfg,
    )
]
