from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import WikiMultihopQADataset, WikiMultihopQAEvaluator
from opencompass.datasets import omniDataset

wikimultihop_reader_cfg = dict(input_columns=["classification_query", "query"], output_column="answer")

wikimultihop_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role="HUMAN", prompt="{classification_query}")],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

wikimultihop_eval_cfg = dict(evaluator=dict(type=WikiMultihopQAEvaluator), pred_role="BOT")

wikimultihop_datasets = [
    dict(
        abbr="WikiMultihopQA",
        type=omniDataset,
        path="/data/zfr/finalTest/opencompass/generate_docs/true_hh_results/",
        name="wiki",
        reader_cfg=wikimultihop_reader_cfg,
        infer_cfg=wikimultihop_infer_cfg,
        eval_cfg=wikimultihop_eval_cfg,
    )
]
