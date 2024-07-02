from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HotpotQADataset, HotpotQAEvaluator
from opencompass.datasets import omniDataset

hotpotqa_reader_cfg = dict(input_columns=["classification_query", "query"], output_column="answer")

hotpotqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role="HUMAN", prompt="{classification_query}")],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

hotpotqa_eval_cfg = dict(evaluator=dict(type=HotpotQAEvaluator), pred_role="BOT")

hotpotqa_datasets = [
    dict(
        type=omniDataset,
        abbr="HotpotQA",
        path="/data/zfr/finalTest/opencompass/generate_docs/true_hh_results/",
        name="hotpotqa",
        reader_cfg=hotpotqa_reader_cfg,
        infer_cfg=hotpotqa_infer_cfg,
        eval_cfg=hotpotqa_eval_cfg,
    )
]
