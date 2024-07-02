from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import webQDataset, webQEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess
from opencompass.datasets import omniDataset

webq_reader_cfg = dict(input_columns=["classification_query", "query"], output_column="answer")

webq_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role="HUMAN", prompt="{classification_query}")],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

webq_eval_cfg = dict(evaluator=dict(type=webQEvaluator), pred_role="BOT")

webq_datasets = [
    dict(
        abbr="WebQ",
        type=omniDataset,
        path="/data/zfr/finalTest/opencompass/generate_docs/true_hh_results/",
        name="webq",
        reader_cfg=webq_reader_cfg,
        infer_cfg=webq_infer_cfg,
        eval_cfg=webq_eval_cfg,
    )
]
