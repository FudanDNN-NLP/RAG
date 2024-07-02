from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import omniDataset
from opencompass.utils.text_postprocessors import first_option_postprocess


ARC_c_reader_cfg = dict(input_columns=["classification_query", "query"], output_column="answer")

ARC_c_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role="HUMAN", prompt="{classification_query}")],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

ARC_c_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_option_postprocess, options="ABCD"),
)

ARC_c_datasets = [
    dict(
        abbr="ARC-c",
        type=omniDataset,
        path="/data/zfr/finalTest/opencompass/generate_docs/true_hh_results/",
        name="arc",
        reader_cfg=ARC_c_reader_cfg,
        infer_cfg=ARC_c_infer_cfg,
        eval_cfg=ARC_c_eval_cfg,
    )
]
