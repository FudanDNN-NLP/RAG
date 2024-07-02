from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import nqDataset, nqEvaluator
from opencompass.datasets import omniDataset

nq_reader_cfg = dict(input_columns=["classification_query", "query"], output_column="answer")

nq_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role="HUMAN", prompt="{classification_query}")],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

nq_eval_cfg = dict(evaluator=dict(type=nqEvaluator), pred_role="BOT")

nq_datasets = [
    dict(
        abbr="nq",
        type=omniDataset,
        path="/data/zfr/finalTest/opencompass/generate_docs/true_hh_results/",
        name="nq",
        reader_cfg=nq_reader_cfg,
        infer_cfg=nq_infer_cfg,
        eval_cfg=nq_eval_cfg,
    )
]
