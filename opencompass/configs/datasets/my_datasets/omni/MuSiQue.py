from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import musiqueDataset, musiqueEvaluator
from opencompass.datasets import omniDataset

musique_reader_cfg = dict(input_columns=["classification_query", "query"], output_column="answer")

musique_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role="HUMAN", prompt="{classification_query}")],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

musique_eval_cfg = dict(evaluator=dict(type=musiqueEvaluator), pred_role="BOT")

musique_datasets = [
    dict(
        abbr="MuSiQue",
        type=omniDataset,
        path="/data/zfr/finalTest/opencompass/generate_docs/true_hh_results/",
        name="musique",
        reader_cfg=musique_reader_cfg,
        infer_cfg=musique_infer_cfg,
        eval_cfg=musique_eval_cfg,
    )
]
