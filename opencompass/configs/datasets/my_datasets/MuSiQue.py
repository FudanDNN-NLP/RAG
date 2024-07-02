from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import musiqueDataset, musiqueEvaluator

musique_reader_cfg = dict(input_columns="question", output_column="answer")

musique_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt="Question: {question}\nAnswer: "),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

musique_eval_cfg = dict(evaluator=dict(type=musiqueEvaluator), pred_role="BOT")

musique_datasets = [
    dict(
        type=musiqueDataset,
        abbr="MuSiQue",
        # path="/data/zfr/finalTest/opencompass/data/my_datasets/MuSiQue",
        reader_cfg=musique_reader_cfg,
        infer_cfg=musique_infer_cfg,
        eval_cfg=musique_eval_cfg,
    )
]
