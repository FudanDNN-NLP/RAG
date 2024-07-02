from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import pubmedQADataset, pubmedQAEvaluator

pubmedQA_reader_cfg = dict(input_columns="question", output_column="answer")

pubmedQA_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt="Based on the background information, determine if the answer to the question is 'YES', 'NO', or 'MAYBE'. Do not generate any other content.\nQuestion: {question}\nAnswer: ",
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

pubmedQA_eval_cfg = dict(evaluator=dict(type=pubmedQAEvaluator), pred_role="BOT")

pubmedQA_datasets = [
    dict(
        type=pubmedQADataset,
        abbr="pubmedQA",
        # path="/data/zfr/finalTest/opencompass/data/my_datasets/pubmedqa",
        reader_cfg=pubmedQA_reader_cfg,
        infer_cfg=pubmedQA_infer_cfg,
        eval_cfg=pubmedQA_eval_cfg,
    )
]
