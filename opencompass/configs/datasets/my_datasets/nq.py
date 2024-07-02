from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import nqDataset,nqEvaluator

nq_reader_cfg = dict(
    input_columns=['question'], output_column='answer')

nq_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Question: {question}\nAnswer: '),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

nq_eval_cfg = dict(evaluator=dict(type=nqEvaluator), pred_role='BOT')

nq_datasets = [
    dict(
        type=nqDataset,
        abbr='nq',
        # path="/data/zfr/finalTest/opencompass/data/my_datasets/nq",
        reader_cfg=nq_reader_cfg,
        infer_cfg=nq_infer_cfg,
        eval_cfg=nq_eval_cfg)
]
