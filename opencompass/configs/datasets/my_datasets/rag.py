from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import ragDataset,ragEvaluator

rag_reader_cfg = dict(
    input_columns=['question'], output_column='gold_answer')

rag_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Question: {question}\nAnswer: '),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

rag_eval_cfg = dict(evaluator=dict(type=ragEvaluator), pred_role='BOT')

rag_datasets = [
    dict(
        type=ragDataset,
        abbr='rag',
        # path="/data/zfr/finalTest/opencompass/data/my_datasets/nq",
        reader_cfg=rag_reader_cfg,
        infer_cfg=rag_infer_cfg,
        eval_cfg=rag_eval_cfg)
]
