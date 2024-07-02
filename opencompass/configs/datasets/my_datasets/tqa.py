from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TQADataset, TQAEvaluator

triviaqa_reader_cfg = dict(input_columns=["question"], output_column="answer")

triviaqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt="Question: {question}\nAnswer: "),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    # inferencer=dict(type=GenInferencer, max_out_len=50),
    inferencer=dict(type=GenInferencer),
)

triviaqa_eval_cfg = dict(evaluator=dict(type=TQAEvaluator), pred_role="BOT")

triviaqa_datasets = [
    dict(
        type=TQADataset,
        abbr="TriviaQA",
        # path="/data/zfr/finalTest/opencompass/data/my_datasets/tqa/",
        reader_cfg=triviaqa_reader_cfg,
        infer_cfg=triviaqa_infer_cfg,
        eval_cfg=triviaqa_eval_cfg,
    )
]
