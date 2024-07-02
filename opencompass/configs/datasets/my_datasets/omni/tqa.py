from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TQADataset, TQAEvaluator
from opencompass.datasets import omniDataset

triviaqa_reader_cfg = dict(input_columns=["classification_query", "query"], output_column="answer")

triviaqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role="HUMAN", prompt="{classification_query}")],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    # inferencer=dict(type=GenInferencer, max_out_len=50),
    inferencer=dict(type=GenInferencer),
)

triviaqa_eval_cfg = dict(evaluator=dict(type=TQAEvaluator), pred_role="BOT")

triviaqa_datasets = [
    dict(
        abbr="TriviaQA",
        type=omniDataset,
        path="/data/zfr/finalTest/opencompass/generate_docs/true_hh_results/",
        name="tqa",
        reader_cfg=triviaqa_reader_cfg,
        infer_cfg=triviaqa_infer_cfg,
        eval_cfg=triviaqa_eval_cfg,
    )
]
