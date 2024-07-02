from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import mmluDataset
from opencompass.datasets import omniDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

mmlu_reader_cfg = dict(input_columns=["classification_query", "query"], output_column="answer")


mmlu_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role="HUMAN", prompt="{classification_query}")],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

mmlu_eval_cfg = dict(evaluator=dict(type=AccEvaluator), pred_postprocessor=dict(type=first_option_postprocess, options="ABCD"))

mmlu_datasets = [
    dict(
        abbr=f"MMLU",
        type=omniDataset,
        path="/data/zfr/finalTest/opencompass/generate_docs/true_hh_results/",
        name="mmlu",
        reader_cfg=mmlu_reader_cfg,
        infer_cfg=mmlu_infer_cfg,
        eval_cfg=mmlu_eval_cfg,
    )
]
