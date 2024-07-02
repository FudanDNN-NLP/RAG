from mmengine.config import read_base

with read_base():
    from ...models.my_models.mymodel import models

    from ...datasets.my_datasets.omni.mmlu import mmlu_datasets
    from ...datasets.my_datasets.omni.openbookqa import obqa_datasets
    from ...datasets.my_datasets.omni.arc_c import ARC_c_datasets

    from ...datasets.my_datasets.omni.fever import fever_datasets
    from ...datasets.my_datasets.omni.pubhealth import pubhealth_datasets

    from ...datasets.my_datasets.omni.nq import nq_datasets
    from ...datasets.my_datasets.omni.webquestions import webq_datasets
    from ...datasets.my_datasets.omni.tqa import triviaqa_datasets

    from ...datasets.my_datasets.omni.hotpotqa import hotpotqa_datasets
    from ...datasets.my_datasets.omni.MuSiQue import musique_datasets
    from ...datasets.my_datasets.omni.WikiMultihop import wikimultihop_datasets

    from ...datasets.my_datasets.omni.pubmedqa import pubmedQA_datasets


datasets = [
    # *mmlu_datasets,
    # *obqa_datasets,
    # *ARC_c_datasets,
    # *fever_datasets,
    # *pubhealth_datasets,
    # *nq_datasets,
    # *webq_datasets,
    # *triviaqa_datasets,
    # *hotpotqa_datasets,
    # *musique_datasets,
    # *wikimultihop_datasets,
    *pubmedQA_datasets,
]
