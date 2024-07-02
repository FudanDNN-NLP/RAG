from mmengine.config import read_base

with read_base():
    from ..models.my_models.mymodel import models

    from ..datasets.my_datasets.mmlu import mmlu_datasets
    from ..datasets.my_datasets.openbookqa import obqa_datasets
    from ..datasets.my_datasets.arc_c import ARC_c_datasets


    from ..datasets.my_datasets.fever import fever_datasets
    from ..datasets.my_datasets.pubhealth import pubhealth_datasets


    from ..datasets.my_datasets.nq import nq_datasets
    from ..datasets.my_datasets.webquestions import webq_datasets
    from ..datasets.my_datasets.tqa import triviaqa_datasets


    from ..datasets.my_datasets.hotpotqa import hotpotqa_datasets
    from ..datasets.my_datasets.MuSiQue import musique_datasets
    from ..datasets.my_datasets.WikiMultihop import wikimultihop_datasets


    from ..datasets.my_datasets.pubmedqa import pubmedQA_datasets


datasets = [
    *mmlu_datasets,
    *obqa_datasets,
    *ARC_c_datasets,
    *fever_datasets,
    *pubhealth_datasets,
    *nq_datasets,
    *webq_datasets,
    *triviaqa_datasets,
    *hotpotqa_datasets,
    *musique_datasets,
    *wikimultihop_datasets,
    *pubmedQA_datasets,
]
