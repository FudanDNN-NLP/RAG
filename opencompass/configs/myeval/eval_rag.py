from mmengine.config import read_base

with read_base():
    from ..models.my_models.mymodel import models

    from ..datasets.my_datasets.rag import rag_datasets


datasets = [*rag_datasets]
