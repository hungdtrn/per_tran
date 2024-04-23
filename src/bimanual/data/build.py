from torch import nn
from yacs.config import CfgNode
from src.shared.tools.registry import Registry

DATA_REGISTRY = Registry()
def build_dataset(data_cfg: CfgNode, dataset_type: str, training: bool):
    """
    build model
    :param model_cfg: model config blob
    :return: model
    """
    dataset_name = data_cfg.name
    
    return DATA_REGISTRY.get(dataset_name)(data_cfg, dataset_type, training)