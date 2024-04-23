from src.shared.tools.registry import Registry
from torch import nn
from yacs.config import CfgNode

PERSISTENT_REGISTRY = Registry()

def build_persistent(model_cfg: CfgNode, **kwargs):
    """
    build backbone
    :param backbone_cfg:
    :return:
    """

    return PERSISTENT_REGISTRY[model_cfg.name](model_cfg, **kwargs)