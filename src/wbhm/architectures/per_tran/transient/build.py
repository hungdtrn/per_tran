from src.shared.tools.registry import Registry
from torch import nn
from yacs.config import CfgNode

TRANSIENT_REGISTRY = Registry()

def build_transient(model_cfg: CfgNode, **kwargs):
    """
    build backbone
    :param backbone_cfg:
    :return:
    """

    return TRANSIENT_REGISTRY[model_cfg.name](model_cfg, **kwargs)