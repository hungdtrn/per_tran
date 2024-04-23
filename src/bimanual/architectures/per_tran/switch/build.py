from src.shared.tools.registry import Registry
from torch import nn
from yacs.config import CfgNode

SWITCH_REGISTRY = Registry()

def build_switch(model_cfg: CfgNode, **kwargs):
    """
    build backbone
    :param backbone_cfg:
    :return:
    """

    return SWITCH_REGISTRY[model_cfg.name](model_cfg, **kwargs)