from torch import nn
from yacs.config import CfgNode
from src.shared.tools.registry import Registry

META_ARCH_WBHM_REGISTRY = Registry()



def build_model(model_cfg: CfgNode) -> nn.Module:
    """
    build model
    :param model_cfg: model config blob
    :return: model
    """
    name = model_cfg.name
    model = META_ARCH_WBHM_REGISTRY.get(name)(model_cfg)
    
    return model