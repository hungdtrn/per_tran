from .build import SWITCH_REGISTRY
from src.shared.architectures.per_tran import BaseSwitch

@SWITCH_REGISTRY.register("switch")
def build_model(model_cfg, **kwargs):
    return BaseSwitch(model_cfg)