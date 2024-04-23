from .build import TRANSIENT_REGISTRY
from src.shared.architectures.per_tran import BaseTransient

@TRANSIENT_REGISTRY.register("transient")
def build_model(model_cfg, **kwargs):
    return BaseTransient(model_cfg)