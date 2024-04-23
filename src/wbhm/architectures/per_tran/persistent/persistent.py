from .build import PERSISTENT_REGISTRY
from src.shared.architectures.per_tran import BasePersistent

@PERSISTENT_REGISTRY.register("persistent")
def build_model(model_cfg, **kwargs):
    return BasePersistent(model_cfg)

