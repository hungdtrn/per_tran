import torch
import numpy as np
import dgl
import dgl.function as fn
from src.shared.architectures.layers.layers import build_mlp

from ..build import META_ARCH_BIMANUAL_REGISTRY
from src.shared.architectures.crnn.model import BaseModel
from src.shared.data.utils import normalize_data, unNormalizeData
from src.bimanual.data.utils import CoordinateHandler

class CRNN(BaseModel):
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg, **kwargs)
        self.coordinate_handler = CoordinateHandler()
        
    def rotate_and_translate(self, seq_data, augment_m=None, mask=None):
        # return seq_data, None
        batch, seq_len, dim = seq_data.size()
        new_data = seq_data.reshape(-1, 2)
        tmp = torch.ones_like(new_data)[:, 0:1]

        new_data = torch.cat([new_data, tmp], -1)
        if mask is not None:
            mask = mask.reshape(batch, 1, 1)

        if augment_m is None:
            cos, sin = 1.0, 0.0
            translate = np.random.randint(-300, 300, (2, 1))
            augment_m = np.concatenate([np.array([[cos, -sin], [sin, cos]]), translate], -1)
            augment_m = torch.Tensor(augment_m).to(seq_data.device)

        new_data = torch.matmul(new_data, augment_m.T).reshape(batch, seq_len, dim)
        if mask is not None:
            new_data = new_data * mask

        return new_data, augment_m
    
    def scale(self, data, additional_data):
        shift, scale = additional_data["data_stats"]["scalar_shift"], additional_data["data_stats"]["scalar_scale"]
        return data * 1.0 / scale
    
    def unscale(self, data, additional_data):
        shift, scale = additional_data["data_stats"]["scalar_shift"], additional_data["data_stats"]["scalar_scale"]
        return data * scale
    
    def normalize_data(self, data, mean, std, mask=None):
        return normalize_data(data, mean, std, mask)
    
    def unNormalizeData(self, data, mean, std, mask=None):
        return unNormalizeData(data, mean, std, mask)
    
@META_ARCH_BIMANUAL_REGISTRY.register("crnn")
def build_crnn(model_cfg, **kwargs):
    return CRNN(model_cfg, **kwargs)