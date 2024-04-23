import torch
import numpy as np

from ..build import META_ARCH_WBHM_REGISTRY
from src.shared.architectures.crnn.model import BaseModel
from src.shared.data.utils import normalize_data, unNormalizeData
from src.wbhm.data.utils import CoordinateHandler

class CRNN(BaseModel):
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg, **kwargs)
        self.coordinate_handler = CoordinateHandler()
        
    def rotate_and_translate(self, seq_data, augment_m=None, mask=None):
        """Augment the data
        
        Rotate along the Z-axis
        Translate over the XYZ axes

        Args:
            seq_data ([tensor]): (seq_len, dim)
        """    
        
        batch, seq_len, dim = seq_data.size()
        
        # Only rotate those 
        
        if mask is not None:
            mask = mask.reshape(batch, 1, 1)
        
        # homogeneous coordinate
        tmp = torch.ones((batch, seq_len, dim//3))
        tmp = tmp.reshape(-1, 1).to(seq_data.device)
        
        new_data = seq_data.reshape(-1, 3)
        new_data = torch.cat([new_data, tmp], dim=1)
        
        if augment_m is None:
            # get random angle and translation
            angle = np.random.randint(-180, 180) * np.pi / 180
            cos, sin = np.cos(angle), np.sin(angle)
            
            # Only rotate the X and Y, keeping Z intact
            rotation_m = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
            
            # Translation matrix
            # translate_m = np.random.randint(-1500, 1500, (3, 1))
            translate_m = np.random.randint(-1500, 1500, (2, 1))
            # translate_m = np.random.randint(-100, 100, (2, 1))
            translate_m = np.array([translate_m[0, 0], translate_m[1, 0], 0])[:, np.newaxis]
            
            augment_m = np.concatenate([rotation_m, translate_m], 1)
            
            augment_m = torch.Tensor(augment_m).to(seq_data.device)

        # zero out the fake data
        new_data = torch.matmul(new_data, augment_m.T).reshape(batch, seq_len, dim)
        
        if mask is not None:
            new_data = new_data * mask
        
        return new_data, augment_m
    
    def scale(self, data, additional_data):
        scale = 1e3
        return data / scale
        
    def unscale(self, data, additional_data):
        scale = 1e3
        return data * scale

    
    def normalize_data(self, data, mean, std, mask=None):
        return normalize_data(data, mean, std, mask)
    
    def unNormalizeData(self, data, mean, std, mask=None):
        return unNormalizeData(data, mean, std, mask)
    
@META_ARCH_WBHM_REGISTRY.register("crnn")
def build_crnn(model_cfg, **kwargs):
    return CRNN(model_cfg, **kwargs)