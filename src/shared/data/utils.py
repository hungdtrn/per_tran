import dgl
from dgl.udf import EdgeBatch

import torch
import numpy as np
import cv2
from scipy.spatial.transform.rotation import Rotation
from scipy import interpolate
import json

from library.geometry import skeleton_to_box_torch, get_box_center, apply_transformation, transpose_sequence_of_matrix
from library.utils import unsqueeze_and_repeat
from library.geometry import get_box_center

class CoordinateHandler:
    def distance_from_human_to_obj(self, human, obj):
        raise NotImplementedError
    
    def decompose_coordinate(self, coordinate, global_component):
        raise NotImplementedError
    
    def compose_coordinate(self, coordinate, global_component):
        raise NotImplementedError
    
    def skeleton_to_box_torch(self, coordinatee):
        raise NotImplementedError

def normalization_stats(completeData):
  """"
  Args
    completeData: Tensor of shape (n_instances, seq_len, dim)
  Returns
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
  """
  data_mean = torch.mean(completeData, dim=0).float()
  data_std  =  torch.std(completeData, dim=0).float()

  return data_mean, data_std

def normalize_data( data, data_mean, data_std, mask=None):
    """
    Normalize input data by removing unused dimensions, subtracting the mean and
    dividing by the standard deviation

    Args
        data: Tensor of shape (n_instances, seq_len, dim)
        data_mean: Tensor of shape (dim,)
        data_std: Tensor of shape (dim,)
        dim_to_use: vector with dimensions used by the model
        actions: list of strings with the encoded actions
    Returns
        data_out: Tensor of shape (n_instances, seq_len, dim)
    """
    batch, seq_len, dim = data.size()
    device = data.device
    
    if mask is not None:
        mask = mask.reshape(batch, 1, 1)

    
    n, seq_len, dim = data.size()
    flat_data = data.reshape(-1, dim)

    norm_data = (flat_data - data_mean) / data_std
  
    norm_data = norm_data.reshape(n, seq_len, dim).float()
    if mask is not None:
        norm_data = norm_data * mask
        
    return norm_data

def unNormalizeData(normalizedData, data_mean, data_std, mask=None):    
    """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
    Args
        normalizedData: Tensor of shape (n_instances, seq_len, dim)
        data_mean: Tensor of shape (dim,)
        data_std: Tensor of shape (dim,)
    Returns
        origData: data originally used to
    """
    device = normalizedData.device
    n, seq_len, dim = normalizedData.size()
  
    if mask is not None:
        mask = mask.reshape(n, 1, 1)  

    flat_data = normalizedData.reshape(-1, dim)
    original_data = flat_data * data_std + data_mean
    
    original_data = original_data.reshape(n, seq_len, dim)
    if mask is not None:
        original_data = original_data * mask
        
    return original_data

def to_one_hot(classes, current_class):
    one_hot = np.zeros((len(classes)))
    class_idx = classes.index(current_class)
    
    one_hot[class_idx] = 1
    return one_hot

