from itertools import groupby
from typing import Dict, Iterable, Optional
import torch
import json
import numpy as np

def run_length_encoding(iterable: Iterable):
    """Make an iterator that returns a label and the number of appearances in its run-length encoding."""
    for k, v in groupby(iterable):
        yield k, len(list(v))

def convert_id_to_one_hot(id, num_idx):
    out = np.zeros((len(id), num_idx), dtype=np.float)
    np.put_along_axis(out, id[:, np.newaxis], 1.0, 1)
    
    return out

def numpy_to_torch(*arrays, device='cpu'):
    """Convert any number of numpy arrays to PyTorch tensors."""
    return [torch.from_numpy(array).to(device) for array in arrays]

def read_dictionary(filepath: str) -> Dict[str, str]:
    """Read a dictionary from a file, where each file line is in the format 'key value'."""
    with open(filepath) as f:
        return json.load(f)
