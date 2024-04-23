import numpy as np
from numpy.lib.function_base import select
import torch

def merge_params(loaded_params, new_params):
    for key, value in new_params.items():
        if value is not None:
            if type(value) == tuple and len(value) == 0:
                continue
            
            if key in loaded_params:
                loaded_params[key] = value
                
    return loaded_params

def torch_to_numpy(torch_data):
    out = torch_data
    if out.is_cuda:
        out = out.cpu()
    
    if out.requires_grad:
        out = out.detach()
        
    return out.numpy()

def find_where_worse(dataset, metric_fn, our_fn, their_fn, better="lower"):
    """Compare the performance of the two model, find where our model is 
    worse than their model

    Args:
        dataset ([type]): [description]
        metric_fn ([type]): Function to compute the performance mettrics
        our ([type]): Function to compute the output of our model
        them ([type]): Function to compute the output of their model
        better (str, optional): [description]. Defaults to "lower".
    """
    
    our_rs, their_rs = [], []
    for data in dataset:
        our_rs.append(metric_fn(our_fn(data), data))
        their_rs.append(metric_fn(their_fn(data), data))
        
    our_rs, their_rs = np.array(our_rs), np.array(their_rs)
    distance = our_rs - their_rs

    indices = np.argsort(distance)
    
    if better == "lower":
        # sort where our rs is higher than their rs
        return indices[::-1]
    if better == "higher":
        # sort where our rs is lower than their rs
        return indices
    
def unsqueeze_and_repeat(x: torch.Tensor, dim: int, num_repeat: int) -> torch.Tensor:
    """ Add extra dimension at the speficied position and repeat at that dimension

    Args:
        x (torch.Tensor): Tensor of any dimension
        dim (int): Dimension to unsqueeze and repeat
        num_repeat (int): [description]

    Returns:
        torch.Tensor: [description]
    """
    
    x = x.unsqueeze(dim)
            
    repeats = [1 for i in range(len(x.shape))]
    repeats[dim] = num_repeat
    
    return x.repeat(repeats)

def straight_through_boolean(boolean: torch.tensor, grad_from: torch.tensor):
    return boolean.float() - grad_from.detach() + grad_from

def straight_through_selection(array: torch.tensor, prob: torch.tensor, dim=1):
    index = prob.max(dim, keepdims=True)[1]
    y_hard = torch.zeros_like(prob, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    y_hard = y_hard - prob.detach() + prob
    masked = array * y_hard.unsqueeze(-1)
    return torch.sum(masked, dim=dim)

def index_select_by_col(array: torch.tensor, index: torch.tensor):
    """For each column, select 1 element by its index.

    Args:
        array (torch.tensor): [description]
        index (torch.tensor): [description]

    Returns:
        [type]: [description]
    """
    assert len(array.size()) == 3
    assert len(index.size()) == 1
    assert len(array) == len(index)
    
    index = index.unsqueeze(-1) 
    col_index = torch.arange(array.size(1)).unsqueeze(0).repeat(array.size(0), 1)
    selection = (col_index == index).float() - index.detach() + index
    
    selected = array * selection.unsqueeze(-1)
    return torch.sum(selected, 1)


def straight_through_max(y_soft, dim=-1):
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        
        return ret

def masked_set(mask: torch.tensor, value: torch.tensor, 
               data: torch.tensor, grad_from: torch.tensor=None) -> torch.tensor:
        if grad_from is None:
            mask = mask.bool()
            return torch.where(mask.bool().unsqueeze(-1), value, data)
        else:
            # print(grad_from.requires_grad, value.requires_grad, data.requires_grad)
            mask = mask.float() - grad_from.detach() + grad_from
            mask = mask.unsqueeze(-1)
            return value * mask + data * (1 - mask)
