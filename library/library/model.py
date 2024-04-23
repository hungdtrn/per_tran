import torch

def masked_select(feat: torch.Tensor, mask: torch.Tensor):
    assert len(feat.size()) == len(mask.size()) + 1
    d = feat.size(-1)
    dim_repeat = tuple([1 for i in range(len(mask.size()))] + [d])
        
    return torch.masked_select(feat, mask.unsqueeze(-1).repeat(*dim_repeat).bool()).reshape(-1, d)
        
def masked_set(mask: torch.Tensor, value: torch.Tensor, data: torch.Tensor):
    assert len(data.size()) == len(mask.size()) + 1
    d = data.size(-1)
    dim_repeat = tuple([1 for i in range(len(mask.size()))] + [d])
        
    mask = mask.unsqueeze(-1).repeat(*dim_repeat).bool()
        
    return torch.where(mask, value, data)
