
import numpy as np
import torch
from library.geometry import get_rotation_from_box

def transpose_sequence_of_matrix(M: torch.Tensor) -> torch.Tensor:
    """

    Args:
        M (torch.Tensor): sequence of matrix, shape (*, M, N)

    Returns:
        torch.Tensor: (*, N, M)
    """
    return torch.einsum('...ij->...ji', M)


def get_rotation_from_skeleton(ske: torch.tensor) -> torch.tensor:
    """ Get the rotation matrix from the skeleton.
    The rotation matrix transform the orginal coordinate to the local coordinate defined by the skeleton
    From [x' z', y'] -> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        Where y' = hip - neck, x' = a x y', where a = head - neck, and z = y' x x'
    Args:
        ske (torch.tensor): Tensor of shape (*, 54)

    Returns:
        torch.tensor: Tensor of shape (*, 3, 3)
    """
    
    assert ske.shape[-1] == 54
    origin_shape = ske.shape
    
    points = ske.reshape(-1, 18, 3)
    head = points[:, 1]
    neck = points[:, 2]
    hip = points[:, 5]
    
    a = head - neck
    y = hip - neck # Note that y go from top -> bottom
    y = y / torch.norm(y, dim=-1, ).unsqueeze(-1).repeat(1, 3)
    
    x = torch.cross(a, y)
    x = x / torch.norm(x, dim=-1, ).unsqueeze(-1).repeat(1, 3)
    
    z = torch.cross(y, x)
    z = z / torch.norm(x, dim=-1, ).unsqueeze(-1).repeat(1, 3)
    
    R = torch.stack([x, z, -y], -1)
    
    origin_shape = list(origin_shape)
    new_shape = origin_shape[:-1] + [3, 3]
    
    return R.reshape(new_shape)

    
def get_transformation_matrix_from_skeleton(ske: torch.Tensor) -> torch.Tensor:
    """[summary]

    Args:
        ske (torch.Tensor): [description]

    Returns:
        torch.Tensor: [description]
    """
    pass

