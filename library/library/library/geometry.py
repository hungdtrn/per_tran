import numpy as np
import torch

from .utils import torch_to_numpy

def transform_coordinate_torch(data, m):
    """Tranform the original coordinate to 
    a new coordinate using the homography matrix

    Args:
        data ([type]): [description]
        m ([type]): [description]
    """
    assert (len(data) == len(m))
    assert (len(data.size()) <= 3)
    
    if type(m) == list:
        m = torch.stack(m)
        
    assert (len(m.size()) == 3)
    
    out = []
    device = data.device
    
    original_shape = data.size()
    if len(data.size()) == 2:
        seq_len = 1
        batch, dim = data.size()
        
        data = data.unsqueeze(1)
    else:
        batch, seq_len, dim = data.size()
        
    # (n, seq, points, 4)
    data = data.reshape(batch, seq_len, -1, 3)
    data = torch.cat([data, torch.ones(batch, seq_len, data.size(2), 1).to(device)], -1)

    # (n, seq, points, 3, 4)
    data = data.unsqueeze(3).repeat(1, 1, 1, 3, 1)
    
    # (n, seq, 3, 4)
    m = m.unsqueeze(1).repeat(1, seq_len, 1, 1)
    
    # (n, seq, points, 3, 4)
    m = m.unsqueeze(2).repeat(1, 1, data.size(2), 1, 1)
    
    out = data * m
    
    return torch.sum(out, -1).reshape(original_shape)
    
def transform_coordinate_numpy(data, m):
    out = []
    for (rdata, rm) in zip(data, m):
        origin_shape = rdata.shape
        flat = rdata.reshape(-1, 3)
        flat = np.concatenate([flat, np.ones((len(flat), 1))], 1)
        flat = np.matmul(flat, np.transpose(rm))
        
        out.append(flat.reshape(origin_shape))
        
    return np.stack(out)

def distance_to_box_3d_relative(point, box):
    assert (len(point) == len(box))
    assert (point.shape[-1] == 3)

    if torch.is_tensor(point):
        point = torch_to_numpy(point)
    
    if torch.is_tensor(box):
        box = torch_to_numpy(box)
                
    box = box.reshape(len(box), -1, 3)

    xmin, xmax = np.min(box[:, :, 0], 1), np.max(box[:, :, 0], 1)
    ymin, ymax = np.min(box[:, :, 1], 1), np.max(box[:, :, 1], 1)
    zmin, zmax = np.min(box[:, :, 2], 1), np.max(box[:, :, 2], 1)
    
    centerx, centery, centerz = (xmin + xmax) / 2, (ymin  + ymax) / 2, (zmin + zmax) / 2
    center = np.stack([centerx, centery, centerz], 1)

    center_length = np.repeat(np.expand_dims(center, 1), box.shape[1], 1)
    center_length = center_length - box
    center_length = np.max(np.sqrt(np.sum(center_length**2, 2)), 1)
    
    distance = np.sqrt(np.sum((center - point)**2, 1))
    distance = distance - center_length

    return distance

def distance_to_box_3d(point, box):
    assert (len(point) == len(box))
    assert (point.shape[-1] == 3)
                
    box = box.view(len(box), -1, 3)

    # xmin, xmax = torch.min(box[:, :, 0], 1)[0], torch.max(box[:, :, 0], 1)[0]
    # ymin, ymax = torch.min(box[:, :, 1], 1)[0], torch.max(box[:, :, 1], 1)[0]
    # zmin, zmax = torch.min(box[:, :, 2], 1)[0], torch.max(box[:, :, 2], 1)[0]
    
    # centerx, centery, centerz = (xmin + xmax) / 2, (ymin  + ymax) / 2, (zmin + zmax) / 2
    # center = torch.stack([centerx, centery, centerz], 1)
    
    center = torch.mean(box, 1)
    distance = torch.sqrt(torch.sum((center - point)**2, 1))

    return distance


def distance_to_boxsurface_3d(point, box):
    """Compute the minimum distance of a point to the surface of a 3D box

    Args:
        point ([type]): [description]
        box ([type]): [description]
    """
    assert (len(point) == len(box))
    assert (point.shape[-1] == 3)


    if torch.is_tensor(point):
        point = torch_to_numpy(point)
    
    if torch.is_tensor(box):
        box = torch_to_numpy(box)
                
    box = box.reshape(len(box), -1, 3)

    # print(box.shape)
    xmin, xmax = np.min(box[:, :, 0], 1), np.max(box[:, :, 0], 1)
    ymin, ymax = np.min(box[:, :, 1], 1), np.max(box[:, :, 1], 1)
    zmin, zmax = np.min(box[:, :, 2], 1), np.max(box[:, :, 2], 1)
    
    centerx, centery, centerz = (xmin + xmax) / 2, (ymin  + ymax) / 2, (zmin + zmax) / 2
    
    # surface_center
    surface_center = [[centerx, centery, zmin],
                      [centerx, centery, zmax],
                      [centerx, ymin, centerz],
                      [centerx, ymax, centerz],
                      [xmin, centery, centerz],
                      [xmax, centery, centerz]]
    
    # edge centers
    edge_center = [[centerx, ymin, zmin],
                   [centerx, ymax, zmin],
                   [centerx, ymin, zmax],
                   [centerx, ymax, zmax],
                   [xmin, centery, zmin],
                   [xmin, centery, zmax],
                   [xmax, centery, zmin],
                   [xmax, centery, zmax],
                   [xmin, ymin, centerz],
                   [xmin, ymax, centerz],
                   [xmax, ymin, centerz],
                   [xmax, ymax, centerz]]
    
    # ppoints
    points = [[xmin, ymin, zmin], 
              [xmax, ymin, zmin],
              [xmax, ymax, zmin],
              [xmin, ymax, zmin],
              [xmin, ymin, zmax],
              [xmax, ymin, zmax],
              [xmax, ymax, zmax],
              [xmin, ymax, zmax]]    
    
    box = surface_center + edge_center + points
    box = [np.stack(x, -1) for x in box]
    box = np.stack(box, 1)
    
    dim_to_expand = len(point.shape) - 1
    point = np.repeat(np.expand_dims(point, dim_to_expand), box.shape[1], dim_to_expand)
    
    assert (point.shape == box.shape)
    
    distance = np.sqrt(np.sum((point - box)**2, 2))
    return np.min(distance, 1)

def get_iou_3d(bb1, bb2):
    """Compute the overlapping percentage between two 3D boxes

    Args:
        bb1: Box 1
        bb2: Box 2
    """
    
    if torch.is_tensor(bb1):
        bb1 = torch_to_numpy(bb1)
    
    if torch.is_tensor(bb2):
        bb2 = torch_to_numpy(bb2)
        
    if bb1.shape[-1] != 3:
        bb1 = bb1.reshape(len(bb1), -1, 3)
        
    if bb2.shape[-1] != 3:
        bb2 = bb2.reshape(len(bb2), -1,  3)
        
    x1_min, x1_max = np.min(bb1[:, :, 0], 1), np.max(bb1[:, :, 0], 1)
    x2_min, x2_max = np.min(bb2[:, :, 0], 1), np.max(bb1[:, :, 0], 1)

    y1_min, y1_max = np.min(bb1[:, :, 1], 1), np.max(bb1[:, :, 1], 1)
    y2_min, y2_max = np.min(bb2[:, :, 1], 1), np.max(bb2[:, :, 1], 1)
    
    z1_min, z1_max = np.min(bb1[:, :, 2], 1), np.max(bb1[:, :, 2], 1)
    z2_min, z2_max = np.min(bb2[:, :, 2], 1), np.max(bb2[:, :, 2], 1)
            
    xmin = np.maximum(x1_min, x2_min)
    xmax = np.minimum(x1_max, x2_max)

    ymin = np.maximum(y1_min, y2_min)
    ymax = np.minimum(y1_max, y2_max)
    
    zmin = np.maximum(z1_min, z2_min)
    zmax = np.minimum(z1_max, z2_max)
        
    overlap = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    bb1_volume = (x1_max - x1_min) * (y1_max - y1_min) * (z1_max - z1_min)
    bb2_volume = (x2_max - x2_min) * (y2_max - y2_min) * (z2_max - z2_min)
    
    iou = overlap / (bb1_volume + bb2_volume - overlap + 1e-4)

    # Not overlap in the x dimension
    iou[xmin > xmax] = -1
    
    # Not overlap in the y dimension
    iou[ymin > ymax] = -1
    
    # Not overlap in the z dimension
    iou[zmin > zmax] = -1


    return iou

def skeleton_to_box(ske_seq, num_box_vec=8):
    n_human, seq_len, dim = ske_seq.shape
    seq = ske_seq.reshape(n_human, seq_len, -1, 3)

    min_x = np.min(seq[:, :, :, 0], 2)
    max_x = np.max(seq[:, :, :, 0], 2)
    min_y = np.min(seq[:, :, :, 1], 2)
    max_y = np.max(seq[:, :, :, 1], 2)
    min_z = np.min(seq[:, :, :, 2], 2)
    max_z = np.max(seq[:, :, :, 2], 2)
                
    if num_box_vec == 8:
        return np.stack([min_x, min_y, min_z, 
                        max_x, min_y, min_z, 
                        max_x, max_y, min_z, 
                        min_x, max_y, min_z, 
                        min_x, min_y, max_z, 
                        max_x, min_y, max_z, 
                        max_x, max_y, max_z, 
                        min_x, max_y, max_z], -1)
    elif num_box_vec == 4:
        return np.stack([min_x, min_y, min_z, 
                         max_x, min_y, min_z, 
                         min_x, max_y, min_z, 
                         min_x, min_y, max_z], -1)

def skeleton_to_box_torch(ske_seq: torch.Tensor, 
                          num_box_vec: int=8) -> torch.Tensor:
    """ Get the bounding box that cover the whole skeleton

    Args:
        ske_seq (torch.Tensor): Tensor of shape (*, dim), where dim % 3 == 0
        num_box_vec (int, optional): Number of vectors used to represent the box. 
            This value is either 4 or 8. Defaults to 8.

    Returns:
        torch.Tensor: [description]. Tensor of shape (*, 12) or (*, 24)
    """
    original_shape = ske_seq.shape
    assert original_shape[-1] % 3 == 0
    assert num_box_vec == 8 or num_box_vec == 4

    num_points = original_shape[-1] // 3
    seq = ske_seq.reshape(-1, num_points, 3)

    min_x, _ = torch.min(seq[:, :, 0], 1)
    max_x, _ = torch.max(seq[:, :, 0], 1)
    min_y, _ = torch.min(seq[:, :, 1], 1)
    max_y, _ = torch.max(seq[:, :, 1], 1)
    min_z, _ = torch.min(seq[:, :, 2], 1)
    max_z, _ = torch.max(seq[:, :, 2], 1)
           
    if num_box_vec == 8:     
        box = torch.stack([min_x, min_y, min_z, 
                            max_x, min_y, min_z, 
                            max_x, max_y, min_z, 
                            min_x, max_y, min_z, 
                            min_x, min_y, max_z, 
                            max_x, min_y, max_z, 
                            max_x, max_y, max_z, 
                            min_x, max_y, max_z], -1).to(ske_seq.device)
    elif num_box_vec == 4:
        box = torch.stack([min_x, min_y, min_z,
                            max_x, min_y, min_z,
                            min_x, max_y, min_z,
                            min_x, min_y, max_z], -1).to(ske_seq.device)
    
    out_shape = list(original_shape)
    out_shape[-1] = num_box_vec * 3
    
    return box.reshape(out_shape)
    
def apply_transformation(M: torch.tensor, vec: torch.tensor) -> torch.Tensor:
    """ Apply a sequence of transformation matrix to a sequence of vectors

    Args:
        M (torch.tensor): (*, 3, 3) or (*, 3, 4)
        vec (torch.tensor): (*, dim), where dim % 3 == 0
    Returns:
        out_vec (torch..Tensor): Torch tensor of shape (*, dim)
    """
    assert M.shape[-2] == 3
    assert M.shape[-1] == 4 or M.shape[-1] == 3
    assert vec.shape[-1] % 3 == 0
    
    original_vec_shape = vec.shape
    original_M_shape = M.shape
    
    num_points = original_vec_shape[-1] // 3
    
    # (*, num_points, 3)
    vec = vec.reshape(-1,num_points, 3)
    M = M.reshape(-1, original_M_shape[-2], original_M_shape[-1])
    
    # (*, num_points, 3, 4) or (*, num_points, 3, 3)
    M = M.unsqueeze(1).repeat(1, num_points, 1, 1)
        
    # If M has the shape of (3, 4) -> create homogeneous coordinate representation (*, num_points, 4)
    if original_M_shape[-1] == 4:
        vec = torch.cat([vec, torch.ones(vec.shape[0], vec.shape[1], 1).to(vec.device)], -1)
    
    # (*, num_points, 3)
    out_vec = torch.einsum("hkij,hkj->hki", M, vec)
    
    return out_vec.reshape(original_vec_shape)
    
def transpose_sequence_of_matrix(M: torch.Tensor) -> torch.Tensor:
    """

    Args:
        M (torch.Tensor): sequence of matrix, shape (*, M, N)

    Returns:
        torch.Tensor: (*, N, M)
    """
    return torch.einsum('...ij->...ji', M)

def transpose_sequence_of_matrix_np(M: np.ndarray) -> np.ndarray:
    """

    Args:
        M (np.ndarray): sequence of matrix, shape (*, M, N)

    Returns:
        np.ndarray: (*, N, M)
    """
    return np.einsum('...ij->...ji', M)

        
def get_box_center(box: torch.Tensor) -> torch.Tensor:
    """ Get the center of the box.
    There are two types of box representation.
    1. 8 3-D vectors => (*, 24)
    2. 4 3-D vectors => (*, 12)

    If the box is represented by 4 vectors. The box representation will have this format (p00, p01, p02, p10), where
                    p01
        p00
                        p03     
            p02 

                p11
    p10           

                    p13
        p12            

    Args:
        box (torch.Tensor): [description]

    Returns:
        torch.Tensor: (*, 3)
    """
    assert box.shape[-1] % 3 == 0
    
    box_shape = box.shape    
    num_points = box.shape[-1] // 3
    assert num_points == 4 or num_points == 8

    new_box_shape =  [-1, num_points, 3]
        
    box = box.reshape(new_box_shape)
    
    if num_points == 8:
        center = torch.mean(box, 1)
        out_shape = list(box_shape)
        out_shape[-1] = 3
        
        return center.reshape(out_shape)
    else:
        point11 = box[:, 3] - box[:, 0] + box[:, 1]
        point12 = box[:, 3] - box[:, 0] + box[:, 2]
        
        center0 = (box[:, 1] + box[:, 2]) / 2
        center1 = (point11 + point12) / 2
        
        center = (center0 + center1) / 2
        out_shape = list(box_shape)
        out_shape[-1] = 3
        
        return center.reshape(out_shape)
    
def get_box_xyz(box: torch.Tensor) -> torch.Tensor:
    origin_shape = box.shape
    assert origin_shape[-1] % 3 == 0
    num_points = origin_shape[-1] // 3
    box = box.reshape(-1, num_points, 3)

    if num_points == 4:
        y = box[:, 3] - box[:, 0]
        x = box[:, 1] - box[:, 0]
        z = box[:, 2] - box[:, 0]
    else:
        y = box[:, 4] - box[:, 0]
        x = box[:, 1] - box[:, 0]
        z = box[:, 3] - box[:, 0]
    
    out_shape = list(origin_shape)
    out_shape[-1] = 3
    
    return x.reshape(out_shape), y.reshape(out_shape), z.reshape(out_shape)
    
def get_rotation_from_box(box: torch.tensor) -> torch.tensor:
    """[summary]

    Args:
        box (torch.tensor): Tensor of shape (*, dim)

    Returns:
        torch.tensor: [description]
    """
    origin_shape = box.shape
    assert origin_shape[-1] % 3 == 0
    num_points = origin_shape[-1] // 3
    box = box.reshape(-1, num_points, 3)
    
    assert num_points == 4 or num_points == 8
    
    if num_points == 4:
        y = box[:, 3] - box[:, 0]
        x = box[:, 1] - box[:, 0]
        z = box[:, 2] - box[:, 0]
    else:
        y = box[:, 4] - box[:, 0]
        x = box[:, 1] - box[:, 0]
        z = box[:, 3] - box[:, 0]
        
    y = y / torch.norm(y, dim=-1, ).unsqueeze(-1).repeat(1, 3)
    x = x / torch.norm(x, dim=-1, ).unsqueeze(-1).repeat(1, 3)
    z = z / torch.norm(z, dim=-1, ).unsqueeze(-1).repeat(1, 3)

    R = torch.stack([x, z, y], -1)
    origin_shape = list(origin_shape)
    new_shape = origin_shape[:-1] + [3, 3]
    
    return R.reshape(new_shape)

