import dgl
from dgl.udf import EdgeBatch

import torch
import numpy as np
import cv2
from scipy.spatial.transform.rotation import Rotation
from scipy import interpolate
import json

from library.geometry import get_box_center, skeleton_to_box_torch
from library.utils import unsqueeze_and_repeat
from src.shared.data.utils import CoordinateHandler as BaseCoordinateHandler, normalization_stats


def get_skeleton_connection(dim=54):
    if dim == 18:
        return [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (3, 6), (4, 7), (6, 8), (7, 9), 
                (5, 10), (5, 11), (10, 12), (11, 13), (12, 14), (13, 15), (14, 16), (15, 17)]
    elif dim == 20:
        return [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (3, 6), (4, 7), (6, 8), (7, 9), (8, 10), (9, 11),
                    (5, 12), (5, 13), (12, 14), (13, 15), (14, 16), (15, 17), (16, 18), (17, 19)]


def keep_or_extrapolate(array, start, end):
    if end <= len(array):
        return array[start:end]
    else:
        f = interpolate.interp1d(np.arange(0, len(array) - start), array[start:], 
                                 fill_value="extrapolate", axis=0)
        
        return f(np.arange(0, end - start))


def sample(np_array, rate=10):
    """sample the raw data with sample rate
    The raw data was recorded with the rate of 100 FPS.
    In the paper, we sample every 100ms (0.1s) => sample 1 in every 10 frames.

    Args:
        np_array ([np.array]): Input of shape (num_frames, dim)
        rate (int)
        
    Returns:
        out ([np.array]): Output of shape (num_frames, dim+1)
    """    
    total_frames = len(np_array)
    sampled_array = []
    idx_array = []
   
    for i in range(total_frames // rate):
        idx = i * rate
        sampled_array.append(np_array[idx])
        idx_array.append(idx)
        
    sampled_array = np.stack(sampled_array)
        
    return sampled_array


def sliding_windows(vid_humans, vid_objs, obs_len=10, pred_len=20, skip=1):
    """[summary]

    Args:
        vid_humans ([list[np.array]]): Sequence of human skeletons in a video, (n_human, n_frame, dim)
        vid_objs ([list[np.array]]): Sequence of object bounding boxes in a video (n_object, n_frame, dim)
        obs_len (int, optional): [description]. Defaults to 10.
        pred_len (int, optional): [description]. Defaults to 20.
        skip (int, optional): [description]. Defaults to 1.

    Returns:
        seq_humans [list[np.array]]: List of human sequences (n_sequence, n_human, seq_len, dim)
        seq_objects [list[np.array]]: List of object sequences (n_sequence, n_object, seq_len, dim)
        frame_idx [list]: List of frame indices of each sequence in the two list (n_sequence, sequence_len)
    """    
    
    seq_len = obs_len + pred_len
    total_seq_len = len(vid_humans[0])
    
    num_seq = int((total_seq_len - seq_len) / skip + 1)
    # if num_seq <= 0:
    #     num_seq = 1
    
    seq_humans = []
    seq_objs = []
    seq_inv_m = []
    seq_m = [] 
    
    frame_idx = []
    
    for i in range(num_seq):
        start = i * skip
        end = i * skip + seq_len
                
        if end > total_seq_len:
            print(end, skip, i, num_seq, total_seq_len)
        
        frame_idx.append(vid_humans[0][start:end, 0].astype(np.int))
                
        # Get object sequence
        tmp = []
        for i, obj in enumerate(vid_objs):
            tmp.append(keep_or_extrapolate(obj, start, end)[:, 1:])

        seq_objs.append(tmp)
        
        # Get human sequence
        tmp = []
        for i, human in enumerate(vid_humans):
            tmp.append(keep_or_extrapolate(human, start, end)[:, 1:])

        seq_humans.append(tmp)
            
    return seq_humans, seq_objs, frame_idx


def get_hand_and_foot_idx(ske):
    assert (ske.shape[-1] == 54 or ske.shape[-1] == 60 or ske.shape[-2] == 18 or ske.shape[-2] == 20)
    if ske.shape[-1] == 54 or ske.shape[-2] == 18:
        hand_idx = [8, 9]
        foot_idx = [16, 17]
    else:
        hand_idx = [10, 11]
        foot_idx = [18, 19]

    return hand_idx, foot_idx


def get_hand_and_foot(ske):
    hand_idx, foot_idx = get_hand_and_foot_idx(ske)

    if len(ske.size()) == 2:
        flat_ske = ske.reshape(len(ske), -1, 3)
        return flat_ske[:, hand_idx + foot_idx]
    elif len(ske.size()) == 3:
        batch, seq_len, _ = ske.size()
        flat_ske = ske.reshape(batch, seq_len, -1, 3)
        return flat_ske[:, :, hand_idx + foot_idx]
    elif len(ske.size()) == 4:
        batch, seq_len, num_human, _ = ske.size()
        flat_ske = ske.reshape(batch, seq_len, num_human, -1, 3)
        return flat_ske[:, :, :, hand_idx + foot_idx]


class CoordinateHandler(BaseCoordinateHandler):
    def distance_from_human_to_obj(self, human, obj):
        # Prepare section.
        # source node is human (center). Although the leaf could be another human. we only consider the "true obj" leaves.        
        # (batch, 4, 3)
        original_shape = human.shape
        human = human.reshape(-1, human.size(-1))
        
        obj = obj.reshape(-1, obj.size(-1))
        
        hand_foot = get_hand_and_foot(human)
        
        # (batch, 3)
        box_center = get_box_center(obj)
                
        # (batch, 4, 3)
        box_center = box_center.unsqueeze(1).repeat(1, hand_foot.size(1), 1)
        
        distance = torch.sqrt(torch.sum((box_center - hand_foot)**2, -1) + 1e-5)
        distance = torch.min(distance, 1)[0]

        distance = distance.reshape(tuple(list(original_shape)[:-1]))
        return distance
    
    def decompose_coordinate(self, coordinate, global_component=None):
        assert coordinate.shape[-1] % 3 == 0
        origin_shape = coordinate.shape
        num_points = origin_shape[-1] // 3
    
        # (batch, *, dim) -> (*, dim)
        coordinate = coordinate.reshape(-1, origin_shape[-1])
        center = None

        if global_component is None:
            box = self.skeleton_to_box_torch(coordinate)
        else:
            assert global_component.shape[-1] % 3 == 0

            # (batch, *, dim) -> (*, dim)
            global_component = global_component.reshape(-1, global_component.shape[-1])
            assert len(coordinate) == len(global_component)
            box = global_component

        global_component_shape = list(origin_shape)
        global_component_shape[-1] = global_component.shape[-1]
        
        center = get_box_center(box)
        center = unsqueeze_and_repeat(center, 1, num_points).view(-1, num_points * 3)
        local_component = (coordinate - center).float()
        
        return local_component.reshape(origin_shape), global_component.reshape(global_component_shape)    

    def compose_coordinate(self, coordinate, global_component):
        assert len(coordinate) == len(global_component)
        
        original_coordinate_shape = coordinate.shape
        original_global_shape = global_component.shape
        
        assert original_coordinate_shape[-1] % 3 == 0
        assert original_global_shape[-1] % 3 ==0
        
        coordinate = coordinate.reshape(-1, original_coordinate_shape[-1])
        num_coordinate_points = original_coordinate_shape[-1] // 3
        
        box = global_component.reshape(-1, original_global_shape[-1])
        center = get_box_center(box)
        center = unsqueeze_and_repeat(center, 1, num_coordinate_points).view(-1, num_coordinate_points * 3)
        coordinate = coordinate + center
        
        return coordinate.reshape(original_coordinate_shape)
        
    def skeleton_to_box_torch(self, coordinate):
        return skeleton_to_box_torch(coordinate)

def rotate_and_translate_wbhm(seq_data, augment_m=None, mask=None, translate_range=1500):
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
        translate_m = np.random.randint(-translate_range, translate_range, (2, 1))
        # translate_m = np.random.randint(-100, 100, (2, 1))
        translate_m = np.array([translate_m[0, 0], translate_m[1, 0], 0])[:, np.newaxis]
        
        augment_m = np.concatenate([rotation_m, translate_m], 1)
        
        augment_m = torch.Tensor(augment_m).to(seq_data.device)

    # zero out the fake data
    new_data = torch.matmul(new_data, augment_m.T).reshape(batch, seq_len, dim)
    
    if mask is not None:
        new_data = new_data * mask
    
    return new_data, augment_m
