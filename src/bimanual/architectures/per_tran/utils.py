import torch
import torch.nn.functional as F

# def compute_box_center_wh_vel(box, center_vel, wh_vel):
#     current_w = box[:, 2] - box[:, 0]
#     current_h = box[:, 3] - box[:, 1]
    
#     w_vel, h_vel = wh_vel[:, 0], wh_vel[:, 1]
    
#     # zero out the veloicy that is is negative and greater than current value
#     # to ensure w and h always > 0
#     w_mask = ((w_vel + current_w) > 0).float()
#     h_mask = ((h_vel + current_h) > 0).float()
    
#     w_vel = w_mask * w_vel + (1 - w_mask) * torch.zeros_like(w_vel)
#     h_vel = h_mask * h_vel + (1 - h_mask) * torch.zeros_like(h_vel)
    
#     tmp = torch.stack([w_vel, h_vel], -1) / 2
#     return torch.cat([center_vel - tmp, center_vel + tmp], -1)

def compute_box_center_wh_vel(box, center_vel, wh_vel):
    current_w = box[:, 2] - box[:, 0]
    current_h = box[:, 3] - box[:, 1]
    current_wh = torch.stack([current_w, current_h], -1)
    
    # Clamp 
    # Prevent the boxes to have 0 width or 0 height
    wh_vel = torch.clamp(wh_vel, min=-0.99 * current_wh)
    
    return torch.cat([center_vel - wh_vel / 2, center_vel + wh_vel / 2], -1)


def compute_box_center_wh_loc(box, center, wh_vel):
    current_w = box[:, 2] - box[:, 0]
    current_h = box[:, 3] - box[:, 1]
    current_wh = torch.stack([current_w, current_h], -1)
    
    updated_wh = current_wh + wh_vel
    mask = (updated_wh > 0).float()
    
    updated_wh = mask * updated_wh + (1 - mask) * current_wh

    return torch.cat([center - updated_wh / 2, center + updated_wh / 2], -1)

def compute_obj_center_wh_vel(obj, obj_h ,center_embedding_fn, wh_embedding_fn):
    obj_center_vel = center_embedding_fn(obj_h)
    obj_wh_vel = wh_embedding_fn(obj_h)
    return compute_box_center_wh_vel(obj, obj_center_vel, obj_wh_vel)

def compute_human_center_wh_vel(human, human_h, arm_embedding_fn, center_embedding_fn, wh_embedding_fn):
    human_center_vel = center_embedding_fn(human_h)
    human_wh_vel = wh_embedding_fn(human_h)
    human_arm_vel = arm_embedding_fn(human_h)
    human_hand_vel = compute_box_center_wh_vel(human[:, -4:], human_center_vel, human_wh_vel)
    
    return torch.cat([human_arm_vel, human_hand_vel], -1)

def compute_box_vel(box, vel):    
    tmp = box + vel
    mask = ((tmp[:, 2] > tmp[:, 0]) & (tmp[:, 3] > tmp[:, 1]))
        
    mask = mask.unsqueeze(-1).float()
    vel = mask * vel + (1 - mask) * torch.zeros_like(vel)
    
    return vel
    
def compute_obj_vel(obj, obj_h, embedding_fn):
    vel = embedding_fn(obj_h)
    return compute_box_vel(obj, vel)

def compute_human_vel(human, human_h, arm_embedding_fn, box_embedding_fn):
    arm_vel = arm_embedding_fn(human_h)
    box_vel = box_embedding_fn(human_h)
    
    box_vel = compute_box_vel(human[:, -4:], box_vel)
    return torch.cat([arm_vel, box_vel], -1)


def compute_obj_center_wh_loc(obj, obj_h ,center_embedding_fn, wh_embedding_fn):
    obj_center = center_embedding_fn(obj_h)
    obj_wh = wh_embedding_fn(obj_h)
    return compute_box_center_wh_loc(obj, obj_center, obj_wh)

def compute_human_center_wh_loc(human, human_h, arm_embedding_fn, center_embedding_fn, wh_embedding_fn):
    human_center = center_embedding_fn(human_h)
    human_wh = wh_embedding_fn(human_h)
    human_arm = arm_embedding_fn(human_h)
    
    human_hand = compute_box_center_wh_loc(human[:, -4:], human_center, human_wh)
    return torch.cat([human_arm, human_hand], -1)