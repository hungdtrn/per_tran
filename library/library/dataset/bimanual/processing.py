import os
import numpy as np
import json
import cv2
import torch

parts = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist']
connections = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 0]]
arm_connections = [[0, 1], [1, 2]]
arm_oks = [0.079, 0.072, 0.062]

def iou(boxA, boxB, eps=1e-7):
    """

    Args:
        boxA: (N, 4)
        boxB:(N, 4)

    Returns:
        _type_: _description_
    """
    x1, y1, x2, y2 = boxA.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxB.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"
    assert (x2g >= x1g).all(), "bad box: x1 larger than x2"
    assert (y2g >= y1g).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    # Interaction
    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    
    # Union
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)
    
    return iouk


def split_train_test(training_data: list, test_fraction: float = 0.2, seed: int = 42):
    np.random.seed(seed)
    np.random.shuffle(training_data)
    num_testing_videos = round(len(training_data) * test_fraction)
    testing_data = training_data[:num_testing_videos]
    training_data = training_data[num_testing_videos:]
    return training_data, testing_data


def get_box_center_torch(data):
    original_shape = data.shape
    assert data.shape[-1] == 4
    data = data.reshape(-1, 2, 2)
    data = data.mean(1)
    
    return data.reshape(list(original_shape)[:-1] + [2])

def skeleton_to_box_torch(data):
    original_shape = data.shape
    assert original_shape[-1] % 2 == 0
    num_points = original_shape[-1] // 2
    points = data.reshape(-1, num_points, 2)
    
    x1, x2 = torch.min(points[:, :, 0], dim=-1)[0], torch.max(points[:, :, 0], dim=-1)[0]
    y1, y2 = torch.min(points[:, :, 1], dim=-1)[0], torch.max(points[:, :, 1], dim=-1)[0]

    # (n, 4)
    box_data = torch.stack([x1, y1, x2, y2], -1)
    return box_data.reshape(list(original_shape)[:-1] + [4])

def get_box_center_numpy(data):
    original_shape = data.shape
    assert data.shape[-1] == 4
    data = np.reshape(data, (-1, 2, 2))
    data = np.mean(data, 1)
    
    return np.reshape(data, list(original_shape)[:-1] + [2])


def skeleton_to_box_numpy(data):
    original_shape = data.shape
    assert original_shape[-1] % 2 == 0
    num_points = original_shape[-1] // 2
    points = np.reshape(data, (-1, num_points, 2))

    x1, x2 = np.min(points[:, :, 0], axis=-1), np.max(points[:, :, 0], axis=-1)
    y1, y2 = np.min(points[:, :, 1], axis=-1), np.max(points[:, :, 1], axis=-1)

    # (n, 4)
    box_data = np.stack([x1, y1, x2, y2], -1)
    return np.reshape(box_data, list(original_shape)[:-1] + [4])


def get_hands_from_arm_pytorch(data):
    original_shape = data.shape
    assert original_shape[-1] == 10
    flat_data = data.reshape(-1, original_shape[-1])
    hands = flat_data[:, -4:]
    hands = hands.reshape(list(original_shape)[:-1] + [4])
    
    return hands

def get_arm_from_arm_pytorch(data):
    original_shape = data.shape
    assert original_shape[-1] == 10
    flat_data = data.reshape(-1, original_shape[-1])
    arm = flat_data[:, :-4]
    arm = arm.reshape(list(original_shape)[:-1] + arm.shape[-1])

    return arm
    


def get_hands_from_skeleton_pytorch(data):
    original_shape = data.shape
    flat_data = data.reshape(-1, original_shape[-1])
    points = flat_data.reshape(len(flat_data), -1, 2)
    hands = [points[:, parts.index("LWrist")], 
             points[:, parts.index("RWrist")]]
    hands = torch.cat(hands, -1)
    hands = hands.reshape(list(original_shape)[:-1] + [4])




def get_hands_from_skeleton_numpy(data):
    original_shape = data.shape
    
    flat_data = np.reshape(data, (-1, original_shape[-1]))
    points = np.reshape(flat_data, (len(flat_data), -1, 2))

    hands = [points[:, parts.index("LWrist")], 
             points[:, parts.index("RWrist")]]

    hands = np.concatenate(hands, -1)
    print(hands.shape, original_shape)

    hands = np.reshape(hands, list(original_shape)[:-1] + [-1])
    
    print(hands.shape)
    return hands

def json_to_np_array(json_path):
    with open(json_path) as f:
        json_data = json.load(f)[0]
        
    array_data = [np.array([json_data[x]['x'], json_data[x]['y']]) for x in parts]
    confidence = [json_data[x]['confidence'] for x in parts]
    
    array_data = np.concatenate(array_data)
    
    return array_data

def draw_skeleton(frame, array_data, color=(0, 0, 0), thickness=3):
    points = np.reshape(array_data, (-1, 2))
    
    for c in connections:
        p1, p2 = points[c[0]], points[c[1]]
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))
        frame = cv2.circle(frame, p1, thickness, color, -1)
        frame = cv2.circle(frame, p2, thickness, color, -1)
        frame = cv2.line(frame, p1, p2, color, thickness)

    hands = get_hands_numpy(array_data).reshape(-1, 2)
    frame = cv2.circle(frame, (int(hands[0][0]), int(hands[0][1])), 3, (255, 0, 0), -1)
    frame = cv2.circle(frame, (int(hands[1][0]), int(hands[1][1])), 3, (0, 0, 255), -1)
    boxs = skeleton_to_box_numpy(array_data)
    center = get_box_center_numpy(boxs)
    # print(boxs.shape)
    # frame = cv2.rectangle(frame, (int(boxs[0]), int(boxs[1])), (int(boxs[2]), int(boxs[3])), (0, 255, 0), 3)
    # frame = cv2.circle(frame, (int(center[0]), int(center[1])), 3, (0, 0, 0), -1)

    return frame


def draw_arm(frame, arm, color=(0, 0, 0), thickness=2, line_thickness=2):
    hand = arm[-4:]
    points = np.reshape(arm[:-4], (-1, 2))
    
    for c in arm_connections:
        p1, p2 = points[c[0]], points[c[1]]
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))
        frame = cv2.circle(frame, p1, 2, color, -1)
        frame = cv2.circle(frame, p2, 2, color, -1)
        frame = cv2.line(frame, p1, p2, color, line_thickness)

    frame = cv2.rectangle(frame, 
                          (int(hand[0]), int(hand[1])), 
                          (int(hand[2]), int(hand[3])), color, thickness)
    
    return frame
    
    
def get_arm_and_hand_numpy(pose, hands):
    print(pose.shape)
    assert pose.shape[1] == 16
    
    left_arm_index = [parts.index("LShoulder"), 
                      parts.index("LElbow"),
                      parts.index("LWrist")]
    
    right_arm_index = [parts.index("RShoulder"), 
                      parts.index("RElbow"),
                      parts.index("RWrist")]
    
    original_shape = pose.shape
    
    flat_data = np.reshape(pose, (-1, original_shape[-1]))
    points = np.reshape(flat_data, (len(flat_data), -1, 2))
    
    left_arm = points[:, left_arm_index]
    right_arm = points[:, right_arm_index]
    
    left_arm = np.reshape(left_arm, list(original_shape)[:-1] + [-1])
    right_arm = np.reshape(right_arm, list(original_shape)[:-1] + [-1])

    left_arm = np.concatenate([left_arm, hands[:, 0]], -1)
    right_arm = np.concatenate([right_arm, hands[:, 1]], -1)
    
    return np.stack([left_arm, right_arm], 1)

def segmentation_from_output_class(y, segmentation_type='input'):
    x_segmentation = np.array(y, dtype=np.float32)
    original_missing_mask = y == -1.0
    x_segmentation = np.where(original_missing_mask, np.nan, x_segmentation)
    end_indices = (x_segmentation[1:] - x_segmentation[:-1]) != 0.0    
    end_indices = np.concatenate([end_indices, np.full_like(end_indices, fill_value=True)[-1:]], axis=0)
    x_segmentation[end_indices] = 1.0
    x_segmentation[~end_indices & ~np.isnan(x_segmentation)] = 0.0
    x_segmentation[np.isnan(x_segmentation)] = 1.0
    if segmentation_type == 'output':
        x_segmentation[original_missing_mask] = -1.0
    return x_segmentation

def find_nearest(candidate, num_idx):
    array_idx = np.array([i for i in range(num_idx) if i not in candidate])
    array_idx = np.repeat(array_idx[np.newaxis, :], len(candidate), 0)
    dist = array_idx - candidate[:, np.newaxis]

    left_dist = np.select([dist < 0, dist >= 0], [dist, -1000])
    right_dist = np.select([dist > 0, dist <= 0], [dist, 1000])
    
    left_idx = np.squeeze(np.take_along_axis(array_idx, np.argmax(left_dist, 1)[:, np.newaxis], axis=1), -1)
    right_idx = np.squeeze(np.take_along_axis(array_idx, np.argmin(right_dist, 1)[:, np.newaxis], axis=1), -1)
    
    weight = (candidate - left_idx) / (right_idx - left_idx)

    # The begining of the video. BOth left and right idx are 0
    weight[right_idx == left_idx] = 0.0
    
    # The end of the video. Right idx is 0 => Negative
    weight[weight < 0] = 0.0
    
    return left_idx, right_idx, weight

def fill_nan_each_box(box):
    is_nan_idx = np.where(np.isnan(box[:, 0]))[0]
    new_box = box.copy()
    if len(is_nan_idx) > 0:
        left_idx, right_idx, weight = find_nearest(is_nan_idx, len(box))   

        new_box[is_nan_idx] = box[left_idx] +(box[right_idx] - box[left_idx]) * weight[:, np.newaxis]        

        assert np.all(weight >= 0) 
           
    assert not np.any(np.isnan(new_box))
    return new_box


def handle_nan_boxes(boxes):
    if len(boxes.shape) == 3:
        for i in range(boxes.shape[1]):
            boxes[:, i] = fill_nan_each_box(boxes[:, i])
        return boxes
    elif len(boxes.shape) == 2:
        return fill_nan_each_box(boxes)
    else:
        print(boxes.shape)
        assert 0 == 1

def handle_nan_ske(skeleton):
    assert len(skeleton.shape) == 3
    assert skeleton.shape[1] == 1
    original_shape = skeleton.shape
    points = np.reshape(skeleton, (len(skeleton), -1, 2))
    
    for i in range(points.shape[1]):
        points[:, i] = fill_nan_each_box(points[:, i])
    
    assert not np.any(np.isnan(points))
    return points.reshape(original_shape)


def smooth_each_box(box):
    kernel_size = 3
    kernel = np.array([1 for i in range(kernel_size)])
    
    out = box.copy()
    seq_len = box.shape[0]
    num_dim = box.shape[1]
    
    for i in range(num_dim):
        convolved = np.convolve(box[:, i], kernel, mode='valid') / kernel_size
        offset = (seq_len - len(convolved)) // 2
        out[offset:(seq_len - offset), i] = convolved
        
    return out

def smooth(data):
    if len(data.shape) == 3:
        for i in range(data.shape[1]):
            data[:, i] = smooth_each_box(data[:, i])

        return data
    elif len(data.shape) == 2:
        return smooth_each_box(data)
    else:
        assert 0 == 1