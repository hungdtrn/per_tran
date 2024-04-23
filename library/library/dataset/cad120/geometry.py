import numpy as np
import torch

JOINT_MAPPING = {
    1: "head",
    2: "neck",
    3: "torso",
    4: "left_shoulder",
    5: "left_elbow",
    6: "right_shoulder",
    7: "right_elbow",
    8: "left_hip",
    9: "left_knee",
    10: "right_hip",
    11: "rigth_knee",
    12: "left_hand",
    13: "right_hand",
    14: "left_foot",
    15: "right_foot"
}

def from_3d_to_2d(cloud: np.array):
    """Convert from 3D world coordinates to 2D image coordinates.

    Arguments:
        data: Tensor of shape (num_points, 3) containing the x, y, z world coordinates of a set of points.
    Returns:
        A tensor of shape (num_points, 2) containing the x, y image coordinates of the input points.
    """
    horizontal_scale = 640 / 1.1147
    vertical_scale = 480 / 0.8336
    depth = cloud[:, -1]

    valid_depth = depth != 0
    depth = depth[valid_depth]
    
    out = np.zeros((len(cloud), 2))
    
    x = cloud[valid_depth, 0] * horizontal_scale / depth + 320
    y = -cloud[valid_depth, 1] * vertical_scale / depth + 240
    out[valid_depth] = np.stack([x, y], axis=-1)
    
    return out

def rgbd_to_point_cloud(rgbd):
    manual_pcd = np.zeros((480, 640, 3))
    manual_pcd[:, :, 0] = np.repeat(np.expand_dims(np.arange(640), 0), 480, axis=0)
    manual_pcd[:, :, 1] = np.repeat(np.expand_dims(np.arange(480), 1), 640, axis=1)
    manual_pcd[:, :, 2] = rgbd[:, :, 3]

    manual_pcd[:, :, 0] = (manual_pcd[:, :, 0] - 640 * 0.5) * manual_pcd[:, :, 2] * 1.147 / 640;
    manual_pcd[:, :, 1] = (480 * 0.5 - manual_pcd[:, :, 1]) * manual_pcd[:, :, 2] * 0.8336 / 480;

    manual_pcd = manual_pcd.reshape(-1, 3)
    
    return manual_pcd[:, :3]

def get_skeleton_connection():
    return [[1, 2], [2, 3], [2, 4], [4, 5], 
            [5, 12], [2, 6], [6, 7], [7, 13],
            [3, 8], [8, 9], [9, 14], [3, 10], 
            [10, 11], [11, 15]]
    
def get_rotation_from_skeleton(ske: torch.tensor) -> torch.tensor:
    pass
    
def get_transformation_matrix_from_skeleton(ske: torch.Tensor) -> torch.Tensor:
    pass

