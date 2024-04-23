import cv2
import numpy as np
import torch

from src.shared.visualization.utils import GeneralFrameDrawer
from src.wbhm.data.utils import get_skeleton_connection

def draw_human(frame, human, type, color=(0, 0, 255), thickness=2, line_thickness=2):
    human = human.detach().cpu().numpy()
    connectivity = get_skeleton_connection(dim=len(human))

    for src_idx, dst_idx in connectivity:
        src_p, dst_p = human[src_idx].astype(int), human[dst_idx].astype(int)
        cv2.line(frame, 
                 (src_p[0], src_p[1]), 
                 (dst_p[0], dst_p[1]),
                 color, line_thickness)
        
    # cv2.circle(frame, tuple(human[1].astype(int).tolist()), 3, (255, 0, 0), -1)
    # cv2.circle(frame, tuple(human[2].astype(int).tolist()), 3, (0, 255, 0), -1)
    # cv2.circle(frame, tuple(human[5].astype(int).tolist()), 3, (0, 0, 255), -1)
    # cv2.circle(frame, tuple(human[8].astype(int).tolist()), 3, (0, 255, 0), -1)
    # cv2.circle(frame, tuple(human[9].astype(int).tolist()), 3, (0, 255, 0), -1)
    # cv2.circle(frame, tuple(human[16].astype(int).tolist()), 3, (0, 255, 0), -1)
    # cv2.circle(frame, tuple(human[17].astype(int).tolist()), 3, (0, 255, 0), -1)
    
    
    for point in human:
        cv2.circle(frame, tuple(point.astype(int).tolist()), thickness, color, -1)
    
    # hand_idx, foot_idx = get_hand_and_foot_idx(human)
    # for idx in (hand_idx + foot_idx):
    
    #     cv2.circle(frame, tuple(human[idx].astype(int).tolist()), 1, (0, 255, 0), -1)

    return frame


def draw_object(frame, obj, type, color, thickness=2):
    obj = obj.detach().cpu().numpy()
    bounding_box = type != "Table"

    if len(obj) == 8:
        p00 = obj[0]
        p01 = obj[1]
        p02 = obj[2]
        p03 = obj[3]
        
        p10 = obj[4]
        p11 = obj[5]
        p12 = obj[6]
        p13 = obj[7]
    else:
        p00 = obj[0]
        p01 = obj[1]
        p03 = obj[2]
        p02 = (p03 - p00) + p01
            
        p10 = obj[3]
        p11 = p10 - p00 + p01
        p12 = p10 - p00 + p02
        p13 = p10 - p00 + p03

    parts = [[p00, p01, p02, p03],
                [p10, p11, p12, p13],
                [p00, p01, p11, p10],
                [p03, p02, p12, p13]]
    
    if not bounding_box:
        for p in parts:
            p = np.stack(p).reshape((-1, 1, 2)).astype(int)
            frame = cv2.polylines(frame, [p], True, color, 2)
    else:
        min_x, max_x = np.min(obj[:, 0]).astype(int), np.max(obj[:, 0]).astype(int)
        min_y, max_y = np.min(obj[:, 1]).astype(int), np.max(obj[:, 1]).astype(int)

        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, thickness)
    
    return frame


def to_pixel(oh_graph, params=None):
    DESIRED_W = 640
    DESIRED_H = 480

    hloc = oh_graph.nodes["human"].data["seq"]
    oloc = oh_graph.nodes["obj"].data["seq"]
    obs_len = oh_graph.nodes["obj"].data["x"].size(1)
    
    for k in ["y_pred", "pred_seq"]:
        if k in oh_graph.nodes["human"].data:
            hloc = torch.cat([hloc, oh_graph.nodes["human"].data[k]], 1)

        if k in oh_graph.nodes["obj"].data:
            oloc = torch.cat([oloc, oh_graph.nodes["obj"].data[k]], 1)

    nh, seq_len, _ = hloc.size()
    no = oloc.size(0)
    
    hloc = hloc.view(nh, seq_len, -1, 3).detach().cpu()
    oloc = oloc.view(no, seq_len, -1, 3).detach().cpu()

    hloc = hloc.view(nh, seq_len, -1, 3).detach().cpu()
    oloc = oloc.view(no, seq_len, -1, 3).detach().cpu()
    

    # In the C3D file (original data), the axis from bottom to top is +Z, from left to right is +X
    # Only select the X and Z axes for visualization, flip the Z axis
    
    # Drop the Y-axis
    hloc = hloc[:, :, :, [0, 2]]
    oloc = oloc[:, :, :, [0, 2]]
    
    loc = torch.cat([hloc.reshape(-1, 2), oloc.reshape(-1, 2)], 0)
    
    if params is None:
        offset = 40
        min_x, max_x = torch.min(loc[:, 0]) - offset * 3, torch.max(loc[:, 0])
        min_y, max_y = torch.min(loc[:, 1]), torch.max(loc[:, 1])
        
        scale_w = DESIRED_W / (max_x - min_x + 1e-4)
        scale_h = DESIRED_H / (max_y - min_y + 1e-4)
        
        params = [min_x, min_y, max_x, max_y, scale_w, scale_h]
    else:
        print("hello")
        min_x, min_y, max_x, max_y, scale_w, scale_h = params 
    
    for node_name in ["human", "obj"]:
        for data_name in ["x", "y", "seq", "y_pred", "pred_seq"]:
            if data_name not in oh_graph.nodes[node_name].data:
                continue
            
            scaled = oh_graph.nodes[node_name].data[data_name]
            n, l, _ = scaled.shape
            scaled = scaled.reshape(n, l, -1, 3)
            scaled = scaled[:, :, :, [0, 2]]
            
            scaled[:, :, :, 0] = scaled[:, :, :, 0] - min_x
            scaled[:, :, :, 1] = max_y - scaled[:, :, :, 1]
            
            scaled[:, :, :, 0] = scaled[:, :, :, 0] * scale_w
            scaled[:, :, :, 1] = scaled[:, :, :, 1] * scale_h
            
            # print(node_name, data_name, torch.min(scaled[:, :, :, 0]), torch.max(scaled[:, :, :, 0]))
            # print(node_name, data_name, torch.min(scaled[:, :, :, 1]), torch.max(scaled[:, :, :, 1]))
            oh_graph.nodes[node_name].data[data_name] = scaled
    return oh_graph, params


