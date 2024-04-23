import cv2

from .geometry import get_skeleton_connection

def draw_skeleton(img, data_2d, color=(255, 0, 0), thickness=2):
    connections = get_skeleton_connection()
    for (srcid, dstid) in connections:
        src_pt, dst_pt = data_2d[srcid - 1], data_2d[dstid - 1]
        
        cv2.line(img, (int(src_pt[0]), int(src_pt[1])),
                 (int(dst_pt[0]), int(dst_pt[1])),
                 color, thickness)
        
    return img