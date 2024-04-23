import cv2
import numpy as np
import torch

from src.bimanual.data.utils import CoordinateHandler
from src.shared.visualization.utils import GeneralFrameDrawer
from library.dataset.bimanual.processing import draw_arm

coordinate_handler = CoordinateHandler()
def draw_human(frame, human, type, color=(0, 0, 255), thickness=2, line_thickness=2):
    return draw_arm(frame, human.cpu().detach().numpy().astype(np.int), color, thickness, line_thickness)

def draw_object(frame, obj, type, color, thickness=2):
    min_x, min_y, max_x, max_y = obj.cpu().detach().numpy().astype(np.int)
    frame = cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, thickness)
    # frame = cv2.circle(frame, ((min_x + max_x) // 2, (min_y + max_y) // 2), 4, color, -1)

    return frame


def to_pixel(oh_graph):
    return oh_graph