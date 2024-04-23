import os
import re

import numpy as np

from ..io import *


def read_raw_obj(path):
    raw_object = txt_to_numpy(path, delim=" ")
    
    output = []
    for i in range(len(raw_object)):
        current_frame = raw_object[i]
        tmp = current_frame.reshape(-1, 3)
        
        min_x, max_x = np.min(tmp[:, 0]), np.max(tmp[:, 0])
        min_y, max_y = np.min(tmp[:, 1]), np.max(tmp[:, 1])
        min_z, max_z = np.min(tmp[:, 2]), np.max(tmp[:, 2])

        output.append([min_x, min_y, min_z, max_x, min_y, min_z, max_x, max_y, min_z, min_x, max_y, min_z, 
                       min_x, min_y, max_z, max_x, min_y, max_z, max_x, max_y, max_z, min_x, max_y, max_z])
        
    output = np.array(output)
    frame_idx = np.arange(0, len(output))[..., np.newaxis]
    
    return np.concatenate([frame_idx, output], axis=1)

def read_raw_human(path):
    raw_human = txt_to_numpy(path, delim=" ")
    frame_idx = np.arange(0, len(raw_human))[..., np.newaxis]
    
    return np.concatenate([frame_idx, raw_human], axis=1)