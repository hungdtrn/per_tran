import os
import re

import numpy as np

from ..io import *

def get_annotation_paths(annotation_path):
    out = {}

    for actor in os.listdir(annotation_path):
        actor_path = os.path.join(annotation_path, actor)
        
        if not os.path.isdir(actor_path):
            continue 

        actor_name = actor.split("_")[0]
        
        if actor_name not in out:
            out[actor_name] = {}
        
        
        for activity in os.listdir(actor_path):
            activity_path = os.path.join(actor_path, activity)
            
            if not os.path.isdir(activity_path):
                continue
            
            if activity not in out[actor_name]:
                out[actor_name][activity] = {}
                
            vid_names = _get_video_names_from_path(activity_path)
            for vid in vid_names:
                vid_paths = {
                    "obj": [],
                    "human": os.path.join(activity_path, "{}.txt".format(vid)),
                    "global_m": os.path.join(activity_path, "{}_globalTransform.txt".format(vid))
                }
                
                for filename in os.listdir(activity_path):
                    if "obj" in filename and vid in filename:
                        vid_paths["obj"].append(os.path.join(activity_path, filename))
                
                out[actor_name][activity][vid] = vid_paths
                
    return out


def read_human(path):
    raw = txt_to_numpy(path, delim=",")
    return _process_human(raw)
    
def read_objs2d(path):
    raw = txt_to_numpy(path, delim=",")
    return raw[:, :6]

def write(path, data):
    out = []
    for row in data:
        row_txt = ",".join([str(x) for x in row.tolist()])
        out.append(row_txt + "\n")
    with open(path, "w") as f:
        f.writelines(out)

def read_objs3d(path):
    raw = txt_to_numpy(path, delim=",")
    return raw
    
def _process_human(raw_data: np.array) -> np.array:
    """
        Input: raw_data
        Output: processed data
        
        The raw data has the format of 
        
            Frame#,ORI(1),P(1),ORI(2),P(2),...,P(11),J(11),P(12),...,P(15)
            Where
                - Frame# : the frame idx, start from 1
                - ORI(i): 3x3 flatterned orientation matrix of 
                        the joint (from joint 1 - to 11)

                - P(i): The 3D position of each joint
                - CONF: The confidence of the skeleton frame work

            In short:
                - from joint 1 -> 11, each joint has the feature of 14
                - from joint 11 -> 15, each joint has the feature of 4
                
        The processed data has the format of
            Frame#, J(1), ..., J(15),
            Where J(i) = (xi, yi, zi)
    """
    skeleton_position = []
    skeleton_position.append(np.expand_dims(raw_data[:, 0], 1))
    for i in range(1, 12):
        # #Frameid, prev_joints, orientation
        # World coordinate starts at position 10
        
        start = 1 + (i - 1) * 14 + 10
        end = start + 3
        skeleton_position.append(raw_data[:, start:end])

    for i in range(1, 5):
        # #FrammeId, prev_body_joints, prev_hand_foot_joints
        # World coordinate starts at position 0
        
        start = 1 + 14 * 11 + (i - 1) * 4 
        end = start + 3
        skeleton_position.append(raw_data[:, start:end])
        
    skeleton_position = np.concatenate(skeleton_position, axis=1)
    
    return skeleton_position
    
def _get_video_names_from_path(path):
    video_regex = r"\d+\.txt"
    
    file_lists = os.listdir(path)
    vid_list = list(filter(lambda x: re.match(video_regex, x) is not None, 
                           file_lists))
    vid_list = list(map(lambda x: x.split('.')[0], vid_list))
    
    return vid_list

    