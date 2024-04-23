import os
import json
import re

import numpy as np

def txt_to_numpy(txt_path, delim, max_line=-1):
    """[summary]

    Args:
        txt_path ([str]): Path to the txt string
    """

    with open(txt_path) as f:
        lines = f.readlines()

    nums = []
    num_cols = 0
    
    row_cnt = 0
    for line in lines:
        row = re.split(delim, line.strip().replace("\n", ""))
        
        # Assume that the first row always has the clean version of the data
        # Get the number of columns from the first row
        if len(nums) == 0:
            num_cols = len(row)

        if len(row) != num_cols:
            continue
        
        row = [float(x) for x in row if len(x) > 0]
        nums.append(row)
        row_cnt += 1
        
        if max_line > 0 and row_cnt == max_line:
            break
        # if row_cnt == 10:
        #     break
    
    return np.array(nums)

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    
    return data

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def mkdir_if_not_exist(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)