import os
from functools import wraps
from time import time

import dgl
import torch
import random
import numpy as np


def timing_decorator(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(te - ts)
        return result
    return wrap

def timing(f):
    def run(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        return result, te - ts
    
    return run
    

def set_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    np.random.seed(seed)  
    
    if deterministic:
        torch.backends.cudnn.deterministic = True  

def mkdir_if_not_exist(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def blend(b, g, r, flag):
    colors=[]
    for i in range(0, 250, 25):
        if flag == 0:
            r = i
            colors = colors + blend(b, g, r, flag + 1)
        elif flag == 1:
            g = i
            colors = colors + blend(b, g, r, flag + 1)
        else:
            b = i
            colors.append(np.array([b, g, r]))

    return colors            
        
def rand_color():
    colors = []
    colors = np.random.randint(0, 255, (100, 3))
    colors[:, 0] = 0
    colors = np.concatenate([np.array([[170, 68, 0],
                                       [0, 0, 170],
                                       [0, 128, 0],
                                       [0, 0, 0],
                                       [0, 85, 212]]), colors])
    # for value in range(0, 150, 50):        
    #     for i in range(3):
    #         color = [0, 0, 0]
            
    #         color[i] = 255 - value
    #         colors.append(np.array(color))

    # colors = colors + blend(0, 0, 0, 0)
    # colors = np.stack(colors, axis=0)

    return colors

def will_be_loaded(name, load_name):
    for n in load_name:
        if n in name:
            return True
        
    return False

def freeze_model(model, freeze_part):
    def toggle_child_model(m, grad):
        for name, child in m.named_children():
            if will_be_loaded(name, freeze_part):
                for param in child.parameters():
                    param.requires_grad = False
                toggle_child_model(child, False)
                print(name, False)   

            else:
                for param in child.parameters():
                    param.requires_grad = grad
                toggle_child_model(child, grad)
                print(name, grad)   



    toggle_child_model(model, True)
