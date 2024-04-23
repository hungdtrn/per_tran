import cv2
import numpy as np
from omegaconf import OmegaConf

from src.utils import mkdir_if_not_exist, set_seed
from .utils import GeneralFrameDrawer, get_key_from_frame, id_to_file_name
import os

def visualize(kwargs, build_dataset_fn, 
              draw_obj_fn, draw_human_fn, pixel_fn,
              drawer_class=GeneralFrameDrawer):
    data_cfg_path = kwargs.pop("data_cfg_path")
    part = kwargs.pop("part")
    seed = kwargs.pop("seed")
    
    is_save = kwargs.pop("is_save")
    is_show = kwargs.pop("is_show")

    if is_save:
        save_path = kwargs.pop("save_path")
        mkdir_if_not_exist(save_path)

    set_seed(seed)
 
    shuffle = kwargs["shuffle"]

    data_cfg = OmegaConf.load(data_cfg_path)
    kwargs["include"] = ["subject_6-task_3_k_pouring-take_8"]

    override_cfg = OmegaConf.create(kwargs)
    
    cfg = OmegaConf.merge(data_cfg, override_cfg)
    print(cfg)
    dataset, data_loader = build_dataset_fn(cfg, part, training=False)

    idx = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(idx)

    drawer = drawer_class(draw_obj_fn, draw_human_fn, pixel_fn,
                          cfg, dataset, kwargs)
    
    break_code = 0
    frame_cnt = 0
    for gidx in idx:
        data_drawer = drawer.get_data_drawer(gidx)
        
        frame_drawers = data_drawer["frame"]
        switch_drawer = data_drawer.get("switch", None)
        
        break_code = 0
        while break_code == 0:
            for i, fd in enumerate(frame_drawers):
                cfame = next(fd)
            
                if cfame is None:
                    break_code = 1
                    break
                
                if is_show:
                    cv2.imshow(str(i), cfame)
                
                if is_save:
                    file_name = id_to_file_name(frame_cnt, 10)
                    file_path = os.path.join(save_path, "{}.png".format(file_name))
                    cv2.imwrite(file_path, cfame)
                    frame_cnt += 1
                    # break_code=1

            if break_code == 1:
                break
            
            if switch_drawer is not None:
                switch_frames = []
                for _, v in switch_drawer.items():
                    current_frame = next(v)
                    
                    if current_frame is not None:
                        switch_frames.append(current_frame)
                    
                if len(switch_frames) > 0:
                    switch_frames = np.concatenate(switch_frames, 1)
                    switch_frames = cv2.resize(switch_frames, (1280, 480))
                    
                    if is_show:
                        cv2.imshow("switch", switch_frames)
                
            break_code = get_key_from_frame()
            if break_code:
                break

            if frame_cnt > 2000 and is_save:
                break_code = 2
                break

                
        if  break_code == 2:
            break
