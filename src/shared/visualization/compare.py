import os
import torch
import cv2
import numpy as np
from omegaconf import OmegaConf

from src.utils import mkdir_if_not_exist, set_seed
from .utils import GeneralFrameDrawer, get_key_from_frame, id_to_file_name

def get_results(data_arr, model, get_results_fn, device, kwargs):
    all_idx = []
    all_results = []
    
    dataset, loader = data_arr
    for batch in loader:
        cpu_graph, idx = batch
        results = get_results_fn(model, batch, dataset, device, kwargs)
        
        all_idx.append(idx)
        all_results.append(results)
        
    all_idx = np.concatenate(all_idx)
    all_results = np.concatenate(all_results)
    
    return all_idx, all_results

def get_sorted_idx(idx1, rs1, idx2, rs2, descending):
    assert np.all(idx1 == idx2)
    
    diff = rs1 - rs2
    arg_idx = np.argsort(diff)
    
    if descending:
        arg_idx = arg_idx[::-1]
    
    return idx1[arg_idx]

def visualize(kwargs, build_dataset_fn, build_model_fn, 
              get_results_fn, 
              draw_obj_fn, draw_human_fn, pixel_fn,
              drawer_class=GeneralFrameDrawer):

    data_cfg_path = kwargs.pop("data_cfg_path")
    model1_path = kwargs.pop("model1_path")
    model2_path = kwargs.pop("model2_path")
    descending = kwargs.pop("descending")
    
    part = kwargs.pop("part")
    seed = kwargs.pop("seed")
    
    set_seed(seed)
 
    is_save = kwargs.pop("is_save")
    is_show = kwargs.pop("is_show")
    gpu = kwargs.pop("gpu")
    
    if is_save:
        save_path = kwargs.pop("save_path")
        mkdir_if_not_exist(save_path)
    
    if gpu is not None:
        device = torch.device("cuda:{}".format(gpu))
    else:
        device = "cpu"

    tmp = OmegaConf.load(data_cfg_path)
    kwargs["graph_path"] = tmp.graph_path
    kwargs["numpy_path"] = tmp.numpy_path
    kwargs["filename"] = tmp.filename
    
    override_dataset_cfg = {}
    for k, v in kwargs.items():
        if v is not None:
            override_dataset_cfg[k] = v
        
    # override_dataset_cfg["include"] = ["pour"]
        
    override_dataset_cfg = OmegaConf.create(override_dataset_cfg)
    datasets, models = [], []

    for checkpoint_path in [model1_path, model2_path]:
        saved = torch.load(checkpoint_path, map_location=device)
        saved_cfg = OmegaConf.create(saved["config"])
        saved_dataset_cfg = saved_cfg.dataset

        dataset_cfg = OmegaConf.merge(saved_dataset_cfg, override_dataset_cfg)
        dataset, data_loader = build_dataset_fn(dataset_cfg, part, training=False)
        print(saved_cfg.model.architecture)
        model = build_model_fn(saved_cfg.model.architecture)
        model.load_state_dict(saved["model_state"])
        model.eval()
        model = model.to(device)
        model.visualize = True
        
        datasets.append([dataset, data_loader])
        models.append(model)

    list_idx = [1057, 1058, 2421, 2420, 1059, 2419, 534, 
                2773, 2774, 546, 2245, 1487, 1978, 540,
                2702, 415, 1484, 539, 2248, 2251, 363]
    idx1, rs1 = get_results(datasets[0], models[0],
                            get_results_fn, device, kwargs)
    idx2, rs2 = get_results(datasets[1], models[1],
                            get_results_fn, device, kwargs)

    idx = get_sorted_idx(idx1, rs1, idx2, rs2, descending)
    model_drawers = [drawer_class(draw_obj_fn, draw_human_fn, pixel_fn,
                                  dataset_cfg, datasets[i][0], model=models[i], model_name=kwargs.pop("model{}_name".format(i+1)), kwargs=kwargs)
                     for i in range(len(models))]

    frame_cnt = 0
    for gidx in idx:
        print(f"{gidx}, rs1: {rs1[gidx]}, rs2: {rs2[gidx]}")
        model_pred_drawers = []
        model_switch_drawers = []
        for drawer in model_drawers:
            data_drawer = drawer.get_model_drawer(gidx, device)
            
            frame_drawers = data_drawer["frame"]
            cpred_drawers = data_drawer["pred"]
            cswitch_drawer = data_drawer.get("switch", None)
            
            model_pred_drawers.append(cpred_drawers)
            
            if cswitch_drawer is not None:
                model_switch_drawers.append(cswitch_drawer)
            

        break_code = 0
        while break_code == 0:
            frames = []
            
            show_frame = None
            for i, fd in enumerate(frame_drawers):
                cfame = next(fd)
            
                if cfame is None:
                    break_code = 1
                    break
                
                frames.append(cfame)
            
            print(len(frames))
            if len(frames) > 0:
                show_frame = np.concatenate(frames)

            # if show_frame is not None:
            #     if is_show:
            #         cv2.imshow("obs", show_frame)
            #         break_code = get_key_from_frame()
            
            for pred_drawers in model_pred_drawers:
                frames = []
                for i, fd in enumerate(pred_drawers):
                    cfame = next(fd)
                
                    if cfame is None:
                        break_code = 1
                        break
                    
                    frames.append(cfame)

                if len(frames) > 0:
                    pred_frame = np.concatenate(frames)
                    if show_frame is None:
                        show_frame = pred_frame
                    else:
                        show_frame = np.concatenate([show_frame, pred_frame], 1)
            
            if len(model_switch_drawers):
                for switch_drawer in model_switch_drawers:
                    switch_frames = None
                    for i, (k, v) in enumerate(switch_drawer.items()):
                        current_frame = next(v)
                        if current_frame is None:
                            break_code = 1
                            break
                        
                        if switch_frames is None:
                            switch_frames = []
                            
                        switch_frames.append(current_frame)
                    
                    if switch_frames is not None:
                        out = []
                        for i, f in enumerate(switch_frames):
                            if i % 2 == 0:
                                out.append([])
                            out[-1].append(f)
                        
                        if len(switch_frames) % 2 == 1:
                            out[-1].append(np.ones_like(out[-1][0]) * 255)
                            
                        out = [np.concatenate(x, 1) for x in out]
                        # print(out[0].shape)
                        out = np.concatenate(out)

                        out = cv2.resize(out, (640*3, 320))

                        if is_show:
                            cv2.imshow("switch", out)
                        
                        if is_save:
                            show_frame = np.concatenate([show_frame, out])

            if break_code:
                break
            
            if show_frame is not None:
                if is_show:
                    cv2.imshow("model", show_frame)
                    break_code = get_key_from_frame()

                    if break_code:
                        break
            
                if is_save:
                    file_name = id_to_file_name(frame_cnt, 10)
                    file_path = os.path.join(save_path, "{}.png".format(file_name))
                    cv2.imwrite(file_path, show_frame)
                
            frame_cnt += 1
            if frame_cnt > 3000 and is_save:
                break_code = 2
                break

            
        if  break_code == 2:
            break
