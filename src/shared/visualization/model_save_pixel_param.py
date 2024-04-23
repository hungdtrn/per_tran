import os
import torch
import cv2
import numpy as np
import pickle
from omegaconf import OmegaConf
import networkx as nx
import matplotlib.pyplot as plt

from src.utils import mkdir_if_not_exist, set_seed
from .utils import GeneralFrameDrawer, get_key_from_frame, id_to_file_name

node_size = 1000
edge_width = 4
arrowsize = 40
def covert_color_format(c, is_switch):
    if is_switch:
        return '#%02x%02x%02x' % (c[2],c[1],c[0])
    else:
        return "grey"

def _draw_persistent(pos, n_nodes, colors):
    nodes = [i for i in range(n_nodes)]
    nodes_with_color = [(0, {"color": colors[0]})] + [(i+1, {"color": colors[i+3]}) for i in range(n_nodes - 1)]
    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes_with_color)
    converted_colors = []
    edges = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                G.add_edge(j, i)
                edges.append((j, i))
                
        converted_colors.append(covert_color_format(nodes_with_color[i][1]["color"], True))

    if pos is None:
        pos0 = np.array([0, 3])
        sub_graph = G.subgraph(nodes[1:])
        pos = nx.spring_layout(sub_graph, 
                                  center=[0, 6])
        pos[0] = pos0

    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_shape="s", node_size=node_size, node_color=converted_colors)
    arc_rad = 0.25
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=covert_color_format([0, 0, 0], True), width=edge_width, arrowsize=arrowsize, connectionstyle=f'arc3, rad = {arc_rad}')
    # nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=covert_color_format([0, 0, 0], True), width=2)

    return G, pos

def _draw_transient(G, start_idx, center_pos, is_inter, is_aware, is_switch, colors, object_type):
    object_colors = colors[3:]
    nodes = [i+start_idx+1 for i in range(len(is_inter))]
    nodes_with_color = [(start_idx, {"color": colors[0]})] + [(i+1+start_idx, {"color": colors[i+3]}) for i in range(len(is_inter))]
    G.add_nodes_from(nodes_with_color)

    aware_edge = []
    aware_edge_color = []
    inter_edge = []
    center = np.array([center_pos[0], center_pos[1]-3])
    pos = nx.circular_layout(nodes, center=center, scale=1.2)
    pos[start_idx] = center
    
    for i in range(len(is_inter)):   
        is_not_table = object_type[i] != "table" 
        drawing_options = {"node_color": "grey"}     
        if is_aware[i] == 1 and is_switch:
            G.add_edge(i+1+start_idx, start_idx, type="aware")
            aware_edge.append((i+1+start_idx, start_idx))
            aware_edge_color.append(covert_color_format(colors[i+3], is_switch))
            drawing_options = {
                "node_color": covert_color_format([255,255,255], is_switch),
                "edgecolors": covert_color_format(colors[i+3], is_switch),
                "linewidths": 2.0
            }
        if is_inter[i] == 1 and is_switch and is_not_table:
            G.add_edge(start_idx, i+1+start_idx, type="interacts")
            inter_edge.append((start_idx, i+1+start_idx))
            drawing_options = {
                "node_color": covert_color_format(colors[i+3], is_switch),
            }

        
        nx.draw_networkx_nodes(G, pos, nodelist=[i+1+start_idx], node_shape="^", node_size=node_size, **drawing_options)
        
    nx.draw_networkx_nodes(G, pos, nodelist=[start_idx], node_size=node_size * 2,  node_color=covert_color_format(colors[0], is_switch))
    arc_rad = 0.25
    nx.draw_networkx_edges(G, pos, edgelist=aware_edge, edge_color=aware_edge_color, width=edge_width, arrowsize=arrowsize, connectionstyle=f'arc3, rad = {arc_rad}')
    arc_rad = 0.25
    nx.draw_networkx_edges(G, pos, edgelist=inter_edge, edge_color=covert_color_format(colors[0], is_switch), width=edge_width, arrowsize=arrowsize, connectionstyle=f'arc3, rad = {arc_rad}')
    
    return G, pos

def draw(pos, is_inter, is_aware, is_switch, colors, object_type):
    plt.figure(figsize=(12, 22))
    n_nodes = len(is_inter)+1
    G, pos = _draw_persistent(pos, n_nodes, colors)
    G, pos1 = _draw_transient(G, n_nodes, pos[0], is_inter, is_aware, is_switch, colors, object_type)
    pos.update(pos1)
    print(is_switch)
    if is_switch == 1:
        arc_rad = 0.25
        nx.draw_networkx_edges(G, pos, edgelist=[(0, n_nodes)], edge_color=covert_color_format(colors[0], is_switch), width=edge_width, arrowsize=arrowsize, connectionstyle=f'arc3, rad = {arc_rad}')
        nx.draw_networkx_edges(G, pos, edgelist=[(n_nodes, 0)], edge_color=covert_color_format(colors[0], is_switch), width=edge_width, arrowsize=arrowsize, connectionstyle=f'arc3, rad = {arc_rad}')

    plt.axis("off")
    # plt.title("Transient Graph")
    plt.tight_layout()
    return pos
    

def draw_transient(is_inter, is_aware, is_switch, colors, object_type):
    object_colors = colors[3:]
    nodes = [i for i in range(len(is_inter))]
    nodes_with_color = [(0, {"color": colors[0]})] + [(i+1, {"color": colors[i+3]}) for i in range(len(is_inter))]
    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes_with_color)
    aware_edge = []
    aware_edge_color = []
    inter_edge = []
    pos = nx.circular_layout(G.subgraph([i+1 for i in nodes]), center=[0, 0])
    pos[0] = np.array([0, 0])
    
    plt.figure()
    for i in range(len(is_inter)): 
        drawing_options = {"node_color": "grey"}    
        is_not_table = object_type[i] != "table" 
        if is_aware[i] == 1 and is_switch:
            G.add_edge(i+1, 0, type="aware")
            aware_edge.append((i+1, 0))
            aware_edge_color.append(covert_color_format(colors[i+3], is_switch))
            drawing_options = {
                "node_color": covert_color_format([255,255,255], is_switch),
                "edgecolors": covert_color_format(colors[i+3], is_switch),
                "linewidths": 2.0
            }
        if is_inter[i] == 1 and is_switch and is_not_table:
            G.add_edge(0, i+1, type="interacts")
            inter_edge.append((0, i+1))
            drawing_options = {
                "node_color": covert_color_format(colors[i+3], is_switch),
            }

        
        nx.draw_networkx_nodes(G, pos, nodelist=[i+1], node_shape="^", **drawing_options)
        
    nx.draw_networkx_nodes(G, pos, nodelist=[0], node_size=900,  node_color=covert_color_format(colors[0], is_switch))
    arc_rad = 0.25
    nx.draw_networkx_edges(G, pos, edgelist=aware_edge, edge_color=aware_edge_color, width=2, connectionstyle=f'arc3, rad = {arc_rad}')
    arc_rad = 0.25
    nx.draw_networkx_edges(G, pos, edgelist=inter_edge, edge_color=covert_color_format(colors[0], is_switch), width=2, connectionstyle=f'arc3, rad = {arc_rad}')
    plt.axis("off")
    # plt.title("Transient Graph")
    # plt.tight_layout()

            


def visualize(kwargs, build_dataset_fn, build_model_fn,
              draw_obj_fn, draw_human_fn, pixel_fn,
              drawer_class=GeneralFrameDrawer):
    data_cfg_path = kwargs.pop("data_cfg_path")
    checkpoint_path = kwargs.pop("checkpoint_path")

    part = kwargs.pop("part")
    seed = kwargs.pop("seed")
    
    set_seed(seed)
 
    is_save = kwargs.pop("is_save")
    is_show = kwargs.pop("is_show")
    gpu = kwargs.pop("gpu")
    
    save_path = None
    if is_save:
        save_path = kwargs.pop("save_path")
        mkdir_if_not_exist(save_path)
    
    if gpu is not None:
        device = torch.device("cuda:{}".format(gpu))
    else:
        device = "cpu"
    print(device)

    tmp = OmegaConf.load(data_cfg_path)
    kwargs["graph_path"] = tmp.graph_path
    kwargs["numpy_path"] = tmp.numpy_path
    kwargs["filename"] = tmp.filename
    
    override_dataset_cfg = {}
    for k, v in kwargs.items():
        if v is not None:
            override_dataset_cfg[k] = v
        
    # override_dataset_cfg["include"] = ["subject_6-task_3_k_pouring-take_8"]
        
    override_dataset_cfg = OmegaConf.create(override_dataset_cfg)
    saved = torch.load(checkpoint_path, map_location=device)
    saved_cfg = OmegaConf.create(saved["config"])
    saved_dataset_cfg = saved_cfg.dataset

    dataset_cfg = OmegaConf.merge(saved_dataset_cfg, override_dataset_cfg)
    print(dataset_cfg)
    crnn_format = dataset_cfg.crnn_format

    dataset, _ = build_dataset_fn(dataset_cfg, part, training=False)
    model = build_model_fn(saved_cfg.model.architecture)
    model.load_state_dict(saved["model_state"])
    model.eval()
    model = model.to(device)
    model.visualize = True

    idx = [i for i in range(0, len(dataset), 1)]
    # if kwargs["shuffle"]:
    #     np.random.shuffle(idx)

    drawer = drawer_class(draw_obj_fn, draw_human_fn, pixel_fn,
                          dataset_cfg, dataset, model=model, kwargs=kwargs)
    
    # target = [1057, 1058, 2421, 2420, 1059, 2419, 534, 
    #             2773, 2774, 546, 2245, 1487, 540,
    #             2702, 415, 1484, 539, 2248, 2251, 363, 2717]
    target = [2227, 2239, 2255, 2700, 2765, 2780]
    pixel_params = {}
    # off_set = 50
    # for x in target:
    #     target = target + list(range(x-off_set, x+off_set))
        
    target = list(set(target))
    frame_cnt = 0
    for gidx in idx:
        # if gidx < 2650:
        #     continue

        if gidx not in target:
            continue
        print(gidx)

        # if not (gidx > 300 and gidx < 600):
        #     continue
        # if gidx != 10:
        #     continue
        if is_save:
            os.makedirs(os.path.join(save_path, str(gidx)), exist_ok=True)
        
        data_drawer = drawer.get_model_drawer(gidx, device)
        
        frame_drawers = data_drawer["frame"]
        pred_drawers = data_drawer["pred"]
        switch_drawer = data_drawer.get("switch", None)
        ego_drawer = data_drawer.get("ego", None)
        colors = data_drawer.get("colors")
        pixel_params[gidx] = data_drawer.get("pixel_params")
        additional_output = data_drawer.get("additional_output") 
        object_type = [dataset.additional_info["object_types"][i.item()].lower() for i in additional_output["object_type"]]
        # if is_save:
        #     if not os.path.exists(os.path.join(save_path, "colors.p")):
        #         with open(os.path.join(save_path, "colors.p"), "wb") as f:
        #             pickle.dump(colors, f)
        if "pred_switch_label" in additional_output and len(additional_output["pred_switch_label"]) > 1:
            continue
        break_code = 0
        saving_frames = {
            "gt": None,
            "pred": None,
            "switch": None,
        }
        per_pos = None
        frame_idx = 0
        while break_code == 0:
            frames = []
            ego_frames = []                
            
            show_frame = None
            for i, fd in enumerate(frame_drawers):
                cfame = next(fd)
            
                if cfame is None:
                    break_code = 1
                    break
                
                frames.append(cfame)
                
            if len(frames) > 0:
                gt_frame = np.concatenate(frames)
                
            if is_save:
                saving_frames["gt"] = gt_frame

            frames = []
            for i, fd in enumerate(pred_drawers):
                cfame = next(fd)
            
                if cfame is None:
                    break_code = 1
                    break
                
                frames.append(cfame)

            if len(frames) > 0:
                pred_frame = np.concatenate(frames)
                show_frame = np.concatenate([gt_frame, pred_frame], 1)
                
                if is_save:
                    saving_frames["pred"] = pred_frame
                
                if is_show:
                    cv2.imshow("model", show_frame)
            
            if break_code == 1:
                break

            if ego_drawer is not None:
                ego_frames = None
                for i, d in enumerate(ego_drawer):
                    current_frame = next(d)
                    if current_frame is None:
                        break
                    
                    if ego_frames is None:
                        ego_frames = []
                        
                    ego_frames.append(current_frame)
                
                if ego_frames is not None and len(ego_frames) > 0:
                    ego_frames = np.concatenate(ego_frames, 1)

                    if is_show:
                        cv2.imshow("ego", ego_frames)
                    
                    if is_save:
                        show_frame = np.concatenate([show_frame, ego_frames])
                    
            
            if switch_drawer is not None:
                switch_frames = None
                for i, (k, v) in enumerate(switch_drawer.items()):
                    current_frame = next(v)
                    if current_frame is None:
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
                    
                    # if len(switch_frames) % 2 == 1:
                    #     out[-1].append(np.ones_like(out[-1][0]) * 255)
                        
                    out = [np.concatenate(x, 1) for x in out]
                    # print(out[0].shape)
                    out = np.concatenate(out)
                    
                    if is_save:
                        saving_frames["switch"] = out

                    out = cv2.resize(out, (1280, 160))

                    if is_show:
                        cv2.imshow("switch", out)
                    
                    if is_save:
                        show_frame = np.concatenate([show_frame, out])
 
            if show_frame is None:
                break
 
            if is_save:
                # file_name = id_to_file_name(frame_cnt, 10)
                # file_name = "{}_{}".format(file_name, gidx)
                # file_path = os.path.join(save_path, "{}.png".format(file_name))
                # cv2.imwrite(file_path, show_frame)
                for k, v in saving_frames.items():
                    if v is not None:
                        # file_path = os.path.join(save_path, "{}_{}.png".format(file_name, k))
                        file_path = os.path.join(save_path, str(gidx), f"{frame_idx}_{k}.png")
                        cv2.imwrite(file_path, v)
                
                
                save_idx = frame_idx
                if "transient_is_aware" in additional_output:
                    if frame_idx == len(additional_output["transient_is_aware"]):
                        save_idx = frame_idx - 1

                    per_pos = draw(per_pos, is_inter=additional_output["transient_is_inter"][save_idx],
                                is_aware=additional_output["transient_is_aware"][save_idx],
                                is_switch=additional_output["pred_switch_label"][:, save_idx].item(),
                                colors=colors, object_type=object_type)
                    plt.savefig(os.path.join(save_path, str(gidx), f"{frame_idx}_both.png"))
                    plt.close()
                    
                    draw_transient(is_inter=additional_output["transient_is_inter"][save_idx],
                                is_aware=additional_output["transient_is_aware"][save_idx],
                                is_switch=additional_output["pred_switch_label"][:, save_idx].item(),
                                colors=colors,
                                object_type=object_type)
                    plt.savefig(os.path.join(save_path, str(gidx), f"{frame_idx}_transient.png"))
                    plt.close()

                    # with open(os.path.join(save_path, "{}_transient.p".format(file_name)), "wb") as f:
                    #     pickle.dump({"is_aware": additional_output["transient_is_aware"][save_idx],
                    #                  "is_inter": additional_output["transient_is_inter"][save_idx],
                    #                  "is_switch": additional_output["pred_switch_label"][:, save_idx],
                    #                  "switch_score": additional_output["pred_switch_score"][:, save_idx],
                    #                  }, f)
                        
            saving_frames = {
                "gt": None,
                "pred": None,
                "switch": None,
            }
            
            frame_cnt += 1
            if frame_cnt > 10000 and is_save:
                break_code = 2
                break
                
            break_code = get_key_from_frame()
            if break_code:
                break
            
            frame_idx = frame_idx + 1
                
        if  break_code == 2:
            break
        
    with open("tmp.p", "wb") as f:
        pickle.dump(pixel_params, f)