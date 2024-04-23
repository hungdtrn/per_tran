import cv2
import click
from src.shared.visualization.utils import SwitchDrawer

from src.wbhm.data import build_dataset
from src.wbhm.architectures import build_model
import networkx as nx
import matplotlib.pyplot as plt


from src.shared.visualization.model import visualize
from .utils import draw_human, draw_object, to_pixel
import os
import torch
import cv2
import numpy as np
from omegaconf import OmegaConf

from src.utils import mkdir_if_not_exist, rand_color, set_seed

set_seed(0)
colors = rand_color()
colors = [x.tolist() for x in colors]

node_size = 1000
edge_width = 4
arrowsize = 40
def covert_color_format(c, is_switch):
    if is_switch:
        return '#%02x%02x%02x' % (c[2],c[1],c[0])
    else:
        return "grey"

def to_pixel(x_human, y_human, y_pred_human, x_obj, y_obj, y_pred_obj, scale_param=None):
    DESIRED_W = 640
    DESIRED_H = 480
    
    if scale_param is None:
        hloc = torch.cat([x_human, y_human, y_pred_human], dim=1)
        oloc = torch.cat([x_obj, y_obj, y_pred_obj], dim=1)
        
        obs_len = x_human.size(1)
        
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
        min_x, max_x = torch.min(loc[:, 0]), torch.max(loc[:, 0])
        min_y, max_y = torch.min(loc[:, 1]), torch.max(loc[:, 1])
    else:
        min_x, max_x, min_y, max_y = scale_param

        
    scale_w = DESIRED_W / (max_x - min_x + 1e-4)
    scale_h = DESIRED_H / (max_y - min_y + 1e-4)
    
    out = []
    for t1 in [[x_human, y_human, y_pred_human], [x_obj, y_obj, y_pred_obj]]:
        for t2 in t1:
            scaled = t2
            n, l, _ = scaled.shape
            scaled = scaled.reshape(n, l, -1, 3)
            scaled = scaled[:, :, :, [0, 2]]
            
            scaled[:, :, :, 0] = scaled[:, :, :, 0] - min_x
            scaled[:, :, :, 1] = max_y - scaled[:, :, :, 1]
            
            scaled[:, :, :, 0] = scaled[:, :, :, 0] * scale_w
            scaled[:, :, :, 1] = scaled[:, :, :, 1] * scale_h
            
            out.append(scaled)
            
    out = out + [[min_x, max_x, min_y, max_y]]
            
    return out

def post_process(y_pred_human, y_pred_obj, g, scale_param=None):
    data = []
    for node_name in ["human", "obj"]:
        for data_name in ["x", "y"]:
            data.append(g.nodes[node_name].data[data_name])
    
    x_human, y_human, x_obj, y_obj = data
    x_human, y_human, y_pred_human, x_obj, y_obj, y_pred_obj, scale_param = to_pixel(x_human, y_human, y_pred_human,
                                                                            x_obj, y_obj, y_pred_obj, scale_param)

    return x_human, y_human, y_pred_human, x_obj, y_obj, y_pred_obj, scale_param

def draw(y_pred_human, y_pred_obj, seq_idx, prev_human, prev_obj, obj_type, switch, start_idx = 0):
    time_window = 5
    current_frame = np.ones((480, 640, 3)).astype(np.uint8) * 255
    switch_frames = []
    weight = 0.2
    for j, prev_h in enumerate(prev_human):
        prev_frame = current_frame.copy()
        for i, h in enumerate(prev_h):
            prev_frame = draw_human(prev_frame, h, type="human", color=colors[i+start_idx], thickness=2)
        
        prev_o = prev_obj[j]
        
        for i, o in enumerate(prev_o):
            # prev_frame = draw_object(prev_frame, o, type=obj_type[i], color=colors[len(human_loc) + i], thickness=2)
            prev_frame = draw_object(prev_frame, o, type=obj_type[i], color=colors[i+1], thickness=2)

        current_frame = np.uint8(prev_frame * weight + current_frame * (1-weight))


    for i, h in enumerate(y_pred_human):
        current_frame = draw_human(current_frame, h[seq_idx], type="human", color=colors[i+start_idx], thickness=5)
        switch_frames.append(switch[i].draw(seq_idx))

    for i, o in enumerate(y_pred_obj):
        current_frame = draw_object(current_frame, o[seq_idx], type=obj_type[i], color=colors[i+1], thickness=3)

    prev_human.append(y_pred_human[:, seq_idx])
    prev_obj.append(y_pred_obj[:, seq_idx])
    
    if len(prev_human) > time_window:
        prev_human.pop(0)
        
    if len(prev_obj) > time_window:
        prev_obj.pop(0)


    return current_frame, np.concatenate(switch_frames,0), prev_human, prev_obj

def _draw_persistent(pos, n_nodes, colors):
    nodes = [i for i in range(n_nodes)]
    nodes_with_color = [(0, {"color": colors[0]})] + [(i+1, {"color": colors[i+1]}) for i in range(n_nodes - 1)]
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
    nodes = [i+start_idx+1 for i in range(len(is_inter))]
    nodes_with_color = [(start_idx, {"color": colors[0]})] + [(i+1+start_idx, {"color": colors[i+1]}) for i in range(len(is_inter))]
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
            aware_edge_color.append(covert_color_format(colors[i+1], is_switch))
            drawing_options = {
                "node_color": covert_color_format([255,255,255], is_switch),
                "edgecolors": covert_color_format(colors[i+1], is_switch),
                "linewidths": 2.0
            }
        if is_inter[i] == 1 and is_switch and is_not_table:
            G.add_edge(start_idx, i+1+start_idx, type="interacts")
            inter_edge.append((start_idx, i+1+start_idx))
            drawing_options = {
                "node_color": covert_color_format(colors[i+1], is_switch),
            }

        
        nx.draw_networkx_nodes(G, pos, nodelist=[i+1+start_idx], node_shape="^", node_size=node_size, **drawing_options)
        
    nx.draw_networkx_nodes(G, pos, nodelist=[start_idx], node_size=node_size * 2,  node_color=covert_color_format(colors[0], is_switch))
    arc_rad = 0.25
    nx.draw_networkx_edges(G, pos, edgelist=aware_edge, edge_color=aware_edge_color, width=edge_width, arrowsize=arrowsize, connectionstyle=f'arc3, rad = {arc_rad}')
    arc_rad = 0.25
    nx.draw_networkx_edges(G, pos, edgelist=inter_edge, edge_color=covert_color_format(colors[0], is_switch), width=edge_width, arrowsize=arrowsize, connectionstyle=f'arc3, rad = {arc_rad}')
    
    return G, pos

def draw_architecture(pos, is_inter, is_aware, is_switch, colors, object_type):
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

@click.command()
@click.option("--data_cfg_path", default="src/wbhm/config/dataset/wbhm.yaml")
@click.option("--checkpoint_path")
@click.option("--video_path")
@click.option("--part", default="test")
@click.option("--seed", default=0)
@click.option("--shuffle", type=bool, default=True)
@click.option("--batch_size", default=128)
@click.option("--crnn_format", type=bool, default=False)
@click.option("--duality_format", default=False)
@click.option("--is_ego", default=False)
@click.option("--is_save", type=bool, default=False)
@click.option("--is_show", type=bool, default=True)
@click.option("--crnn_format", default=False)
@click.option("--gpu", type=int)
@click.option("--is_draw_switch", type=bool, default=False)
@click.option("--model_name", type=str)
@click.option("--save_path")
def visualize_counterfactual(**kwargs):
    data_cfg_path = kwargs.pop("data_cfg_path")
    checkpoint_path = kwargs.pop("checkpoint_path")
    part = kwargs.pop("part")
    seed = kwargs.pop("seed") 
    is_save = kwargs.pop("is_save")
    is_show = kwargs.pop("is_show")
    gpu = kwargs.pop("gpu")
    
    # if is_save:
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
        
    # override_dataset_cfg["include"] = ["subject_6-task_3_k_pouring-take_8"]
        
    override_dataset_cfg = OmegaConf.create(override_dataset_cfg)
    saved = torch.load(checkpoint_path, map_location=device)
    saved_cfg = OmegaConf.create(saved["config"])
    saved_dataset_cfg = saved_cfg.dataset

    dataset_cfg = OmegaConf.merge(saved_dataset_cfg, override_dataset_cfg)
    print(dataset_cfg)
    crnn_format = dataset_cfg.crnn_format

    dataset, _ = build_dataset(dataset_cfg, part, training=False)
    model = build_model(saved_cfg.model.architecture)
    model.load_state_dict(saved["model_state"])
    model.eval()
    model = model.to(device)
    model.visualize = True

    idx = [i for i in range(0, len(dataset), 1)]
    # if kwargs["shuffle"]:
    #     np.random.shuffle(idx)

    frame_cnt = 0
    for gidx in idx:
        print(gidx)
        # if gidx < 1440:
        #     continue
        #695, 805
        if gidx != 805:
            continue
        
        save_path = os.path.join(save_path, f"{gidx}")
        mkdir_if_not_exist(save_path)
        
        t_force_persistent = -1
        t_force_transient = -1
        per_pos = None

        g, _ = dataset[gidx]
        g = g.to(device)
        obj_type_name = dataset.additional_info["object_types"]
        seq_len = g.nodes["human"].data["y"].size(1) + g.nodes["human"].data["x"].size(1)
        object_type = [dataset.additional_info["object_types"][i.item()].lower() for i in torch.argmax(g.nodes["obj"].data["type"], dim=1)]
        pred_len = 30
        
        data_stats = {}
        for k, v in dataset.additional_info["data_stats"].items():
            if ("human" in k) or ("obj" in k) or ("point" in k):
                data_stats[k] = v.to(device)
            else:
                data_stats[k] = v
        
        additional_info = {}
        additional_info["post_process"] = True
        additional_info["data_stats"] = data_stats

        gt_switch, pred_switch, pred_count_switch = None, None, None
        additional_output_count = None
        with g.local_scope():
            _,  (y_pred_human, y_pred_obj), additional_output = model(g, pred_len, 
                                                                        additional_info, is_training=False,
                                                                        is_visualize=True)


        
        x_human, y_human, y_pred_human, x_obj, y_obj, y_pred_obj, scale_param = post_process(y_pred_human, y_pred_obj, g)
        y_pred_human_count, y_pred_obj_count = None, None

        obj_type = torch.argmax(g.nodes["obj"].data["type"], dim=1)
        obj_type = [obj_type_name[int(i)] for i in obj_type]


        prev_human, prev_obj = [], []
        prev_human_cnt, prev_obj_cnt = [], []

        gt_switch = [SwitchDrawer(x, "obs", color) for (x, color) in zip(additional_output["gt_switch_score"],
                                                                         colors)]
        
        obs_len = x_human.shape[1]
        for seq_idx in range(obs_len):
            current_frame, switch_frame, prev_human, prev_obj = draw(x_human, x_obj, seq_idx, prev_human=prev_human, prev_obj=prev_obj, obj_type=obj_type, switch=gt_switch)
            # cv2.imshow("pred", current_frame)
            cv2.imwrite(os.path.join(save_path, "obs_{}.png".format(seq_idx)), current_frame)
            cv2.imwrite(os.path.join(save_path, "obs_switch_{}.png".format(seq_idx)), switch_frame)
            per_pos = draw_architecture(per_pos, is_inter=additional_output["transient_is_inter"][seq_idx],
                        is_aware=additional_output["transient_is_aware"][seq_idx],
                        is_switch=additional_output["pred_switch_label"][:, seq_idx].item(),
                        colors=colors, object_type=object_type)
            plt.savefig(os.path.join(save_path, f"obs_{seq_idx}_both.png"))
            plt.close()
        
        
        seq_len = y_pred_human.shape[1]
        pred_switch = [SwitchDrawer(x, "pred", color) for (x, color) in zip(additional_output["pred_switch_score"][:, obs_len:],
                                                                         colors)]

        for seq_idx in range(seq_len):
            current_frame, current_switch, prev_human, prev_obj = draw(y_pred_human, y_pred_obj, seq_idx, prev_human=prev_human, prev_obj=prev_obj, obj_type=obj_type, switch=pred_switch)
            # cv2.imshow("pred", current_frame)
            cv2.imwrite(os.path.join(save_path, "pred_{}.png".format(seq_idx)), current_frame)
            cv2.imwrite(os.path.join(save_path, "pred_switch_{}.png".format(seq_idx)), current_switch)
            
            
            per_pos = draw_architecture(per_pos, is_inter=additional_output["transient_is_inter"][seq_idx + obs_len],
                        is_aware=additional_output["transient_is_aware"][seq_idx + obs_len],
                        is_switch=additional_output["pred_switch_label"][:, seq_idx + obs_len].item(),
                        colors=colors, object_type=object_type)
            plt.savefig(os.path.join(save_path, f"pred_{seq_idx}_both.png"))
            plt.close()
        
            if y_pred_human_count is not None:
                current_count_frame, current_count_switch, prev_human_cnt, prev_obj_cnt = draw(y_pred_human_count, y_pred_obj_count, seq_idx, prev_human=prev_human_cnt, prev_obj=prev_obj_cnt, obj_type=obj_type, switch=pred_count_switch)
                # cv2.imshow("pred_count", current_count_frame)
                cv2.imwrite(os.path.join(save_path, "pred_count_{}.png".format(seq_idx)), current_count_frame)
                cv2.imwrite(os.path.join(save_path, "pred_count_switch_{}.png".format(seq_idx)), current_count_switch)
                per_pos = draw_architecture(per_pos, is_inter=additional_output_count["transient_is_inter"][seq_idx + obs_len],
                            is_aware=additional_output_count["transient_is_aware"][seq_idx + obs_len],
                            is_switch=additional_output_count["pred_switch_label"][:, seq_idx + obs_len].item(),
                            colors=colors, object_type=object_type)
                plt.savefig(os.path.join(save_path, f"pred_{seq_idx}_count_both.png"))
                plt.close()



            key_pressed = cv2.waitKey(0) & 0xff
            if seq_idx == 0:
                key_pressed = ord('t')
            else:
                key_pressed = ord('a')
                
            if key_pressed == ord('q'):
                break
            elif key_pressed == ord('p') or key_pressed == ord('t'):    
                if key_pressed == ord('p'):
                    t_force_transient = -1
                    t_force_persistent = seq_idx
                elif key_pressed == ord('t'):
                    t_force_transient = seq_idx
                    t_force_persistent = -1                
                    
                with g.local_scope():
                    _,  (y_pred_human_count, y_pred_obj_count), additional_output_count = model(g, pred_len, 
                                                                                additional_info, is_training=False,
                                                                                is_visualize=True, t_force_persistent=t_force_persistent,
                                                                                t_force_transient=t_force_transient)
                    pred_count_switch = [SwitchDrawer(x, "pred counterfactual", color) for (x, color) in zip(additional_output_count["pred_switch_score"][:, obs_len:],
                                                                                    colors)]

                _, _, y_pred_human_count, _, _, y_pred_obj_count, _ = post_process(y_pred_human_count, y_pred_obj_count, g, scale_param=scale_param)
                
                prev_human_cnt = prev_human.copy()
                prev_obj_cnt = prev_obj.copy()

                current_count_frame, current_count_switch, prev_human_cnt, prev_obj_cnt = draw(y_pred_human_count, y_pred_obj_count, seq_idx, prev_human=prev_human_cnt, prev_obj=prev_obj_cnt, obj_type=obj_type, switch=pred_count_switch)
                # cv2.imshow("pred_count", current_count_frame)
                cv2.imwrite(os.path.join(save_path, "pred_count_{}.png".format(seq_idx)), current_count_frame)
                cv2.imwrite(os.path.join(save_path, "pred_count_switch_{}.png".format(seq_idx)), current_count_switch)

                # key_pressed = cv2.waitKey(0) & 0xff
                per_pos = draw_architecture(per_pos, is_inter=additional_output_count["transient_is_inter"][seq_idx + obs_len],
                            is_aware=additional_output_count["transient_is_aware"][seq_idx + obs_len],
                            is_switch=additional_output_count["pred_switch_label"][:, seq_idx + obs_len].item(),
                            colors=colors, object_type=object_type)
                plt.savefig(os.path.join(save_path, f"pred_{seq_idx}_count_both.png"))
                plt.close()


            

if __name__ == "__main__":
    visualize_counterfactual()