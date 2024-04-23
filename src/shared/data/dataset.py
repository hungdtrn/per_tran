from typing import List
import pickle
import dgl
import copy

import os
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from yacs.config import CfgNode
from omegaconf import DictConfig, OmegaConf

from .graph_utils import process_switch, process_switch_v2
from .utils import CoordinateHandler

def compare_config(saved_config, load_config):
    for k in saved_config.keys():
        if "path" in k:
            continue
        
        if k not in load_config:
            print("{} not found".format(k))
            return False
        
        if saved_config[k] != load_config[k]:
            print("{} is different: saved: {}, now {}".format(k, saved_config[k], load_config[k]))
            return False
        
    return True

def convert_crnn_format(graphs):
    def _msg_fn(edges):
        return {
            "x": edges.src["x"],
            "y": edges.src["y"]
        }
        
    def _reduce_fn(nodes):
        x, y = nodes.mailbox["x"], nodes.mailbox["y"]
        
        assert x.size(1) == 1
        assert y.size(1) == 1
                
        return {
            "x_ske": x.squeeze(1),
            "y_ske": y.squeeze(1),
        }
        
    new_graphs = []
    for i in range(len(graphs)):
        g = graphs[i]
        obj_data = g.nodes["obj"].data
        
        n_human = g.num_nodes("human")
        n_obj = g.num_nodes("obj")
        human_type = g.nodes["human"].data["type"]
        
        num_obj_obj_edges = g.num_edges(("obj", "interacts", "obj"))
        graph_data = {
            ('obj', 'interacts', 'obj'): [[], []],
            ('human', 'to', 'obj'): [[], []],
            ('obj', 'to', 'human'): [[], []],
        }
        
        obj_data = g.nodes["obj"].data
        
        new_obj_data = {}
        new_human_data = {}
        
        
        for i in range(n_obj):
            for j in range(n_obj):
                graph_data[('obj', 'interacts', 'obj')][0].append(i)
                graph_data[('obj', 'interacts', 'obj')][1].append(j)

        
        for i in range(n_human):
            # Add edges to convert from human to object bounding box
            graph_data[('human', 'to', 'obj')][0].append(i)
            graph_data[('human', 'to', 'obj')][1].append(i + n_obj)
            graph_data[('obj', 'to', 'human')][0].append(i + n_obj)
            graph_data[('obj', 'to', 'human')][1].append(i)

            # Add edges between new human box with other objects
            for j in range(n_obj):
                # From the human box to the object
                graph_data[('obj', 'interacts', 'obj')][0].append(i + n_obj)
                graph_data[('obj', 'interacts', 'obj')][1].append(j)
                
                # From the object to the human box
                graph_data[('obj', 'interacts', 'obj')][0].append(j)
                graph_data[('obj', 'interacts', 'obj')][1].append(i + n_obj)

            # Add edges between human box
            for j in range(n_human):
                graph_data[('obj', 'interacts', 'obj')][0].append(i + n_obj)
                graph_data[('obj', 'interacts', 'obj')][1].append(j + n_obj)
        
        for data_name, data in obj_data.items():
            if data_name == "x" or data_name == "y" or data_name == "mask":
                first_row = data[0]
                shape = [n_human] + list(first_row.shape)
                new_obj_data[data_name] = torch.cat([data, torch.zeros(shape, dtype=first_row.dtype)])
            elif data_name == "type":
                new_obj_data[data_name] = torch.cat([data, human_type])
        
        new_human_data["x"] = g.nodes["human"].data["x"]
        new_human_data["y"] = g.nodes["human"].data["y"]

        if "mask" in g.nodes["human"].data:
            new_human_data["mask"] = g.nodes["human"].data["mask"]
        
        
        for k, v in graph_data.items():
            graph_data[k] = (torch.tensor(v[0]), torch.tensor(v[1]))
        
        g = dgl.heterograph(graph_data)
        new_num_obj_obj_edges = g.num_edges(("obj", "interacts", "obj"))

        assert new_num_obj_obj_edges == (num_obj_obj_edges + 2 * n_human * n_obj + n_human**2)
        
        for dname, d in new_obj_data.items():
            g.nodes["obj"].data[dname] = d
        for dname, d in new_human_data.items():
            g.nodes["human"].data[dname] = d

        g.update_all(_msg_fn, _reduce_fn, etype=('human', 'to', 'obj'))
        g.nodes["obj"].data["is_human"] = torch.tensor([0 for i in range(n_obj)] + [1 for i in range(n_human)]).float()
        
        new_graphs.append(g)
       
    return new_graphs

def convert_pertran_format(graphs, cfg):
    new_graphs = []
    print(cfg.future_window)
    try:
        is_format_v2 = cfg.is_per_tran_v2
    except:
        is_format_v2 = False
        
    if is_format_v2:
        print("Using per_tran v2")
        
    for i in range(len(graphs)):
        g = graphs[i]
        
        if not is_format_v2:
            switch_fn = process_switch
        else:
            switch_fn = process_switch_v2
        
        score, flag = switch_fn(g.nodes["human"].data["switch_flag"],
                                    cfg.future_window,
                                    cfg.weight_gauss_kernel,
                                    cfg.weight_gauss_std)
        
        g.nodes["human"].data["switch_label"] = flag
        g.nodes["human"].data["switch_score"] = score

        new_graphs.append(g)
    
    return new_graphs

def remove_self_loop(g):
    for etype in g.canonical_etypes:
        if etype[0] == etype[-1]:
            g = dgl.remove_self_loop(g, etype=etype)
    
    return g

def convert_duality_format(graphs, cfg):
    # remove self loops:
    new_graphs = []

    for i in range(len(graphs)):
        g = graphs[i]
        
        g = remove_self_loop(g)
        
        score, flag = process_switch(g.nodes["human"].data["switch_flag"],
                                     cfg.future_window,
                                     cfg.weight_gauss_kernel,
                                     cfg.weight_gauss_std)
        g.nodes["human"].data["switch_label"] = flag
        g.nodes["human"].data["switch_score"] = score

        new_graphs.append(g)
    return new_graphs

def get_switch_stats(graphs):
    class_cnt = None
    for g in graphs:
        score = g.nodes["human"].data["switch_score"]
        label = g.nodes["human"].data["switch_label"]
        
        gtscore = torch.stack([1 - score, score], -1)
        gtlabel = torch.zeros_like(gtscore).scatter_(-1, label.unsqueeze(-1).long(), 1)

        gtlabel = gtlabel.view(-1, 2)
        
        if class_cnt is None:
            class_cnt = torch.sum(gtlabel, 0)
        else:
            class_cnt = class_cnt + torch.sum(gtlabel, 0)
    
    label_weight = torch.max(class_cnt) / class_cnt
    return label_weight

class GraphDataset(Dataset):
    def __init__(self, cfg: CfgNode, graph_path: str) -> None:
        super().__init__()
        
        self.coordinate_handler = CoordinateHandler()
        g = self.load_graph(cfg, graph_path)
        g = self.convert_to_desired_format(cfg, g)
        
        self.graphs = g

    def load_graph(self, cfg, graph_path):
        with open(graph_path, "rb") as f:
            raw_data = pickle.load(f)
        
        self.drop = cfg.drop    
        g = raw_data["graphs"]
        
        self.data_cfg = raw_data["cfg"]
        self.additional_info = raw_data["additional_info"]        
        
        if not compare_config(self.data_cfg, 
                              yaml.safe_load(OmegaConf.to_yaml(cfg))):
            print("saved config", self.data_cfg)
            print("data config", yaml.safe_load(OmegaConf.to_yaml(cfg)))
            raise Exception("Saved config must match load config")

        return g

    def convert_pertran_format(self, g, cfg):
        return convert_pertran_format(g, cfg)

    def convert_to_desired_format(self, cfg, g):
        self.crnn_format = cfg.get("crnn_format", False)
        self.duality_format = cfg.get("duality_format", False)
                    
        
        if self.crnn_format:
            print("Converting to crnn format")
            g = convert_crnn_format(g)
        elif self.duality_format:
            print("Converting to duality format")

            g = convert_duality_format(g, cfg)
            self.additional_info["switch_label_stats"] =  get_switch_stats(g)            
        else:
            print("Converting to per tran format")

            g = self.convert_pertran_format(g, cfg)
            self.additional_info["switch_label_stats"] =  get_switch_stats(g)

        return g

    def __getitem__(self, index):
        # g = self.clone(self.graphs[index], index)        
        return self.graphs[index], index
        
    def __len__(self):
        return len(self.graphs)
        
class DataCollator(object):
    def __init__(self) -> None:
        super().__init__()
                
    def __call__(self, batch: List) -> List:
        graphs, idx = map(list, zip(*batch))
        return dgl.batch(graphs), idx
