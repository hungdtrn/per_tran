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


from src.shared.data.dataset import GraphDataset, DataCollator
from ...shared.data.graph_utils import is_future_happen, smooth_weight
from .utils import CoordinateHandler
from .build import DATA_REGISTRY

def smooth_segment_label(weight, kernel, cfg):
    assert kernel % 2 == 1
    
    original_shape = weight.shape
    
    try:
        version = cfg.seg_label_version
    except:
        version = 1
    
    # window = [i for i in range(kernel // 2 + 1, kernel)] + [kernel] + [(kernel - i -1) for i in range(kernel // 2)]
    
    if version == 1:
        window = [i + 1 for i in range(kernel // 2)] + [kernel] + [(kernel // 2 - i) for i in range(kernel // 2)]
    elif version == 2:
        window = [0 for i in range(kernel // 2)] + [kernel] + [kernel - (2 * (i + 1)) for i in range(kernel // 2)]
    elif version == 3:
        window = [2 * i + 1 for i in range(kernel // 2)] + [kernel] + [kernel - (2 * (i + 1)) for i in range(kernel // 2)]

    # window = [2 * i + 1 for i in range(kernel // 2)] + [kernel] + [kernel - (2 * (i + 1)) for i in range(kernel // 2)]

    window = torch.tensor(window).to(weight.device).float() / kernel
    window = window.reshape(1, 1, window.size(-1))
    
    pad_left = weight[:, 0:1].repeat(1, kernel // 2)
    pad_right = weight[:, -2:-1].repeat(1, kernel // 2)
    weight = torch.cat([pad_left, weight, pad_right], -1)

    weight = weight.view(-1, 1, weight.shape[-1]).float()

    out = torch.nn.functional.conv1d(weight, window)
    out = torch.clamp(out, max=1.0)
    return out.reshape(original_shape)


def process_segment_label(data, cfg):
    # future_window = cfg.future_window
    # if not is_format_v2:
    #     if future_window:
    #         data = is_future_happen(data, future_window)
    # else:
    #     tmp = data.clone()
    #     if future_window:
    #         data[:, :-future_window] = tmp[:, future_window:]       
    
    try:
        is_delay_segment = cfg.is_delay_segment
    except:
        is_delay_segment = False
    
    if is_delay_segment:
        new_data = torch.zeros_like(data)
        new_data[:, 1:] = data[:, :-1]
        data = new_data
    
    score = smooth_segment_label(data, 7, cfg)
    data = score >= 0.5
    return score, data    

    
def get_segment_stats(graphs):
    class_cnt = None
    for g in graphs:
        score = g.nodes["human"].data["segment_score"]
        label = score > 0
        
        gtscore = torch.stack([1 - score, score], -1)
        gtlabel = torch.zeros_like(gtscore).scatter_(-1, label.unsqueeze(-1).long(), 1)

        gtlabel = torch.sum(gtlabel, 1)

        if class_cnt is None:
            class_cnt = torch.sum(gtlabel, 0)
        else:
            class_cnt = class_cnt + torch.sum(gtlabel, 0)
    
    label_weight = torch.max(class_cnt) / class_cnt
    print("Segment Class distribution", label_weight)
    return label_weight


class BimanualDataset(GraphDataset):
    def __init__(self, cfg: CfgNode, graph_path: str) -> None:
        super().__init__(cfg, graph_path)
        self.coordinate_handler = CoordinateHandler()
        
    def load_graph(self, cfg, graph_path):
        g = super().load_graph(cfg, graph_path)
        try:
            to_consider = []
            if len(cfg.include) > 0:
                for i in range(len(self.additional_info["video_idx"])):
                    for to_include in cfg.include:
                        if to_include in self.additional_info["video_idx"][i]:
                            to_consider.append(i)
        
                self.additional_info["video_idx"] = [self.additional_info["video_idx"][x] for x in to_consider]
                self.additional_info["frame_idx"] = [self.additional_info["frame_idx"][x] for x in to_consider]

                return [g[x] for x in to_consider]
            else:
                return g
        except:
            return g
            
    def convert_pertran_format(self, g, cfg):
        graphs = super().convert_pertran_format(g, cfg)

        for g in graphs:
            g.nodes["human"].data["segment_flag"] = g.nodes["human"].data["segment_flag"].float() * g.nodes["human"].data["switch_label"].float()
            score, flag = process_segment_label(g.nodes["human"].data["segment_flag"],
                                                cfg)

            g.nodes["human"].data["segment_label"] = flag
            g.nodes["human"].data["segment_score"] = score

        self.additional_info["segment_label_stats"] = get_segment_stats(graphs)

        return graphs

@DATA_REGISTRY.register('bimanual')
def build_bimanual_dataset(data_cfg: CfgNode, dataset_type="train", training=True):
    shuffle = data_cfg.shuffle
    collator = DataCollator()
    
    drop = data_cfg.drop
    batch_size = data_cfg.batch_size
    filename = "{}_sampling{}_min{}_obs{}_pred{}_drop{}".format(data_cfg.filename, data_cfg.sampling_rate,
                                                             data_cfg.min_num_frame,
                                                data_cfg.obs_len,
                                                data_cfg.pred_len,
                                                data_cfg.drop)
    graph_path = data_cfg.graph_path

    if dataset_type == "train":
        dataset = BimanualDataset(data_cfg, os.path.join(graph_path, "train_{}.p".format(filename)))
        dataloader = DataLoader(dataset, batch_size=batch_size,
                              collate_fn=collator, shuffle=shuffle)
    elif dataset_type == "val":
        if training:
            shuffle = False
        
        dataset = BimanualDataset(data_cfg, os.path.join(graph_path, "val_{}.p".format(filename)))
        dataloader = DataLoader(dataset, batch_size=batch_size,
                              collate_fn=collator, shuffle=shuffle)
    elif dataset_type == "test":
        if training:
            shuffle = False

        dataset = BimanualDataset(data_cfg, os.path.join(graph_path, "test_{}.p".format(filename)))
        dataloader = DataLoader(dataset, batch_size=batch_size,
                              collate_fn=collator, shuffle=False)

        
    return dataset, dataloader

