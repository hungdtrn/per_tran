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
from .utils import CoordinateHandler
from .build import DATA_REGISTRY

class WBHMDataset(GraphDataset):
    def __init__(self, cfg: CfgNode, graph_path: str) -> None:
        super().__init__(cfg, graph_path)
        self.coordinate_handler = CoordinateHandler()

@DATA_REGISTRY.register('wbhm')
def build_wbhm_dataset(data_cfg: CfgNode, dataset_type="train", training=True):
    shuffle = data_cfg.shuffle
    collator = DataCollator()
    
    drop = data_cfg.drop
    batch_size = data_cfg.batch_size
    filename = "drop{}_{}".format(drop, data_cfg.filename)
    graph_path = data_cfg.graph_path

    if dataset_type == "train":
        dataset = GraphDataset(data_cfg, os.path.join(graph_path, "train_{}.p".format(filename)))
        dataloader = DataLoader(dataset, batch_size=batch_size,
                              collate_fn=collator, shuffle=shuffle)
    elif dataset_type == "val":
        if training:
            shuffle = False
        
        dataset = GraphDataset(data_cfg, os.path.join(graph_path, "val_{}.p".format(filename)))
        dataloader = DataLoader(dataset, batch_size=batch_size,
                              collate_fn=collator, shuffle=shuffle)
    elif dataset_type == "test":
        if training:
            shuffle = False

        dataset = GraphDataset(data_cfg, os.path.join(graph_path, "test_{}.p".format(filename)))
        dataloader = DataLoader(dataset, batch_size=batch_size,
                              collate_fn=collator, shuffle=False)

        
    return dataset, dataloader


    