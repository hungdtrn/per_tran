from omegaconf.omegaconf import OmegaConf
import torch
from torch import nn
from yacs.config import CfgNode
import dgl
import dgl.function as fn
from dgl.udf import EdgeBatch, NodeBatch
import torch.nn.functional as F

from src.shared.architectures.layers.layers import build_mlp, attention, GAT
from library.utils import masked_set, straight_through_boolean

class SwitchModule(nn.Module):
    def __init__(self, model_cfg: CfgNode):
        super().__init__()
        
        self.d_inter = model_cfg.d_inter
        self.switch_type = model_cfg.switch_type
        hidden_size = model_cfg.hidden_size
        
        switch_hidden_size = model_cfg.get("switch_hidden_size", hidden_size)
        self.has_switch_skip_connection = model_cfg.get("has_switch_skip_connection", False)
        switch_act = model_cfg.get("switch_act", "elu")
        gru_size = model_cfg.persistent.human_gru_size

        if "rule" not in self.switch_type:
            if "distance" in self.switch_type:
                self.human_distance_beta = nn.parameter.Parameter(torch.FloatTensor(1, 1))
                self.human_distance_beta.data.fill_(model_cfg.initial_distance_beta)
            
            if "temp" in self.switch_type:
                if self.has_switch_skip_connection:
                    self.temp_mlp = build_mlp([gru_size + hidden_size, switch_hidden_size, 1], [switch_act, 'sigmoid'])
                else:
                    self.temp_mlp = build_mlp([gru_size, switch_hidden_size, 1], [switch_act, 'sigmoid'])

        self.visualize = False
        self.visualize_data = {}

    def initialize(self, init_fn, graph: dgl.DGLGraph):
        self.visualize_data = {}
        return graph

    def update_visualization_data(self, name, data):
        # print(temp_score)

        if self.visualize:
            if name not in self.visualize_data:
                self.visualize_data[name] = []
                
            self.visualize_data[name].append(data)
                
    def prepare_node_features(self, graph: dgl.DGLGraph, t, is_obs):                
        return graph
    
    def compute_switch_distance_score(self, graph: dgl.DGLGraph):
        switch_output = {}
        
        graph.update_all(fn.copy_src("distance_to_center", "distance"),
                            fn.min("distance", "min_distance"), etype=("leaf", "interacts", "center"))
        min_distance = graph.nodes["center"].data.pop("min_distance").unsqueeze(-1)                 
        distance_weight = self.human_distance_beta ** 2
        distance_score = torch.exp(-distance_weight * min_distance)
                            
        distance_score = distance_score.squeeze(-1)
        switch_output["pred_distance_score"] = distance_score
                
        return switch_output

    def compute_switch_temporal_score(self, graph: dgl.DGLGraph):
        switch_output = {}
        if self.has_switch_skip_connection:
            temp_score = self.temp_mlp(torch.cat([graph.nodes["human"].data["h_gru"],
                                                  graph.nodes["human"].data.pop("gru_inp")], -1))
        else:
            temp_score = self.temp_mlp(graph.nodes["human"].data["h_gru"])

            
        temp_score = temp_score.squeeze(-1)

        self.update_visualization_data("temp_score", temp_score)
        switch_output["pred_temp_score"] = temp_score
        return switch_output

    
    def message_passing(self, graph, t, is_obs):        
        switch_output = {}
        if "rule" in self.switch_type:
            graph.update_all(fn.copy_src("distance_to_center", "distance"),
                                    fn.min("distance", "min_distance"), etype=("leaf", "interacts", "center"))
            min_distance = graph.nodes["center"].data.pop("min_distance")
            switch_score = (min_distance <= self.d_inter).float()
        else:
            switch_score = None
            switch_output = {}
            if "distance" in self.switch_type:
                switch_output.update(self.compute_switch_distance_score(graph))
                switch_score = switch_output["pred_distance_score"]
                
            if "temp" in self.switch_type:
                switch_output.update(self.compute_switch_temporal_score(graph))
                if switch_score is None:
                    switch_score = switch_output["pred_temp_score"]
                else:
                    switch_score = switch_score * switch_output["pred_temp_score"]

        switch_label = straight_through_boolean(switch_score >= 0.5,
                                                switch_score)

        switch_output["pred_switch_score"] = switch_score
        switch_output["pred_switch_label"] = switch_label

        return graph, switch_output
    
    def get_distance_beta(self):
        if "distance" in self.switch_type and "aggregate" not in self.switch_type:
            return self.human_distance_beta
        else:
            return None
          
            
