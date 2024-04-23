import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl
import dgl.function as fn
from dgl.udf import EdgeBatch, NodeBatch

from src.shared.architectures.layers.layers import build_mlp

class BaseModel(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        
        human_input_size = model_cfg.human_input_size
        human_embedding_size = model_cfg.human_embedding_size
        human_gru_size = model_cfg.human_gru_size

        obj_input_size = model_cfg.obj_input_size
        obj_embedding_size  = model_cfg.obj_embedding_size
        obj_gru_size = model_cfg.obj_gru_size
        
        hidden_size = model_cfg.hidden_size
        num_obj = model_cfg.num_obj
        dropout = model_cfg.dropout
        self.obj_gru_size = obj_gru_size
        self.human_gru_size = human_gru_size

        self.learn_interaction = model_cfg.get("learn_interaction", True)
        self.human_embedding = build_mlp([human_input_size, human_embedding_size], dropout=dropout)
        self.obj_embedding = build_mlp([obj_input_size + human_input_size + num_obj, hidden_size, obj_embedding_size], dropout=dropout)
        
        self.cfg = model_cfg

        # For computing the weights of edges
        if self.learn_interaction:
            self.attn_fn = build_mlp([obj_gru_size * 2, hidden_size, 1],
                                    activations=['leakyrelu', 'identity'],
                                    bias=False, dropout=dropout)
        else:
            self.distance_threshold = 1
            assert not self.cfg.normalize

        self.dropout = nn.Dropout(dropout)

        self.W = build_mlp([2*obj_embedding_size, hidden_size], bias=False, dropout=dropout)

        # For object
        self.obj_gru = nn.GRUCell(hidden_size, obj_gru_size)
        self.obj_out_fc = build_mlp([obj_gru_size, obj_input_size], dropout=dropout)

        
        # For human
        self.human_gru = nn.GRUCell(human_embedding_size, human_gru_size)
        self.human_out_fc = build_mlp([human_gru_size + obj_gru_size, human_input_size], dropout=dropout)
        
    def initialize_hidden(self, batch_size, hidden_dim, device):
        return torch.zeros(batch_size, hidden_dim).to(device)

    def initialize(self, graph):
        n_human = graph.num_nodes("human")
        n_obj = graph.num_nodes("obj")
        device = graph.device
        
        graph.nodes["obj"].data["h"] = self.initialize_hidden(n_obj, self.obj_gru_size, device)
        graph.nodes["human"].data["h"] = self.initialize_hidden(n_human, self.human_gru_size, device)
        
        return graph
    
    def rotate_and_translate(self, data, augment_m=None, mask=None):
        raise NotImplementedError
    
    def normalize_data(self, data, mean, std, mask):
        raise NotImplementedError
    
    def unNormalizeData(self, data, mean, std, mask):
        raise NotImplementedError
    
    def scale(self, data, additional_data):
        return NotImplementedError
    
    def unscale(self, data, additional_data):
        return NotImplementedError
    
    def preprocess(self, graph, additional_data, is_training=True):
        # 1. Perform data normalization and agumentation
        x_human, y_human = graph.nodes["human"].data["x"], graph.nodes["human"].data["y"]
        x_obj, y_obj = graph.nodes["obj"].data["x"], graph.nodes["obj"].data["y"]
        x_obj_ske, y_obj_ske = graph.nodes["obj"].data["x_ske"], graph.nodes["obj"].data["y_ske"]
        obj_is_human = graph.nodes["obj"].data["is_human"].float()
        
        if self.cfg.augment & is_training:
            x_human, augment_m = self.rotate_and_translate(x_human)
            y_human, _ = self.rotate_and_translate(y_human, augment_m=augment_m)
            x_obj, _ = self.rotate_and_translate(x_obj, augment_m=augment_m, mask=1-obj_is_human)
            y_obj, _ = self.rotate_and_translate(y_obj, augment_m=augment_m, mask=1-obj_is_human)
            x_obj_ske, _ = self.rotate_and_translate(x_obj_ske, augment_m=augment_m, mask=obj_is_human)
            y_obj_ske, _ = self.rotate_and_translate(y_obj_ske, augment_m=augment_m, mask=obj_is_human)
            
            
        if self.cfg.normalize:
            data_stats = additional_data["data_stats"]
            x_human = self.normalize_data(x_human, data_stats["human_mean"], data_stats["human_std"])
            y_human = self.normalize_data(y_human, data_stats["human_mean"], data_stats["human_std"])
            x_obj_ske = self.normalize_data(x_obj_ske, data_stats["human_mean"], data_stats["human_std"], mask=obj_is_human)
            y_obj_ske = self.normalize_data(y_obj_ske, data_stats["human_mean"], data_stats["human_std"], mask=obj_is_human)

            x_obj = self.normalize_data(x_obj, data_stats["obj_mean"], data_stats["obj_std"], mask=1-obj_is_human)
            y_obj = self.normalize_data(y_obj, data_stats["obj_mean"], data_stats["obj_std"], mask=1-obj_is_human)
        else:
            [x_human, y_human, x_obj_ske, y_obj_ske, x_obj, y_obj] = [self.scale(item, additional_data) for item in [x_human, y_human, x_obj_ske, y_obj_ske, x_obj, y_obj]]
        
        graph.nodes["human"].data["x"] = x_human
        graph.nodes["human"].data["y"] = y_human
        graph.nodes["obj"].data["x"] = x_obj
        graph.nodes["obj"].data["y"] = y_obj
        graph.nodes["obj"].data["x_ske"] = x_obj_ske
        graph.nodes["obj"].data["y_ske"] = y_obj_ske

        
        return graph
    def postprocess(self, graph, additional_data):
        y_pred_human = graph.nodes["human"].data["y_pred"]
        y_pred_obj = graph.nodes["obj"].data["y_pred"]
        data_stats = additional_data["data_stats"]
        obj_is_human = graph.nodes["obj"].data["is_human"].float()

        if self.cfg.normalize:
            y_human = self.unNormalizeData(graph.nodes["human"].data["y"], data_stats["human_mean"], data_stats["human_std"])
            y_obj = self.unNormalizeData(graph.nodes["obj"].data["y"], data_stats["obj_mean"], data_stats["obj_std"], mask=1-obj_is_human)
            
            y_pred_human = self.unNormalizeData(y_pred_human, data_stats["human_mean"], data_stats["human_std"])
            y_pred_obj = self.unNormalizeData(y_pred_obj, data_stats["obj_mean"], data_stats["obj_std"], mask=1-obj_is_human)
        else:
            [y_human, y_obj, y_pred_human, y_pred_obj] = [self.unscale(x, additional_data) for x in [graph.nodes["human"].data["y"], graph.nodes["obj"].data["y"], 
                                                                                                     y_pred_human, y_pred_obj]]

        graph.nodes["human"].data["y_pred"] = y_pred_human
        graph.nodes["human"].data["y"] = y_human
        
        graph.nodes["obj"].data["y_pred"] = y_pred_obj
        graph.nodes["obj"].data["y"] = y_obj
        
        return graph
    def prepare_node_features(self, graph, t, is_obs=True):
        if is_obs:
            graph.nodes["human"].data["current_loc"] = graph.nodes["human"].data["x"][:, t]
            graph.nodes["obj"].data["current_loc"] = graph.nodes["obj"].data["x"][:, t]
            graph.nodes["obj"].data["current_loc_ske"] = graph.nodes["obj"].data["x_ske"][:, t]
            
            graph.nodes["obj"].data["current_feat"] = torch.cat([graph.nodes["obj"].data["x_ske"][:, t],
                                                                graph.nodes["obj"].data["type"],
                                                                graph.nodes["obj"].data["current_loc"]], -1)
        else:
            graph.update_all(fn.copy_src("current_loc", "current_loc"), 
                             fn.sum("current_loc", "current_loc_ske"),
                             etype=('human', 'to', 'obj'))
            graph.nodes["obj"].data["current_feat"] = torch.cat([graph.nodes["obj"].data["current_loc_ske"],
                                                                 graph.nodes["obj"].data["type"],
                                                                 graph.nodes["obj"].data["current_loc"]], -1)

        graph.nodes["obj"].data["embed_feat"] = self.obj_embedding(graph.nodes["obj"].data["current_feat"])
        graph.nodes["human"].data["embed_feat"] = self.human_embedding(graph.nodes["human"].data["current_loc"])

        return graph
    
     
    def message_passing(self, graph, t, is_obs=True, is_start=False):
        def _msg_fn(edges: EdgeBatch) -> dict:
            srcidx, dstidx, _ = edges.edges()
            out = {
                "edge_feat": torch.cat([edges.dst["embed_feat"],
                                        edges.dst["embed_feat"] - edges.src["embed_feat"]], -1)
            }            
            
            if is_start:
                out["weight"] = srcidx == dstidx
            else:
                if self.learn_interaction:
                    out["attn_feat"] = torch.cat([edges.dst["h"],
                                                edges.dst["h"] - edges.src["h"]], -1)
                else:
                    # Obj to obj distance
                    obj_to_obj_distance = torch.norm(edges.dst["current_loc"] - edges.src["current_loc"], 2, 1)
                    is_obj_to_obj = (1 - edges.src["is_human"].float()) * (1 - edges.dst["is_human"].float())
                    
                    distance = is_obj_to_obj.float() * obj_to_obj_distance
                    out["weight"] = distance < self.distance_threshold

            return out
            
        def _reduce_fn(nodes: NodeBatch) -> dict:
            if is_start:
                weight = nodes.mailbox["weight"]
            else:
                if self.learn_interaction:
                    weight = self.attn_fn(nodes.mailbox["attn_feat"]).squeeze(-1)
                    weight = F.softmax(weight, dim=1)
                else:
                    weight = nodes.mailbox["weight"] / (torch.sum(nodes.mailbox["weight"], 1).unsqueeze(1) + 1e-5)
                                
            weight = weight.unsqueeze(-1)
            edge_feat = self.W(nodes.mailbox["edge_feat"])
            
            msg = torch.sum(weight * edge_feat, dim=1)
            
            return {
                "msg": msg
            }
            
        graph = self.prepare_node_features(graph, t, is_obs)
        graph.update_all(_msg_fn, _reduce_fn, etype=('obj', 'interacts', 'obj'))
        
        # Update hidden states
        graph.nodes["obj"].data["h"] = self.obj_gru(graph.nodes["obj"].data["msg"],
                                                       graph.nodes["obj"].data["h"])
        
        graph.nodes["human"].data["h"] = self.human_gru(graph.nodes["human"].data["embed_feat"],
                                                        graph.nodes["human"].data["h"])

        return graph
    
    def predict_location(self, graph):
        graph.update_all(fn.copy_src("h", "h_obj"), fn.sum("h_obj", "h_obj"),
                         etype=('obj', 'to', 'human'))
        
        human_h = graph.nodes["human"].data["h"]
        human_obj_h = graph.nodes["human"].data.pop("h_obj")
        
        obj_h = graph.nodes["obj"].data["h"]
        
        obj_vel = self.obj_out_fc(obj_h)
        human_vel = self.human_out_fc(torch.cat([human_h, human_obj_h], -1))
        
        graph.nodes["human"].data["current_loc"] = graph.nodes["human"].data["current_loc"] + human_vel
        graph.nodes["obj"].data["current_loc"] = graph.nodes["obj"].data["current_loc"] + obj_vel

        return graph

    def forward(self, graph, pred_len, additional_data={}, is_training=True, is_visualize=False, **kwargs):
        post_process = False
        if not is_training:
            post_process = True
            
        self.is_post_process = post_process
            
        with graph.local_scope():

            # Preprocess the data
            graph = self.preprocess(graph, additional_data, is_training)
            
            # Intialize the hidden states
            graph = self.initialize(graph)
            
            obs_len = graph.nodes["human"].data["x"].size(1)

            for i in range(obs_len):
                graph = self.message_passing(graph, i, is_obs=True, is_start=i==0)

            y_pred_human, y_pred_obj = [], []
            
            graph = self.predict_location(graph)
            y_pred_human.append(graph.nodes["human"].data["current_loc"].clone())
            y_pred_obj.append(graph.nodes["obj"].data["current_loc"].clone())
            for i in range(pred_len-1):
                graph = self.message_passing(graph, i + obs_len, is_obs=False, is_start=False)
                graph = self.predict_location(graph)
                y_pred_human.append(graph.nodes["human"].data["current_loc"].clone())
                y_pred_obj.append(graph.nodes["obj"].data["current_loc"].clone())

            graph.nodes["human"].data["y_pred"] = torch.stack(y_pred_human, 1)
            graph.nodes["obj"].data["y_pred"] = torch.stack(y_pred_obj, 1)

            if post_process:
                graph = self.postprocess(graph, additional_data)
                
            is_real_obj = (1 - graph.nodes["obj"].data["is_human"]).bool()
            is_real_obj = is_real_obj.reshape(len(is_real_obj), 1, 1)
            
            y_pred_human = graph.nodes["human"].data["y_pred"]
            y_pred_obj = graph.nodes["obj"].data["y_pred"]
            y_human, y_obj = graph.nodes["human"].data["y"], graph.nodes["obj"].data["y"]
            
            _, seq_len, dim = y_obj.size()

            if not additional_data.get("crnn_return_all", False):
                y_obj = torch.masked_select(y_obj, is_real_obj).reshape(-1, seq_len, dim)
                y_pred_obj = torch.masked_select(y_pred_obj, is_real_obj).reshape(-1, seq_len, dim)

            
        return (y_human, y_obj), (y_pred_human, y_pred_obj), {}
        
    def select_obj(self, data, weight):
        num_dim = len(data.shape)

        data_size = list(data.size())
        data_size[0] = -1
        for i in range(num_dim - 1):
            weight = weight.unsqueeze(-1)
        
        return torch.masked_select(data, weight).reshape(data_size)
