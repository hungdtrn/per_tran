import torch
from torch import device, nn
from yacs.config import CfgNode
import dgl
from dgl.udf import EdgeBatch, NodeBatch

from src.shared.architectures.layers.layers import build_mlp, HeteGAT


class Persistent(nn.Module):
    def __init__(self, model_cfg: CfgNode, **kwargs):
        super().__init__()
        
        human_input_size = model_cfg.human_input_size
        obj_input_size = model_cfg.obj_input_size
        
        human_gru_size = model_cfg.human_gru_size
        obj_gru_size = model_cfg.obj_gru_size
        
        hidden_size = model_cfg.hidden_size
        
        attention_act = model_cfg.attention_act

        human_embedding_size = model_cfg.get("human_embedding_size", hidden_size)
        obj_embedding_size = model_cfg.get("obj_embedding_size", hidden_size)
        self.has_skip_connection = model_cfg.get("has_skip_connection", False)
        self.has_switch_skip_connection = model_cfg.get("has_switch_skip_connection", False)        
        num_head = model_cfg.num_head
        hete_sem_act = model_cfg.get("hete_sem_act", 'tanh')
        
        
        self.human_gat = HeteGAT({
            ('obj', 'interacts', 'human'): [obj_embedding_size + obj_gru_size, human_embedding_size + human_gru_size],
            ('human', 'interacts', 'human'): [human_embedding_size + human_gru_size, human_embedding_size + human_gru_size]
        }, hidden_size, [hidden_size], attention_act, num_head=num_head, sem_act=hete_sem_act)
        
        self.obj_gat = HeteGAT({
            ('human', 'interacts', 'obj'): [human_embedding_size + human_gru_size, obj_embedding_size + obj_gru_size],
            ('obj', 'interacts', 'obj'): [obj_embedding_size + obj_gru_size, obj_embedding_size + obj_gru_size]
        }, hidden_size, [hidden_size], attention_act, num_head=num_head, sem_act=hete_sem_act)

        self.human_gru_size = human_gru_size
        self.obj_gru_size = obj_gru_size
        
        human_gru_inp_size = hidden_size +  human_embedding_size

        self.has_cross_graph_msg = model_cfg.has_cross_graph_msg
        if self.has_cross_graph_msg:
            human_gru_inp_size += hidden_size
            self.cross_graph_msg_embedding = build_mlp([human_gru_size, hidden_size])
        
        
        self.human_gru = nn.GRUCell(human_gru_inp_size, human_gru_size)
        self.obj_gru = nn.GRUCell(hidden_size + obj_embedding_size, obj_gru_size)
        
        if not self.has_skip_connection:
            self.human_output_embedding = build_mlp([human_gru_size, human_input_size])
            self.obj_output_embedding = build_mlp([obj_gru_size, obj_input_size])
        else:
            self.human_output_embedding = build_mlp([human_gru_size + human_gru_inp_size, human_input_size])
            self.obj_output_embedding = build_mlp([obj_gru_size + hidden_size + obj_embedding_size, obj_input_size])

        if self.has_switch_skip_connection:
            self.switch_skip_mlp = build_mlp([human_gru_inp_size, hidden_size])



    def initialize(self, init_fn, graph: dgl.DGLGraph):
        num_human, num_obj = graph.num_nodes("human"), graph.num_nodes("obj")
        device = graph.device
        graph.nodes["human"].data["h_gru"] = init_fn(num_human,
                                                     self.human_gru_size,
                                                     device=device) 
        
        graph.nodes["obj"].data["h_gru"] = init_fn(num_obj,
                                                    self.obj_gru_size,
                                                    device=device) 

        return graph
    
    def prepare_node_features(self, graph: dgl.DGLGraph, t, obs):                
        return graph

    def message_passing(self, graph, t, is_obs):        
        human_embed_feat = graph.nodes["human"].data["embed_feat"]
        obj_embed_feat = graph.nodes["obj"].data["embed_feat"]
        
        human_msg = self.human_gat(graph)
        obj_msg = self.obj_gat(graph)        
        
        human_h = graph.nodes["human"].data["h_gru"]
        obj_h = graph.nodes["obj"].data["h_gru"]
        
        human_gru_inp = torch.cat([human_embed_feat, human_msg], -1)
        if self.has_cross_graph_msg:
            human_gru_inp = torch.cat([human_gru_inp,
                                       graph.nodes["human"].data.pop("center_human_msg")], -1)
            
        human_h = self.human_gru(human_gru_inp, human_h)
        
        obj_gru_inp = torch.cat([obj_embed_feat, obj_msg], -1)
        obj_h = self.obj_gru(obj_gru_inp, obj_h)

        graph.nodes["human"].data["h_gru"] = human_h
        graph.nodes["obj"].data["h_gru"] = obj_h
        
        if self.has_cross_graph_msg:
            graph.nodes["center"].data["human_center_msg"] = self.cross_graph_msg_embedding(human_h)
            
        if self.has_skip_connection:
            graph.nodes["human"].data["gru_inp"] = human_gru_inp
            graph.nodes["obj"].data["gru_inp"] = obj_gru_inp
            
        if self.has_switch_skip_connection:
            graph.nodes["human"].data["gru_inp"] = self.switch_skip_mlp(human_gru_inp)
            
        return graph
        
    def predict(self, graph: dgl.DGLGraph):
        current_human = graph.nodes["human"].data["current_persistent_loc"]
        current_obj = graph.nodes["obj"].data["current_persistent_loc"]
        
        if self.has_skip_connection:
            human_mlp_inp = torch.cat([graph.nodes["human"].data["h_gru"],
                                       graph.nodes["human"].data.pop("gru_inp")], -1)
            obj_mlp_inp = torch.cat([graph.nodes["obj"].data["h_gru"],
                                       graph.nodes["obj"].data.pop("gru_inp")], -1)
        else:
            human_mlp_inp = graph.nodes["human"].data["h_gru"]
            obj_mlp_inp = graph.nodes["obj"].data["h_gru"]
        
        current_human_vel = self.human_output_embedding(human_mlp_inp)
        current_obj_vel = self.obj_output_embedding(obj_mlp_inp)
        
        return current_human + current_human_vel, current_obj + current_obj_vel