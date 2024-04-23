
from queue import PriorityQueue
import torch
from torch import nn
from yacs.config import CfgNode
import dgl
import dgl.function as fn
from dgl.udf import EdgeBatch, NodeBatch
from src.shared.architectures.layers.layers import GAT, MultiheadGat, build_mlp, attention
from library.utils import masked_set



class Transient(nn.Module):
    def __init__(self, model_cfg: CfgNode, **kwargs):
        super().__init__()
        
        self.d_aware = model_cfg.d_aware
        self.d_inter = model_cfg.d_inter
        
        num_obj = model_cfg.num_obj
        activation = model_cfg.activation
        attention_act = model_cfg.attention_act
        
        hidden_size = model_cfg.hidden_size
        center_gru_size = model_cfg.center_gru_size
        leaf_gru_size = model_cfg.leaf_gru_size
        
        self.has_cross_graph_msg = model_cfg.has_cross_graph_msg
            

        human_embedding_size = model_cfg.get("human_embedding_size", hidden_size)
        obj_embedding_size = model_cfg.get("object_embedding_size", hidden_size)
        
        self.grad_flow = model_cfg.get("grad_flow", False)
        self.has_skip_connection = model_cfg.get("has_skip_connection", False)
        self.use_multihead = model_cfg.get("use_multihead", False)


        self.center_inp_embedding = build_mlp([model_cfg.human_input_size, human_embedding_size], [activation])
        self.leaf_loc_embedding = build_mlp([model_cfg.obj_input_size, obj_embedding_size], [activation])
        self.type_embedding = build_mlp([num_obj, obj_embedding_size], [activation])
        self.leaf_inp_embedding = build_mlp([obj_embedding_size + obj_embedding_size, obj_embedding_size], [activation])

        self.hidden_size = hidden_size
        modify_gat = model_cfg.get("modify_gat", True)

        if self.use_multihead:
            self.leaf_center_gat = MultiheadGat(num_heads=model_cfg.num_head,
                                                etype=('leaf', 'interacts', 'center'), 
                                                src_dim=obj_embedding_size + leaf_gru_size,
                                                dst_dim=human_embedding_size + center_gru_size,
                                                out_dim=hidden_size,
                                                attn_hidden_dim=[hidden_size],
                                                attn_act=attention_act)

        else:
            self.leaf_center_gat = GAT(('leaf', 'interacts', 'center'), 
                                    src_dim=obj_embedding_size + leaf_gru_size,
                                    dst_dim=human_embedding_size + center_gru_size,
                                    out_dim=hidden_size,
                                    attn_hidden_dim=[hidden_size],
                                    attn_act=attention_act,
                                    modified=modify_gat)
        
        self.center_leaf_msg_embedding = build_mlp([center_gru_size + human_embedding_size, hidden_size], [activation])
        
        center_gru_inp_size = human_embedding_size + hidden_size
        leaf_gru_inp_size = obj_embedding_size + hidden_size
    
        center_gru_size = model_cfg.center_gru_size
        leaf_gru_size = model_cfg.leaf_gru_size
        
        if self.has_cross_graph_msg:
            center_gru_inp_size += hidden_size
            self.cross_graph_msg_embedding = build_mlp([center_gru_size, hidden_size], [activation])
            self.center_human_msg_size = hidden_size

        self.center_gru_size = center_gru_size
        self.leaf_gru_size = leaf_gru_size 
            
        self.leaf_gru = nn.GRUCell(leaf_gru_inp_size, leaf_gru_size)
                    
        self.center_gru = nn.GRUCell(center_gru_inp_size, center_gru_size)

        if not self.has_skip_connection:
            self.center_output_embedding = build_mlp([center_gru_size, model_cfg.human_input_size])
            self.leaf_output_embedding = build_mlp([leaf_gru_size, model_cfg.obj_input_size])
        else:
            self.center_output_embedding = build_mlp([center_gru_size + center_gru_inp_size, model_cfg.human_input_size])
            self.leaf_output_embedding = build_mlp([leaf_gru_size + leaf_gru_inp_size, model_cfg.obj_input_size])

        self.visualize = False
        self.weights = {}
        self.coordinate_handler = None       


    def initialize(self, init_fn, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        graph.nodes["center"].data["h_gru"] = init_fn(graph.num_nodes("center"), 
                                                      self.center_gru_size, 
                                                      graph.device)
        graph.nodes["leaf"].data["h_gru"] = init_fn(graph.num_nodes("leaf"), 
                                                      self.leaf_gru_size, 
                                                      graph.device)
                
        if self.has_cross_graph_msg:
            graph.nodes["human"].data["center_human_msg"] = torch.zeros(graph.num_nodes("center"), self.center_human_msg_size).to(graph.device)
        
        if self.visualize:
            self.is_inter = []
        
        return graph
    
    def prepare_node_features(self, graph: dgl.DGLGraph, t, is_obs) -> dgl.DGLGraph:
        graph.nodes["center"].data["embed_feat"] = self.center_inp_embedding(graph.nodes["center"].data["current_transient_loc"])
        graph.nodes["leaf"].data["embed_feat"] = self.leaf_inp_embedding(torch.cat([
            self.type_embedding(graph.nodes["leaf"].data["type"]),
            self.leaf_loc_embedding(graph.nodes["leaf"].data["current_transient_loc"])
        ], -1))

        return graph
    
    def determine_structure(self, graph, current_switch_score, current_switch_label,
                            is_all_transient):
        switch_label = current_switch_label[graph.nodes["leaf"].data["center_idx"]]
        switch_score = current_switch_score[graph.nodes["leaf"].data["center_idx"]]

        distance = graph.nodes["leaf"].data["distance_to_center"]
        is_aware = ((distance <= self.d_aware) & switch_label.bool()).float()
        is_inter = ((distance <= self.d_inter) & switch_label.bool()).float()

        if is_all_transient:

            is_inter = ((distance == torch.min(distance)) | (distance <= self.d_inter)).float()
            is_aware = (distance <= self.d_aware).float()
            # is_inter = torch.ones(graph.num_nodes("leaf")).to(graph.device)
                        
        graph.nodes["leaf"].data["is_aware"] = is_aware
        graph.nodes["leaf"].data["is_inter"] = is_inter
        
        if self.visualize:
            self.is_inter.append(is_inter.clone())
        
        # print(is_aware, is_inter, self.d_aware, self.d_inter)            
        graph.nodes["leaf"].data["confidence"] = switch_score.detach() * is_inter

        return graph
    
    def send_msg_to_persistent(self, graph, current_switch_label):
        msg = self.cross_graph_msg_embedding(graph.nodes["center"].data["h_gru"])
        
        # Zero when not active
        msg = masked_set(current_switch_label > 0,
                         msg,
                         torch.zeros_like(msg))
        
        graph.nodes["human"].data["center_human_msg"] = msg
        
        return graph
        
    def message_passing(self, graph, t, obs, current_switch_output,
                        is_all_transient):
        
        current_switch_score = current_switch_output["pred_switch_score"]
        current_switch_label = current_switch_output["pred_switch_label"]
        
        graph = self.determine_structure(graph, current_switch_score, current_switch_label,
                                         is_all_transient)
        
        leaf_nodes = graph.nodes["leaf"]
        center_nodes = graph.nodes["center"]
        
        # Update the center first
        center_candidate = current_switch_label.float()
        center_h = center_nodes.data["h_gru"]
        leaf_h = leaf_nodes.data["h_gru"]
        
        if torch.any(center_candidate > 0):                    
            leaf_center_msg = self.leaf_center_gat(graph,
                                                srch=torch.cat([leaf_nodes.data["embed_feat"],
                                                                leaf_nodes.data["h_gru"]], -1),
                                                dsth=torch.cat([center_nodes.data["embed_feat"],
                                                                center_nodes.data["h_gru"]], -1),
                                                mask_fn=lambda edges: edges.src["is_aware"])
            
            
            center_gru_inp = torch.cat([center_nodes.data["embed_feat"],
                                        leaf_center_msg], -1)
                        
            if self.has_cross_graph_msg:
                center_gru_inp = torch.cat([center_gru_inp,
                                            center_nodes.data.pop("human_center_msg")], -1)
                    
            new_center_h = self.center_gru(center_gru_inp, center_h)
            
            center_h = masked_set(center_candidate > 0,
                                new_center_h,
                                center_h,
                                grad_from=center_candidate)
            
            if self.has_skip_connection:
                graph.nodes["center"].data["gru_inp"] = center_gru_inp

        # Then update the leaves
        leaf_candidate = leaf_nodes.data["is_inter"]

        if torch.any(leaf_candidate > 0):
            center_leaf_msg = self.center_leaf_msg_embedding(torch.cat([
                center_nodes.data["embed_feat"][leaf_nodes.data["center_idx"]],
                center_nodes.data["h_gru"][leaf_nodes.data["center_idx"]]
            ], -1))
            leaf_gru_inp = torch.cat([leaf_nodes.data["embed_feat"],
                                    center_leaf_msg], -1)
            
            
            new_leaf_h = self.leaf_gru(leaf_gru_inp, leaf_h)

            leaf_h = masked_set(leaf_candidate > 0,
                                new_leaf_h,
                                leaf_h)
            
            if self.has_skip_connection:
                graph.nodes["leaf"].data["gru_inp"] = leaf_gru_inp

            
        graph.nodes["center"].data["h_gru"] = center_h
        graph.nodes["leaf"].data["h_gru"] = leaf_h

        if self.has_cross_graph_msg:
            graph = self.send_msg_to_persistent(graph, current_switch_label)

        return graph
        
    def convert_center_prediction(self, graph: dgl.DGLGraph, center_pred):
        base_data = graph.nodes["human"].data["current_persistent_loc"]
        return self.coordinate_handler.convert_center_to_human(center_pred, base_data)
    
    def convert_leaf_prediction(self, graph, leaf_pred):
        base_data = graph.nodes["human"].data["current_persistent_loc"]        
        base_data = base_data[graph.nodes["leaf"].data["center_idx"]]
        
        graph.nodes["leaf"].data["converted_pred"] = self.coordinate_handler.convert_leaf_to_obj(leaf_pred, base_data,
                                                                                                 self.grad_flow)
    
        def _msg_fn(edges: EdgeBatch) -> dict:
            return {
                "converted_pred": edges.src["converted_pred"],
                "confidence": edges.src["confidence"],
            }
            
        def _reduce_fn(nodes: NodeBatch) -> dict:
            confidence = nodes.mailbox["confidence"]
            converted_pred = nodes.mailbox["converted_pred"]
            
            if confidence.size(1) == 1:
                max_confidence = confidence.squeeze(1)
                converted_pred = converted_pred.squeeze(1)
            else:
                max_confidence, max_idx = torch.max(confidence, 1)
                mask = torch.zeros_like(confidence).scatter_(1, max_idx.unsqueeze(-1), 1.0)
                mask = mask.unsqueeze(-1).bool()

                dim = converted_pred.size(-1)
                converted_pred = torch.masked_select(converted_pred, mask).reshape(-1, dim)

            return {
                "converted_pred": converted_pred,
                "confidence": max_confidence
            }

        graph.update_all(_msg_fn, _reduce_fn, etype=('leaf', 'to', 'obj'))
        
        graph.nodes["leaf"].data.pop("converted_pred")
        graph.nodes["leaf"].data.pop("confidence")

        return graph.nodes["obj"].data.pop("converted_pred"), graph.nodes["obj"].data.pop("confidence")
    
        
    def predict(self, graph: dgl.DGLGraph, current_switch_output):
        current_switch_label = current_switch_output["pred_switch_label"]
        
        current_center = graph.nodes["center"].data["current_transient_loc"]
        current_leaf = graph.nodes["leaf"].data["current_transient_loc"]
        
        center_candidate = current_switch_label
        leaf_candidate = graph.nodes["leaf"].data["is_inter"].float()

        if torch.any(center_candidate > 0):
            if not self.has_skip_connection:
                center_mlp_inp = graph.nodes["center"].data["h_gru"]
            else:
                center_mlp_inp = torch.cat([graph.nodes["center"].data["h_gru"],
                                            graph.nodes["center"].data.pop("gru_inp")], -1)
            
            current_center_vel = self.center_output_embedding(center_mlp_inp)
            current_center_vel = masked_set(center_candidate > 0,
                                current_center_vel,
                                torch.zeros_like(current_center_vel))
        else:
            current_center_vel = torch.zeros_like(current_center)
            
            
        if torch.any(leaf_candidate > 0):
            if not self.has_skip_connection:
                leaf_mlp_inp = graph.nodes["leaf"].data["h_gru"]
            else:
                leaf_mlp_inp = torch.cat([graph.nodes["leaf"].data["h_gru"],
                                        graph.nodes["leaf"].data.pop("gru_inp")], -1)
            

            current_leaf_vel = self.leaf_output_embedding(leaf_mlp_inp)

            current_leaf_vel = masked_set(leaf_candidate > 0,
                                current_leaf_vel,
                                torch.zeros_like(current_leaf_vel))
        else:
            current_leaf_vel = torch.zeros_like(current_leaf)
                

        human_loc = self.convert_center_prediction(graph, current_center + current_center_vel)
        obj_loc, confidence = self.convert_leaf_prediction(graph, current_leaf + current_leaf_vel)

        return human_loc, obj_loc, confidence

