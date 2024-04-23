from yacs.config import CfgNode
import torch
import dgl
import dgl.function as fn

from src.bimanual.architectures.per_tran.utils import compute_human_center_wh_loc, compute_human_center_wh_vel, compute_obj_center_wh_loc, compute_obj_center_wh_vel
from src.bimanual.architectures.per_tran.utils import compute_human_vel, compute_obj_vel

from .build import TRANSIENT_REGISTRY
from src.shared.architectures.per_tran import BaseTransient
from src.shared.architectures.layers.layers import build_mlp
from library.utils import masked_set


class Transient(BaseTransient):
    def __init__(self, model_cfg: CfgNode, **kwargs):
        super().__init__(model_cfg, **kwargs)
        self.is_center_wh_format = model_cfg.get("is_center_wh_format", False)
        self.consider_iou = model_cfg.get("consider_iou", True)
        self.is_select_top_k_leaf = model_cfg.get("is_select_topk_leaf", False)
        self.has_adaptive_structure = model_cfg.has_adaptive_structure
        self.k = model_cfg.get("k", 2)

        self.iou_thresh = model_cfg.iou_thresh
        self.pred_vel = model_cfg.pred_vel

        center_gru_size = model_cfg.center_gru_size
        leaf_gru_size = model_cfg.leaf_gru_size

        del self.center_output_embedding

        if self.is_center_wh_format:        
            del self.leaf_output_embedding            

            # Predict the center and the width and height of the bounding boxes
            self.leaf_out_center_embedding = build_mlp([leaf_gru_size, model_cfg.hidden_size, 2], ['elu', 'identity'])
            self.leaf_out_wh_embedding = build_mlp([leaf_gru_size, model_cfg.hidden_size, 2], ['elu', 'identity'])

            self.center_out_arm_embedding = build_mlp([center_gru_size, model_cfg.hidden_size, 6], ['elu', 'identity'])
            self.center_out_center_embedding = build_mlp([center_gru_size, model_cfg.hidden_size, 2], ['elu', 'identity'])
            self.center_out_wh_embedding = build_mlp([center_gru_size, model_cfg.hidden_size, 2], ['elu', 'identity'])
        else:
            self.center_out_arm_embedding = build_mlp([center_gru_size, 6])
            self.center_out_box_embedding = build_mlp([center_gru_size, 4])

    def select_topk_leaf(self, graph, distance, iou):
        is_inter = torch.zeros_like(distance)
        graph.nodes["leaf"].data["score"] = (1 / (distance + 1e-3))
        if self.consider_iou:
            graph.nodes["leaf"].data["score"] = graph.nodes["leaf"].data["score"] * (iou >= self.iou_thresh).float()
        
        def _select_top_k_edge_fn(edges):
            return {"score": edges.src["score"],
                    "src_idx": edges.edges()[0]}
            
        def _select_top_k_node_fn(nodes):
            _, top_k_indices = torch.topk(nodes.mailbox["score"], k=self.k,  dim=1)
            topk_inter = torch.zeros_like(nodes.mailbox["src_idx"]).scatter_(1, top_k_indices, 1.0)
            indice = nodes.mailbox["src_idx"].flatten()
            is_inter[indice] = topk_inter.flatten().float()
            return {}
            
        graph.update_all(_select_top_k_edge_fn, _select_top_k_node_fn, 
                         etype=("leaf", "interacts", "center"))
        return is_inter

    def determine_structure(self, graph, current_switch_score, current_switch_label, is_all_transient):
        switch_label = current_switch_label[graph.nodes["leaf"].data["center_idx"]]
        switch_score = current_switch_score[graph.nodes["leaf"].data["center_idx"]]

        distance = graph.nodes["leaf"].data["distance_to_center"]
        iou = graph.nodes["leaf"].data["iou_to_center"]

        if self.has_adaptive_structure:
            is_aware = ((distance <= self.d_aware) & switch_label.bool()).float()
        else:
            is_aware = switch_label.float()

        if is_all_transient:
            is_inter = torch.ones(graph.num_nodes("leaf")).to(graph.device)
        else:
            is_inter = ((distance <= self.d_inter) & switch_label.bool())
            
            if self.consider_iou:
                is_inter = (is_inter & (iou >= self.iou_thresh)).float()
            else:
                is_inter = is_inter.float()
                
            if self.is_select_top_k_leaf:
                is_inter = is_inter * self.select_topk_leaf(graph, distance, iou)
        
                        
        graph.nodes["leaf"].data["is_aware"] = is_aware
        graph.nodes["leaf"].data["is_inter"] = is_inter
        
        # print(is_aware, is_inter, self.d_aware, self.d_inter)            
        # graph.nodes["leaf"].data["confidence"] = switch_score.detach() * is_inter
        graph.nodes["leaf"].data["confidence"] = (1 / (graph.nodes["leaf"].data["distance_to_center"] + 1e-4)) * is_inter

        if self.visualize:
            self.is_inter.append((distance <= self.d_inter) & switch_label.bool())


        return graph
    
    def predict(self, graph: dgl.DGLGraph, current_switch_ouput):
        current_switch_label = current_switch_ouput["pred_switch_label"]
        current_center = graph.nodes["center"].data["current_transient_loc"]
        current_leaf = graph.nodes["leaf"].data["current_transient_loc"]

        if self.has_adaptive_structure:
            leaf_candidate = graph.nodes["leaf"].data["is_inter"]
        else:
            leaf_candidate = graph.nodes["leaf"].data["is_aware"]
        
        if self.is_center_wh_format:
            if self.pred_vel:
                current_center_vel = compute_human_center_wh_vel(current_center, 
                                                    graph.nodes["center"].data["h_gru"],
                                                    self.center_out_arm_embedding,
                                                    self.center_out_center_embedding,
                                                    self.center_out_wh_embedding)
                current_leaf_vel = compute_obj_center_wh_vel(current_leaf,
                                                graph.nodes["leaf"].data["h_gru"],
                                                self.leaf_out_center_embedding,
                                                self.leaf_out_wh_embedding)

                center_candidate = current_switch_label
                
            
                current_center_vel = masked_set(center_candidate > 0,
                                        current_center_vel,
                                        torch.zeros_like(current_center_vel))

                current_leaf_vel = masked_set(leaf_candidate > 0,
                                    current_leaf_vel,
                                    torch.zeros_like(current_leaf_vel))
                        

                human_loc = self.convert_center_prediction(graph, current_center + current_center_vel)
                obj_loc, confidence = self.convert_leaf_prediction(graph, current_leaf + current_leaf_vel)
            else:
                updated_center = compute_human_center_wh_loc(current_center, 
                                                    graph.nodes["center"].data["h_gru"],
                                                    self.center_out_arm_embedding,
                                                    self.center_out_center_embedding,
                                                    self.center_out_wh_embedding)
                updated_leaf = compute_obj_center_wh_loc(current_leaf,
                                                graph.nodes["leaf"].data["h_gru"],
                                                self.leaf_out_center_embedding,
                                                self.leaf_out_wh_embedding)

                center_candidate = current_switch_label
            
                updated_center = masked_set(center_candidate > 0,
                                        updated_center,
                                        current_center)

                updated_leaf = masked_set(leaf_candidate > 0,
                                    updated_leaf,
                                    current_leaf)

                human_loc = self.convert_center_prediction(graph, updated_center)
                obj_loc, confidence = self.convert_leaf_prediction(graph, updated_leaf)
        else:
            current_center_vel = compute_human_vel(current_center,
                                                   graph.nodes["center"].data["h_gru"],
                                                   self.center_out_arm_embedding,
                                                   self.center_out_box_embedding)
            current_leaf_vel = compute_obj_vel(current_leaf,
                                               graph.nodes["leaf"].data["h_gru"],
                                               self.leaf_output_embedding)
            center_candidate = current_switch_label
        
            current_center_vel = masked_set(center_candidate > 0,
                                    current_center_vel,
                                    torch.zeros_like(current_center_vel))

            current_leaf_vel = masked_set(leaf_candidate > 0,
                                current_leaf_vel,
                                torch.zeros_like(current_leaf_vel))
                    

            human_loc = self.convert_center_prediction(graph, current_center + current_center_vel)
            obj_loc, confidence = self.convert_leaf_prediction(graph, current_leaf + current_leaf_vel)

        
        return human_loc, obj_loc, confidence


@TRANSIENT_REGISTRY.register("transient")
def build_model(model_cfg, **kwargs):
    return Transient(model_cfg)