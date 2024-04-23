import dgl
import torch
from yacs.config import CfgNode
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from .build import SWITCH_REGISTRY
from src.shared.architectures.per_tran import BaseSwitch
from library.utils import straight_through_boolean

class SwitchModule(BaseSwitch):
    def __init__(self, model_cfg: CfgNode):
        super().__init__(model_cfg)
        
        self.iou_thresh = model_cfg.iou_thresh
        self.lower_bound_beta = 0.1
        
        if "iou" in self.switch_type:
            self.human_iou_beta = nn.parameter.Parameter(torch.FloatTensor(1, 1))
            self.human_iou_gamma = nn.parameter.Parameter(torch.FloatTensor(1, 1))
            self.human_iou_beta.data.fill_(model_cfg.initial_iou_beta)
            self.human_iou_gamma.data.fill_(1e-4)


    def compute_switch_distance_score(self, graph: dgl.DGLGraph):
        def _edge_fn(edges):
            distance_score = None
            if "distance" in self.switch_type:
                # Distance increase -> score decrease
                distance_weight = F.relu(self.human_distance_beta) 
                score = torch.exp(-distance_weight * edges.src["distance_to_center"].detach().unsqueeze(-1)) 

                if distance_score is None:
                    distance_score = score
            
            if "iou" in self.switch_type:
                # Iou increase -> score increase
                iou_weight = F.relu(self.human_iou_beta)
                score = 1 - torch.exp(-iou_weight * 100 * (edges.src["iou_to_center"].detach().unsqueeze(-1) + F.relu(self.human_iou_gamma)))

                if distance_score is None:
                    distance_score = score
                else:
                    distance_score = distance_score * score

            return {
                "distance_score": distance_score
            }

        
        graph.update_all(_edge_fn, fn.max("distance_score", "distance_score"), etype=("leaf", "interacts", "center"))
        self.update_visualization_data("distance_score", graph.nodes["center"].data["distance_score"].squeeze(-1))

        return {
            "pred_distance_score": graph.nodes["center"].data.pop("distance_score").squeeze(-1)
        }

    def message_passing(self, graph, t, is_obs):       
        if "rule" not in self.switch_type:
            return super().message_passing(graph, t, is_obs) 
        else:
            def _msg_fn(edges):
                distance = edges.src["distance_to_center"]
                # print(edges.src["iou_to_center"], self.iou_thresh)
                # print(edges.src["iou_to_center"] >= self.iou_thresh)
                iou_flag = (edges.src["iou_to_center"] >= self.iou_thresh).float()
                distance = iou_flag * distance + (1 - iou_flag) * torch.ones_like(distance) * 1000
                
                return {
                    "distance": distance
                }
            def _reduce_fn(nodes):
                return {
                    "min_distance": torch.min(nodes.mailbox["distance"], 1)[0],
                }
                
            graph.update_all(_msg_fn, _reduce_fn, etype=("leaf", "interacts", "center"))
            min_distance = graph.nodes["center"].data.pop("min_distance")
            switch_score = (min_distance <= self.d_inter).float()

            # print(switch_score)
            # switch_label = straight_through_boolean(switch_score >= 0.5,
            #                                         switch_score)
            switch_label = switch_score.clone()

            switch_ouput = {
                "pred_switch_score": switch_score,
                "pred_switch_label": switch_label,
            }

            return graph, switch_ouput


@SWITCH_REGISTRY.register("switch")
def build_model(model_cfg, **kwargs):
    return SwitchModule(model_cfg)