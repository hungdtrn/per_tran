
from yacs.config import CfgNode
import torch

from .persistent import build_persistent
from .transient import build_transient
from .switch import build_switch
from src.bimanual.data.utils import CoordinateHandler
from src.shared.architectures.per_tran import BasePerTran, BaseOnlyPersistent, BaseOnlyTransient

from ..build import META_ARCH_BIMANUAL_REGISTRY

class PerTranCoordinateHandler(CoordinateHandler):
    def convert_human_to_center(self, human_data):
        return self.decompose_coordinate(human_data,
                                         human_data.detach())[0]


    def convert_obj_to_leaf(self, graph, obj_data, base_data):
        obj_idx = graph.nodes["leaf"].data["obj_idx"]
        center_idx = graph.nodes["leaf"].data["center_idx"]
        
        raw = obj_data[obj_idx]
        base = base_data[center_idx].detach()
        
        return self.decompose_coordinate(raw, base)[0]

    def convert_center_to_human(self, center_data, base_data):
        return self.compose_coordinate(center_data, 
                                       base_data.detach())
        
    def convert_leaf_to_obj(self, leaf_data, base_data, grad_flow=False):
        if not grad_flow:
            base_data = base_data.detach()
            
        return self.compose_coordinate(leaf_data, 
                                       base_data)
        
    def convert_human_to_leaf(self, graph, human_data, base_data):
        center_idx = graph.nodes["human_leaf"].data["center_idx"]
        raw = human_data
        base = base_data[center_idx].detach()
        return self.decompose_coordinate(raw, base)[0]
    
class NoegocoordinateHandler(CoordinateHandler):
    def convert_human_to_center(self, human_data):
        return human_data


    def convert_obj_to_leaf(self, graph, obj_data, base_data):
        obj_idx = graph.nodes["leaf"].data["obj_idx"]
        
        raw = obj_data[obj_idx]
        return raw
    
    def convert_center_to_human(self, center_data, base_data):
        return center_data
        
    def convert_leaf_to_obj(self, leaf_data, base_data, grad_flow=False):
        return leaf_data
        
    def convert_human_to_leaf(self, graph, human_data, base_data):
        return human_data
    
class HandBasedCoordinateHandler(PerTranCoordinateHandler):
    def get_arm_center(self, arm):
        assert len(arm.shape) == 2
        assert arm.shape[-1] == 10
        return arm[:, -2:]

class CenterBasedCoordinateHandler(PerTranCoordinateHandler):
    def get_arm_center(self, arm):
        assert len(arm.shape) == 2
        assert arm.shape[-1] == 10
        arm = arm[:, :6]
        arm = arm.reshape(len(arm), -1, 2)
        min_x, max_x = torch.min(arm[:, 0], -1, True)[0], torch.max(arm[:, 0], -1, True)[0]
        min_y, max_y = torch.min(arm[:, 1], -1, True)[0], torch.max(arm[:, 1], -1, True)[0]
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2 
        
        center = torch.cat([center_x, center_y], -1)
        return center
    


class PerTran(BasePerTran):
    def __init__(self, model_cfg: CfgNode, build_persistent_fn=build_persistent,
                 build_transient_fn=build_transient, build_switch_fn=build_switch,
                 coordinate_handler=PerTranCoordinateHandler()) -> None:
        super().__init__(model_cfg, build_persistent_fn, build_transient_fn, build_switch_fn, coordinate_handler)
        
        ego_type = model_cfg.get("ego_type", "normal")
        if ego_type == "none":
            self.coordinate_handler = NoegocoordinateHandler()
        elif ego_type == "hand":
            self.coordinate_handler = HandBasedCoordinateHandler()
        elif ego_type == "center":
            self.coordinate_handler = CenterBasedCoordinateHandler()
            
        self.transient.coordinate_handler = self.coordinate_handler
        self.thresh_scaled = False

    def compute_distance(self, graph):
        leaf = graph.nodes["leaf"]
        human = graph.nodes["human"].data["current_persistent_loc"][leaf.data["center_idx"]]
        obj = graph.nodes["obj"].data["current_persistent_loc"][leaf.data["obj_idx"]]
        
        distance = self.coordinate_handler.distance_from_human_to_obj(human, obj)
        iou = self.coordinate_handler.iou_from_human_to_obj(human, obj)
        
        graph.nodes["leaf"].data["distance_to_center"] = distance
        graph.nodes["leaf"].data["iou_to_center"] = iou
        
        return graph
        
    def scale(self, graph, additional_data):
        shift, scale = additional_data["data_stats"]["scalar_shift"], additional_data["data_stats"]["scalar_scale"]
        node_to_scale = ["human", "obj"]
        for node_name in node_to_scale:
            graph.nodes[node_name].data["x"] = (graph.nodes[node_name].data["x"]) / scale
            graph.nodes[node_name].data["y"] = (graph.nodes[node_name].data["y"]) / scale

        # Egocentric coordinates are shift invariant
        for node_name in ["leaf", "center"]:
            graph.nodes[node_name].data["x"] = graph.nodes[node_name].data["x"] / scale
            graph.nodes[node_name].data["y"] = graph.nodes[node_name].data["y"] / scale

        if not self.thresh_scaled:
            self.transient.d_inter = self.transient.d_inter / scale
            self.transient.d_aware = self.transient.d_aware / scale
            self.switch.d_inter = self.switch.d_inter / scale
            self.thresh_scaled = True

        return graph

    def unscale(self, graph, additional_data):
        shift, scale = additional_data["data_stats"]["scalar_shift"], additional_data["data_stats"]["scalar_scale"]

        node_to_scale = ["human", "obj"]
        for node_name in node_to_scale:
            graph.nodes[node_name].data["x"] = graph.nodes[node_name].data["x"] * scale
            graph.nodes[node_name].data["y"] = graph.nodes[node_name].data["y"] * scale 
            graph.nodes[node_name].data["y_pred"] = graph.nodes[node_name].data["y_pred"] * scale 

        for node_name in ["leaf", "center"]:
            graph.nodes[node_name].data["x"] = graph.nodes[node_name].data["x"] * scale
            graph.nodes[node_name].data["y"] = graph.nodes[node_name].data["y"] * scale


        return graph

class OnlyPersistent(BaseOnlyPersistent):
    def __init__(self, model_cfg: CfgNode, build_persistent_fn=build_persistent,
                 build_transient_fn=build_transient, build_switch_fn=build_switch,
                 coordinate_handler=PerTranCoordinateHandler()) -> None:
        super().__init__(model_cfg, build_persistent_fn, build_transient_fn, build_switch_fn, coordinate_handler)

        ego_type = model_cfg.get("ego_type", "normal")
        if ego_type == "none":
            self.coordinate_handler = NoegocoordinateHandler()
        elif ego_type == "hand":
            self.coordinate_handler = HandBasedCoordinateHandler()
        elif ego_type == "center":
            self.coordinate_handler = CenterBasedCoordinateHandler()
        self.thresh_scaled = False

    def compute_distance(self, graph):
        leaf = graph.nodes["leaf"]
        human = graph.nodes["human"].data["current_persistent_loc"][leaf.data["center_idx"]]
        obj = graph.nodes["obj"].data["current_persistent_loc"][leaf.data["obj_idx"]]
        
        distance = self.coordinate_handler.distance_from_human_to_obj(human, obj)
        iou = self.coordinate_handler.iou_from_human_to_obj(human, obj)
        
        graph.nodes["leaf"].data["distance_to_center"] = distance
        graph.nodes["leaf"].data["iou_to_center"] = iou
        
        return graph
        
    def scale(self, graph, additional_data):
        shift, scale = additional_data["data_stats"]["scalar_shift"], additional_data["data_stats"]["scalar_scale"]
        node_to_scale = ["human", "obj"]
        for node_name in node_to_scale:
            graph.nodes[node_name].data["x"] = (graph.nodes[node_name].data["x"]) / scale
            graph.nodes[node_name].data["y"] = (graph.nodes[node_name].data["y"]) / scale

        # Egocentric coordinates are shift invariant
        for node_name in ["leaf", "center"]:
            graph.nodes[node_name].data["x"] = graph.nodes[node_name].data["x"] / scale
            graph.nodes[node_name].data["y"] = graph.nodes[node_name].data["y"] / scale

        if not self.thresh_scaled:
            self.thresh_scaled = True

        return graph

    def unscale(self, graph, additional_data):
        shift, scale = additional_data["data_stats"]["scalar_shift"], additional_data["data_stats"]["scalar_scale"]

        node_to_scale = ["human", "obj"]
        for node_name in node_to_scale:
            graph.nodes[node_name].data["x"] = graph.nodes[node_name].data["x"] * scale
            graph.nodes[node_name].data["y"] = graph.nodes[node_name].data["y"] * scale 
            graph.nodes[node_name].data["y_pred"] = graph.nodes[node_name].data["y_pred"] * scale 

        for node_name in ["leaf", "center"]:
            graph.nodes[node_name].data["x"] = graph.nodes[node_name].data["x"] * scale
            graph.nodes[node_name].data["y"] = graph.nodes[node_name].data["y"] * scale


        return graph

class OnlyTransient(BaseOnlyTransient):
    def __init__(self, model_cfg: CfgNode, build_persistent_fn=build_persistent,
                 build_transient_fn=build_transient, build_switch_fn=build_switch,
                 coordinate_handler=PerTranCoordinateHandler()) -> None:
        super().__init__(model_cfg, build_persistent_fn, build_transient_fn, build_switch_fn, coordinate_handler)
        
        ego_type = model_cfg.get("ego_type", "normal")
        if ego_type == "none":
            self.coordinate_handler = NoegocoordinateHandler()
        elif ego_type == "hand":
            self.coordinate_handler = HandBasedCoordinateHandler()
        elif ego_type == "center":
            self.coordinate_handler = CenterBasedCoordinateHandler()
            
        self.transient.coordinate_handler = self.coordinate_handler
        self.thresh_scaled = False

    def compute_distance(self, graph):
        leaf = graph.nodes["leaf"]
        human = graph.nodes["human"].data["current_persistent_loc"][leaf.data["center_idx"]]
        obj = graph.nodes["obj"].data["current_persistent_loc"][leaf.data["obj_idx"]]
        
        distance = self.coordinate_handler.distance_from_human_to_obj(human, obj)
        iou = self.coordinate_handler.iou_from_human_to_obj(human, obj)
        
        graph.nodes["leaf"].data["distance_to_center"] = distance
        graph.nodes["leaf"].data["iou_to_center"] = iou
        
        return graph
        
    def scale(self, graph, additional_data):
        shift, scale = additional_data["data_stats"]["scalar_shift"], additional_data["data_stats"]["scalar_scale"]
        node_to_scale = ["human", "obj"]
        for node_name in node_to_scale:
            graph.nodes[node_name].data["x"] = (graph.nodes[node_name].data["x"]) / scale
            graph.nodes[node_name].data["y"] = (graph.nodes[node_name].data["y"]) / scale

        # Egocentric coordinates are shift invariant
        for node_name in ["leaf", "center"]:
            graph.nodes[node_name].data["x"] = graph.nodes[node_name].data["x"] / scale
            graph.nodes[node_name].data["y"] = graph.nodes[node_name].data["y"] / scale

        if not self.thresh_scaled:
            self.transient.d_inter = self.transient.d_inter / scale
            self.transient.d_aware = self.transient.d_aware / scale
            self.thresh_scaled = True

        return graph

    def unscale(self, graph, additional_data):
        shift, scale = additional_data["data_stats"]["scalar_shift"], additional_data["data_stats"]["scalar_scale"]

        node_to_scale = ["human", "obj"]
        for node_name in node_to_scale:
            graph.nodes[node_name].data["x"] = graph.nodes[node_name].data["x"] * scale
            graph.nodes[node_name].data["y"] = graph.nodes[node_name].data["y"] * scale 
            graph.nodes[node_name].data["y_pred"] = graph.nodes[node_name].data["y_pred"] * scale 

        for node_name in ["leaf", "center"]:
            graph.nodes[node_name].data["x"] = graph.nodes[node_name].data["x"] * scale
            graph.nodes[node_name].data["y"] = graph.nodes[node_name].data["y"] * scale


        return graph



        
@META_ARCH_BIMANUAL_REGISTRY.register("per_tran")
def build_per_tran(model_cfg, **kwargs):
    return PerTran(model_cfg)

@META_ARCH_BIMANUAL_REGISTRY.register("only_persistent")
def build_only_persistent(model_cfg, **kwargs):
    return OnlyPersistent(model_cfg)

@META_ARCH_BIMANUAL_REGISTRY.register("only_transient")
def build_only_transient(model_cfg, **kwargs):
    return OnlyTransient(model_cfg)