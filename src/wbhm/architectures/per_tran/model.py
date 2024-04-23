
from yacs.config import CfgNode

from .persistent import build_persistent
from .transient import build_transient
from .switch import build_switch
from src.wbhm.data.utils import CoordinateHandler, rotate_and_translate_wbhm
from src.shared.architectures.per_tran import BasePerTran, BaseOnlyPersistent, BaseOnlyTransient

from ..build import META_ARCH_WBHM_REGISTRY

class PerTranCoordinateHandler(CoordinateHandler):
    def convert_human_to_center(self, human_data):
        return self.decompose_coordinate(human_data,
                                    self.skeleton_to_box_torch(human_data).detach())[0]


    def convert_obj_to_leaf(self, graph, obj_data, base_data, **kwargs):
        obj_idx = graph.nodes["leaf"].data["obj_idx"]
        center_idx = graph.nodes["leaf"].data["center_idx"]
        
        raw = obj_data[obj_idx]
        base = self.skeleton_to_box_torch(base_data[center_idx]).detach()
        
        return self.decompose_coordinate(raw, base)[0]

    def convert_center_to_human(self, center_data, base_data, **kwargs):
        return self.compose_coordinate(center_data, 
                                self.skeleton_to_box_torch(base_data).detach())
        
    def convert_leaf_to_obj(self, leaf_data, base_data, grad_flow, **kwargs):
        return self.compose_coordinate(leaf_data, 
                                  self.skeleton_to_box_torch(base_data).detach())
        
        
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
    
    
        
class PerTran(BasePerTran):
    def __init__(self, model_cfg: CfgNode) -> None:
        is_ego = model_cfg.get("is_ego", True)
        
        if is_ego:
            coordinate_handler_class = PerTranCoordinateHandler
        else:
            coordinate_handler_class = NoegocoordinateHandler
            
        super().__init__(model_cfg, build_persistent, build_transient, build_switch, coordinate_handler_class())
        self.transient.coordinate_handler = coordinate_handler_class()
        
    def augment(self, graph):
        x_human, y_human = graph.nodes["human"].data["x"], graph.nodes["human"].data["y"]
        x_obj, y_obj = graph.nodes["obj"].data["x"], graph.nodes["obj"].data["y"]

        x_human, augment_m = self.rotate_and_translate(x_human)
        y_human, _ = self.rotate_and_translate(y_human, augment_m=augment_m)
        x_obj, _ = self.rotate_and_translate(x_obj, augment_m=augment_m)
        y_obj, _ = self.rotate_and_translate(y_obj, augment_m=augment_m)

        graph.nodes["human"].data["x"] = x_human
        graph.nodes["human"].data["y"] = y_human
        graph.nodes["obj"].data["x"] = x_obj
        graph.nodes["obj"].data["y"] = y_obj
        
        return graph
    
    def rotate_and_translate(self, seq_data, augment_m=None, mask=None):
        return rotate_and_translate_wbhm(seq_data, augment_m, mask, translate_range=1000)

class OnlyPersistent(BaseOnlyPersistent):
    def __init__(self, model_cfg: CfgNode) -> None:
        is_ego = model_cfg.get("is_ego", True)
        
        if is_ego:
            coordinate_handler_class = PerTranCoordinateHandler
        else:
            coordinate_handler_class = NoegocoordinateHandler
            
        super().__init__(model_cfg, build_persistent, build_transient, build_switch, coordinate_handler_class())

    def augment(self, graph):
        x_human, y_human = graph.nodes["human"].data["x"], graph.nodes["human"].data["y"]
        x_obj, y_obj = graph.nodes["obj"].data["x"], graph.nodes["obj"].data["y"]

        x_human, augment_m = self.rotate_and_translate(x_human)
        y_human, _ = self.rotate_and_translate(y_human, augment_m=augment_m)
        x_obj, _ = self.rotate_and_translate(x_obj, augment_m=augment_m)
        y_obj, _ = self.rotate_and_translate(y_obj, augment_m=augment_m)

        graph.nodes["human"].data["x"] = x_human
        graph.nodes["human"].data["y"] = y_human
        graph.nodes["obj"].data["x"] = x_obj
        graph.nodes["obj"].data["y"] = y_obj
        
        return graph
    
    def rotate_and_translate(self, seq_data, augment_m=None, mask=None):
        return rotate_and_translate_wbhm(seq_data, augment_m, mask, translate_range=1000)

class OnlyTransient(BaseOnlyTransient):
    def __init__(self, model_cfg: CfgNode) -> None:
        is_ego = model_cfg.get("is_ego", True)
        
        if is_ego:
            coordinate_handler_class = PerTranCoordinateHandler
        else:
            coordinate_handler_class = NoegocoordinateHandler
            
        super().__init__(model_cfg, build_persistent, build_transient, build_switch, coordinate_handler_class())
        self.transient.coordinate_handler = coordinate_handler_class()

    def augment(self, graph):
        x_human, y_human = graph.nodes["human"].data["x"], graph.nodes["human"].data["y"]
        x_obj, y_obj = graph.nodes["obj"].data["x"], graph.nodes["obj"].data["y"]

        x_human, augment_m = self.rotate_and_translate(x_human)
        y_human, _ = self.rotate_and_translate(y_human, augment_m=augment_m)
        x_obj, _ = self.rotate_and_translate(x_obj, augment_m=augment_m)
        y_obj, _ = self.rotate_and_translate(y_obj, augment_m=augment_m)

        graph.nodes["human"].data["x"] = x_human
        graph.nodes["human"].data["y"] = y_human
        graph.nodes["obj"].data["x"] = x_obj
        graph.nodes["obj"].data["y"] = y_obj
        
        return graph
    
    def rotate_and_translate(self, seq_data, augment_m=None, mask=None):
        return rotate_and_translate_wbhm(seq_data, augment_m, mask, translate_range=1000)

class PerTranProperNorm(PerTran):
    def __init__(self, model_cfg: CfgNode) -> None:
        super().__init__(model_cfg)
        self.transient.d_inter = self.transient.d_inter * 1e3
        self.transient.d_aware = self.transient.d_aware * 1e3
        self.switch.d_inter = self.switch.d_inter * 1e3
        self.thresh_scaled = False

    def scale(self, graph, additional_data):
        shift, scale = additional_data["data_stats"]["scalar_shift"], additional_data["data_stats"]["scalar_scale"]
        node_to_scale = ["human", "obj"]
        for node_name in node_to_scale:
            graph.nodes[node_name].data["x"] = (graph.nodes[node_name].data["x"] - shift) / scale
            graph.nodes[node_name].data["y"] = (graph.nodes[node_name].data["y"] - shift) / scale

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
            graph.nodes[node_name].data["x"] = graph.nodes[node_name].data["x"] * scale + shift
            graph.nodes[node_name].data["y"] = graph.nodes[node_name].data["y"] * scale + shift
            graph.nodes[node_name].data["y_pred"] = graph.nodes[node_name].data["y_pred"] * scale + shift

        for node_name in ["leaf", "center"]:
            graph.nodes[node_name].data["x"] = graph.nodes[node_name].data["x"] * scale
            graph.nodes[node_name].data["y"] = graph.nodes[node_name].data["y"] * scale


        return graph
        
@META_ARCH_WBHM_REGISTRY.register("per_tran")
def build_per_tran(model_cfg, **kwargs):
    return PerTran(model_cfg)

@META_ARCH_WBHM_REGISTRY.register("only_persistent")
def build_only_persistent(model_cfg, **kwargs):
    return OnlyPersistent(model_cfg)

@META_ARCH_WBHM_REGISTRY.register("only_transient")
def build_only_transient(model_cfg, **kwargs):
    return OnlyTransient(model_cfg)

@META_ARCH_WBHM_REGISTRY.register("per_tran_proper_norm")
def build_per_tran_proper_norm(model_cfg, **kwargs):
    return PerTranProperNorm(model_cfg)
