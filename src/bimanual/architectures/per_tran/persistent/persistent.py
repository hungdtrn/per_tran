from yacs.config import CfgNode
import dgl
import torch

from src.bimanual.architectures.per_tran.utils import compute_human_center_wh_loc, compute_obj_center_wh_loc, compute_obj_center_wh_vel, compute_human_center_wh_vel
from src.bimanual.architectures.per_tran.utils import compute_human_vel, compute_obj_vel

from src.shared.architectures.layers.layers import build_mlp
from .build import PERSISTENT_REGISTRY
from src.shared.architectures.per_tran import BasePersistent
from library.utils import masked_set

class Persistent(BasePersistent):
    def __init__(self, model_cfg: CfgNode, **kwargs):
        super().__init__(model_cfg, **kwargs)
        
        try:
            self.is_center_wh_format = model_cfg.is_center_wh_format
        except:
            self.is_center_wh_format = True
            
        human_gru_size = model_cfg.human_gru_size
        obj_gru_size = model_cfg.obj_gru_size

        del self.human_output_embedding
        self.human_out_arm_embedding = build_mlp([human_gru_size, 6])

        if self.is_center_wh_format:
            del self.obj_output_embedding

                    
            # Predict the center and the width and height of the bounding boxes
            self.object_out_center_embedding = build_mlp([obj_gru_size, 2])
            self.object_out_wh_embedding = build_mlp([obj_gru_size, 2])

            self.human_out_center_embedding = build_mlp([human_gru_size, 2])
            self.human_out_wh_embedding = build_mlp([human_gru_size, 2])
            
            self.pred_vel = model_cfg.pred_vel
        else:
            self.human_out_box_embedding = build_mlp([human_gru_size, 4])

    def predict(self, graph: dgl.DGLGraph):
        current_human = graph.nodes["human"].data["current_persistent_loc"]
        current_obj = graph.nodes["obj"].data["current_persistent_loc"]

        if self.is_center_wh_format:
            if self.pred_vel:
                obj_vel = compute_obj_center_wh_vel(current_obj, graph.nodes["obj"].data["h_gru"],
                                        self.object_out_center_embedding,
                                        self.object_out_wh_embedding)
                human_vel = compute_human_center_wh_vel(current_human, 
                                            graph.nodes["human"].data["h_gru"],
                                            self.human_out_arm_embedding,
                                            self.human_out_center_embedding,
                                            self.human_out_wh_embedding)
                        
                return current_human + human_vel, current_obj + obj_vel
            else:
                human_loc = compute_human_center_wh_loc(current_human, 
                                            graph.nodes["human"].data["h_gru"],
                                            self.human_out_arm_embedding,
                                            self.human_out_center_embedding,
                                            self.human_out_wh_embedding)
                obj_loc = compute_obj_center_wh_loc(current_obj, graph.nodes["obj"].data["h_gru"],
                                        self.object_out_center_embedding,
                                        self.object_out_wh_embedding)
                
                return human_loc, obj_loc
        else:
            obj_vel = compute_obj_vel(current_obj, graph.nodes["obj"].data["h_gru"], self.obj_output_embedding)
            human_vel = compute_human_vel(current_human, graph.nodes["human"].data["h_gru"], self.human_out_arm_embedding, self.human_out_box_embedding)
            
            return current_human + human_vel, current_obj + obj_vel

# class PersistentHete(BasePersistentHete):
#     def __init__(self, model_cfg: CfgNode, **kwargs):
#         super().__init__(model_cfg, **kwargs)
        
#         try:
#             self.is_center_wh_format = model_cfg.is_center_wh_format
#         except:
#             self.is_center_wh_format = True
            
#         human_gru_size = model_cfg.human_gru_size
#         obj_gru_size = model_cfg.obj_gru_size

#         del self.human_output_embedding
#         self.human_out_arm_embedding = build_mlp([human_gru_size, 6])

#         if self.is_center_wh_format:
#             del self.obj_output_embedding

                    
#             # Predict the center and the width and height of the bounding boxes
#             self.object_out_center_embedding = build_mlp([obj_gru_size, 2])
#             self.object_out_wh_embedding = build_mlp([obj_gru_size, 2])

#             self.human_out_center_embedding = build_mlp([human_gru_size, 2])
#             self.human_out_wh_embedding = build_mlp([human_gru_size, 2])
            
#             self.pred_vel = model_cfg.pred_vel
#         else:
#             self.human_out_box_embedding = build_mlp([human_gru_size, 4])
        
#     def predict(self, graph: dgl.DGLGraph):
#         current_human = graph.nodes["human"].data["current_persistent_loc"]
#         current_obj = graph.nodes["obj"].data["current_persistent_loc"]

#         if self.is_center_wh_format:
#             if self.pred_vel:
#                 obj_vel = compute_obj_center_wh_vel(current_obj, graph.nodes["obj"].data["h_gru"],
#                                         self.object_out_center_embedding,
#                                         self.object_out_wh_embedding)
#                 human_vel = compute_human_center_wh_vel(current_human, 
#                                             graph.nodes["human"].data["h_gru"],
#                                             self.human_out_arm_embedding,
#                                             self.human_out_center_embedding,
#                                             self.human_out_wh_embedding)
                        
#                 return current_human + human_vel, current_obj + obj_vel
#             else:
#                 human_loc = compute_human_center_wh_loc(current_human, 
#                                             graph.nodes["human"].data["h_gru"],
#                                             self.human_out_arm_embedding,
#                                             self.human_out_center_embedding,
#                                             self.human_out_wh_embedding)
#                 obj_loc = compute_obj_center_wh_loc(current_obj, graph.nodes["obj"].data["h_gru"],
#                                         self.object_out_center_embedding,
#                                         self.object_out_wh_embedding)
                
#                 return human_loc, obj_loc
#         else:
#             obj_vel = compute_obj_vel(current_obj, graph.nodes["obj"].data["h_gru"], self.obj_output_embedding)
#             human_vel = compute_human_vel(current_human, graph.nodes["human"].data["h_gru"], self.human_out_arm_embedding, self.human_out_box_embedding)
            
#             return current_human + human_vel, current_obj + obj_vel

@PERSISTENT_REGISTRY.register("persistent")
def build_model(model_cfg, **kwargs):
    return Persistent(model_cfg)

# @PERSISTENT_REGISTRY.register("persistent_hete")
# def build_model_hete(model_cfg, **kwargs):
#     return PersistentHete(model_cfg)