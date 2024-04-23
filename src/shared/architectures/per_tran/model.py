from builtins import print
from re import S
import dgl
from dgl.udf import EdgeBatch, NodeBatch
import torch
from torch import nn
from yacs.config import CfgNode
from omegaconf import OmegaConf
import yaml
import dgl.function as fn

from src.shared.architectures.layers.layers import build_mlp
from src.shared.data.graph_utils import copy_from_src_to_dst

from library.utils import masked_set, straight_through_boolean

def _combine_cfg(cfg1: CfgNode, cfg2: CfgNode):
    cfg1 = yaml.safe_load(OmegaConf.to_yaml(cfg1))
    cfg2 = yaml.safe_load(OmegaConf.to_yaml(cfg2))
    
    cfg1.update(cfg2)
    return OmegaConf.create(cfg1)


class Model(nn.Module):
    def __init__(self, model_cfg: CfgNode,
                 build_persistent_fn, 
                 build_transient_fn,
                 build_switch_fn,
                 coordinate_handler) -> None:
        super().__init__()
        
        self.cfg = model_cfg
        self.is_augment = self.cfg.get("augment", False)
        if "manual_all" in model_cfg.switch.switch_type:
            model_cfg.has_cross_graph_msg = False

        human_input_size = model_cfg.human_input_size
        obj_input_size = model_cfg.obj_input_size
        hidden_size = model_cfg.hidden_size
        
        human_embedding_size = model_cfg.get("human_embedding_size", hidden_size)
        obj_embedding_size = model_cfg.get("obj_embedding_size", hidden_size)
                    
        num_obj = model_cfg.num_obj
        activation = model_cfg.activation


        self.human_inp_embedding = build_mlp([human_input_size, human_embedding_size], [activation])
        self.obj_loc_embedding = build_mlp([obj_input_size, obj_embedding_size], [activation])
        self.obj_inp_embedding = build_mlp([obj_embedding_size + obj_embedding_size, obj_embedding_size], [activation])
        self.type_embedding = build_mlp([num_obj, obj_embedding_size], [activation])
        
        self.has_cross_graph_msg = model_cfg.has_cross_graph_msg
        self.persistent = build_persistent_fn(_combine_cfg(model_cfg, model_cfg.persistent))
        self.transient = build_transient_fn(_combine_cfg(model_cfg, model_cfg.transient))
        self.switch = build_switch_fn(_combine_cfg(model_cfg, model_cfg.switch))
        
        self.visualize = False
        self.visualize_data = {}
        
        self.coordinate_handler = coordinate_handler
                
    def initialize_hidden(self, batch_size, hidden_dim, device):
        return torch.zeros(batch_size, hidden_dim).to(device)
    
    def initialize(self, graph):
        self.visualize_data = {}
        graph = self.persistent.initialize(self.initialize_hidden,
                                           graph)
        graph = self.transient.initialize(self.initialize_hidden,
                                          graph)
        graph = self.switch.initialize(self.initialize_hidden,
                                              graph)
        
        return graph
    
    def scale(self, graph, additional_data):
        scale = 1e-3
        
        node_to_scale = ["human", "obj", "leaf", "center"]
        for node_name in node_to_scale:
            graph.nodes[node_name].data["x"] = graph.nodes[node_name].data["x"] * scale
            graph.nodes[node_name].data["y"] = graph.nodes[node_name].data["y"] * scale

        return graph
        
    def augment(self, graph):
        raise NotImplementedError
    
    def preprocess(self, graph, additional_data, is_training):
        # Steps: (1) convert to meter (2) set the center idx and obj idx to the leaf
        
        if self.is_augment & is_training:
            graph = self.augment(graph)
        
        # Scaling
        graph = self.scale(graph, additional_data)

        # At first, ensure that no center and obj idx are assigned to the leaf
        assert "center_idx" not in graph.nodes["leaf"].data
        assert "obj_idx" not in graph.nodes["leaf"].data

        # Ensure that the idx of the center is the same with the human
        # So that we can directly assign the values between the two nodes 
        human_idx, center_idx = graph.edges(etype=("human", "to", "center"))
        assert torch.all(human_idx == center_idx)
        
        # Assign the center and obj node idx to the leaf
        # These idx are used to compute the distances
        # and converting the values between persistent and transient channels
        device = graph.device
        graph.nodes["center"].data["idx"] = torch.arange(graph.num_nodes("center")).float().to(device)
        # graph.nodes["human_leaf"].data["idx"] = torch.arange(graph.num_nodes("human_leaf")).float().to(device)
        graph.nodes["obj"].data["idx"] = torch.arange(graph.num_nodes("obj")).float().to(device)
        
        leaf_center_idx = copy_from_src_to_dst(graph, data_name="idx", etype=("center", "interacts", "leaf"))
        leaf_obj_idx = copy_from_src_to_dst(graph, data_name="idx", etype=("obj", "to", "leaf"))
        
        assert torch.all(leaf_center_idx == leaf_center_idx.long())
        assert torch.all(leaf_obj_idx == leaf_obj_idx.long())

        
        graph.nodes["leaf"].data["center_idx"] = leaf_center_idx.long()
        graph.nodes["leaf"].data["obj_idx"] = leaf_obj_idx.long()
        
        graph.nodes["center"].data.pop("idx")
        graph.nodes["obj"].data.pop("idx")

        return graph
    
    def unscale(self, graph, additional_data):
        inv_scale = 1e3

        node_to_scale = ["human", "obj", "leaf", "center"]
        for node_name in node_to_scale:
            graph.nodes[node_name].data["x"] = graph.nodes[node_name].data["x"] * inv_scale
            graph.nodes[node_name].data["y"] = graph.nodes[node_name].data["y"] * inv_scale

        for node_name in ["human", "obj"]:
            graph.nodes[node_name].data["y_pred"] = graph.nodes[node_name].data["y_pred"] * inv_scale

        return graph
    
    def postprocess(self, graph, additional_data):
        return self.unscale(graph, additional_data)

    def compute_distance(self, graph):
        leaf = graph.nodes["leaf"]
        human = graph.nodes["human"].data["current_persistent_loc"][leaf.data["center_idx"]]
        obj = graph.nodes["obj"].data["current_persistent_loc"][leaf.data["obj_idx"]]
        
        distance = self.coordinate_handler.distance_from_human_to_obj(human, obj)
        graph.nodes["leaf"].data["distance_to_center"] = distance

        return graph
    
    def prepare_for_components(self, graph, t, is_obs):
        graph = self.persistent.prepare_node_features(graph, t, is_obs)
        graph = self.transient.prepare_node_features(graph, t, is_obs)
        graph = self.switch.prepare_node_features(graph, t, is_obs)
        
        return graph

    
    def prepare_node_features(self, graph, t, is_obs, is_forcing=False):
        if is_obs:
            node_name_list = ["human", "obj"]
            for name in node_name_list:
                graph.nodes[name].data["current_persistent_loc"] = graph.nodes[name].data["x"][:, t]

        elif is_forcing:
            node_name_list = ["human", "obj"]
            obs_len = graph.nodes["human"].data["x"].size(1)
            for name in node_name_list:
                graph.nodes[name].data["current_persistent_loc"] = graph.nodes[name].data["y"][:, t - obs_len]

                
        # convert persistent loc to transient loc
        current_center_loc = self.coordinate_handler.convert_human_to_center(graph.nodes["human"].data["current_persistent_loc"])
        current_leaf_loc = self.coordinate_handler.convert_obj_to_leaf(graph, 
                                                obj_data=graph.nodes["obj"].data["current_persistent_loc"],
                                                base_data=graph.nodes["human"].data["current_persistent_loc"])
        
        # print(t, current_leaf_loc)        
        
        graph.nodes["center"].data["current_transient_loc"] = current_center_loc
        graph.nodes["leaf"].data["current_transient_loc"] = current_leaf_loc
                    
        graph = self.compute_distance(graph)
        
        graph.nodes["human"].data["embed_feat"] = self.human_inp_embedding(graph.nodes["human"].data["current_persistent_loc"])
        graph.nodes["obj"].data["embed_feat"] = self.obj_inp_embedding(torch.cat([
            self.type_embedding(graph.nodes["obj"].data["type"]),
            self.obj_loc_embedding(graph.nodes["obj"].data["current_persistent_loc"])
        ], -1))
        

        return self.prepare_for_components(graph, t, is_obs)
        
    def persistent_message_passing(self, graph, t, is_obs):
        graph = self.persistent.message_passing(graph, t, is_obs)
        return graph
    
    def switch_and_transient_message_passing(self, graph, t, is_obs, is_all_transient, is_all_persistent):
        if (not is_all_persistent) and (not is_all_transient):
            graph, switch_output = self.switch.message_passing(graph, t, is_obs)
        elif is_all_persistent:            
            switch_output = {
                "pred_switch_score": torch.zeros(graph.num_nodes("human")).float().to(graph.device),
                "pred_switch_label": torch.zeros(graph.num_nodes("human")).float().to(graph.device),
                "pred_distance_score": torch.zeros(graph.num_nodes("human")).float().to(graph.device)
            }
            
        elif is_all_transient:
            def edge_distance_fn(edges):
                obj_loc = edges.src["current_persistent_loc"]
                human_loc = edges.dst["current_persistent_loc"]
                distance = self.coordinate_handler.distance_from_human_to_obj(human_loc, obj_loc).detach()
                return {"distance" : distance}
            
            graph.update_all(edge_distance_fn, fn.min("distance", "min_distance"),
                             etype=("obj", "interacts", "human"))
            
            switch_output = {
                "pred_distance_score": 0.5 + 1.0 / (graph.nodes["human"].data.pop("min_distance") + 1e-4),
                "pred_switch_label": torch.ones(graph.num_nodes("human")).float().to(graph.device),
                "pred_switch_score": torch.ones(graph.num_nodes("human")).float().to(graph.device)
            }

        graph = self.transient.message_passing(graph, t, is_obs, switch_output, is_all_transient)
        return graph, switch_output

        
    def message_passing(self, graph, t, is_obs=True, is_start=False, is_forcing=False, is_all_transient=False, is_all_persistent=False):
        graph = self.prepare_node_features(graph, t, is_obs, is_forcing=is_forcing)

        # Do message passing
        graph = self.persistent_message_passing(graph, t, is_obs)
        graph, switch_output = self.switch_and_transient_message_passing(graph, t, is_obs, is_all_transient, is_all_persistent)

        return graph, switch_output
    
    def predict(self, graph, switch_output):
        switch_label = switch_output["pred_switch_label"]
        human_tran, obj_tran, obj_tran_confidence = self.transient.predict(graph, switch_output)
        human_per, obj_per = self.persistent.predict(graph)
        
        obj_tran_candidate = straight_through_boolean(obj_tran_confidence >= 0.5,
                                                      obj_tran_confidence)
        
        human_loc = masked_set(switch_label > 0,
                               human_tran,
                               human_per)
        
        # obj_loc  = obj_per
        obj_loc = masked_set(obj_tran_candidate > 0,
                             obj_tran,
                             obj_per)
                
        graph.nodes["human"].data["current_persistent_loc"] = human_loc
        graph.nodes["obj"].data["current_persistent_loc"] = obj_loc
        
        return graph 

    def get_distance_beta(self):
        return self.switch.get_distance_beta()
    
    def select_obj(self, graph, data, weight):
        num_dim = len(data.shape)

        data_size = list(data.size())
        data_size[0] = -1
        for i in range(num_dim - 1):
            weight = weight.unsqueeze(-1)
        
        return torch.masked_select(data, weight).reshape(data_size)

    def prepare_switch_for_teacher_forcing(self, graph, t):
        # score = graph.nodes["human"].data["switch_score"][:, t]
        # label = graph.nodes["human"].data["switch_label"][:, t]

        # return score, label.float()
        return {
            "pred_switch_score": graph.nodes["human"].data["switch_score"][:, t],
            "pred_switch_label": graph.nodes["human"].data["switch_label"][:, t].float()
        }

    def set_visualization(self, flag):
        self.visualize = flag
        self.transient.visualize = flag
        self.persistent.visualize = flag
        self.switch.visualize = flag
        

    def forward(self, graph, pred_len, 
                additional_data={}, is_training=True, 
                is_forcing=False, is_all_transient=False, 
                is_all_persistent=False, is_visualize=False, t_force_persistent=-1, 
                t_force_transient=-1, **kwargs):
        
        self.set_visualization(is_visualize)
        self.is_post_process = False
        # print(self.switch.get_distance_beta())
        if not is_training:
            self.is_post_process = True
        # if "post_process" in additional_data:
        #     post_process = additional_data["post_process"]

        with graph.local_scope():
            graph = self.preprocess(graph, additional_data, is_training)
            graph = self.initialize(graph)
            obs_len = graph.nodes["human"].data["x"].size(1)

            pred_switch_output = {}
            is_inter, is_aware = [], []
            
            for i in range(obs_len):
                graph, current_switch_output = self.message_passing(graph, i, is_obs=True, is_start=i==0, is_forcing=is_forcing,
                                                             is_all_transient=is_all_transient, is_all_persistent=is_all_persistent)
                
                is_aware.append(graph.nodes["leaf"].data["is_aware"])
                is_inter.append(graph.nodes["leaf"].data["is_inter"])

                for k, v in current_switch_output.items():
                    if v is not None:
                        if k not in pred_switch_output:
                            pred_switch_output[k] = []
                            
                        pred_switch_output[k].append(v.clone())
                
                if is_forcing:
                    current_switch_output = self.prepare_switch_for_teacher_forcing(graph, i)

            y_pred_human, y_pred_obj = [], []
            
            graph = self.predict(graph, current_switch_output)
            y_pred_human.append(graph.nodes["human"].data["current_persistent_loc"].clone())
            y_pred_obj.append(graph.nodes["obj"].data["current_persistent_loc"].clone())

            for i in range(pred_len-1):
                if i == t_force_persistent:
                    is_all_persistent = True
                elif i == t_force_transient:
                    is_all_transient = True
                    
                graph, current_switch_output = self.message_passing(graph, i + obs_len, is_obs=False, is_start=False, is_forcing=is_forcing,
                                                             is_all_transient=is_all_transient, is_all_persistent=is_all_persistent)
                

                is_aware.append(graph.nodes["leaf"].data["is_aware"])
                is_inter.append(graph.nodes["leaf"].data["is_inter"])

                graph = self.predict(graph, current_switch_output)

                for k, v in current_switch_output.items():
                    if v is not None:
                        if k not in pred_switch_output:
                            pred_switch_output[k] = []
                            
                        pred_switch_output[k].append(v.clone())
                
                if is_forcing:
                    current_switch_output = self.prepare_switch_for_teacher_forcing(graph, i)
                
                y_pred_human.append(graph.nodes["human"].data["current_persistent_loc"].clone())
                y_pred_obj.append(graph.nodes["obj"].data["current_persistent_loc"].clone())

            graph.nodes["human"].data["y_pred"] = torch.stack(y_pred_human, 1)
            graph.nodes["obj"].data["y_pred"] = torch.stack(y_pred_obj, 1)

            if self.is_post_process:
                graph = self.postprocess(graph, additional_data)

            y_pred_human = graph.nodes["human"].data["y_pred"]
            y_pred_obj = graph.nodes["obj"].data["y_pred"]
            y_human, y_obj = graph.nodes["human"].data["y"], graph.nodes["obj"].data["y"]
        
            start_switch = 0
            for k in pred_switch_output.keys():
                pred_switch_output[k] = torch.stack(pred_switch_output[k], 1)
                pred_switch_output[k] = pred_switch_output[k][:, start_switch:]
                         
            gt_score = graph.nodes["human"].data["switch_score"][:, start_switch:-1]
            gt_label = gt_score >= 0.5
        
            additional_output = {
                "gt_switch_label": gt_label,
                "gt_switch_score": gt_score,
                "distance_beta": self.get_distance_beta(),
                "transient_is_aware": torch.stack(is_aware, 0),
                "transient_is_inter": torch.stack(is_inter, 0),
            }
            
            additional_output.update(pred_switch_output)

        if self.visualize:
            self.visualize_data.update(self.switch.visualize_data)
            graph.nodes["leaf"].data["is_inter_seq"] = torch.stack(self.transient.is_inter, 1)
            
            for k in self.visualize_data.keys():
                self.visualize_data[k] = torch.stack(self.visualize_data[k], 1)

        # print(additional_output.keys())
        return (y_human, y_obj), (y_pred_human, y_pred_obj), additional_output


class OnlyPersistent(Model):
    def __init__(self, model_cfg: CfgNode, build_persistent_fn, build_transient_fn, build_switch_fn, coordinate_handler) -> None:
        model_cfg.has_cross_graph_msg = False
        super().__init__(model_cfg, build_persistent_fn, build_transient_fn, build_switch_fn, coordinate_handler)
        
        del self.transient
        del self.switch
        
    def initialize(self, graph):
        self.visualize_data = {}
        graph = self.persistent.initialize(self.initialize_hidden,
                                           graph)
        return graph
    
    def prepare_for_components(self, graph, t, is_obs):
        graph = self.persistent.prepare_node_features(graph, t, is_obs)
        return graph
    
    def switch_and_transient_message_passing(self, graph, t, is_obs, is_all_transient, is_all_persistent):
        return graph, {}
    
    def predict(self, graph, switch_output):
        human_per, obj_per = self.persistent.predict(graph)
        graph.nodes["human"].data["current_persistent_loc"] = human_per
        graph.nodes["obj"].data["current_persistent_loc"] = obj_per
        
        return graph
    
    def set_visualization(self, flag):
        self.visualize = flag
        self.persistent.visualize = flag

    def get_distance_beta(self):
        return None
    
    
class OnlyTransient(Model):
    def __init__(self, model_cfg: CfgNode, build_persistent_fn, build_transient_fn, build_switch_fn, coordinate_handler) -> None:
        model_cfg.has_cross_graph_msg = False
        super().__init__(model_cfg, build_persistent_fn, build_transient_fn, build_switch_fn, coordinate_handler)
        
        del self.persistent
        del self.switch
        
    def initialize(self, graph):
        self.visualize_data = {}
        graph = self.transient.initialize(self.initialize_hidden,
                                           graph)
        return graph

    def prepare_for_components(self, graph, t, is_obs):
        graph = self.transient.prepare_node_features(graph, t, is_obs)
        return graph
    
    def persistent_message_passing(self, graph, t, is_obs):
        return graph
    
    def switch_and_transient_message_passing(self, graph, t, is_obs, is_all_transient, is_all_persistent):
        return super().switch_and_transient_message_passing(graph, t, is_obs, True, False)

    def predict(self, graph, switch_output):
        human_tran, obj_tran, obj_tran_confidence = self.transient.predict(graph, switch_output)
        graph.nodes["human"].data["current_persistent_loc"] = human_tran
        graph.nodes["obj"].data["current_persistent_loc"] = obj_tran

        return graph

    def set_visualization(self, flag):
        self.visualize = flag
        self.transient.visualize = flag

    def get_distance_beta(self):
        return None