import dgl
import dgl.function as fn
from dgl.udf import EdgeBatch, NodeBatch
import torch
from scipy import signal

# from src.data.hoi.utils import compose_coordinate, decompose_coordinate, skeleton_to_box_torch

def is_future_happen(event, window):
    original_shape = event.shape
    event = event.view(-1, 1, event.shape[-1]).float()

    right_kernel = torch.ones(1, 1, 2 * window + 1).float().to(event.device)
    right_kernel[:, :, :window] = 0

    out_right = torch.nn.functional.conv1d(event, right_kernel, padding=window)
    return (out_right >= 1).reshape(original_shape)

def smooth_weight(weight, kernel=5, std=7):
    """Smooth an input segmentation.

    Arguments:
        weight - torch tensor of shape (N, seq_len)
        mu - foat
        sigma - float
    Returns:
        The smoothed weight
    """
    original_shape = weight.shape
    pad_left = weight[:, 0:1].repeat(1, kernel // 2)
    pad_right = weight[:, -2:-1].repeat(1, kernel // 2)
    weight = torch.cat([pad_left, weight, pad_right], -1)

    weight = weight.view(-1, 1, weight.shape[-1]).float()

    window = signal.windows.gaussian(kernel, std=std)
    window = torch.tensor(window).to(weight.device).float()
    window = window.reshape(1, 1, window.size(-1)) / torch.sum(window)

    out = torch.nn.functional.conv1d(weight, window)

    return out.reshape(original_shape)

def process_switch(switch_flag, future_window, weight_gauss_kernel, weight_gauss_std):
    if future_window:
        switch_flag = is_future_happen(switch_flag, future_window)
    
    switch_score = smooth_weight(switch_flag, weight_gauss_kernel,
                                    weight_gauss_std)

    return switch_score, switch_flag

def process_switch_v2(switch_flag, future_window, weight_gauss_kernel, weight_gauss_std):
    if future_window:
        tmp = switch_flag.clone()
        switch_flag[:, :-future_window] = tmp[:, future_window:]
            
    switch_score = smooth_weight(switch_flag, weight_gauss_kernel, weight_gauss_std)
    return switch_score, switch_flag

def copy_from_src_to_dst(graph, data_name, etype):
    # The dst nodes in these edges only have 1 in-degree
    assert etype in [('human', 'to', 'center'),
                     ('center', 'to', 'human'),
                     ('obj', 'to', 'leaf'),
                     ('center', 'interacts', 'leaf'),
                     ("center", "interacts", "human_leaf"),
                     ("human_leaf", "interacts", "center")]
    
    graph.update_all(fn.copy_src(data_name, 'tmp'),
                     fn.mean('tmp', 'tmp'),
                     etype=etype)
    
    dst_name = etype[-1]
    return graph.nodes[dst_name].data.pop("tmp")
    

def verify_one_in_degree(graph, node_type, etype):
    num_nodes = graph.num_nodes(node_type)
    in_degrees = graph.in_degrees([i for i in range(num_nodes)],
                                  etype=etype)
    
    assert torch.all(in_degrees == 1)
    
def test_from_leaf_to_obj(coorinate_handler, graph, leaf_data_name, obj_data_name, base_human_data_name):
    center_idx = graph.nodes["leaf"].data["center_idx"]
    base = coorinate_handler.skeleton_to_box_torch(graph.nodes["human"].data[base_human_data_name][center_idx]).detach()

    graph.nodes["leaf"].data["tmp_{}".format(obj_data_name)] = coorinate_handler.compose_coordinate(graph.nodes["leaf"].data[leaf_data_name],
                                                                                  base)

    graph.update_all(fn.copy_src("tmp_{}".format(obj_data_name), "tmp_{}".format(obj_data_name)),
                     fn.mean("tmp_{}".format(obj_data_name), "tmp_{}".format(obj_data_name)),
                     etype=("leaf", "to", "obj"))
    
    return graph.nodes["obj"].data.pop("tmp_{}".format(obj_data_name))
    
def loc_from_leaf_to_obj(coorinate_handler, graph, leaf_data_name, obj_data_name, base_human_data_name, confidence_name,  inplace=True):
    # convert each leaf to obj first
    center_idx = graph.nodes["leaf"].data["center_idx"]
    base = coorinate_handler.skeleton_to_box_torch(graph.nodes["human"].data[base_human_data_name][center_idx]).detach()
        
    graph.nodes["leaf"].data["tmp_{}".format(obj_data_name)] = coorinate_handler.compose_coordinate(graph.nodes["leaf"].data[leaf_data_name],
                                                                                  base)
    
    
    def _msg_fn(edges: EdgeBatch) -> dict:
        return {
            "tmp_{}".format(obj_data_name): edges.src["tmp_{}".format(obj_data_name)],
            confidence_name: edges.src[confidence_name],
        }
        
    def _reduce_fn(nodes: NodeBatch) -> dict:
        confidence = nodes.mailbox[confidence_name]
        max_confidence, max_idx = torch.max(confidence, 1)
        mask = torch.zeros_like(confidence).scatter_(1, max_idx.unsqueeze(-1), 1.0)
        mask = mask.unsqueeze(-1).bool()
        
        leaf_data = nodes.mailbox["tmp_{}".format(obj_data_name)]
        dim = leaf_data.size(-1)
        obj_data = torch.masked_select(leaf_data, mask).reshape(-1, dim)

        return {
            obj_data_name: obj_data,
            "confidence": max_confidence
        }
    
    graph.update_all(_msg_fn, _reduce_fn, etype=("leaf", "to", "obj"))
    
    
    graph.nodes["leaf"].data.pop("tmp_{}".format(obj_data_name))
    
    if inplace:
        return graph
    else:
        return graph.nodes["obj"].data.pop(obj_data_name), graph.nodes["obj"].data.pop("confidence")
        
    
    