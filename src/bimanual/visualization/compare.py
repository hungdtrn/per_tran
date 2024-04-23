import cv2
import click

import dgl
from src.bimanual.data import build_dataset
from src.bimanual.architectures import build_model
from src.bimanual.tools.evaluate import compute_human_metrics, compute_obj_metrics

import torch
import numpy as np
from src.shared.visualization.compare import visualize
from .utils import draw_human, draw_object, to_pixel

def compute_metric(h_gt, o_gt, h_pred, o_pred, h_weight, o_weight,  is_mean=False):
    n, seq_len, _ = h_gt.shape
    h_gt = h_gt.view(-1, h_gt.size(-1))
    o_gt = o_gt.view(-1, o_gt.size(-1))
    h_pred = h_pred.view(-1, h_pred.size(-1))
    o_pred = o_pred.view(-1, o_pred.size(-1))
    
    h_kp_loss, h_box_loss = compute_human_metrics(h_gt, h_pred)
    o_loss = compute_obj_metrics(o_gt, o_pred)
    
    # print(h_loss, h_weight)

    h_kp_loss = h_kp_loss.reshape(-1, seq_len).mean(1) * h_weight
    h_box_loss = h_box_loss.reshape(-1, seq_len).mean(1) * h_weight

    o_loss = o_loss.reshape(-1, seq_len).mean(1) * o_weight
    
    return ((h_kp_loss + h_box_loss) / 2, o_loss)
    
    


def get_results(model, batch, dataset, device, kwargs):
    cpu_graph, idx = batch
    graphs = cpu_graph.to(device)
    
    data_stats = {}
    for k, v in dataset.additional_info["data_stats"].items():
        if ("human" in k) or ("obj" in k) or ("point" in k):
            data_stats[k] = v.to(device)
        else:
            data_stats[k] = v

    additional_data = {
        "data_stats": data_stats, 
        "post_process": True
    }

    
    with graphs.local_scope():
        graphs = dgl.unbatch(graphs)
        losses = []

        for g in graphs:
            n_human, n_obj = g.num_nodes("human"), g.num_nodes("obj")
                                
            if dataset.crnn_format:
                # In crnn humans are treated as objects. We want to exclude the dummy human from tthe objects
                n_obj = n_obj - n_human
                
            g.nodes["obj"].data["weight"] = torch.ones(g.num_nodes("obj"), device=device).float() / n_obj
            g.nodes["human"].data["weight"] = torch.ones(n_human, device=device).float() /  n_human
        
        graphs = dgl.batch(graphs)
        
        pred_len = g.nodes["human"].data["y"].size(1)
        (yh, yo), (ypred_h, ypred_o), _ = model(graphs, pred_len, additional_data, is_training=False)
        h_weight, o_weight = graphs.nodes["human"].data["weight"], graphs.nodes["obj"].data["weight"]

        if dataset.crnn_format:
            is_true_obj = (1 - graphs.nodes["obj"].data["is_human"]).bool()
            o_weight = model.select_obj(o_weight, is_true_obj)

        h_loss, o_loss = compute_metric(yh, yo, ypred_h, ypred_o, h_weight, o_weight, is_mean=True)
        
        if dataset.crnn_format:
            tmp_obj = torch.zeros(graphs.num_nodes("obj")).to(o_loss.device)
            tmp_obj[is_true_obj] = o_loss
            o_loss = tmp_obj
            
        graphs.nodes["human"].data["loss"] = h_loss
        graphs.nodes["obj"].data["loss"] = o_loss
        for g in dgl.unbatch(graphs):
            # loss = g.nodes["human"].data["loss"].sum() + g.nodes["obj"].data["loss"].sum()
            loss = g.nodes["obj"].data["loss"].sum()
            
            losses.append(loss.detach().cpu().numpy())
    
    return np.array(losses)

@click.command()
@click.option("--data_cfg_path", default="src/bimanual/config/dataset/bimanual_clip.yaml")
@click.option("--checkpoint_path")
@click.option("--video_path")
@click.option("--part", default="test")
@click.option("--seed", default=0)
@click.option("--shuffle", type=bool, default=True)
@click.option("--batch_size", default=128)
@click.option("--duality_format", default=False)
@click.option("--is_ego", default=False, type=bool)
@click.option("--is_save", type=bool, default=False)
@click.option("--descending", type=bool, default=True)
@click.option("--is_show", type=bool, default=True)
@click.option("--gpu", type=int, default=0)
@click.option("--is_draw_switch", type=bool, default=False)
@click.option("--model_name", type=str)
@click.option("--save_path")
@click.option("--model1_path")
@click.option("--model1_name")
@click.option("--model2_path")
@click.option("--model2_name")
@click.option("--drop", type=float)
def visualize_compare(**kwargs):
    visualize(kwargs, build_dataset, build_model,
              get_results, draw_object, draw_human,
              to_pixel)
    

if __name__ == "__main__":
    visualize_compare()