from email.policy import default
import os

import re
from typing import Tuple
from pyrsistent import l

from torch.utils.data import dataset

import numpy as np
import dgl
from src.wbhm.architectures.build import build_model
from torch._C import device
import yaml
import click
import torch
from omegaconf import OmegaConf

from src.wbhm.data import build_dataset
from src.utils import set_seed

import pandas as pd

mse_loss = torch.nn.MSELoss(reduction='none')
def compute_metric(h_gt, o_gt, h_pred, o_pred, h_weight, o_weight,  is_mean=False):
    _, seq_len, _ = h_gt.size()

    h_loss = mse_loss(h_pred.reshape(len(h_gt), seq_len, -1, 3),
                      h_gt.reshape(len(h_gt), seq_len, -1, 3))
                
    o_loss = mse_loss(o_pred.reshape(len(o_gt), seq_len, -1, 3),
                      o_gt.reshape(len(o_gt), seq_len, -1, 3))
    
    h_loss = torch.sqrt(torch.sum(h_loss, -1)).mean(dim=-1)
    o_loss = torch.sqrt(torch.sum(o_loss, -1)).mean(dim=-1)
    
    h_loss = torch.sum(h_loss * h_weight.unsqueeze(-1), dim=0)
    o_loss = torch.sum(o_loss * o_weight.unsqueeze(-1), dim=0)

    return h_loss, o_loss

def copy_to_k_interval(data: torch.Tensor, k: int) -> torch.Tensor:
    # Example: data: 0010010, k=1
    # prev: 0100100, after: 0001001
    # output = 0100100 + 0010010 + 0001001 = 0111111
    prev, after = [], []
    for i in range(k):
        pos = i + 1
        # shift data left k positions
        data_prev = torch.zeros_like(data)
        data_prev[:, :-pos] = data[:, pos:].clone()

        # shift data right k positions
        data_after = torch.zeros_like(data)
        data_after[:, pos:] = data[:, :-pos].clone()

        prev.append(data_prev)
        after.append(data_after)

    out = torch.stack(prev + [data] + after, dim=1)
    out = torch.sum(out, dim=1)
    out[out > 1] = 1
    return out

def counting(gt: torch.Tensor, pred: torch.Tensor, k: int) -> Tuple[torch.Tensor]:
    gt_k = copy_to_k_interval(gt, k)
    pred_k = copy_to_k_interval(pred, k)
    epsilon = 1e-5

    tp = torch.sum(gt_k * pred_k, dim=1)
    fp = torch.sum((1 - gt_k) * pred_k, dim=1)
    fn = torch.sum(gt_k * (1 - pred_k), dim=1)

    return tp, fp, fn

def compute_switch_metric(gt_switch: torch.Tensor, pred_switch: torch.Tensor, weight: torch.Tensor, num_k: int) -> dict:
    predicted_on, predicted_off = get_switch_on_off_label(pred_switch)
    gt_on, gt_off = get_switch_on_off_label(gt_switch)
    mask_gt = torch.any(gt_switch == 0, dim=1) & torch.any(gt_switch == 1, dim = 1)
    mask_pred = torch.any(pred_switch == 0, dim=1) & torch.any(pred_switch == 1, dim = 1)
    
    mask = mask_gt & mask_pred
    mask = mask.float()
    out = {}
    for k in range(num_k):                
        tp_on, fp_on, fn_on = counting(gt_on, predicted_on, k=k)
        tp_off, fp_off, fn_off = counting(gt_off, predicted_off, k=k)
        
        out[k] = {
            "on": {
                "tp": torch.sum(tp_on * weight * mask).detach().cpu().numpy(),
                "fp": torch.sum(fp_on * weight * mask).detach().cpu().numpy(),
                "fn": torch.sum(fn_on * weight * mask).detach().cpu().numpy(),
            },
            "off": {
                "tp": torch.sum(tp_off * weight * mask).detach().cpu().numpy(),
                "fp": torch.sum(fp_off * weight * mask).detach().cpu().numpy(),
                "fn": torch.sum(fn_off * weight * mask).detach().cpu().numpy(),
            }
        }

    return out


def get_switch_on_off_label(switch):
    switch_on = torch.zeros_like(switch)
    switch_off = torch.zeros_like(switch)

    # from off --> on
    switch_on[:, 1:] = (switch[:, :-1] == 0) & (switch[:, 1:] == 1)

    # from on --> off
    switch_off[:, 1:] = (switch[:,  :-1] == 1) & (switch[:, 1:] == 0)

    return switch_on, switch_off
    
    

@click.command()
@click.option("--data_cfg_path", default="src/wbhm/config/dataset/wbhm.yaml")
@click.option("--checkpoint_folder")
@click.option("--out_name", default="out")
@click.option("--batch_size", type=int, default=128)
@click.option("--exclude_table", type=bool, default=True)
@click.option("--exclude_stationary", type=bool, default=False)
@click.option("--pred_len", type=int)
@click.option("--obs_len", type=int)
@click.option("--gpu")
@click.option("--seed", default=100)
def main(data_cfg_path, checkpoint_folder, exclude_table, exclude_stationary, batch_size, gpu, seed, obs_len, pred_len, out_name):    
    set_seed(seed, deterministic=True)
    torch.set_num_threads(1)
    if gpu is not None:
        device = torch.device("cuda:{}".format(gpu))
    else:
        device = "cpu"
        
    data_cfg = OmegaConf.load(data_cfg_path)
    overload_cfg = {        
        "graph_path": data_cfg.graph_path,
        "numpy_path": data_cfg.numpy_path,
        "filename": data_cfg.filename,
        "batch_size": batch_size,
        "shuffle": False,
        "augment": False,
    }
    
    if pred_len is None:
        pred_len = 20
    
    if obs_len is None:
        obs_len = 10
    
    overload_cfg["pred_len"] = pred_len
    overload_cfg["obs_len"] = obs_len
    overload_cfg["filename"] = "obs{}_pred{}".format(obs_len, pred_len)
        
    overload_cfg = OmegaConf.create(overload_cfg)

    models = []
    assert os.path.isdir(checkpoint_folder)
    for fname in os.listdir(checkpoint_folder):
        models.append(os.path.join(checkpoint_folder, fname))
    
    out = {}
    data = {}
    df_index = []            
    for fpath in models:
        if not fpath.endswith(".pt"):
            continue
        
        out[fpath] = {}
        saved = torch.load(fpath, map_location=device)
        saved_cfg = OmegaConf.create(saved["config"])
        dataset_cfg = saved_cfg.dataset

        # Make sure all the data config use the same data source
        dataset_cfg = OmegaConf.merge(dataset_cfg, overload_cfg)
                
        test_dataset, test_loader = build_dataset(dataset_cfg, "test", training=True)

        data_stats = {}
        for k, v in test_dataset.additional_info["data_stats"].items():
            if ("human" in k) or ("obj" in k) or ("point" in k):
                data_stats[k] = v.to(device)
            else:
                data_stats[k] = v

        additional_data = {
            "data_stats": data_stats, 
            "post_process": True
        }
        
        if test_dataset.crnn_format:
            additional_data["crnn_return_all"] = True
        
        # Load the model
        model = build_model(saved_cfg.model.architecture)
        model.load_state_dict(saved["model_state"])
        model.eval()
        model = model.to(device)
        
        # load evaluator
        loss_human, loss_obj = None, None
        total_switch_loss = {}
        num_graphs = 0
        for batch in test_loader:
            cpu_graph, idx = batch
            graphs = cpu_graph.to(device)
            
            pred_len = overload_cfg.pred_len
            (yh, yo), (ypred_h, ypred_o), additional_output = model(graphs, pred_len, additional_data, is_training=False)
            
            pred_switch, gt_switch = additional_output["pred_switch_label"], additional_output["gt_switch_label"]
            
            
            graphs.nodes["obj"].data["yo"] = yo
            
            graphs = dgl.unbatch(graphs)
            for g in graphs:
                obj_mask = torch.ones(g.num_nodes("obj"), device=device).float()
                human_mask = torch.ones(g.num_nodes("human"), device=device).float()
                
                                    
                if exclude_table:
                    node_type = torch.argmax(g.nodes["obj"].data["type"], dim=1)
                    node_type = [test_dataset.additional_info["object_types"][i].lower() for i in node_type]
                    is_not_stationary = torch.tensor([(x != "table" and x != "ladder_big") for x in node_type])
                    
                    obj_mask = is_not_stationary.float()

                if exclude_stationary:
                    yo_vel = g.nodes["obj"].data["yo"][:, 1:] - g.nodes["obj"].data["yo"][:, :-1]
                    yo_vel = yo_vel.mean(-1)
                    obj_mask = obj_mask * torch.any(yo_vel > 0.1, -1).float()


                if test_dataset.crnn_format:
                    # In CRNN, we create dummy objects to represent humans. 
                    # We need to exclude them while computing the loss
                    # print("before", obj_mask)
                    obj_mask = obj_mask * (1 - g.nodes["obj"].data["is_human"])                                                    
                    # print("after", obj_mask)
                    
                n_objs = torch.sum(obj_mask)
                if n_objs == 0:
                    n_objs = 1
                                        
                g.nodes["obj"].data["weight"] = obj_mask / n_objs
                g.nodes["human"].data["weight"] = human_mask /  g.num_nodes("human")
                
            num_graphs += len(graphs)
            
            graphs = dgl.batch(graphs)

            
            h_weight, o_weight = graphs.nodes["human"].data["weight"], graphs.nodes["obj"].data["weight"]
            
            h_loss, o_loss = compute_metric(yh, yo, ypred_h, ypred_o, h_weight, o_weight, is_mean=True)
            switch_losses = compute_switch_metric(gt_switch, pred_switch, h_weight, num_k=10)
            
            for k in switch_losses.keys():
                if k not in total_switch_loss:
                    total_switch_loss[k] = switch_losses[k]
                else:
                    for ty in switch_losses[k].keys():
                        for me in switch_losses[k][ty].keys():
                            total_switch_loss[k][ty][me] = total_switch_loss[k][ty][me] + switch_losses[k][ty][me]
                        
            h_loss = h_loss.detach().cpu().numpy()
            o_loss = o_loss.detach().cpu().numpy()
            
                    
            if loss_human is None:
                loss_human = h_loss
            else:
                loss_human = loss_human + h_loss
            if loss_obj is None:
                loss_obj = o_loss
            else:
                loss_obj = loss_obj + o_loss
            
        loss_human = np.round(loss_human / num_graphs)
        loss_obj = np.round(loss_obj / num_graphs)

        mean_human = np.mean(loss_human)
        mean_obj = np.mean(loss_obj)

        for k in total_switch_loss.keys():
            for ty in total_switch_loss[k].keys():                
                tp = total_switch_loss[k][ty].pop("tp")
                fp = total_switch_loss[k][ty].pop("fp")
                fn = total_switch_loss[k][ty].pop("fn")
                
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * precision * recall / (precision + recall)
                
                total_switch_loss[k][ty] = {
                    "precision": np.round(precision, 2),
                    "recall": np.round(recall, 2),
                    "f1": np.round(f1, 2),
                }
                
        print("Switch loss", total_switch_loss)


        print(fpath)
        print("Loss_human:", loss_human, "mean:", mean_human)
        print("Loss obj:", loss_obj, "mean:", mean_obj)
        
        for t in range(len(loss_human)):
            if t not in data:
                data[t] = []
            
            data[t].append(loss_human[t])
            data[t].append(loss_obj[t])
            
            for k in total_switch_loss.keys():
                for ty in total_switch_loss[k].keys():
                    for me in total_switch_loss[k][ty].keys():
                        data[t].append(total_switch_loss[k][ty][me])
                        
            
            out[fpath]["human"] = loss_human
            out[fpath]["obj"] = loss_obj
            
        df_index += ["human_{}".format(fpath), "obj_{}".format(fpath)]
        for k in total_switch_loss.keys():
            for ty in total_switch_loss[k].keys():
                for me in total_switch_loss[k][ty].keys():         
                    df_index.append("k={}_type={}_metric={}_{}".format(k, ty, me, fpath))
                    
        del model
        del test_dataset
        del test_loader
        torch.cuda.empty_cache()
    df = pd.DataFrame(data=data, index=df_index)
    df = df.sort_index(axis=0)
    df.to_csv(os.path.join(checkpoint_folder, "{}.csv".format(out_name)))
    print(df)

if __name__ == "__main__":
    main()
