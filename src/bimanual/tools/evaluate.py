import os

import re

from torch.utils.data import dataset
import numpy as np
import dgl
from src.bimanual.architectures.build import build_model
from torch._C import device
import yaml
import click
import torch
from omegaconf import OmegaConf

from src.bimanual.data import build_dataset
from src.utils import set_seed
from fvcore.nn import giou_loss
import pandas as pd

arm_oks_k = torch.tensor([0.079, 0.072, 0.062])


def iou(boxA, boxB, eps=1e-7):
    """

    Args:
        boxA: (N, 4)
        boxB:(N, 4)

    Returns:
        _type_: _description_
    """
    x1, y1, x2, y2 = boxA.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxB.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    # Interaction
    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    
    # Union
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)
    
    return iouk

mse_loss = torch.nn.MSELoss(reduction='none')
def mse(target, inp, average=True):
    loss = mse_loss(inp, target)
    n, _ = loss.shape
    loss = loss.reshape(n, -1, 2)
    loss = torch.sqrt(loss.sum(-1))
    
    if average:
        loss = torch.mean(loss, -1)
    
    return loss

# def convert_human(human):
#     h_box_center = human[:, -4:].reshape(-1, 2, 2).mean(1)
#     return torch.cat([human[:, :-4], h_box_center], -1)

# def convert_obj(obj):
#     box_center = obj.reshape(-1, 2, 2).mean(1)
#     return box_center

def get_square_root_area_arm(arm):
    xmin, xmax = torch.min(arm[:, :, 0], dim=-1)[0], torch.max(arm[:, :, 0], dim=-1)[0]
    ymin, ymax = torch.min(arm[:, :, 1], dim=-1)[0], torch.max(arm[:, :, 1], dim=-1)[0]
    
    return torch.sqrt((xmax - xmin) * (ymax - ymin))

def compute_keypoint_metric(gt, pred):
    gt = gt.reshape(len(gt), -1, 2)
    pred = pred.reshape(len(pred), -1, 2)
    
    # N, 1
    s = get_square_root_area_arm(gt)
    std = s.view(-1, 1) * arm_oks_k.to(s.device).view(1, -1)
        
    # N, 3
    d = torch.linalg.norm(gt - pred, dim=-1)
    
    # (N,)
    oks = torch.mean(torch.exp(-d**2 / (2 * std**2)), 1)
    
    return oks
        

def compute_human_metrics(gt, pred):
    gt_arm, gt_hand = gt[:, :-4], gt[:, -4:]
    pred_arm, pred_hand = pred[:, :-4], pred[:, -4:]
    
    return compute_keypoint_metric(gt_arm, pred_arm), iou(gt_hand, pred_hand)

def compute_obj_metrics(gt, pred):
    return iou(gt, pred)

def compute_metric(h_gt, o_gt, h_pred, o_pred, h_weight, o_weight,  is_mean=False):
    """_summary_

    Args:
        h_gt (tensor): Size of (N, seq_len, 4)
        o_gt (tensor): Size of (N, seq_len, 4)
        h_pred (tensor): Size of (N, seq_len, 4)
        o_pred (tensor): Size of (N, seq_len, 4)
        h_weight (tensor): Size of (N, seq_len)
        o_weight (tensor): Size of (N, seq_len)
        is_mean (bool, optional)

    Returns:
        losses: tuple
    """
    
    h_mask = (h_weight > 0).view(-1, 1).float()
    o_mask = (o_weight > 0).view(-1, 1).float()
    
    
    # print(h_mask.shape, o_mask.shape, h_weight.shape, o_weight.shape, h_gt.shape, o_gt.shape)
    
    n, seq_len, _ = h_gt.shape
    h_gt = h_gt.view(-1, h_gt.size(-1))
    o_gt = o_gt.view(-1, o_gt.size(-1))
    h_pred = h_pred.view(-1, h_pred.size(-1))
    o_pred = o_pred.view(-1, o_pred.size(-1))
    
    h_kp_loss, h_box_loss = compute_human_metrics(h_gt, h_pred)
    o_loss = compute_obj_metrics(o_gt, o_pred)
    
    # print(h_loss, h_weight)

    h_kp_loss, h_box_loss = [x.reshape(-1, seq_len) for x in [h_kp_loss, h_box_loss]]
    o_loss = o_loss.reshape(-1, seq_len)

    # Average over the sequence and num nodes each graph
    h_kp_loss, h_box_loss = [torch.sum(x * h_weight.unsqueeze(-1), 0) for x in [h_kp_loss, h_box_loss]]
    o_loss = torch.sum(o_loss * o_weight.unsqueeze(-1), 0)

    return (h_kp_loss, h_box_loss, o_loss)
    
    
    
    
    

@click.command()
@click.option("--data_cfg_path", default="src/bimanual/config/dataset/bimanual_clip.yaml")
@click.option("--checkpoint_path")
@click.option("--multiple", type=bool, default=False)
@click.option("--folder_pattern")
@click.option("--file_name")
@click.option("--gpu", type=int, default=0)
@click.option("--seed", default=100)
def main(data_cfg_path, checkpoint_path, multiple, folder_pattern, file_name, gpu, seed):    
    set_seed(seed, deterministic=True)
    torch.set_num_threads(1)
    if gpu is not None:
        device = torch.device("cuda:{}".format(gpu))
    else:
        device = "cpu"
        
    data_cfg = OmegaConf.load(data_cfg_path)
    overload_cfg = OmegaConf.create({        
        "graph_path": data_cfg.graph_path,
        "numpy_path": data_cfg.numpy_path,
        "filename": data_cfg.filename,
        "shuffle": False,
        "augment": False,
        # "include": ["pour", "wipe"]
    })

    if not multiple:
        models = [checkpoint_path]
    else:
        models = []
        assert os.path.isdir(checkpoint_path)
        for folder_name in os.listdir(checkpoint_path):
            if folder_name.startswith(folder_pattern):
            # if re.match(".*{}.*".format(folder_pattern), folder_name) is not None: 
                if os.path.exists(os.path.join(checkpoint_path, folder_name, file_name)):
                    models.append(os.path.join(checkpoint_path, folder_name, file_name))
    
    out = {}
    data = {}
    df_index = []    
    mean_human_kp, mean_human_box = 0, 0
    mean_obj = 0        
    for fpath in models:
        out[fpath] = {}
        saved = torch.load(fpath, map_location=device)
        saved_cfg = OmegaConf.create(saved["config"])
        dataset_cfg = saved_cfg.dataset

        # Make sure all the data config use the same data source
        dataset_cfg = OmegaConf.merge(dataset_cfg, overload_cfg)
        print(dataset_cfg)
                
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
        
        # Load the model
        model = build_model(saved_cfg.model.architecture)
        model.load_state_dict(saved["model_state"])
        model.eval()
        model = model.to(device)
        
        obs_len = dataset_cfg.obs_len
        
        # load evaluator
        loss_human_kp, loss_human_box, loss_obj = None, None, None
        num_graph = 0
        for batch in test_loader:
            cpu_graph, idx = batch
            graphs = cpu_graph.to(device)
            graphs = dgl.unbatch(graphs)
            for g in graphs:
                n_human, n_obj = g.num_nodes("human"), g.num_nodes("obj")
                                    
                if test_dataset.crnn_format:
                    # In crnn humans are treated as objects. We want to exclude the dummy human from tthe objects
                    n_obj = n_obj - n_human
                    
                g.nodes["obj"].data["weight"] = torch.ones(g.num_nodes("obj"), device=device).float() / n_obj
                g.nodes["human"].data["weight"] = torch.ones(n_human, device=device).float() /  n_human
            
            num_graph += len(graphs)
            graphs = dgl.batch(graphs)
            
            pred_len = g.nodes["human"].data["y"].size(1)
            (yh, yo), (ypred_h, ypred_o), _ = model(graphs, pred_len, additional_data, is_training=False)
            h_weight, o_weight = graphs.nodes["human"].data["weight"], graphs.nodes["obj"].data["weight"]
            
            if test_dataset.crnn_format:
                is_true_obj = (1 - graphs.nodes["obj"].data["is_human"]).bool()
                o_weight = model.select_obj(o_weight, is_true_obj)

            h_kp_loss, h_box_loss, o_loss = compute_metric(yh, yo, ypred_h, ypred_o, h_weight, o_weight, is_mean=True)
            
            # num_human += len(h_loss)
            # num_obj += len(o_loss)


            # h_loss = h_loss.sum(0)
            # o_loss = o_loss.sum(0)
            
            h_kp_loss = h_kp_loss.detach().cpu().numpy()
            h_box_loss = h_box_loss.detach().cpu().numpy()
            o_loss = o_loss.detach().cpu().numpy()
            
                    
            if loss_human_kp is None:
                loss_human_kp = h_kp_loss
                loss_human_box = h_box_loss
            else:
                loss_human_kp = loss_human_kp + h_kp_loss
                loss_human_box = loss_human_box + h_box_loss
            if loss_obj is None:
                loss_obj = o_loss
            else:
                loss_obj = loss_obj + o_loss
    
        
        loss_human_kp = loss_human_kp / num_graph
        loss_human_box = loss_human_box / num_graph
        loss_obj = loss_obj / num_graph

        print(fpath)
        print("Loss_human key points:", loss_human_kp, np.mean(loss_human_kp))
        print("Loss_human boxes:", loss_human_box, np.mean(loss_human_box))

        print("Loss obj:", loss_obj, np.mean(loss_obj))
        mean_human_kp += np.mean(loss_human_kp) / len(models)
        mean_human_box += np.mean(loss_human_box) / len(models)
        
        mean_obj += np.mean(loss_obj) / len(models)
        
    #     for t in range(len(loss_human_kp)):
    #         if t not in data:
    #             data[t] = []
            
    #         data[t].append(loss_human[t])
    #         data[t].append(loss_obj[t])
            
    #         out[fpath]["human"] = loss_human
    #         out[fpath]["obj"] = loss_obj
            
    #     df_index += ["human_{}".format(fpath), "obj_{}".format(fpath)]
         
    #     del model
    #     del test_dataset
    #     del test_loader
    #     torch.cuda.empty_cache()
    # df = pd.DataFrame(data=data, index=df_index)
    # df = df.sort_index(axis=0)
    # print(df)
    
    print("Mean human keypoint loss", mean_human_kp)
    print("Mean human box loss", mean_human_box)
    print("Mean obj loss", mean_obj)


if __name__ == "__main__":
    main()
