import sys
import os
import logging

import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
from fvcore.nn import giou_loss

from src.bimanual.data import build_dataset
from src.bimanual.architectures import build_model
from src.shared.tools.train_utils import start_train, BaseLoss
from src.shared.tools.train_utils import BaseLoss

    
def weighted_bce(output, target, pos_weight):
    eps = 1e-12
    loss = -(pos_weight * target * output.clamp(min=eps).log() + (1 - target)*(1 - output).clamp(min=eps).log())
    return loss

class ClipLoss(BaseLoss):
    def __init__(self, additional_data) -> None:
        super().__init__(additional_data)
        self.bce = torch.nn.BCELoss(reduction="none")
        self.mse = torch.nn.MSELoss(reduction="none")

    def box_loss(self, pred, gt):
        _, seq_len, _ = pred.size()
        
        gt = gt.reshape(-1, gt.size(-1))
        
        loss = giou_loss(pred, gt)
        return loss.reshape(-1, seq_len)

    def mse_loss(self, pred, gt):
        
        loss = self.mse(pred, gt)
        loss = torch.sum(loss, -1)
        
        return loss

    def prediction_loss(self, h_pred, h_gt, o_pred, o_gt, additional_output):
        if self.additional_data.get("is_iou", False):
            h_loss = self.box_loss(h_pred, h_gt) 
            o_loss = self.box_loss(o_pred, o_gt) 
        else:
            h_loss = self.mse_loss(h_pred, h_gt)
            o_loss = self.mse_loss(o_pred, o_gt) 

        # Averate over the sequence 
        # h_loss = torch.mean(h_loss, dim=-1)
        # o_loss = torch.mean(o_loss, dim=-1)
        eps = 1e-5
        h_loss = torch.sqrt(torch.sum(h_loss , dim=-1) + eps)
        o_loss = torch.sqrt(torch.sum(o_loss, dim=-1) + eps)

        # Average over the batch
        h_loss = torch.mean(h_loss)
        o_loss = torch.mean(o_loss)

        return {
            "human_loss": h_loss,
            "object_loss": o_loss,
        }

    def switch_loss(self, additional_output, weighted_switch):
        gt_switch_score, pred_switch_score = additional_output["gt_switch_score"], additional_output["pred_switch_score"]
        obs_len = self.additional_data["obs_len"]
        # gt_switch_score = gt_switch_score[:, :obs_len]
        # pred_switch_score = pred_switch_score[:, :obs_len]


        weight = self.additional_data["dataset_additional_info"]["switch_label_stats"]
        # frac = weight[1] / weight[0]
        # s_loss = weighted_bce(pred_switch_score, gt_switch_score, frac) * 1 / frac

        if not weighted_switch:
            s_loss = self.bce(pred_switch_score, gt_switch_score)
        else:
            weight = self.additional_data["dataset_additional_info"]["switch_label_stats"]
            frac = weight[1] / weight[0]
            s_loss = weighted_bce(pred_switch_score, gt_switch_score, frac) * 1 / frac

        s_loss = torch.mean(torch.mean(s_loss, 1))

        return {
            "switch_loss": s_loss
        }

    def segment_loss(self, additional_output):
        gt_segment_score, pred_segment_score = additional_output["gt_segment_score"], additional_output["pred_segment_score"]
        gt_switch_label = additional_output["gt_switch_score"] >= 0.5
        pred_segment_score = pred_segment_score * gt_switch_label.float()
        gt_segment_score = gt_segment_score * gt_switch_label.float()
        
        # gt_segment_score = torch.stack([1 - gt_segment_score, gt_segment_score], -1)
        # pred_segment_score = torch.stack([1 - pred_segment_score, pred_segment_score], -1)
        weight = self.additional_data["dataset_additional_info"]["segment_label_stats"]
        frac = weight[1] / weight[0]
        # Average by sequence
        s_loss = torch.mean(weighted_bce(pred_segment_score, gt_segment_score, frac), 1)
        # s_loss = torch.mean(self.bce(pred_segment_score, gt_segment_score), 1)

        # s_loss = s_loss * weight.unsqueeze(0).to(s_loss.device)

        s_loss = torch.mean(s_loss)
        return {
            "segment_loss": s_loss
        }


    def get_model_output(self, model, batch, **kwargs):
        is_training = self.additional_data["is_training"]
        device = self.additional_data["device"]
        g, _ = batch
        g = g.to(device)
        pred_len = g.nodes["human"].data["y"].size(1)
        (yh, yo), (ypred_h, ypred_o), additional_output = model(g, pred_len, self.additional_data, is_training=is_training)

        return (yh, yo), (ypred_h, ypred_o), additional_output

    def forward(self, model, batch, **kwargs):
        device = self.additional_data["device"]
        losses = {}
        (yh, yo), (ypred_h, ypred_o), additional_output = self.get_model_output(model, batch)

        lambda_human = self.additional_data.get("lambda_human", 1)
        lambda_obj = self.additional_data.get("lambda_obj", 1)
        
        loss_pred = self.prediction_loss(ypred_h, yh, ypred_o, yo, additional_output)
        total_loss = lambda_human * loss_pred["human_loss"] + lambda_obj * loss_pred["object_loss"]
        losses.update(loss_pred)

        is_use_switch = (additional_output is not None) and ("pred_switch_score" in additional_output) and self.additional_data.get("use_switch_loss", False)
        if is_use_switch:
            lambda_switch = self.additional_data.get("lambda_switch", 1)
            loss_switch = self.switch_loss(additional_output, self.additional_data.get("weighted_switch", False))

            total_loss = total_loss + lambda_switch * loss_switch["switch_loss"]

            losses.update(loss_switch)
            
        is_use_segment = is_use_switch and "pred_segment_score" in additional_output
        if is_use_segment:
            lambda_segment = self.additional_data.get("lambda_segment", 1)
            loss_segment = self.segment_loss(additional_output)
            total_loss = total_loss + lambda_segment * loss_segment["segment_loss"]
            losses.update(loss_segment)
        
        losses["total_loss"] = total_loss
        
        return losses

class VideoLoss(ClipLoss):
    def __init__(self, additional_data) -> None:
        super().__init__(additional_data)
        
    def box_loss(self, pred, gt, mask):
        _, seq_len, _ = pred.size()
        
        gt = gt.reshape(-1, gt.size(-1))
        mask = mask.reshape(-1, 1)
        pred = pred.reshape(-1, pred.size(-1))
        gt = gt.reshape(-1, gt.size(-1))

        loss = giou_loss(pred * mask, gt * mask)
        return loss.reshape(-1, seq_len)
    
    def mse_loss(self, pred, gt, mask):
        return super().mse_loss(pred, gt)
        
    def prediction_loss(self, h_pred, h_gt, o_pred, o_gt, additional_output):
        human_mask, obj_mask = additional_output["human_mask"], additional_output["obj_mask"]
        obs_len = self.additional_data["obs_len"]
        human_mask, obj_mask = human_mask[:, obs_len:], obj_mask[:, obs_len:]
        
        if self.additional_data.get("is_iou", False):
            h_loss = self.box_loss(h_pred, h_gt, human_mask) 
            o_loss = self.box_loss(o_pred, o_gt, obj_mask) 
        else:
            h_loss = self.mse_loss(h_pred, h_gt, human_mask)
            o_loss = self.mse_loss(o_pred, o_gt, obj_mask) 
        human_mask = human_mask / torch.sum(human_mask, 1, keepdim=True)
        obj_mask = obj_mask / torch.sum(obj_mask, 1, keepdim=True)

        # Averate over the sequence 
        h_loss = torch.sum(h_loss * human_mask, dim=-1)
        o_loss = torch.sum(o_loss * obj_mask, dim=-1)
        
        # Average over the batch
        h_loss = torch.mean(h_loss)
        o_loss = torch.mean(o_loss)

        return {
            "human_loss": h_loss,
            "object_loss": o_loss,
        }

    def switch_loss(self, additional_output):
        gt_switch_score, pred_switch_score = additional_output["gt_switch_score"], additional_output["pred_switch_score"]
        switch_mask = self.additional_data["switch_mask"]
        s_loss = self.bce(pred_switch_score, gt_switch_score)
        s_loss = s_loss * (switch_mask[:, :-1] / torch.sum(switch_mask[:, :-1], 1, keepdim=True))
        s_loss = torch.mean(torch.sum(s_loss, 1))
        
        return {
            "switch_loss": s_loss
        }

    def segment_loss(self, additional_output):
        gt_segment_score, pred_segment_score = additional_output["gt_segment_score"], additional_output["pred_segment_score"]
        switch_mask = self.additional_data["switch_mask"]
        segment_loss = self.bce(pred_segment_score, gt_segment_score)
        segment_loss = segment_loss * (switch_mask[:, :-1] / torch.sum(switch_mask[:, :-1], 1, keepdim=True))
        segment_loss = torch.mean(torch.sum(segment_loss, 1))

        return {
            "segment_loss": segment_loss
        }


    def get_model_output(self, model, batch):
        g, _ = batch
        self.additional_data["obs_len"] = g.nodes["human"].data["x"].size(1)
        if self.additional_data.get("use_switch_loss", False):
            self.additional_data["switch_mask"] = g.nodes["human"].data["switch_mask"].to(self.additional_data["device"])
            
        return super().get_model_output(model, batch)

    def forward(self, model, batch):
        device = self.additional_data["device"]
        losses = {}
        (yh, yo), (ypred_h, ypred_o), additional_output = self.get_model_output(model, batch)

        lambda_human = self.additional_data.get("lambda_human", 1)
        lambda_obj = self.additional_data.get("lambda_obj", 1)
        
        loss_pred = self.prediction_loss(ypred_h, yh, ypred_o, yo, additional_output)
        total_loss = lambda_human * loss_pred["human_loss"] + lambda_obj * loss_pred["object_loss"]
        losses.update(loss_pred)

        is_use_switch = (additional_output is not None) and ("pred_switch_score" in additional_output) and self.additional_data.get("use_switch_loss", False)
        if is_use_switch:
            lambda_switch = self.additional_data.get("lambda_switch", 1)
            loss_switch = self.switch_loss(additional_output)

            total_loss = total_loss + lambda_switch * loss_switch["switch_loss"]

            losses.update(loss_switch)

        is_use_segment = (additional_output is not None) and ("pred_segment_score") in additional_output and self.additional_data.get("use_segment_loss", False)
        if is_use_segment:
            lambda_segment = self.additional_data.get("lambda_segment", 1)
            loss_segment = self.segment_loss(additional_output)
            total_loss = total_loss + lambda_segment * loss_segment["segment_loss"]

            losses.update(loss_segment)

        losses["total_loss"] = total_loss
        
        return losses

@hydra.main(config_path="../config", config_name="train")
def start(cfg : DictConfig):
    train_loss_kwargs = {
        "obs_len": cfg.dataset.obs_len,
        "use_switch_loss": ("per_tran" in cfg.model.architecture.name) and ("rule" not in cfg.model.architecture.switch.switch_type),
        "weighted_switch": cfg.model.solver.loss.weighted_switch
    }
    
    val_loss_kwargs = {
        "obs_len": cfg.dataset.obs_len,
    }
    start_train(cfg, build_model_fn=build_model,
                build_dataset_fn=build_dataset,
                loss_class=ClipLoss,
                train_loss_kwargs=train_loss_kwargs,
                val_loss_kwargs=val_loss_kwargs,)

if __name__ == "__main__":
    start()
