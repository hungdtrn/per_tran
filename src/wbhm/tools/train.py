import sys
import os
import logging

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from src.wbhm.data import build_dataset
from src.wbhm.architectures import build_model
from src.shared.tools.train_utils import start_train, BaseLoss

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FORMAT = '[%(levelname)s: %(name)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)


class ClipLoss(BaseLoss):
    def __init__(self, additional_data) -> None:
        super().__init__(additional_data)
        self.bce = torch.nn.BCELoss(reduction="none")
        self.mse = torch.nn.MSELoss(reduction="none")

    def mse_loss(self, pred, gt):
        
        loss = self.mse(pred, gt)
        loss = torch.sum(loss, -1)
        
        return loss

    def prediction_loss(self, h_pred, h_gt, o_pred, o_gt, additional_output):
        h_loss = self.mse_loss(h_pred, h_gt)
        o_loss = self.mse_loss(o_pred, o_gt) 

        # Averate over the sequence 
        # h_loss = torch.mean(h_loss, dim=-1)
        # o_loss = torch.mean(o_loss, dim=-1)
        h_loss = torch.sqrt(torch.sum(h_loss, dim=-1))
        o_loss = torch.sqrt(torch.sum(o_loss, dim=-1))

        # Average over the batch
        h_loss = torch.mean(h_loss)
        o_loss = torch.mean(o_loss)

        return {
            "human_loss": h_loss,
            "object_loss": o_loss,
        }

    def switch_loss(self, additional_output):
        gt_switch_score, pred_switch_score = additional_output["gt_switch_score"], additional_output["pred_switch_score"]

        s_loss = self.bce(pred_switch_score, gt_switch_score)
        s_loss = torch.mean(torch.mean(s_loss, 1))
        distance_beta = additional_output["distance_beta"]
        out =  {
            "switch_loss": s_loss,
        }

        if distance_beta is not None:
            out["distance_reg"] = 1 / distance_beta ** 2

        return out

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
        (yh, yo), (ypred_h, ypred_o), additional_output = self.get_model_output(model, batch, **kwargs)

        lambda_human = self.additional_data.get("lambda_human", 1)
        lambda_obj = self.additional_data.get("lambda_obj", 1)
        # if not self.additional_data["is_training"]:
        #     lambda_obj = 2
        
        loss_pred = self.prediction_loss(ypred_h, yh, ypred_o, yo, additional_output)
        total_loss = lambda_human * loss_pred["human_loss"] + lambda_obj * loss_pred["object_loss"]
        losses.update(loss_pred)

        is_use_switch = (additional_output is not None) and ("pred_switch_score" in additional_output) and self.additional_data.get("use_switch_loss", False)
        if is_use_switch:
            lambda_switch = self.additional_data.get("lambda_switch", 1)
            lambda_reg = self.additional_data.get("lambda_reg", 0.1)
            loss_switch = self.switch_loss(additional_output)
            #print(lambda_reg)
            total_loss = total_loss + lambda_switch * loss_switch["switch_loss"]

            if "distance_reg" in loss_switch:
                total_loss = total_loss + lambda_reg * loss_switch["distance_reg"]

            losses.update(loss_switch)
        
        losses["total_loss"] = total_loss
        
        return losses

@hydra.main(config_path="../config", config_name="train")
def start(cfg : DictConfig):
    train_loss_kwargs = {
        "obs_len": cfg.dataset.obs_len,
        "use_switch_loss": ("per_tran" in cfg.model.architecture.name) and ("rule" not in cfg.model.architecture.switch.switch_type),
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
