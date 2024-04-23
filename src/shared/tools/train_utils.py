import os
import sys
import time
import logging

import yaml
import torch
import torch.nn as nn
import wandb
from yacs.config import CfgNode
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from .solver import EarlyStopping, GradClipper, GradExplodeChecker, build_optimizer, build_scheduler
from src.utils import set_seed, mkdir_if_not_exist

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FORMAT = '[%(levelname)s: %(name)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)


class LossTracker:
    def __init__(self, num_batches) -> None:
        self.losses = {}
        self.num_batches = num_batches
        
    def update(self, losses):
        for k in losses.keys():
            if k not in self.losses:
                self.losses[k] = 0
            
            self.losses[k] = self.losses[k] + losses[k].item() / self.num_batches

    def get(self, name=None):
        if name is None:
            return self.losses
        
        return self.losses.get(name)


class BaseLoss(nn.Module):
    def __init__(self, additional_data) -> None:
        super().__init__()
        self.additional_data = additional_data

    def forward(self, model, batch):
        pass


class CustomLogger:
    # Contain three things
    # 1. The logger of the system
    # 2. The wandb
    # 3. The tensorboard
    def __init__(self, cfg, model) -> None:
        self.wandb = None
        self.logger = None
        self.board_writer = None

        project_name = cfg.project_name
        logging_dir = os.getcwd()
        experiment_name = os.path.basename(logging_dir)

        if cfg.is_wandb:
            wandb.init(project=project_name, 
                    name=experiment_name,
                    dir=PROJECT_PATH,
                    config=yaml.safe_load(OmegaConf.to_yaml(cfg)),
                    settings=wandb.Settings(start_method="fork"))
            wandb.watch(model, log='gradients')
            self.wandb = wandb

        if cfg.is_tensorboard:
            tensorboard_path = os.path.join(logging_dir, "tensorboard")
            mkdir_if_not_exist(os.path.join(logging_dir, "tensorboard"))
            
            self.board_writer = SummaryWriter(log_dir=tensorboard_path)
        
        self.logger = logging.getLogger(experiment_name)
        if cfg.is_logging:
            logger_path = os.path.join(logging_dir, "{}.txt".format(experiment_name))
            fn = logging.FileHandler(logger_path, mode='w')
            self.logger.addHandler(fn)        

        self.cfg = cfg

    def log(self, msg):
        self.logger.info(msg)

    def log_param(self, model):
        pass

    def log_grad(self, model):
        for name, param in model.named_parameters():
            if 'bn' not in name:
                if param.grad is None:
                    if self.cfg.debug_gradient:
                        print("None", name)
                    # board_writer.add_histogram("Gradient/{}".format(name), torch.zeros_like(param), c_iter)
                else:
                    if self.cfg.debug_gradient:
                        if torch.all(param.grad == 0):
                            print("Zero", name)
                    # board_writer.add_histogram("Gradient/{}".format(name), param.grad, c_iter)
                    
                # board_writer.add_histogram("Param/{}".format(name), param, c_iter)

    def log_losses(self, losses, stage):
        for k, v in losses.items():
            self.logger.info("{} {}: {}".format(stage, k, v))

            if self.wandb is not None:
                self.wandb.log({"{}/{}".format(stage, k): v})
       
        
class ModelIO:
    def __init__(self, cfg, model_cfg, build_model_fn,
                 optimizer_cfg=None, scheduler_cfg=None, save_path="", 
                 device='cpu', is_saving=True, filename="") -> None:
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.save_path = save_path
        self.device = device
        self.is_saving = is_saving
        self.filename = filename
        self.build_model_fn = build_model_fn

    def save(self, epoch, model, optimizer, scheduler):
        if self.is_saving:
            cp = {
                "epoch": epoch + 1,
                "config": yaml.safe_load(OmegaConf.to_yaml(self.cfg)),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }

            if scheduler is not None:
                cp["scheduler_state"] = scheduler.state_dict()

            if self.filename != "":
                filename = "{}_best.pt".format(self.filename)
            else:
                filename = "best.pt"

            torch.save(cp, os.path.join(self.save_path, filename))
            if epoch < 200:
                torch.save(cp, os.path.join(self.save_path, "epoch200_{}".format(filename)))



    def reload(self, additional_info):
        num_data = additional_info["num_data"]

        # Rebuild the model, optimizer, and scheduler
        model = self.build_model_fn(self.model_cfg.architecture)
        model = model.to(self.device)
        optimizer = build_optimizer(model, self.optimizer_cfg)
        scheduler = build_scheduler(optimizer, self.scheduler_cfg, num_data=num_data, batch_size=self.cfg.dataset.batch_size)
        
        # Reload the checkpoint
        if self.filename != "":
            filename = "{}_best.pt".format(self.filename)
        else:
            filename = "best.pt"

        checkpoint = torch.load(os.path.join(self.save_path, filename), map_location=self.device)
        current_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        return model, optimizer, scheduler, current_epoch


def run_val(model, data_loader, loss_fn):
    model.eval()
    batch_loss_tracker = LossTracker(len(data_loader))
    with torch.no_grad():
        for batch in data_loader:
            losses = loss_fn(model, batch)
            batch_loss_tracker.update(losses)
        
    model.train()
    return batch_loss_tracker


def run_train(model, dataset_dict, solver_dict, model_io_dict, train_loss_fn, val_loss_fn, num_epoches, initial_epoch, logger):
    train_dataset, train_loader = dataset_dict["train_dataset"], dataset_dict["train_loader"]
    val_dataset, val_loader = dataset_dict["val_dataset"], dataset_dict["val_loader"]
    test_dataset, test_loader = dataset_dict["test_dataset"], dataset_dict["test_loader"]
    
    model_io = model_io_dict["model_io"]
    
    # TODO: Remove
    test_model_io = model_io_dict["test_model_io"]
    
    optimizer, scheduler = solver_dict["optimizer"], solver_dict["scheduler"]
    grad_clipper = solver_dict["grad_clipper"]
    early_stopper = solver_dict["early_stopper"]
    
    explode_checker = GradExplodeChecker(logger=logger.logger)
    
    epoch = initial_epoch
    end_epoch = initial_epoch + num_epoches
    
    logger.log("Number of training data: {}".format(len(train_dataset)))
    logger.log("Number of validating data: {}".format(len(val_dataset)))
    logger.log("Model: {}".format(model))
    num_params = sum(p.numel()
                     for p in model.parameters() if p.requires_grad)
    logger.log("Number of training parameters: {}".format(num_params))

    
    while(epoch < end_epoch):
        batch_loss_tracker = LossTracker(len(train_loader))

        logger.log('Epoch: {}'.format(epoch))

        explode = False
        model.train()
        
        epoch_time = time.time()
        for bidx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            losses = train_loss_fn(model, batch, current_epoch=epoch)
            total_loss = losses["total_loss"]

            if explode_checker.check(total_loss):
                logger.log("Gradient explode. Loss: {}. Reloading the previous checkpoint".format(total_loss))
                explode = True
                break
            
            total_loss.backward()
            grad_clipper(model)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            logger.log_param(model)
            logger.log_grad(model)
            batch_loss_tracker.update(losses)

        epoch_time = time.time() - epoch_time
            
        if explode:
            model, optimizer, scheduler, epoch = model_io.reload({"num_data": len(train_dataset)})
            logger.log("Reload from epoch: {}".format(epoch))
            continue
        
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        explode_checker.set_average_loss(batch_loss_tracker.get("total_loss"))
        logger.log("Train time: {}, LR: {}".format(epoch_time, current_lr))
        logger.log("Train losses:")
        logger.log_losses(batch_loss_tracker.get(), stage="train")

        val_time = time.time()
        val_losses = run_val(model, val_loader, val_loss_fn)
        val_time = time.time() - val_time
        logger.log("Val time: {}".format(val_time))
        logger.log("Val losses:")
        logger.log_losses(val_losses.get(), stage="val")
        

        early_stopper.update(val_losses.get("total_loss"), epoch=epoch)

        if early_stopper.is_stop():
            logger.log("Early stopping...")
            break
        elif early_stopper.is_save():
            logger.log("Saving model")
            model_io.save(epoch, model, optimizer, scheduler)

        epoch = epoch + 1

    logger.log("Returning the best model")
    model, _, _, _ = model_io.reload({"num_data": len(train_dataset)})
    
    return model, epoch


def start_train(cfg : DictConfig, build_model_fn, build_dataset_fn, loss_class,
                train_loss_kwargs, val_loss_kwargs):
    torch.set_num_threads(cfg.num_thread)
    print(PROJECT_PATH)
    cfg.override_cfg = os.path.join(PROJECT_PATH, cfg.override_cfg)
    if os.path.isfile(cfg.override_cfg):
        override_cfg = OmegaConf.load(cfg.override_cfg)
        print("override", override_cfg)
        cfg = OmegaConf.merge(cfg, override_cfg)
        print("final", cfg)
    set_seed(cfg.seed)

    cfg.dataset.numpy_path = os.path.join(PROJECT_PATH, cfg.dataset.numpy_path)
    cfg.dataset.graph_path = os.path.join(PROJECT_PATH, cfg.dataset.graph_path)
    cfg.output_path = os.path.join(PROJECT_PATH, cfg.output_path)
    print(cfg)

    if cfg.debug:
        torch.autograd.set_detect_anomaly(True)

    device = "cpu"
    if cfg.gpu != -1:
        device = torch.device("cuda:{}".format(cfg.gpu))


    logging_dir = os.getcwd()
    experiment_name = os.path.basename(logging_dir)
        
    project_name = cfg.project_name

    # Build model
    model = build_model_fn(cfg.model.architecture)
    model = model.to(device)

    # Build logger
    logger = CustomLogger(cfg, model)
    logger.log("Config: {}".format(yaml.safe_load(OmegaConf.to_yaml(cfg))))
    
    # Build dataset
    train_dataset, train_loader = build_dataset_fn(cfg.dataset, "train", training=True)
    val_dataset, val_loader = build_dataset_fn(cfg.dataset, "val", training=False)
    test_dataset, test_loader = build_dataset_fn(cfg.dataset, "test", training=False)
    
    dataset_dict = {
        "train_dataset": train_dataset,
        "train_loader": train_loader,
        "val_dataset": val_dataset,
        "val_loader": val_loader,
        "test_dataset": test_dataset,
        "test_loader": test_loader,
    }


    
    # Optimizer, scheduler, amp
    solver_cfg = cfg.model.solver
    opti_cfg, sche_cfg = solver_cfg.optimizer, solver_cfg.scheduler
    optimizer = build_optimizer(model, opti_cfg)
    scheduler = build_scheduler(optimizer, sche_cfg, num_data=len(train_dataset), batch_size=cfg.dataset.batch_size)
    early_stop_fn = EarlyStopping(cfg.model.solver.early_stop, logger=logger.logger)

    grad_clipper = GradClipper(solver_cfg.clip_grad)
    solver_dict = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "grad_clipper": grad_clipper,
        "early_stopper": early_stop_fn,
    }
    

    # ModelIO For saving and reloading
    output_path = cfg.output_path
    if cfg.is_saving:
        mkdir_if_not_exist(output_path)
        mkdir_if_not_exist(os.path.join(output_path, project_name))
        
    output_path = os.path.join(output_path, project_name, experiment_name)
    if cfg.is_saving:
        mkdir_if_not_exist(output_path)
        mkdir_if_not_exist(os.path.join(output_path, "test"))

    model_io = {
        "model_io": ModelIO(cfg, cfg.model, build_model_fn, solver_cfg.optimizer, 
                       solver_cfg.scheduler, save_path=output_path, 
                       device=device, is_saving=cfg.is_saving),
        
        "test_model_io": ModelIO(cfg, cfg.model, build_model_fn, solver_cfg.optimizer, 
                       solver_cfg.scheduler, save_path=os.path.join(output_path, "test"), 
                       device=device, is_saving=cfg.is_saving)
    }
    
    # Build Loss
    data_stats = {}
    
    for k, v in train_dataset.additional_info["data_stats"].items():
        if ("human" in k) or ("obj" in k) or ("point" in k):
            data_stats[k] = v.to(device)
        else:
            data_stats[k] = v


    train_loss_kwargs.update({
        "dataset_additional_info": train_dataset.additional_info,
        "data_stats": data_stats,
        "lambda_human": cfg.model.solver.loss.lambda_human,
        "lambda_obj": cfg.model.solver.loss.lambda_obj,
        "lambda_switch": cfg.model.solver.loss.lambda_switch,
        "lambda_reg": cfg.model.solver.loss.lambda_reg,
        "batch_size": cfg.dataset.batch_size,
        "device": device,
        "is_training": True,
    })
    
    val_loss_kwargs.update({
        "data_stats": data_stats,
        "use_switch_loss": False,
        "is_training": False,
        "device": device,
        "batch_size": cfg.dataset.batch_size
    })
    
    train_loss_fn = loss_class(train_loss_kwargs)
    
    val_loss_fn = loss_class(val_loss_kwargs)
    
    num_epoches = solver_cfg.total_epochs
    
    run_train(model, dataset_dict, solver_dict, model_io, train_loss_fn, val_loss_fn, num_epoches=num_epoches,
              initial_epoch=0, logger=logger)

    
def start_train_curriculum(cfg : DictConfig, build_model_fn, build_dataset_fn, loss_class,
                train_loss_kwargs, val_loss_kwargs):
    torch.set_num_threads(cfg.num_thread)
    print(PROJECT_PATH)
    cfg.override_cfg = os.path.join(PROJECT_PATH, cfg.override_cfg)
    if os.path.isfile(cfg.override_cfg):
        override_cfg = OmegaConf.load(cfg.override_cfg)
        print("override", override_cfg)
        cfg = OmegaConf.merge(cfg, override_cfg)
        print("final", cfg)
    set_seed(cfg.seed)

    cfg.dataset.numpy_path = os.path.join(PROJECT_PATH, cfg.dataset.numpy_path)
    cfg.dataset.graph_path = os.path.join(PROJECT_PATH, cfg.dataset.graph_path)
    cfg.output_path = os.path.join(PROJECT_PATH, cfg.output_path)
    print(cfg)

    if cfg.debug:
        torch.autograd.set_detect_anomaly(True)

    device = "cpu"
    if cfg.gpu != -1:
        device = torch.device("cuda:{}".format(cfg.gpu))


    logging_dir = os.getcwd()
    experiment_name = os.path.basename(logging_dir)
        
    project_name = cfg.project_name

    # Build model
    model = build_model_fn(cfg.model.architecture)
    model = model.to(device)

    # Build logger
    logger = CustomLogger(cfg, model)
    logger.log("Config: {}".format(yaml.safe_load(OmegaConf.to_yaml(cfg))))
    
    # Build dataset
    train_dataset, train_loader = build_dataset_fn(cfg.dataset, "train", training=True)
    val_dataset, val_loader = build_dataset_fn(cfg.dataset, "val", training=False)
    test_dataset, test_loader = build_dataset_fn(cfg.dataset, "test", training=False)
    
    dataset_dict = {
        "train_dataset": train_dataset,
        "train_loader": train_loader,
        "val_dataset": val_dataset,
        "val_loader": val_loader,
        "test_dataset": test_dataset,
        "test_loader": test_loader,
    }

    # Optimizer, scheduler, amp
    solver_cfg = cfg.model.solver

    # ModelIO For saving and reloading
    output_path = cfg.output_path
    if cfg.is_saving:
        mkdir_if_not_exist(output_path)
        mkdir_if_not_exist(os.path.join(output_path, project_name))
        
    output_path = os.path.join(output_path, project_name, experiment_name)
    if cfg.is_saving:
        mkdir_if_not_exist(output_path)
        mkdir_if_not_exist(os.path.join(output_path, "test"))
    
    # Build Loss
    data_stats = {}
    
    for k, v in train_dataset.additional_info["data_stats"].items():
        if ("human" in k) or ("obj" in k) or ("point" in k):
            data_stats[k] = v.to(device)
        else:
            data_stats[k] = v


    train_loss_kwargs.update({
        "dataset_additional_info": train_dataset.additional_info,
        "data_stats": data_stats,
        "lambda_human": cfg.model.solver.loss.lambda_human,
        "lambda_obj": cfg.model.solver.loss.lambda_obj,
        "lambda_switch": cfg.model.solver.loss.lambda_switch,
        "lambda_reg": cfg.model.solver.loss.lambda_reg,
        "batch_size": cfg.dataset.batch_size,
        "device": device,
        "is_training": True,
        "num_probs": len(cfg.model.solver.problem_epoch),
    })
    
    val_loss_kwargs.update({
        "dataset_additional_info": train_dataset.additional_info,
        "data_stats": data_stats,
        "use_switch_loss": False,
        "is_training": False,
        "device": device,
        "batch_size": cfg.dataset.batch_size,
        "num_probs": len(cfg.model.solver.problem_epoch),
    })

    
    model_io = {
        "model_io": ModelIO(cfg, cfg.model, build_model_fn, solver_cfg.optimizer, 
                       solver_cfg.scheduler, save_path=output_path, 
                       device=device, is_saving=cfg.is_saving),
        
        "test_model_io": ModelIO(cfg, cfg.model, build_model_fn, solver_cfg.optimizer, 
                       solver_cfg.scheduler, save_path=os.path.join(output_path, "test"), 
                       device=device, is_saving=cfg.is_saving)
    }
    
    initial_epoch = 0
    epoches = cfg.model.solver.problem_epoch
    epoch_lr = cfg.model.solver.problem_lr
    
    for i, (num_epoches, epoch_lr) in enumerate(zip(epoches, epoch_lr)):
        solver_cfg = cfg.model.solver
        opti_cfg, sche_cfg = solver_cfg.optimizer, solver_cfg.scheduler
        opti_cfg.base_lr = epoch_lr
        optimizer = build_optimizer(model, opti_cfg)
        scheduler = build_scheduler(optimizer, sche_cfg, num_data=len(train_dataset), batch_size=cfg.dataset.batch_size)

        early_stop_fn = EarlyStopping(cfg.model.solver.early_stop, logger=logger.logger)
        grad_clipper = GradClipper(solver_cfg.clip_grad)
        solver_dict = {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "grad_clipper": grad_clipper,
            "early_stopper": early_stop_fn,

        }    
        train_loss_kwargs["prob_id"] = i
        val_loss_kwargs["prob_id"] = i

        train_loss_fn = loss_class(train_loss_kwargs)
        
        val_loss_fn = loss_class(val_loss_kwargs)

        logger.log("Problem: {}".format(i))
        model, initial_epoch = run_train(model, dataset_dict, solver_dict, model_io, train_loss_fn, val_loss_fn, num_epoches=num_epoches,
                initial_epoch=initial_epoch, logger=logger)

    
def start_train_multistage(cfg : DictConfig, build_model_fn, build_dataset_fn, losses,
                train_loss_kwargs, val_loss_kwargs):
    torch.set_num_threads(cfg.num_thread)
    print(PROJECT_PATH)
    cfg.override_cfg = os.path.join(PROJECT_PATH, cfg.override_cfg)
    if os.path.isfile(cfg.override_cfg):
        override_cfg = OmegaConf.load(cfg.override_cfg)
        print("override", override_cfg)
        cfg = OmegaConf.merge(cfg, override_cfg)
        print("final", cfg)
    set_seed(cfg.seed)

    cfg.dataset.numpy_path = os.path.join(PROJECT_PATH, cfg.dataset.numpy_path)
    cfg.dataset.graph_path = os.path.join(PROJECT_PATH, cfg.dataset.graph_path)
    cfg.output_path = os.path.join(PROJECT_PATH, cfg.output_path)
    print(cfg)

    if cfg.debug:
        torch.autograd.set_detect_anomaly(True)

    device = "cpu"
    if cfg.gpu != -1:
        device = torch.device("cuda:{}".format(cfg.gpu))


    logging_dir = os.getcwd()
    experiment_name = os.path.basename(logging_dir)
        
    project_name = cfg.project_name

    # Build model
    model = build_model_fn(cfg.model.architecture)
    model = model.to(device)

    # Build logger
    logger = CustomLogger(cfg, model)
    logger.log("Config: {}".format(yaml.safe_load(OmegaConf.to_yaml(cfg))))
    
    # Build dataset
    train_dataset, train_loader = build_dataset_fn(cfg.dataset, "train", training=True)
    val_dataset, val_loader = build_dataset_fn(cfg.dataset, "val", training=False)
    test_dataset, test_loader = build_dataset_fn(cfg.dataset, "test", training=False)
    
    dataset_dict = {
        "train_dataset": train_dataset,
        "train_loader": train_loader,
        "val_dataset": val_dataset,
        "val_loader": val_loader,
        "test_dataset": test_dataset,
        "test_loader": test_loader,
    }


    
    # Optimizer, scheduler, amp
    solver_cfg = cfg.model.solver
    
    cfgs = {
        "switch": solver_cfg.switch,
        "all": solver_cfg.all
    }


    # ModelIO For saving and reloading
    output_path = cfg.output_path
    if cfg.is_saving:
        mkdir_if_not_exist(output_path)
        mkdir_if_not_exist(os.path.join(output_path, project_name))
        
    output_path = os.path.join(output_path, project_name, experiment_name)
    if cfg.is_saving:
        mkdir_if_not_exist(output_path)
        mkdir_if_not_exist(os.path.join(output_path, "test"))
    
    # Build Loss
    data_stats = {}
    
    for k, v in train_dataset.additional_info["data_stats"].items():
        if ("human" in k) or ("obj" in k) or ("point" in k):
            data_stats[k] = v.to(device)
        else:
            data_stats[k] = v


    train_loss_kwargs.update({
        "dataset_additional_info": train_dataset.additional_info,
        "data_stats": data_stats,
        "lambda_human": cfg.model.solver.loss.lambda_human,
        "lambda_obj": cfg.model.solver.loss.lambda_obj,
        "lambda_switch": cfg.model.solver.loss.lambda_switch,
        "lambda_reg": cfg.model.solver.loss.lambda_reg,
        "batch_size": cfg.dataset.batch_size,
        "device": device,
        "is_training": True,
    })
    
    val_loss_kwargs.update({
        "dataset_additional_info": train_dataset.additional_info,
        "data_stats": data_stats,
        "use_switch_loss": False,
        "is_training": False,
        "device": device,
        "batch_size": cfg.dataset.batch_size
    })
        
    initial_epoch = 0
    for stage_name, stage_cfg in cfgs.items():
        if stage_cfg.total_epochs <= 0:
            continue

        opti_cfg, sche_cfg = stage_cfg.optimizer, stage_cfg.scheduler
        optimizer = build_optimizer(model, opti_cfg)
        scheduler = build_scheduler(optimizer, sche_cfg, num_data=len(train_dataset), batch_size=cfg.dataset.batch_size)
        early_stop_fn = EarlyStopping(cfg.model.solver.early_stop, logger=logger.logger)
        grad_clipper = GradClipper(solver_cfg.clip_grad)

        model_io = {
            "model_io": ModelIO(cfg, cfg.model, build_model_fn, stage_cfg.optimizer, 
                       stage_cfg.scheduler, save_path=output_path, 
                       device=device, is_saving=cfg.is_saving),
            
            "test_model_io": ModelIO(cfg, cfg.model, build_model_fn, stage_cfg.optimizer, 
                       stage_cfg.scheduler, save_path=os.path.join(output_path, "test"), 
                       device=device, is_saving=cfg.is_saving)
        }

        solver_dict = {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "grad_clipper": grad_clipper,
            "early_stopper": early_stop_fn,

        }    
        
        train_loss_fn = losses[stage_name](train_loss_kwargs)
        val_loss_fn = losses[stage_name](val_loss_kwargs)

        logger.log("Stage: {}".format(stage_name))
        model, initial_epoch = run_train(model, dataset_dict, solver_dict, model_io, train_loss_fn, val_loss_fn, num_epoches=stage_cfg.total_epochs,
                initial_epoch=0, logger=logger)