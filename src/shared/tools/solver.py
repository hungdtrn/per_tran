import torch
import torch.nn as nn
from yacs.config import CfgNode
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR

class GradClipper(object):
    def __init__(self, cfg: CfgNode) -> None:
        super().__init__()
        
        self.name = cfg.name
        self.thresh = cfg.thresh
        
    def __call__(self, model: nn.Module):
        if self.name == "none":
            return
        
        if self.name == "norm":
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.thresh)
        elif self.name == "value":
            torch.nn.utils.clip_grad_value_(model.parameters(), self.thresh)
            
        return
    
    
class EarlyStopping:
    """ Early stops the training if the validation loss doesn't imporve after 
    a given patience
    """
    def __init__(self, stopping_cfg: CfgNode, logger=None) -> None:
        self.patience = stopping_cfg.patience
        self.delta = stopping_cfg.delta
        self.verbose = True
        self.best_score = None
        self.criteria = stopping_cfg.criteria
        self.counter = 0
        self.logger = logger
        self.min_epoch = stopping_cfg.min_epoch
        self.score = -2
        
    def update(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            
        if self.criteria == 'lower':
            if score <= self.best_score:    
                if self.verbose:
                    self.logger.info("Updating the best score, {:.2f} -> {:.2f}".format(self.best_score, score))
                
                self.best_score = score
                # self.counter = max(int(0.2 * epoch), self.patience) + 1
                self.counter = self.patience + 1
                self.score = -1
                return None
                
            elif score > self.best_score + self.delta:
                if epoch > self.min_epoch:
                    self.counter -= 1
                    
                    if self.verbose:
                        self.logger.info("Counter decrease by 1, remain: {}, best: {:.2f}, current: {:.2f}".format(self.counter, 
                                                                                                               self.best_score, 
                                                                                                               score))
                
        elif self.criteria == 'higher':
            if score > self.best_score:                
                if self.verbose:
                    self.logger.info("Updating the best score, {:.2f} -> {:.2f}".format(self.best_score, score))

                self.best_score = score
                # self.counter = max(int(0.2 * epoch), self.patience) + 1
                self.counter = self.patience + 1
                self.score = -1
                return None

            elif score < self.best_score - self.delta:
                if epoch > self.min_epoch:
                    self.counter -= 1
                    
                    if self.verbose:
                        self.logger.info("Counter decrease by 1, remain: {}, best: {:.2f}, current: {:.2f}".format(self.counter, 
                                                                                                               self.best_score, 
                                                                                                               score))
        
        self.score = int(self.counter < 0)
        
    def is_stop(self):
        return self.score == 1
    
    def is_save(self):
        return self.score == -1

             
class GradExplodeChecker:
    def __init__(self, logger=None) -> None:
        self.average_loss = None
        self.logger = logger
    
    def set_average_loss(self, loss):
        self.average_loss = loss
    
    def check(self, loss):
        thresh = 8
        if (self.average_loss is None) or (thresh * self.average_loss > loss):
            return False
        
        self.logger.info("Explode!!!, old loss: {}, new loss: {}".format(self.average_loss, loss))
        return True
            
    
def build_optimizer(model: torch.nn.Module, opti_cfg: CfgNode) -> Optimizer:
    """
    simple optimizer builder
    :param model: already gpu pushed model
    :param opti_cfg:  config node
    :return: the optimizer
    """
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    opti_type = opti_cfg.name
    lr = opti_cfg.base_lr
    if opti_type == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, amsgrad=opti_cfg.amsgrad)
    elif opti_type == 'sgd':
        sgd_cfg = opti_cfg.sgd
        momentum = sgd_cfg.momentum
        nesterov = sgd_cfg.nesterov
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, nesterov=nesterov, amsgrad=opti_cfg.amsgrad)
    else:
        raise Exception('invalid optimizer, available choices adam/sgd')
    return optimizer


def build_scheduler(optimizer: Optimizer, scheduler_cfg: CfgNode, **kwargs):
    """

    :param optimizer:
    :param optimizer: Optimizer
    :param scheduler_cfg:
    "param solver_cfg: CfgNode
    :return:
    """
    scheduler_type = scheduler_cfg.name
    scheduler = None

    if scheduler_type == 'unchange':
        scheduler = None

    elif scheduler_type == 'multi_steps':
        gamma = scheduler_cfg.reduce_gamma
        milestones = scheduler_cfg.milestones
        scheduler = MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=-1)

    elif scheduler_type == 'lambda_lr':
        num_data, batch_size = kwargs["num_data"], kwargs["batch_size"]
        
        decay_factor = scheduler_cfg.reduce_gamma
        decay_every = scheduler_cfg.decay_step * num_data // batch_size
        
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda epoch: decay_factor ** (epoch // decay_every)])

    return scheduler