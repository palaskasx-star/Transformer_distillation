# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# 2022.10.14-Changed for building manifold kd
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import logging


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    writer=None, args=None, set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    # NEW: Unpack indices from the dataloader
    for samples, targets, indices in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            # NEW: Simplified loss unpacking based on the updated DistillationLoss
            loss_base, loss_dist, loss_crd = criterion(samples, outputs, targets, indices)
            loss = ((1 - args.distillation_alpha) * loss_base + args.distillation_alpha * loss_dist) + args.distillation_beta *loss_crd

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logging.error("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_base=loss_base.item())
        metric_logger.update(loss_dist=loss_dist.item())
        metric_logger.update(loss_crd=loss_crd.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if writer is not None:
        writer.add_scalar('Train/Loss/total_loss', metric_logger.loss.global_avg, epoch)
        writer.add_scalar('Train/Loss/base_loss', metric_logger.loss_base.global_avg, epoch)
        writer.add_scalar('Train/Loss/distillation_loss', metric_logger.loss_dist.global_avg, epoch)
        writer.add_scalar('Train/Loss/crd_loss', metric_logger.loss_crd.global_avg, epoch)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, criterion_dist: DistillationLoss, writer=None, epoch=0):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    criterion_dist.eval() 

    # NEW: Unpack indices here as well to match the dataset output, even if not explicitly used
    for images, target, indices in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images, require_feat=True)
            loss = criterion(output[0], target)
            target_onehot = torch.zeros_like(output[0]).scatter_(1, target.unsqueeze(1), 1)
            
            # Use the evaluation pass. Note: CRD usually updates memory only in train mode, 
            # so inside your criterion_dist you should ensure CRD memory isn't updated during eval
            loss_base, loss_dist, loss_crd = criterion_dist(images, output, target_onehot, indices)

        acc1, acc5 = accuracy(output[0], target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_base=loss_base.item())
        metric_logger.update(loss_dist=loss_dist.item())
        metric_logger.update(loss_crd=loss_crd.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
          
    if writer is not None:
        writer.add_scalar('Test/Acc@1', metric_logger.acc1.global_avg, epoch)
        writer.add_scalar('Test/Acc@5', metric_logger.acc5.global_avg, epoch)
        writer.add_scalar('Test/Loss', metric_logger.loss.global_avg, epoch)
        writer.add_scalar('Test/Loss/base_loss', metric_logger.loss_base.global_avg, epoch)
        writer.add_scalar('Test/Loss/distillation_loss', metric_logger.loss_dist.global_avg, epoch)
        writer.add_scalar('Test/Loss/crd_loss', metric_logger.loss_crd.global_avg, epoch)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
