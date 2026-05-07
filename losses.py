# 2022.10.14-Changed for building manifold kd
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#
# Modified from Fackbook, Deit
# {haozhiwei1, jianyuan.guo}@huawei.com
#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist

class ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class DistillationLoss(nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module, prototypes: None, projectors_nets: None, args):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert args.distillation_type in ['none', 'soft', 'hard', 'DKD']
        self.distillation_type = args.distillation_type
        self.tau = args.distillation_tau

        self.layer_ids_s = args.s_id
        self.layer_ids_t = args.t_id
        self.alpha = args.distillation_alpha
        self.K = args.K

        self.normalize = args.normalize
        self.distance = args.distance

        self.prototypes = prototypes
        self.projectors_nets = projectors_nets

        self.delta = args.delta

        self.temperature = args.temperature

        self.grad_scale = args.grad_scale

        self.world_size = args.world_size


    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        # only consider the case of [outputs, block_outs_s] or [(outputs, outputs_kd), block_outs_s]
        # i.e. 'require_feat' is always True when we compute loss
        block_outs_s = outputs[1]
        if isinstance(outputs[0], torch.Tensor):
            outputs = outputs_kd = outputs[0]
        else:
            outputs, outputs_kd = outputs[0]

        base_loss = self.base_criterion(outputs, labels)

        if self.distillation_type == 'none':
            return base_loss, torch.tensor(0.), torch.tensor(0.)

        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs, block_outs_t = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='batchmean',
                log_target=True
            ) * (T * T)
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss_base = base_loss
        loss_dist = distillation_loss
        loss_mf_rand= mf_loss(block_outs_s, block_outs_t, self.layer_ids_s, self.layer_ids_t, self.K, normalize=self.normalize, distance=self.distance, prototypes=self.prototypes, projectors_nets=self.projectors_nets, world_size=self.world_size, delta=self.delta, temperature=self.temperature, grad_scale=self.grad_scale)  # manifold distillation loss
        return loss_base, loss_dist, loss_mf_rand


def mf_loss(block_outs_s, block_outs_t, layer_ids_s, layer_ids_t, K, normalize=False, distance='MSE', prototypes=None, projectors_nets=None, world_size=1, beta=0.0, gamma=0.0, delta=0.0, temperature=0.1, grad_scale=0.0):
    losses = [] 

    for idx, (id_s, id_t) in enumerate(zip(layer_ids_s, layer_ids_t)):
        extra_tk_num = block_outs_s[id_s].shape[1] - block_outs_t[id_t].shape[1]
        F_s = block_outs_s[id_s][:, extra_tk_num:, :] 
        F_t = block_outs_t[id_t]

        dev = F_t.device

        if prototypes[idx].protos[0] is not None:
            if delta == 0.0:
                loss_mf_rand = torch.tensor(0.0, device=dev)
            else:
                loss_mf_rand = layer_loss_w_concepts(
                    F_s, F_t, K, normalize=normalize, distance=distance, prototypes=prototypes[idx], projectors_net=projectors_nets[idx], world_size=world_size, temperature=temperature, grad_scale=grad_scale)
        else:  
            if delta == 0.0:
                loss_mf_rand = torch.tensor(0.0, device=dev)
            else:
                loss_mf_rand = layer_loss_wo_concepts(
                    F_s, F_t, K, normalize=normalize, distance=distance, temperature=temperature)

        losses.append(loss_mf_rand)
        
    loss_mf_rand = sum(losses) / len(losses)
    
    return loss_mf_rand

def layer_loss_wo_concepts(F_s, F_t, K, normalize=False, distance='MSE', temperature=0.1, eps=1e-8): 
    bsz, patch_num, _ = F_s.shape
    sampler = torch.randperm(bsz * patch_num)[:K]

    f_s = F_s.reshape(bsz * patch_num, -1)[sampler].unsqueeze(0)
    f_t = F_t.reshape(bsz * patch_num, -1)[sampler].unsqueeze(0)

    if normalize:
        f_s = normalize_mean_std(f_s)
        f_t = normalize_mean_std(f_t)

    M_s = L2_dist(f_s, f_s)
    M_t = L2_dist(f_t, f_t) 

    if distance == 'MSE':
        M_diff = M_t - M_s
        loss_mf_rand = (M_diff * M_diff).mean()
    elif distance == 'KL':
        M_s = F.softmax(-M_t/ temperature, dim=2)
        M_t = F.softmax(-M_s/ temperature, dim=2)
        loss_mf_rand = - torch.mean(torch.sum(p2 * torch.log(p1 + 1e-6), dim=2)) / 2
    dev = loss_mf_rand.device
    
    return loss_mf_rand

def layer_loss_w_concepts(F_s, F_t, K, normalize=False, distance='MSE', eps=1e-8, prototypes=None, projectors_net=None, temperature=0.1, grad_scale=0.0, world_size=1):
    bsz, patch_num, _ = F_s.shape
    sampler = torch.randperm(bsz * patch_num)[:K]

    f_s = F_s.reshape(bsz * patch_num, -1)[sampler].unsqueeze(0)
    f_t = F_t.reshape(bsz * patch_num, -1)[sampler].unsqueeze(0)
    f_s = projectors_net.projs[0](f_s)

    protos = prototypes.protos[0].unsqueeze(0)

    if normalize:
        f_s = normalize_mean_std(f_s)
        f_t = normalize_mean_std(f_t)
        protos_norm = normalize_mean_std(protos)
    else:
        protos_norm = protos

    protos_unscaled = protos_norm 
    protos_scaled = ScaleGradient.apply(protos_norm, grad_scale)

    M_s = L2_dist(f_s, protos_unscaled)
    q1 = distributed_sinkhorn(M_s, nmb_iters=3, epsilon=0.05, world_size=world_size).detach()

    M_t = L2_dist(f_t, protos_unscaled)
    p2 = F.softmax(-M_t / temperature, dim=2)
    q2 = distributed_sinkhorn(M_t, nmb_iters=3, epsilon=0.05, world_size=world_size).detach()

    M_s_scaled = L2_dist(f_s, protos_scaled)
    p1_scaled = F.softmax(-M_s_scaled / temperature, dim=2)

    M_t_scaled = L2_dist(f_t, protos_scaled)
    p2_scaled = F.softmax(-M_t_scaled / temperature, dim=2)
    

    if distance == 'MSE':
        diff12 = q1 - p2
        diff21 = q2 - p1_scaled 
        loss12 = (diff12 * diff12).mean()
        loss21 = (diff21 * diff21).mean()
        loss_mf_rand = (loss12 + loss21) / 2
        
    elif distance == 'KL':
        loss1 = - torch.mean(torch.sum(p2_scaled * torch.log(p1_scaled + 1e-6), dim=2))
        loss3 = - torch.mean(torch.sum(q1 * torch.log(p1_scaled + 1e-6), dim=2))
        loss2 = - torch.mean(torch.sum(q2 * torch.log(p2 + 1e-6), dim=2))

    loss_mf_rand = (loss1 + loss2 + loss3) / 2

    dev = loss_mf_rand.device
    return loss_mf_rand  
    


@torch.no_grad()
def distributed_sinkhorn(out, nmb_iters=3, epsilon=0.05, world_size=1):
    exponential = -out / epsilon
    exponential_max, _ = torch.max(exponential, dim=2, keepdim=True)
    exponential = exponential - exponential_max
    
    Q = torch.exp(exponential).permute(0, 2, 1)
    B = Q.shape[2] * world_size
    K = Q.shape[1]

    sum_Q = Q.sum(dim=(1, 2), keepdim=True)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(nmb_iters):
        sum_of_rows = torch.sum(Q, dim=2, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K
        
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B
        
    Q *= B 
    return Q.permute(0, 2, 1)

def cosine_kernel(x, p):
    x = F.normalize(x, p=2, dim=2)  
    p = F.normalize(p, p=2, dim=2)  
    
    cosine_similarity = torch.bmm(x, p.transpose(1, 2))
    return cosine_similarity

def L2_dist(x, p):
    dist = torch.cdist(x, p, p=2)  
    
    dist_sq = dist.pow(2) / x.shape[2]  
    return dist_sq


def normalize_mean_std(x, eps=1e-6):
    x_norm = (x - x.mean(dim=1, keepdim=True)) /  (x.std(dim=1, keepdim=True) + eps)
    return x_norm

