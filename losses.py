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


class DistillationLoss(nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module, prototypes: None, projectors_nets: None, args):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert args.distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = args.distillation_type
        self.tau = args.distillation_tau

        self.layer_ids_s = args.s_id
        self.layer_ids_t = args.t_id
        self.alpha = args.distillation_alpha
        self.beta = args.distillation_beta
        self.w_sample = args.w_sample
        self.w_patch = args.w_patch
        self.w_rand = args.w_rand
        self.K = args.K

        self.normalize = args.distance
        self.distance = args.distance

        self.prototypes = prototypes
        self.projectors_nets = projectors_nets

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
            return base_loss, torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

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

        loss_base = (1 - self.alpha) * base_loss
        loss_dist = self.alpha * distillation_loss
        loss_mf_sample, loss_mf_patch, loss_mf_rand = mf_loss(block_outs_s, block_outs_t, self.layer_ids_s,
                                  self.layer_ids_t, self.K, self.w_sample, self.w_patch, self.w_rand, normalize=self.normalize, distance=self.distance,
                                  prototypes=self.prototypes, projectors_nets=self.projectors_nets)  # manifold distillation loss
        loss_mf_sample = self.beta * loss_mf_sample
        loss_mf_patch = self.beta * loss_mf_patch
        loss_mf_rand = self.beta * loss_mf_rand
        return loss_base, loss_dist, loss_mf_sample, loss_mf_patch, loss_mf_rand


def mf_loss(block_outs_s, block_outs_t, layer_ids_s, layer_ids_t, K, w_sample, w_patch, w_rand, max_patch_num=0, normalize=False, distance='MSE', prototypes=None, projectors_nets=None):
    losses = [[], [], []]  # loss_mf_sample, loss_mf_patch, loss_mf_rand
    for idx, (id_s, id_t) in enumerate(zip(layer_ids_s, layer_ids_t)):
        extra_tk_num = block_outs_s[0].shape[1] - block_outs_t[0].shape[1]
        F_s = block_outs_s[id_s][:, extra_tk_num:, :]  # remove additional tokens
        F_t = block_outs_t[id_t]
        if max_patch_num > 0:
            F_s = merge(F_s, max_patch_num)
            F_t = merge(F_t, max_patch_num)
        if prototypes[0] is not None:
            loss_mf_patch, loss_mf_sample, loss_mf_rand = layer_mf_loss_prototypes(
                F_s, F_t, K, normalize=normalize, distance=distance, prototypes=prototypes[idx], projectors_net=projectors_nets[idx])
        else:
            loss_mf_patch, loss_mf_sample, loss_mf_rand = layer_mf_loss(
                F_s, F_t, K, normalize=normalize, distance=distance, prototypes=prototypes[idx], projectors_net=projectors_nets[idx])
        losses[0].append(w_sample * loss_mf_sample)
        losses[1].append(w_patch * loss_mf_patch)
        losses[2].append(w_rand * loss_mf_rand)

    loss_mf_sample = sum(losses[0]) / len(losses[0])
    loss_mf_patch = sum(losses[1]) / len(losses[1])
    loss_mf_rand = sum(losses[2]) / len(losses[2])

    return loss_mf_sample, loss_mf_patch, loss_mf_rand


def layer_mf_loss(F_s, F_t, K, normalize=False, distance='MSE', eps=1e-8, prototypes=None, projectors_net=None):
    # manifold loss among different patches (intra-sample)
    f_s = F_s
    f_t = F_t

    if normalize:
        f_s = ((f_s - f_s.mean(dim=1, keepdim=True)) / (f_s.std(dim=1, keepdim=True) + eps))
        f_t = ((f_t - f_t.mean(dim=1, keepdim=True)) / (f_t.std(dim=1, keepdim=True) + eps))


    f_s = F.normalize(f_s, dim=-1, p=2)
    f_t = F.normalize(f_t, dim=-1, p=2)


    M_s = f_s.bmm(f_s.transpose(-1, -2))
    M_t = f_t.bmm(f_t.transpose(-1, -2))


    if distance == 'MSE':
        M_diff = M_t - M_s
        loss_mf_patch = (M_diff * M_diff).mean()
    elif distance == 'KL':
        M_s = (M_s + 1)/2
        M_t = (M_t + 1)/2
        M_s = M_s/torch.sum(M_s, dim=2, keepdim=True)
        M_t = M_t/torch.sum(M_t, dim=2, keepdim=True)
        loss_mf_patch = (M_s * (torch.log(M_s + eps) - torch.log(M_t + eps))).mean()
    

    # manifold loss among different samples (inter-sample)
    f_s = F_s.permute(1, 0, 2)
    f_t = F_t.permute(1, 0, 2)

    if normalize:
        f_s = ((f_s - f_s.mean(dim=1, keepdim=True)) / (f_s.std(dim=1, keepdim=True) + eps))
        f_t = ((f_t - f_t.mean(dim=1, keepdim=True)) / (f_t.std(dim=1, keepdim=True) + eps))

    f_s = F.normalize(f_s, dim=-1, p=2)
    f_t = F.normalize(f_t, dim=-1, p=2)
    

    M_s = f_s.bmm(f_s.transpose(-1, -2))
    M_t = f_t.bmm(f_t.transpose(-1, -2))


    if distance == 'MSE':
        M_diff = M_t - M_s
        loss_mf_sample = (M_diff * M_diff).mean()
    elif distance == 'KL':
        M_s = (M_s + 1)/2
        M_t = (M_t + 1)/2
        M_s = M_s/torch.sum(M_s, dim=2, keepdim=True)
        M_t = M_t/torch.sum(M_t, dim=2, keepdim=True)
        loss_mf_sample = (M_s * (torch.log(M_s + eps) - torch.log(M_t + eps))).mean()
    
    # manifold loss among random sampled patches
    bsz, patch_num, _ = F_s.shape
    sampler = torch.randperm(bsz * patch_num)[:K]

    f_s = F_s.reshape(bsz * patch_num, -1)[sampler].unsqueeze(0)
    f_t = F_t.reshape(bsz * patch_num, -1)[sampler].unsqueeze(0)

    if normalize:
        f_s = ((f_s - f_s.mean(dim=1, keepdim=True)) / (f_s.std(dim=1, keepdim=True) + eps))
        f_t = ((f_t - f_t.mean(dim=1, keepdim=True)) / (f_t.std(dim=1, keepdim=True) + eps))


    f_s = F.normalize(f_s, dim=-1, p=2)
    f_t = F.normalize(f_t, dim=-1, p=2)

    M_s = f_s.bmm(f_s.transpose(-1, -2))
    M_t = f_t.bmm(f_t.transpose(-1, -2))

    if distance == 'MSE':
        M_diff = M_t - M_s
        loss_mf_rand = (M_diff * M_diff).mean()
    elif distance == 'KL':
        M_s = (M_s + 1)/2
        M_t = (M_t + 1)/2
        M_s = M_s/torch.sum(M_s, dim=2, keepdim=True)
        M_t = M_t/torch.sum(M_t, dim=2, keepdim=True)
        loss_mf_rand = (M_s * (torch.log(M_s + eps) - torch.log(M_t + eps))).mean()

    return loss_mf_patch, loss_mf_sample, loss_mf_rand

def layer_mf_loss_prototypes(F_s, F_t, K, normalize=False, distance='MSE', eps=1e-8, prototypes=None, projectors_net=None, temperature=0.1):
    prototypes = F.normalize(prototypes, dim=-1, p=2)

    # manifold loss among different patches (intra-sample)
    f_s = F_s
    f_t = F_t

    if normalize:
        f_s = ((f_s - f_s.mean(dim=1, keepdim=True)) / (f_s.std(dim=1, keepdim=True) + eps))
        f_t = ((f_t - f_t.mean(dim=1, keepdim=True)) / (f_t.std(dim=1, keepdim=True) + eps))

    f_s = projectors_net(f_s)

    f_s = F.normalize(f_s, dim=-1, p=2)
    f_t = F.normalize(f_t, dim=-1, p=2)

    M_s = f_s @ prototypes.t()
    q1 = distributed_sinkhorn(M_s, nmb_iters=3).detach()
    M_t = f_t @ prototypes.t()
    q2 = distributed_sinkhorn(M_t, nmb_iters=3).detach()


    p1 = F.softmax(M_s / temperature, dim=2)
    p2 = F.softmax(M_t / temperature, dim=2)

    loss12 = - torch.mean(torch.sum(q1 * torch.log(p2 + 1e-6), dim=2))
    loss21 = - torch.mean(torch.sum(q2 * torch.log(p1 + 1e-6), dim=2))

    loss_mf_patch = (loss12 + loss21)/2

    # manifold loss among different samples (inter-sample)
    f_s = F_s.permute(1, 0, 2)
    f_t = F_t.permute(1, 0, 2)

    if normalize:
        f_s = ((f_s - f_s.mean(dim=1, keepdim=True)) / (f_s.std(dim=1, keepdim=True) + eps))
        f_t = ((f_t - f_t.mean(dim=1, keepdim=True)) / (f_t.std(dim=1, keepdim=True) + eps))

    f_s = projectors_net(f_s)

    f_s = F.normalize(f_s, dim=-1, p=2)
    f_t = F.normalize(f_t, dim=-1, p=2)
    
    M_s = f_s @ prototypes.t()
    q1 = distributed_sinkhorn(M_s, nmb_iters=3).detach()
    M_t = f_t @ prototypes.t()
    q2 = distributed_sinkhorn(M_t, nmb_iters=3).detach()

    p1 = F.softmax(M_s / temperature, dim=2)
    p2 = F.softmax(M_t / temperature, dim=2)

    loss12 = - torch.mean(torch.sum(q1 * torch.log(p2 + 1e-6), dim=2))
    loss21 = - torch.mean(torch.sum(q2 * torch.log(p1 + 1e-6), dim=2))

    loss_mf_sample = (loss12 + loss21)/2

    # manifold loss among random sampled patches
    bsz, patch_num, _ = F_s.shape
    sampler = torch.randperm(bsz * patch_num)[:K]

    f_s = F_s.reshape(bsz * patch_num, -1)[sampler].unsqueeze(0)
    f_t = F_t.reshape(bsz * patch_num, -1)[sampler].unsqueeze(0)


    if normalize:
        f_s = ((f_s - f_s.mean(dim=1, keepdim=True)) / (f_s.std(dim=1, keepdim=True) + eps))
        f_t = ((f_t - f_t.mean(dim=1, keepdim=True)) / (f_t.std(dim=1, keepdim=True) + eps))

    f_s = projectors_net(f_s)

    f_s = F.normalize(f_s, dim=-1, p=2)
    f_t = F.normalize(f_t, dim=-1, p=2)
    
    M_s = f_s @ prototypes.t()
    q1 = distributed_sinkhorn(M_s, nmb_iters=3).detach()
    M_t = f_t @ prototypes.t()
    q2 = distributed_sinkhorn(M_t, nmb_iters=3).detach()

    p1 = F.softmax(M_s / temperature, dim=2)
    p2 = F.softmax(M_t / temperature, dim=2)

    loss12 = - torch.mean(torch.sum(q1 * torch.log(p2 + 1e-6), dim=2))
    loss21 = - torch.mean(torch.sum(q2 * torch.log(p1 + 1e-6), dim=2))

    loss_mf_rand = (loss12 + loss21)/2

    return loss_mf_patch, loss_mf_sample, loss_mf_rand


def merge(x, max_patch_num=196):
    B, P, C = x.shape
    if P <= max_patch_num:
        return x
    n = int(P ** (1/2))  # original patch num at each dim
    m = int(max_patch_num ** (1/2))  # target patch num at each dim
    merge_num = n // m  # merge every (merge_num x merge_num) adjacent patches
    x = x.view(B, m, merge_num, m, merge_num, C)
    merged = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, m * m, -1)
    return merged


@torch.no_grad()
def distributed_sinkhorn(out, nmb_iters=3, epsilon=0.05):
    """
    out: tensor of shape [batch_size, n_prototypes]
    Returns: balanced assignments Q (batch_size x n_prototypes)
    """
    T, B, K = out.shape
    Q = torch.exp(out / epsilon).permute(0, 2, 1)  # T x K x B


    Q /= Q.sum(dim=(1, 2), keepdim=True)    # normalize

    for _ in range(nmb_iters):
        # normalize rows (prototypes)
        Q /= Q.sum(dim=2, keepdim=True)
        Q /= K

        # normalize columns (samples)
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= B

    Q *= B  # undo normalization over samples
    return Q.permute(0, 2, 1).contiguous()  # bach to T x B x K