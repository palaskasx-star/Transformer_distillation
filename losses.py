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
        self.K = args.K

        self.normalize = args.normalize
        self.distance = args.distance

        self.prototypes = prototypes
        self.projectors_nets = projectors_nets

        self.world_size = args.world_size

        self.KoLeoData = KoLeoLossData()
        self.KoLeoPrototypes = KoLeoLossPrototypes()

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

        loss_base = base_loss
        loss_dist = distillation_loss
        loss_mf_patch, loss_mf_cls, loss_mf_rand, loss_KoLeo_patch_data, loss_KoLeo_cls_data, loss_KoLeo_rand_data, loss_KoLeo_patch_proto, loss_KoLeo_cls_proto, loss_KoLeo_rand_proto = mf_loss(block_outs_s, block_outs_t, self.layer_ids_s,
                                  self.layer_ids_t, self.K, normalize=self.normalize, distance=self.distance,
                                  prototypes=self.prototypes, projectors_nets=self.projectors_nets, KoLeoData=self.KoLeoData, KoLeoPrototypes=self.KoLeoPrototypes, world_size=self.world_size)  # manifold distillation loss

        return loss_base, loss_dist, loss_mf_patch, loss_mf_cls, loss_mf_rand, loss_KoLeo_patch_data, loss_KoLeo_cls_data, loss_KoLeo_rand_data, loss_KoLeo_patch_proto, loss_KoLeo_cls_proto, loss_KoLeo_rand_proto


def mf_loss(block_outs_s, block_outs_t, layer_ids_s, layer_ids_t, K, max_patch_num=0, normalize=False, distance='MSE', prototypes=None, projectors_nets=None, KoLeoData=None, KoLeoPrototypes=None, world_size=1):
    losses = [[], [], []]  # loss_mf_cls, loss_mf_patch, loss_mf_rand
    losses_KoLeo_data = [[], [], []]  # loss_mf_cls, loss_mf_patch, loss_mf_rand
    losses_KoLeo_proto = [[], [], []]  # loss_mf_cls, loss_mf_patch, loss_mf_rand

    for idx, (id_s, id_t) in enumerate(zip(layer_ids_s, layer_ids_t)):
        extra_tk_num = block_outs_s[0].shape[1] - block_outs_t[0].shape[1]
        F_s = block_outs_s[id_s][:, extra_tk_num:, :]  # remove additional tokens
        F_t = block_outs_t[id_t]
        if max_patch_num > 0:
            F_s = merge(F_s, max_patch_num)
            F_t = merge(F_t, max_patch_num)
        if prototypes[idx].protos[0] is not None:
            loss_mf_patch, loss_mf_cls, loss_mf_rand, loss_KoLeo_patch_data, loss_KoLeo_cls_data, loss_KoLeo_rand_data, loss_KoLeo_patch_proto, loss_KoLeo_cls_proto, loss_KoLeo_rand_proto = layer_mf_loss_prototypes(
                F_s, F_t, K, normalize=normalize, distance=distance, prototypes=prototypes[idx], projectors_net=projectors_nets[idx], KoLeoData=KoLeoData, KoLeoPrototypes=KoLeoPrototypes, world_size=world_size)
        else:
            loss_mf_patch, loss_mf_cls, loss_mf_rand = layer_mf_loss(
                F_s, F_t, K, normalize=normalize, distance=distance, prototypes=prototypes[idx], projectors_net=projectors_nets[idx])
        losses[0].append(loss_mf_cls)
        losses[1].append(loss_mf_patch)
        losses[2].append(loss_mf_rand)
        
        losses_KoLeo_data[0].append(loss_KoLeo_cls_data)
        losses_KoLeo_data[1].append(loss_KoLeo_patch_data)
        losses_KoLeo_data[2].append(loss_KoLeo_rand_data)

        losses_KoLeo_proto[0].append(loss_KoLeo_cls_proto)
        losses_KoLeo_proto[1].append(loss_KoLeo_patch_proto)
        losses_KoLeo_proto[2].append(loss_KoLeo_rand_proto)
        
    loss_mf_cls = sum(losses[0]) / len(losses[0])
    loss_mf_patch = sum(losses[1]) / len(losses[1])
    loss_mf_rand = sum(losses[2]) / len(losses[2])
    
    loss_KoLeo_cls_data = sum(losses_KoLeo_data[0]) / len(losses_KoLeo_data[0])
    loss_KoLeo_patch_data = sum(losses_KoLeo_data[1]) / len(losses_KoLeo_data[1])
    loss_KoLeo_rand_data = sum(losses_KoLeo_data[2]) / len(losses_KoLeo_data[2])

    loss_KoLeo_cls_proto = sum(losses_KoLeo_proto[0]) / len(losses_KoLeo_proto[0])
    loss_KoLeo_patch_proto = sum(losses_KoLeo_proto[1]) / len(losses_KoLeo_proto[1])
    loss_KoLeo_rand_proto = sum(losses_KoLeo_proto[2]) / len(losses_KoLeo_proto[2])

    return loss_mf_patch, loss_mf_cls, loss_mf_rand, loss_KoLeo_patch_data, loss_KoLeo_cls_data, loss_KoLeo_rand_data, loss_KoLeo_patch_proto, loss_KoLeo_cls_proto, loss_KoLeo_rand_proto


def layer_mf_loss(F_s, F_t, K, normalize=False, distance='MSE', eps=1e-8, prototypes=None, projectors_net=None):
    # intra-image manifold loss
    f_s = F_s.clone()
    f_t = F_t.clone()

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
    

    # cls token loss
    f_s = F_s[:, 0:1, :].permute(1, 0, 2).clone()  # select only the cls token
    f_t = F_t[:, 0:1, :].permute(1, 0, 2).clone()  # select only the cls token

    if normalize:
        f_s = ((f_s - f_s.mean(dim=1, keepdim=True)) / (f_s.std(dim=1, keepdim=True) + eps))
        f_t = ((f_t - f_t.mean(dim=1, keepdim=True)) / (f_t.std(dim=1, keepdim=True) + eps))

    f_s = F.normalize(f_s, dim=-1, p=2)
    f_t = F.normalize(f_t, dim=-1, p=2)
    

    M_s = f_s.bmm(f_s.transpose(-1, -2))
    M_t = f_t.bmm(f_t.transpose(-1, -2))


    if distance == 'MSE':
        M_diff = M_t - M_s
        loss_mf_cls = (M_diff * M_diff).mean()
    elif distance == 'KL':
        M_s = (M_s + 1)/2
        M_t = (M_t + 1)/2
        M_s = M_s/torch.sum(M_s, dim=2, keepdim=True)
        M_t = M_t/torch.sum(M_t, dim=2, keepdim=True)
        loss_mf_cls = (M_s * (torch.log(M_s + eps) - torch.log(M_t + eps))).mean()
    
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

    return loss_mf_patch, loss_mf_cls, loss_mf_rand

def layer_mf_loss_prototypes(F_s, F_t, K, normalize=False, distance='MSE', eps=1e-8, prototypes=None, projectors_net=None, KoLeoData=None, KoLeoPrototypes=None, temperature=0.1, world_size=1):
    """
    prototypes = F.normalize(prototypes, dim=-1, p=2)
    """
    with torch.no_grad():
        for i in range(len(prototypes.protos)):
            prototypes.protos[i].copy_(F.normalize(prototypes.protos[i], dim=1))
            
    # manifold loss among different patches (intra-sample)
    f_s = F_s
    f_t = F_t

    if normalize:
        f_s = ((f_s - f_s.mean(dim=1, keepdim=True)) / (f_s.std(dim=1, keepdim=True) + eps))
        f_t = ((f_t - f_t.mean(dim=1, keepdim=True)) / (f_t.std(dim=1, keepdim=True) + eps))

    # apply the first projector from the ModuleList
    f_s = projectors_net.projs[0](f_s)

    f_s = F.normalize(f_s, dim=-1, p=2)
    f_t = F.normalize(f_t, dim=-1, p=2)

    loss_KoLeo_patch_data = KoLeoData(f_s)
    loss_KoLeo_patch_proto = KoLeoPrototypes( prototypes.protos[0])

    M_s = f_s @ prototypes.protos[0].t()
    q1 = sinkhorn(M_s, nmb_iters=3).detach()
    M_t = f_t @ prototypes.protos[0].t()
    q2 = sinkhorn(M_t, nmb_iters=3).detach()


    p1 = F.softmax(M_s / temperature, dim=2)
    p2 = F.softmax(M_t / temperature, dim=2)

    loss12 = - torch.mean(torch.sum(q1 * torch.log(p2 + 1e-6), dim=2))
    loss21 = - torch.mean(torch.sum(q2 * torch.log(p1 + 1e-6), dim=2))

    loss_mf_patch = (2*loss12 + loss21)/2

    # cls token loss
    f_s = F_s[:, 0:1, :].permute(1, 0, 2).clone()  # select only the cls token
    f_t = F_t[:, 0:1, :].permute(1, 0, 2).clone()  # select only the cls token

    if normalize:
        f_s = ((f_s - f_s.mean(dim=1, keepdim=True)) / (f_s.std(dim=1, keepdim=True) + eps))
        f_t = ((f_t - f_t.mean(dim=1, keepdim=True)) / (f_t.std(dim=1, keepdim=True) + eps))

    f_s = projectors_net.projs[1](f_s)

    f_s = F.normalize(f_s, dim=-1, p=2)
    f_t = F.normalize(f_t, dim=-1, p=2)

    loss_KoLeo_cls_data = KoLeoData(f_s)
    loss_KoLeo_cls_proto = KoLeoPrototypes( prototypes.protos[1])
    
    M_s = f_s @ prototypes.protos[1].t()
    q1 = sinkhorn(M_s, nmb_iters=3).detach()
    M_t = f_t @ prototypes.protos[1].t()
    q2 = sinkhorn(M_t, nmb_iters=3).detach()

    p1 = F.softmax(M_s / temperature, dim=2)
    p2 = F.softmax(M_t / temperature, dim=2)

    loss12 = - torch.mean(torch.sum(q1 * torch.log(p2 + 1e-6), dim=2))
    loss21 = - torch.mean(torch.sum(q2 * torch.log(p1 + 1e-6), dim=2))

    loss_mf_cls = (2*loss12 + loss21)/2

    # manifold loss among random sampled patches
    bsz, patch_num, _ = F_s.shape
    sampler = torch.randperm(bsz * patch_num)[:K]

    f_s = F_s.reshape(bsz * patch_num, -1)[sampler].unsqueeze(0)
    f_t = F_t.reshape(bsz * patch_num, -1)[sampler].unsqueeze(0)


    if normalize:
        f_s = ((f_s - f_s.mean(dim=1, keepdim=True)) / (f_s.std(dim=1, keepdim=True) + eps))
        f_t = ((f_t - f_t.mean(dim=1, keepdim=True)) / (f_t.std(dim=1, keepdim=True) + eps))

    f_s = projectors_net.projs[2](f_s)

    f_s = F.normalize(f_s, dim=-1, p=2)
    f_t = F.normalize(f_t, dim=-1, p=2)

    loss_KoLeo_rand_data = KoLeoData(f_s)
    loss_KoLeo_rand_proto = KoLeoPrototypes( prototypes.protos[2])

    M_s = f_s @ prototypes.protos[2].t()
    q1 = sinkhorn(M_s, nmb_iters=3).detach()
    M_t = f_t @ prototypes.protos[2].t()
    q2 = sinkhorn(M_t, nmb_iters=3).detach()

    p1 = F.softmax(M_s / temperature, dim=2)
    p2 = F.softmax(M_t / temperature, dim=2)

    loss12 = - torch.mean(torch.sum(q1 * torch.log(p2 + 1e-6), dim=2))
    loss21 = - torch.mean(torch.sum(q2 * torch.log(p1 + 1e-6), dim=2))

    loss_mf_rand = (2*loss12 + loss21)/2

    return loss_mf_patch, loss_mf_cls, loss_mf_rand, loss_KoLeo_patch_data, loss_KoLeo_cls_data, loss_KoLeo_rand_data, loss_KoLeo_patch_proto, loss_KoLeo_cls_proto, loss_KoLeo_rand_proto



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
def sinkhorn(out, nmb_iters=3, epsilon=0.05):
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

@torch.no_grad()
def distributed_sinkhorn(out, nmb_iters=3, epsilon=0.05, world_size=1):
    Q = torch.exp(out / epsilon).permute(0, 2, 1)  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[2] * world_size # number of samples to assign
    K = Q.shape[1] # how many prototypes

    # make the matrix sums to 1
    sum_Q = Q.sum(dim=(1, 2), keepdim=True)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(nmb_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=2, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.permute(0, 2, 1)


class KoLeoLossData(nn.Module):
    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        # x is (B, T, feat_dim)
        dots = torch.bmm(x, x.transpose(1, 2))
        dots.diagonal(dim1=-2, dim2=-1).fill_(-1)
        _, I = torch.max(dots, dim=2)

        return I

    def forward(self, student_output, eps=1e-8):
        # Fix 1: Updated autocast syntax to remove warning
        with torch.amp.autocast('cuda', enabled=False):
            
            # 2. Find nearest neighbors
            I = self.pairwise_NNs_inner(student_output)

            # 3. Gather neighbors
            # Fix 2: Changed variable name 'F' to 'feat_dim' to avoid collision with functional F
            B, T, feat_dim = student_output.shape
            
            batch_indices = torch.arange(B, device=student_output.device).view(-1, 1).expand(-1, T)
            neighbors = student_output[batch_indices, I]


            # 4. Flatten and calculate distance
            flat_student = student_output.view(-1, feat_dim)
            flat_neighbors = neighbors.view(-1, feat_dim)
            
            distances = self.pdist(flat_student, flat_neighbors)

            loss = -torch.log(distances + eps).mean()
        
        return loss

class KoLeoLossPrototypes(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss
