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

from types import MethodType

import torch


def register_forward(model, model_name):
    if model_name.split('_')[0] == 'deit':
        model.forward_features = MethodType(vit_forward_features, model)
        model.forward = MethodType(vit_forward, model)
    elif model_name.split('_')[0] == 'cait':
        model.forward_features = MethodType(cait_forward_features, model)
        model.forward = MethodType(cait_forward, model)
    elif model_name.split('_')[0] == 'regnety':
        model.forward_features = MethodType(regnet_forward_features, model)
        model.forward = MethodType(regnet_forward, model)
    else:
        raise RuntimeError(f'Not defined customized method forward for model {model_name}')


# deit & vit
def vit_forward_features(self, x, require_feat: bool = False):
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    if self.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = self.pos_drop(x + self.pos_embed)

    # x = self.blocks(x)
    block_outs = []
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        block_outs.append(x)

    x = self.norm(x)
    if require_feat:
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), block_outs
        else:
            return (x[:, 0], x[:, 1]), block_outs
    else:
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]


def vit_forward(self, x, require_feat: bool = True):
    if require_feat:
        outs = self.forward_features(x, require_feat=True)
        x = outs[0]
        block_outs = outs[-1]
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return (x, x_dist), block_outs
            else:
                return (x + x_dist) / 2, block_outs
        else:
            x = self.head(x)
        return x, block_outs
    else:
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


# cait
def cait_forward_features(self, x, require_feat: bool = False):
    B = x.shape[0]
    x = self.patch_embed(x)

    cls_tokens = self.cls_token.expand(B, -1, -1)

    x = x + self.pos_embed
    x = self.pos_drop(x)

    block_outs = []
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        block_outs.append(x)

    for i, blk in enumerate(self.blocks_token_only):
        cls_tokens = blk(x, cls_tokens)

    x = torch.cat((cls_tokens, x), dim=1)

    x = self.norm(x)
    if require_feat:
        return x[:, 0], block_outs
    else:
        return x[:, 0]


def cait_forward(self, x, require_feat: bool = True):
    if require_feat:
        x, block_outs = self.forward_features(x, require_feat=True)
        x = self.head(x)
        return x, block_outs
    else:
        x = self.forward_features(x)
        x = self.head(x)
        return x

# --------------------
# RegNetY (from timm)
# --------------------
def regnet_forward_features(self, x, require_feat: bool = False):
    """
    Custom forward_features for timm RegNetY-160.
    Captures intermediate outputs after each stage.
    """
    block_outs = []

    # Stem
    x = self.stem(x)
    block_outs.append(torch.nn.Unfold(kernel_size=8, stride=8)(x).permute(0, 2, 1))

    # Stages (typical timm RegNet has 4 stages: s1-s4)
    x = self.s1(x); block_outs.append(torch.nn.Unfold(kernel_size=4, stride=4)(x).permute(0, 2, 1))
    x = self.s2(x); block_outs.append(torch.nn.Unfold(kernel_size=2, stride=2)(x).permute(0, 2, 1))
    x = self.s3(x); block_outs.append(torch.nn.Unfold(kernel_size=1, stride=1)(x).permute(0, 2, 1))
    x = self.s4(x); block_outs.append(torch.nn.Unfold(kernel_size=1, stride=1)(torch.nn.AdaptiveAvgPool2d(14)(x)).permute(0, 2, 1))


    # Head
    x = self.head.global_pool(x)
    x = self.head.flatten(x)
    x = self.head.fc(x)

    if require_feat:
        return x, block_outs
    else:
        return x


def regnet_forward(self, x, require_feat: bool = True):
    if require_feat:
        logits, feats = self.forward_features(x, require_feat=True)
        return logits, feats
    else:
        return self.forward_features(x)
