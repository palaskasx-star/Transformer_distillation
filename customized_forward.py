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

from typing import Optional


def register_forward(model, model_name, out_indices ):
    if model_name.split('_')[0] == 'deit' or model_name.split('_')[0] == 'deit3':
        model.forward_features = MethodType(vit_forward_features, model)
        model.forward = MethodType(vit_forward, model)
    elif model_name.split('_')[0] == 'cait':
        model.forward_features = MethodType(cait_forward_features, model)
        model.forward = MethodType(cait_forward, model)
    elif model_name.split('_')[0] == 'regnety':
        model.forward_features = MethodType(regnet_forward_features, model)
        model.forward = MethodType(regnet_forward, model)
    elif 'dinov3' in model_name.lower():
        model.forward_features = MethodType(dinov3_forward_features, model)
        model.forward = MethodType(dinov3_forward, model)
    else:
        raise RuntimeError(f'Not defined customized method forward for model {model_name}')

    if out_indices is not None:
        model.out_indices = set(out_indices)
    else:
        model.out_indices = set(range(len(model.blocks)))

def dinov3_forward_features(self, x: torch.Tensor, require_feat: bool = False) -> torch.Tensor:
    """Forward pass through feature extraction layers.

    Args:
        x: Input tensor.

    Returns:
        Feature tensor.
    """
    block_outs = []
    x = self.patch_embed(x)
 
    x, rot_pos_embed = self._pos_embed(x)

    x = self.norm_pre(x)


    num_reg = self.reg_token.shape[1]


    if getattr(self, 'rope_mixed', False) and rot_pos_embed is not None:
        # Handle depth-dependent embeddings for mixed mode
        # pos embed has shape (depth, num_heads, H*W, dim) or (depth, batch_size, num_heads, H*W, dim)
        for i, blk in enumerate(self.blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed[i])
                cls_t = x[:, 0:1] 
                patch_t = x[:, 1+num_reg:] 
                combined = torch.cat([cls_t, patch_t], dim=1)
                block_outs.append(combined.clone())


            else:
                x = blk(x, rope=rot_pos_embed[i])
                cls_t = x[:, 0:1] 
                patch_t = x[:, 1+num_reg:] 
                combined = torch.cat([cls_t, patch_t], dim=1)
                block_outs.append(combined.clone())

    else:
        # Standard path for non-mixed mode
        for idx, blk in enumerate(self.blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
                cls_t = x[:, 0:1] 
                patch_t = x[:, 1+num_reg:] 
                combined = torch.cat([cls_t, patch_t], dim=1)
                block_outs.append(combined.clone())

            else:
                x = blk(x, rope=rot_pos_embed)
                cls_t = x[:, 0:1] 
                patch_t = x[:, 1+num_reg:] 
                combined = torch.cat([cls_t, patch_t], dim=1)
                block_outs.append(combined.clone())



    x = self.norm(x)
    
    return x, block_outs



def dinov3_forward(self, x: torch.Tensor, require_feat: bool = False) -> torch.Tensor:
    """Forward pass.

    Args:
        x: Input tensor.

    Returns:
        Output tensor.
    """
    x, block_outs = self.forward_features(x)
    x = self.forward_head(x)
    return x, block_outs

# deit & vit
def vit_forward_features(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, require_feat: bool = False) -> torch.Tensor:
    """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)."""
    x = self.patch_embed(x)
    x = self._pos_embed(x)
    x = self.patch_drop(x)
    x = self.norm_pre(x)
    block_outs = []

    for idx, blk in enumerate(self.blocks):
        x = blk(x)
        block_outs.append(x)


    x = self.norm(x)
    return x, block_outs


def vit_dist_forward_head(self, x, pre_logits: bool = False) -> torch.Tensor:
    x, x_dist = x[:, 0], x[:, 1]
    if pre_logits:
        return (x + x_dist) / 2
    x = self.head(x)
    x_dist = self.head_dist(x_dist)
    if self.distilled_training and self.training and not torch.jit.is_scripting():
        # only return separate classification predictions when training in distilled mode
        return x, x_dist
    else:
        # during standard train / finetune, inference average the classifier predictions
        return (x + x_dist) / 2

def vit_forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, require_feat: bool = False) -> torch.Tensor:
    x, block_outs = self.forward_features(x, attn_mask=attn_mask)
    x = self.forward_head(x)
    return x, block_outs

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





