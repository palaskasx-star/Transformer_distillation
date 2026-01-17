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
    elif model_name.split('_')[0] == 'deit3':
        model.forward_features = MethodType(vit3_forward_features, model)
        model.forward = MethodType(vit3_forward, model)
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

def dinov3_forward_features(self, x, require_feat: bool = False):
    
    # Initialize lists
    block_outs = []
    num_reg = self.reg_token.shape[1]


    # --------------------------------------------------------
    # 2. Embedding & Token Prep
    # --------------------------------------------------------
    x = self.patch_embed(x)
    x = x.flatten(1, 2) 

    # Expand special tokens
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    reg_tokens = self.reg_token.expand(x.shape[0], -1, -1)
    
    
    # Concatenate: [CLS, REG..., PATCH...]
    x = torch.cat((cls_token, reg_tokens, x), dim=1)
    x = self.pos_drop(x)


    for blk in self.blocks:
        x = blk(x)
        
        if require_feat:
            cls_t = x[:, 0:1] 
            patch_t = x[:, 1+num_reg:] 
            combined = torch.cat([cls_t, patch_t], dim=1)

            block_outs.append(combined.clone())


    x = self.norm(x)

    if require_feat:
        return x[:, 0], block_outs
    else:
        return x[:, 0]

def dinov3_forward(self, x, require_feat: bool = True):
    if require_feat:
        # Get all lists
        cls_feat, block_outs = self.forward_features(x, require_feat=True)
        
        # Compute final logits
        x_cls = self.head(cls_feat)
        
        # Return: (Patches, Registers, CLS, Logits)
        return x_cls, block_outs
    else:
        cls_feat = self.forward_features(x, require_feat=False)
        x_cls = self.head(cls_feat)
        return x_cls

# deit & vit
def vit_forward_features(self, x, require_feat: bool = False):
    x = self.patch_embed(x)
    
    # 1. SAFE ACCESS: Check if dist_token exists (it likely returns None now)
    dist_token = getattr(self, 'dist_token', None)
    
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    
    # Logic: If dist_token exists (Old Timm), use it. If None (New Timm), skip it.
    if dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        x = torch.cat((cls_token, dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    
    x = self.pos_drop(x + self.pos_embed)

    block_outs = []
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        
        # Save intermediate features
        if dist_token is None:
            block_outs.append(x)
        else:
            # Original logic: skip CLS token if dist token exists
            block_outs.append(x[:, 1:]) 

    x = self.norm(x)
    
    # 2. HANDLE PRE_LOGITS:
    # Newer timm often removes 'pre_logits'. Since we ran self.norm(x), 
    # we can usually just take x[:, 0] directly.
    
    if require_feat:
        if dist_token is None:
            # Just return CLS token
            return x[:, 0], block_outs
        else:
            # Return CLS and DIST tokens
            return (x[:, 0], x[:, 1]), block_outs
    else:
        if dist_token is None:
            return x[:, 0]
        else:
            return x[:, 0], x[:, 1]


def vit_forward(self, x, require_feat: bool = True):
    # Call the updated features function
    if require_feat:
        outs = self.forward_features(x, require_feat=True)
        x_feat = outs[0]      # This is x[:, 0] (or tuple if dist exists)
        block_outs = outs[-1] # The list of block outputs
    else:
        x_feat = self.forward_features(x, require_feat=False)

    # 3. SAFE ACCESS: Check if head_dist exists
    head_dist = getattr(self, 'head_dist', None)

    # Logic for Head
    if head_dist is not None:
        # We have two heads (Old Deit style)
        # x_feat must be a tuple: (cls_token, dist_token)
        x, x_dist = self.head(x_feat[0]), head_dist(x_feat[1])
        
        if self.training and not torch.jit.is_scripting():
            if require_feat:
                return (x, x_dist), block_outs
            return x, x_dist
        else:
            # Inference average
            res = (x + x_dist) / 2
            if require_feat:
                return res, block_outs
            return res
            
    else:
        # Standard ViT (Single Head)
        # x_feat is just the tensor x[:, 0]
        x = self.head(x_feat)
        
        if require_feat:
            return x, block_outs
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


def vit3_forward_features(self, x, require_feat: bool = False):
    x = self.patch_embed(x)
    
    x = x + self.pos_embed
    
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    
    x = torch.cat((cls_token, x), dim=1)


    x = self.pos_drop(x)

    block_outs = []
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        
        block_outs.append(x)


    x = self.norm(x)
    

    if require_feat:
        return x[:, 0], block_outs

    else:
        return x[:, 0]


# vit_forward remains the same as the previous version I gave you
def vit3_forward(self, x, require_feat: bool = True):
    if require_feat:
        outs = self.forward_features(x, require_feat=True)
        x_feat = outs[0]
        block_outs = outs[-1]
    else:
        x_feat = self.forward_features(x, require_feat=False)

    head_dist = getattr(self, 'head_dist', None)

    x = self.head(x_feat)
    if require_feat:
        return x, block_outs
    return x


