# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# 2026.5.6-Changed for building ConceptKD (c) 2024 PalaskasChristos

from types import MethodType

import torch

from typing import Optional


def register_forward(model, model_name, out_indices ):
    # Check for keywords anywhere in the name
    model_name_lower = model_name.lower()
    
    if any(x in model_name_lower.lower() for x in ['dinov3', 'eva02']):
        model.forward_features = MethodType(dinov3_forward_features, model)
        model.forward = MethodType(dinov3_forward, model)
    elif any(x in model_name_lower for x in ['deit', 'deit3']):
        model.forward_features = MethodType(vit_forward_features, model)
        model.forward = MethodType(vit_forward, model)
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
                if idx in self.out_indices:
                    cls_t = x[:, 0:1] 
                    patch_t = x[:, 1+num_reg:] 
                    combined = torch.cat([cls_t, patch_t], dim=1)
                    block_outs.append(combined.clone())
                else:
                    block_outs.append([])

            else:
                x = blk(x, rope=rot_pos_embed[i])
                if idx in self.out_indices:
                    cls_t = x[:, 0:1] 
                    patch_t = x[:, 1+num_reg:] 
                    combined = torch.cat([cls_t, patch_t], dim=1)
                    block_outs.append(combined.clone())
                else:
                    block_outs.append([])
    else:
        # Standard path for non-mixed mode
        for idx, blk in enumerate(self.blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
                if idx in self.out_indices:
                    cls_t = x[:, 0:1] 
                    patch_t = x[:, 1+num_reg:] 
                    combined = torch.cat([cls_t, patch_t], dim=1)
                    block_outs.append(combined.clone())
                else:
                    block_outs.append([])
            else:
                x = blk(x, rope=rot_pos_embed)
                if idx in self.out_indices:
                    cls_t = x[:, 0:1] 
                    patch_t = x[:, 1+num_reg:] 
                    combined = torch.cat([cls_t, patch_t], dim=1)
                    block_outs.append(combined.clone())
                else:
                    block_outs.append([])


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
        if idx in self.out_indices:
            block_outs.append(x)
        else:
            block_outs.append([])

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


