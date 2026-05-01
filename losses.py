import torch
import torch.nn as nn
from torch.nn import functional as F

class DistillationLoss(nn.Module):
    """
    Computes Base Task Loss + Logit KD Loss + Review Feature KD Loss
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module, abfs: nn.ModuleList, args):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        
        assert args.distillation_type in ['none', 'soft', 'hard', 'DKD']
        self.distillation_type = args.distillation_type
        self.tau = args.distillation_tau

        self.layer_ids_s = args.s_id
        self.layer_ids_t = args.t_id
        self.abfs = abfs

    def forward(self, inputs, outputs, labels):
        # Handle tuple outputs (standard predictions, distillation predictions, features)
        block_outs_s = outputs[1]
        if isinstance(outputs[0], torch.Tensor):
            outputs = outputs_kd = outputs[0]
        else:
            outputs, outputs_kd = outputs[0]

        # 1. Task Loss (e.g., CrossEntropy)
        base_loss = self.base_criterion(outputs, labels)

        if self.distillation_type == 'none':
            return base_loss, torch.tensor(0., device=inputs.device), torch.tensor(0., device=inputs.device)

        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs, block_outs_t = self.teacher_model(inputs)

        # 2. Logit Distillation Loss
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

        # 3. Review Feature KD Loss
        review_loss = review_feature_loss(
            block_outs_s, block_outs_t, 
            self.layer_ids_s, self.layer_ids_t, 
            self.abfs
        )

        return base_loss, distillation_loss, review_loss


def review_feature_loss(block_outs_s, block_outs_t, layer_ids_s, layer_ids_t, abfs):
    """
    Extracts features, applies top-down ABF fusion, and computes MSE loss.
    """
    # Extract features. 
    # Note: Using [:, 1:, :] skips the [CLS] token. If you want to distill 
    # the [CLS] token as well, remove the `[:, 1:, :]` slice.
    feats_s = [block_outs_s[i][:, 1:, :] for i in layer_ids_s]
    feats_t = [block_outs_t[i][:, 1:, :] for i in layer_ids_t]

    # Review KD: Top-Down Fusion using ABF
    if abfs is not None:
        x = feats_s[::-1]
        abfs_reversed = abfs[::-1] 
        results = []
        
        # 1. Deepest Layer (No deeper context to fuse)
        out_features, res_features = abfs_reversed[0](x[0])
        results.append(out_features)
        
        # 2. Recursive Loop (Deep -> Shallow)
        for idx in range(1, len(x)):
            out_features, res_features = abfs_reversed[idx](x[idx], res_features)
            results.insert(0, out_features) # Restore original [Shallow -> Deep] order
            
        feats_s = results

    # Compute standard MSE loss between fused student features and teacher features
    loss_review = 0.0
    counter = 0
    for f_s, f_t in zip(feats_s, feats_t):
        # L2 normalize features before MSE (optional but recommended for Transformers)
        loss_review += hcl_transformer(f_t, f_s)
        counter = counter + 1
    loss_review = loss_review/counter
    return loss_review

def hcl_transformer(t_feat, s_feat):
    """
    Transformer equivalent of Hierarchical Context Loss (HCL).
    Expects features without the CLS token, shape: [N, L, C].
    """
    assert t_feat.shape == s_feat.shape
    N, L, C = t_feat.shape
    
    # 1. Base loss (flat MSE on the sequence)
    loss = F.mse_loss(t_feat, s_feat, reduction='mean')
    
    # 2. Reshape to 2D spatial grid for multi-scale pooling
    H = int(L ** 0.5)
    W = H
    if H * W != L:
        raise ValueError(f"Sequence length {L} is not a perfect square. Did you forget to remove the [CLS] token?")
        
    t_grid = t_feat.transpose(1, 2).reshape(N, C, H, W)
    s_grid = s_feat.transpose(1, 2).reshape(N, C, H, W)
    
    # 3. Multi-scale hierarchical pooling (4x4, 2x2, 1x1)
    cnt = 1.0
    tot = 1.0
    for level in [4, 2, 1]:
        if level >= H:
            continue
        tmp_t_feat = F.adaptive_avg_pool2d(t_grid, (level, level))
        tmp_s_feat = F.adaptive_avg_pool2d(s_grid, (level, level))
        cnt /= 2.0
        loss += F.mse_loss(tmp_t_feat, tmp_s_feat, reduction='mean') * cnt
        tot += cnt
        
    loss = loss / tot
    return loss
