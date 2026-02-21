import timm
from timm.models import register_model

@register_model
def vit_tiny_patch16_dinov3(pretrained: bool = False, **kwargs):
    # Pass arguments to the existing small function, overriding dims
    return timm.create_model(
        'vit_small_patch16_dinov3', 
        pretrained=pretrained, 
        embed_dim=192, 
        num_heads=3, 
        **kwargs
    )
