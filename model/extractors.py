import torch
import torch.nn as nn
import numpy as np
import math
from torchvision import transforms
from .model_utils import PatchEmbed, TimestepEmbedder, modulate
from typing import Callable
from torch.nn.init import trunc_normal_

def drop_add_residual_stochastic_depth(
    x: torch.Tensor,
    residual_func: Callable[[torch.Tensor], torch.Tensor],
    sample_drop_ratio: float = 0.0,
) -> torch.Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class ViTExtractor(nn.Module):
    FEATURE_DIM = 384
    output_dim = FEATURE_DIM
    def __init__(self, pretrained=False):
        super(ViTExtractor, self).__init__()
        self.is_pretrained = pretrained
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', skip_validation=True)
        self.resize = transforms.Resize((224, 224))
        self.pre_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.model.mask_token.requires_grad = False
        self.model.patch_embed = PatchEmbed(img_size=(224, 224), patch_size=14, in_chans=12, embed_dim=self.FEATURE_DIM)
        for blk in self.model.blocks:
            blk.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.FEATURE_DIM, 6 * self.FEATURE_DIM, bias=True)
            )
        
        self.t_embedder = TimestepEmbedder(self.FEATURE_DIM)
        
        self.__init_weights()
    
    def __init_weights(self):
        if not self.is_pretrained:
            trunc_normal_(self.model.pos_embed, std=0.02)
            nn.init.normal_(self.model.cls_token, std=1e-6)
            if self.model.register_tokens is not None:
                nn.init.normal_(self.model.register_tokens, std=1e-6)
            named_apply(init_weights_vit_timm, self.model)
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for blk in self.model.blocks:
            nn.init.constant_(blk.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(blk.adaLN_modulation[-1].bias, 0)
        
    def blk_forward(self, blk, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = blk.adaLN_modulation(c).chunk(6, dim=1)
        def attn_residual_func(x: torch.Tensor) -> torch.Tensor:
            return blk.ls1(blk.attn(modulate(blk.norm1(x), shift_msa, scale_msa)))

        def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
            return blk.ls2(blk.mlp(modulate(blk.norm2(x), shift_mlp, scale_mlp)))

        if blk.training and blk.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=blk.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=blk.sample_drop_ratio,
            )
        elif blk.training and blk.sample_drop_ratio > 0.0:
            x = x + gate_msa.unsqueeze(1) * blk.drop_path1(attn_residual_func(x))
            x = x + gate_mlp.unsqueeze(1) * blk.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + gate_msa.unsqueeze(1) * attn_residual_func(x)
            x = x + gate_mlp.unsqueeze(1) * ffn_residual_func(x)
        return x
    
        
    def forward(self, x, ts=None):
        assert len(x.shape) == 5 and x.shape[2] == 12, f"Check input shape: {x.shape}"
        B, NC, C, H, W = x.shape

        x = x.view(B * NC, C, H, W)
        
        x_pre = torch.zeros((B * NC, 12, 224, 224), device=x.device, dtype=x.dtype)
        x_pre[:, :3, ...] = self.pre_normalize(self.resize(x[:, :3, ...]))
        x_pre[:, 9:, ...] = self.pre_normalize(self.resize(x[:, 9:, ...]))
        x_pre[:, 3:9, ...] = self.resize(x[:, 3:9, ...])
        x = x_pre
        
        x = self.model.prepare_tokens_with_masks(x, None)
        c = self.t_embedder(ts.view(B * NC))
        
        for blk in self.model.blocks:
            x = self.blk_forward(blk, x, c)
            
        x = self.model.norm(x)
        x = self.model.head(x)
        
        x = x[:, 1:, :, ].view(B, NC, 16, 16, -1)
        x = x.permute(0, 1, 4, 2, 3)
        return x
