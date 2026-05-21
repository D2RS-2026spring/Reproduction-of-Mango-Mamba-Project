#!/usr/bin/env python3

# MangoMamba core model
import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models.vision_transformer import Mlp
from timm.models.registry import register_model
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat

def _cfg(url='', **kwargs):
    return {'url': url,
            'num_classes': 8,
            'input_size': (3, 224, 224),
            'pool_size': None,
            'crop_pct': 0.875,
            'interpolation': 'bicubic',
            'fixed_input_size': True,
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            **kwargs}

def window_partition(x, window_size):
    """Partition feature map into windows"""
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """Reverse window partitioning"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x

class EfficientDownsample(nn.Module):
    """Lightweight downsampling with depthwise separable convolution
    + Coordinate Attention (CA) for spatial-channel localization
    + Global Response Normalization (GRN) for stability
    """
    
    def __init__(self, dim, keep_dim=False, use_eca=True, use_grn=True):
        super().__init__()
        dim_out = dim if keep_dim else 2 * dim
        self.use_eca = use_eca
        self.use_grn = use_grn
        
        # Use depthwise separable conv for efficiency
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 2, 1, groups=dim, bias=False),  # Depthwise
            nn.Conv2d(dim, dim_out, 1, 1, 0, bias=False),          # Pointwise
            nn.BatchNorm2d(dim_out, eps=1e-4)
        )

        # Efficient Channel Attention (ECA) & GRN
        if self.use_eca:
            self.ca = ECALayer(channels=dim_out, k_size=3)
        if self.use_grn:
            self.grn = GRN(channels=dim_out)

    def forward(self, x):
        x = self.reduction(x)
        if self.use_eca:
            x = self.ca(x)
        if self.use_grn:
            x = self.grn(x)
        return x

class LightweightPatchEmbed(nn.Module):
    """Efficient patch embedding with reduced parameters
    + Coordinate Attention (CA)
    + Global Response Normalization (GRN)
    """
    
    def __init__(self, in_chans=3, in_dim=16, dim=48, use_eca=True, use_grn=True):
        super().__init__()
        self.use_eca = use_eca
        self.use_grn = use_grn
        
        # Reduced intermediate dimension for efficiency
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU(inplace=True)
        )

        # ECA & GRN on early features to emphasize disease regions
        if self.use_eca:
            self.ca = ECALayer(channels=dim, k_size=3)
        if self.use_grn:
            self.grn = GRN(channels=dim)

    def forward(self, x):
        x = self.conv_down(x)
        if self.use_eca:
            x = self.ca(x)
        if self.use_grn:
            x = self.grn(x)
        return x

class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2)
    Normalizes response energy to stabilize training on conv-like tensors.
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):  # x: (B, C, H, W)
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return x + self.gamma * (x * nx) + self.beta

class ECALayer(nn.Module):
    """Efficient Channel Attention (ECA-Net)
    Very-low parameter channel attention using 1D conv on channel descriptors.
    """
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (B, C, H, W)
        y = self.avg_pool(x)                  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(1, 2)     # (B, 1, C)
        y = self.conv1(y)                     # (B, 1, C)
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y

class DiseaseAwareMixer(nn.Module):
    """Lightweight Mamba mixer optimized for disease pattern recognition
    """
    def __init__(self, d_model, d_state=8, d_conv=3, expand=1.5):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = max(math.ceil(self.d_model / 16), 1)
        
        # Reduced parameters for efficiency
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=False)
        self.x_proj = nn.Linear(self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True)
        
        # Initialize parameters
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        dt = torch.exp(torch.rand(self.d_inner//2) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        
        # State space parameters
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner//2)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2))
        self.D._no_weight_decay = True
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        
        # Efficient 1D convolutions
        self.conv1d_x = nn.Conv1d(self.d_inner//2, self.d_inner//2, d_conv, 
                                 padding=(d_conv-1)//2, groups=self.d_inner//2, bias=False)
        self.conv1d_z = nn.Conv1d(self.d_inner//2, self.d_inner//2, d_conv, 
                                 padding=(d_conv-1)//2, groups=self.d_inner//2, bias=False)

    def forward(self, hidden_states):
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        
        A = -torch.exp(self.A_log.float())
        x = F.silu(self.conv1d_x(x))
        z = F.silu(self.conv1d_z(z))
        
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
        y = selective_scan_fn(x, dt, A, B, C, self.D.float(), z=None, 
                             delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)

class MultiScaleMambaMixer(nn.Module):
    """MS2D-inspired multi-scale SSM mixer
    Processes window tokens at native resolution and a downscaled branch,
    then fuses them. Keeps (B, L, C) I/O.
    """
    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 3, expand: float = 1.5):
        super().__init__()
        # Keep high-res branch capacity; use a lighter low-res branch (smaller expand)
        self.high = DiseaseAwareMixer(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.low = DiseaseAwareMixer(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=1.0)
        # Lightweight gated fusion instead of 2C->C projection to avoid parameter growth
        self.gate = nn.Parameter(torch.zeros(1, 1, d_model))  # sigmoid gate per channel

    def forward(self, x):  # x: (B, L, C), L = S*S per window
        B, L, C = x.shape
        S = int(L ** 0.5)
        # High-res SSM
        y_high = self.high(x)  # (B, L, C)

        # Build low-res branch via bilinear downscale to S//2, robust to odd S
        x_img = x.transpose(1, 2).reshape(B, C, S, S)
        Sl = max(1, S // 2)
        x_low = F.interpolate(x_img, size=(Sl, Sl), mode='bilinear', align_corners=False)
        x_low_seq = x_low.flatten(2).transpose(1, 2)  # (B, Sl*Sl, C)
        y_low = self.low(x_low_seq)                   # (B, Sl*Sl, C)
        # Upsample back to S
        y_low_img = y_low.transpose(1, 2).reshape(B, C, Sl, Sl)
        y_low_up = F.interpolate(y_low_img, size=(S, S), mode='bilinear', align_corners=False)
        y_low_up = y_low_up.flatten(2).transpose(1, 2)  # (B, L, C)

        g = torch.sigmoid(self.gate)
        return (1.0 - g) * y_high + g * y_low_up

class LargeKernelAttention(nn.Module):
    """Large Kernel Attention (VAN)
    Depthwise conv + dilated conv + pointwise to create attention map.
    """
    def __init__(self, dim: int, k1: int = 5, k2: int = 7, dilation: int = 3):
        super().__init__()
        # Use depthwise convs only (no PW) and residual gating to cut parameters
        self.dw1 = nn.Conv2d(dim, dim, kernel_size=k1, padding=k1 // 2, groups=dim, bias=False)
        self.dw2 = nn.Conv2d(dim, dim, kernel_size=k2, padding=dilation * (k2 // 2), groups=dim, dilation=dilation, bias=False)
        self.gate = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):  # (B, L, C) with L=S*S
        B, L, C = x.shape
        S = int(L ** 0.5)
        x2 = x.transpose(1, 2).reshape(B, C, S, S)
        a = self.dw2(self.dw1(x2))
        y = (x2 + torch.sigmoid(self.gate) * a).flatten(2).transpose(1, 2)
        return y

class MangoMambaBlock(nn.Module):
    """Unified block with Mamba mixer and attention"""
    
    def __init__(self, dim, use_attention=False, mlp_ratio=2., drop=0., drop_path=0., layer_scale=None, use_swiglu=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # Mixer choices:
        # - Attention path: Large Kernel Attention (no QK cost)
        # - Non-attention path: MS2D-inspired MultiScaleMambaMixer
        if use_attention:
            # Use smaller kernels to reduce param count in LKA
            self.mixer = LargeKernelAttention(dim, k1=3, k2=5, dilation=2)
        else:
            self.mixer = MultiScaleMambaMixer(d_model=dim, d_state=8, d_conv=3, expand=1.5)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP choice: SwiGLU or Standard GELU-based MLP
        mlp_hidden_dim = int(dim * (mlp_ratio * 0.8))
        if use_swiglu:
            self.mlp = SwiGLU(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        else:
            self.mlp = StandardMLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
        # Layer scale for training stability
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class SwiGLU(nn.Module):
    """SwiGLU MLP block (GLU variant)
    """
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(in_features, hidden_features * 2)
        self.proj = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):  # (B, L, C)
        u, v = self.fc(x).chunk(2, dim=-1)
        y = self.proj(F.silu(v) * u)
        return self.dropout(y)

class StandardMLP(nn.Module):
    """Standard MLP with GELU activation (for ablation study)"""
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):  # (B, L, C)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.dropout(x)

class MangoMambaStage(nn.Module):
    """Efficient stage with windowed processing"""
    
    def __init__(self, dim, depth, num_heads, window_size, downsample=True,
                 mlp_ratio=2., drop=0., attn_drop=0., drop_path=0., layer_scale=None,
                 mixer_mode='hybrid', use_grn=True, use_eca=True, use_swiglu=True):
        super().__init__()
        
        self.use_grn = use_grn
        
        # Determine mixer type for each block based on mixer_mode
        self.blocks = nn.ModuleList([
            MangoMambaBlock(
                dim=dim,
                use_attention=self._get_use_attention(i, depth, mixer_mode),
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                layer_scale=layer_scale,
                use_swiglu=use_swiglu
            ) for i in range(depth)
        ])
        
        self.downsample = EfficientDownsample(dim=dim, use_eca=use_eca, use_grn=use_grn) if downsample else None
        self.window_size = window_size
        
        # GRN on 4D features at the end of stage computation
        if self.use_grn:
            self.stage_grn = GRN(channels=dim)

    def _get_use_attention(self, block_idx, depth, mixer_mode):
        """Determine whether to use attention for a given block"""
        if mixer_mode == 'pure_lka':
            return True
        elif mixer_mode == 'pure_mamba':
            return False
        else:  # hybrid
            return block_idx >= depth // 2

    def forward(self, x):
        _, _, H, W = x.shape
        
        # Window partitioning for efficiency
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))
            _, _, Hp, Wp = x.shape
        else:
            Hp, Wp = H, W
        
        x = window_partition(x, self.window_size)  # (Bwin, T, C), T = window_size*window_size
        
        for blk in self.blocks:
            x = blk(x)
        
        x = window_reverse(x, self.window_size, Hp, Wp)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()
        
        # Stage-level GRN stabilization
        if self.use_grn:
            x = self.stage_grn(x)

        if self.downsample is not None:
            x = self.downsample(x)
        
        return x

class MangoMamba(nn.Module):
    """Lightweight MambaVision for plant disease classification"""
    
    def __init__(self, dim=48, in_dim=16, depths=[2, 3, 6, 3], window_size=[8, 8, 14, 7],
                 mlp_ratio=2., num_heads=[2, 4, 8, 8], drop_path_rate=0.1,
                 in_chans=3, num_classes=8, drop_rate=0., attn_drop_rate=0., layer_scale=1e-5,
                 mixer_mode='hybrid', use_eca=True, use_grn=True, use_swiglu=True):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Lightweight patch embedding
        self.patch_embed = LightweightPatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim, 
                                                 use_eca=use_eca, use_grn=use_grn)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build stages
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = MangoMambaStage(
                dim=int(dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                downsample=(i < len(depths) - 1),
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                layer_scale=layer_scale,
                mixer_mode=mixer_mode,
                use_grn=use_grn,
                use_eca=use_eca,
                use_swiglu=use_swiglu
            )
            self.stages.append(stage)
        
        # Final feature dimension
        num_features = int(dim * 2 ** (len(depths) - 1))
        
        # Global pooling and normalization
        self.norm = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head only
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        
        # Classification output only
        logits = self.head(features)
        
        return {
            'logits': logits,
            'features': features
        }

@register_model
def mango_mamba_tiny(**kwargs):
    # Set default values, allow override from kwargs
    defaults = {
        'dim': 48, 'in_dim': 16, 'depths': [2,3,6,3], 
        'window_size': [7, 7, 7, 7], 'mlp_ratio': 2.5,
        'num_heads': [2, 4, 8, 8], 'drop_path_rate': 0.15,
        'layer_scale': 1e-5,
        'mixer_mode': 'hybrid', 'use_eca': True, 'use_grn': True, 'use_swiglu': True
    }
    # Update defaults with kwargs, kwargs take precedence
    defaults.update(kwargs)
    
    model = MangoMamba(**defaults)
    return model
