# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++, adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/models/layerspp.py
"""
import math

from .basic_ops import (
    linear,
    conv_nd,
    avg_pool_nd,
    normalization,
    timestep_embedding,
)
from . import layers
from . import up_or_down_sampling, dense_layer
from .transformer_utils import PatchEmbed, PatchUnEmbed, TransformerBlock
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from . import swin_transformer
from .DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init
dense = dense_layer.dense
BasicLayer = swin_transformer.BasicLayer
SwinTransformerBlock_AdaFM = swin_transformer.SwinTransformerBlock_AdaFM
class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups,in_channel, style_dim):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups, in_channel, affine=False, eps=1e-6)
        self.style = dense(style_dim, in_channel * 2)

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style_proj = self.style(style)
        style_proj = style_proj.unsqueeze(2).unsqueeze(3)
        gamma, beta = style_proj.chunk(2, 1)

        out = self.norm(input)
        assert gamma.shape[1] == out.shape[1], f"Mismatch: gamma {gamma.shape[1]}, out {out.shape[1]}"
        out = gamma * out + beta

        return out


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
    """Combine information from skip connections."""

    def __init__(self, dim1, dim2, method='cat'):
        super().__init__()
        self.Conv_0 = conv1x1(dim1, dim2)
        self.method = method

    def forward(self, x, y):
        h = self.Conv_0(x)
        if self.method == 'cat':
            return torch.cat([h, y], dim=1)
        elif self.method == 'sum':
            return h + y
        else:
            raise ValueError(f'Method {self.method} not recognized.')


class AttnBlockpp(nn.Module):
    """Channel-wise self-attention block. Modified from DDPM."""

    def __init__(self, channels, skip_rescale=False, init_scale=0.):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                        eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum('bhwij,bcij->bchw', w, v)
        h = self.NIN_3(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)



class Upsample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
                 fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch)
        else:
            if with_conv:
                self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                           kernel=3, up=True,
                                                           resample_kernel=fir_kernel,
                                                           use_bias=True,
                                                           kernel_init=default_init())
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            h = F.interpolate(x, (H * 2, W * 2), 'nearest')
            if self.with_conv:
                h = self.Conv_0(h)
        else:
            if not self.with_conv:
                h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = self.Conv2d_0(x)

        return h


class Downsample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
                 fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
        else:
            if with_conv:
                self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                           kernel=3, down=True,
                                                           resample_kernel=fir_kernel,
                                                           use_bias=True,
                                                           kernel_init=default_init())
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            if self.with_conv:
                x = F.pad(x, (0, 1, 0, 1))
                x = self.Conv_0(x)
            else:
                x = F.avg_pool2d(x, 2, stride=2)
        else:
            if not self.with_conv:
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                x = self.Conv2d_0(x)

        return x


class WaveletDownsample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.weight = nn.Parameter(torch.zeros(out_ch, in_ch * 4, 3, 3))
        self.weight.data = default_init()(self.weight.data.shape)
        self.bias = nn.Parameter(torch.zeros(out_ch))

        self.dwt = DWT_2D("haar")

    def forward(self, x):
        xLL, xLH, xHL, xHH = self.dwt(x)

        x = torch.cat((xLL, xLH, xHL, xHH), dim=1) / 2.

        x = F.conv2d(x, self.weight, stride=1, padding=1)
        x = x + self.bias.reshape(1, -1, 1, 1)

        return x


class HFRefinementRes(nn.Module):
    """Residual HF Refinement Block (2D version) for filtering high-frequency coefficients before IDWT."""

    def __init__(self, in_channels, init_alpha=0.3, network_config=None):
        """
        初始化高频细化残差模�?(2D�?

        Args:
            in_channels (int): 输入通道�?            init_alpha (float): 初始alpha值（可扩展）
            network_config (dict): 网络配置字典，可控制激活函数等选项
        """
        super().__init__()

        self.network_config = network_config or {}
        hf_config = self.network_config.get('hf_refinement', {})

        # 深度可分离卷积（Depthwise Conv�?        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,  # depthwise
            bias=True
        )

        # Instance Normalization (2D)
        self.norm = nn.InstanceNorm2d(in_channels, affine=True)

        # ReLU 激�?        self.relu = nn.ReLU(inplace=True)

        # 1x1 卷积用于通道融合
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

        # 可�?Sigmoid 激活，用作掩码
        if hf_config.get('use_sigmoid', True):
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None

    def forward(self, x):
        """
        前向传播：生成自适应高频权重掩码并对输入进行抑制�?        """
        refined = self.conv1(x)
        refined = self.norm(refined)
        refined = self.relu(refined)
        refined = self.conv2(refined)

        if self.sigmoid is not None:
            refined = self.sigmoid(refined)

        # 与输入逐点相乘：保留有用高频特征，抑制噪声
        out = x * refined
        return out


class FusionBlock(nn.Module):
    def __init__(self, dec_ch, enc_ch, out_ch=None):
        super().__init__()
        self.dec_ch = dec_ch
        self.out_ch = out_ch if out_ch else dec_ch

        self.align_enc = layers.ddpm_conv1x1(enc_ch, dec_ch)

        self.fuse = nn.Sequential(
            layers.ddpm_conv1x1(dec_ch * 2, dec_ch),
            nn.ReLU(inplace=True),
            layers.ddpm_conv1x1(dec_ch, self.out_ch)
        )

    def forward(self, dec_feat, enc_feat):
        enc_feat = self.align_enc(enc_feat)
        fused = torch.cat([dec_feat, enc_feat], dim=1)
        return dec_feat + self.fuse(fused)  # 残差融合

class ResnetBlockDDPMpp_Adagn(nn.Module):
    """ResBlock adapted from DDPM."""

    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, conv_shortcut=False,
                 dropout=0.1, skip_rescale=False, init_scale=0.):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        # self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = AdaptiveGroupNorm(min(out_ch // 4, 32), out_ch, zemb_dim)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None, zemb=None):
        h = self.act(self.GroupNorm_0(x, zemb))
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h, zemb))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if x.shape[1] != self.out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)


class WaveletResnetBlockBigGANpp_Adagn(nn.Module):
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, up=False, down=False,
                 dropout=0.1, skip_rescale=True, init_scale=0., hi_in_ch=None,
                 num_heads=8, window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=normalization, emb_channels=64 * 4, fft_patch_size=4,
                 img_size=64, patch_size=1, patch_norm=True, depth=6, enc_low_chs=None, fusion_mode="fusion_block", fuse_levels=[1,2,3]):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch

        self.up = up
        self.down = down
        self.Conv_0 = conv3x3(in_ch, out_ch)

        self.blocks = nn.ModuleList([
            SwinTransformerBlock_AdaFM(
                dim=in_ch,
                input_resolution=(img_size // patch_size, img_size // patch_size),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                dropout=dropout,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                emb_channels=emb_channels,
                fft_patch_size=fft_patch_size
            )
            for i in range(depth)
        ])

        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)

        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

        if self.up:
            self.convH_0 = conv3x3(hi_in_ch * 3, out_ch * 3)

        self.dwt = DWT_2D("haar")
        self.iwt = IDWT_2D("haar")

    def forward(self, x, temb=None, zemb=None, skipH=None, enc_ll=None, enc_id=None):
        h = x

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        hH = None
        if self.up:
            D = h.size(1)
            skipH = self.convH_0(torch.cat(skipH, dim=1) / 2.) * 2.
            h = self.iwt(2. * h, skipH[:, :D],
                         skipH[:, D: 2 * D], skipH[:, 2 * D:])
            x = self.iwt(2. * x, skipH[:, :D],
                         skipH[:, D: 2 * D], skipH[:, 2 * D:])

        elif self.down:
            h, hLH, hHL, hHH = self.dwt(h)
            x, xLH, xHL, xHH = self.dwt(x)
            hH, _ = (hLH, hHL, hHH), (xLH, xHL, xHH)
            h, x = h / 2., x / 2.  # shift range of ll

        # 经过多个 Swin Transformer Block
        for blk in self.blocks:
            h = blk(h, temb, zemb)
        h = self.Conv_0(h)

        # h = self.Dropout_0(h)
        # h = self.Conv_1(h)

        if not self.skip_rescale:
            out = x + h
        else:
            out = (x + h) / np.sqrt(2.)

        if not self.down:
            return out  # self.down == False
        return out, hH


class ResnetBlockBigGANpp_Adagn(nn.Module):
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, up=False, down=False,
                 dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
                 skip_rescale=True, init_scale=0.):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)

        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel

        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = AdaptiveGroupNorm(min(out_ch // 4, 32), out_ch, zemb_dim)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, temb=None, zemb=None):
        h = self.act(self.GroupNorm_0(x, zemb))

        if self.up:
            if self.fir:
                h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

        h = self.Conv_0(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h, zemb))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)


class ResnetBlockBigGANpp_Adagn_one(nn.Module):
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, up=False, down=False,
                 dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
                 skip_rescale=True, init_scale=0.):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)

        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel

        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)

        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, temb=None, zemb=None):
        h = self.act(self.GroupNorm_0(x, zemb))

        if self.up:
            if self.fir:
                h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

        h = self.Conv_0(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)
