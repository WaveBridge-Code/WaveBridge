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

""" Codes adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py
"""

from . import layers, layerspp, dense_layer
from . import swin_transformer
import torch.nn as nn
import functools
import torch
import numpy as np
import torch.nn.functional as F
from .DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp_Adagn
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp_Adagn
WaveletResnetBlockBigGAN = layerspp.WaveletResnetBlockBigGANpp_Adagn
ResnetBlockBigGAN_one = layerspp.ResnetBlockBigGANpp_Adagn_one
BasicLayer = swin_transformer.BasicLayer
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
default_initializer = layers.default_init
dense = dense_layer.dense


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


def exists(x):
    return x is not None


class NCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(
        self,
        self_recursion=True,
        z_emb_dim=256,
        ch_mult=[1, 2, 2, 4, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resamp_with_conv=True,
        image_size=256,
        conditional=True,
        fir=True,
        fir_kernel=[1, 3, 3, 1],
        skip_rescale=True,
        resblock_type="biggan",
        progressive="none",
        progressive_input="residual",
        embedding_type="positional",
        combine_method="sum",
        fourier_scale=16,
        nf=64,
        num_channels=2,
        nz=100,
        n_mlp=3,
        centered=True,
        not_use_tanh=False,
        no_use_freq=False,
        no_use_fbn=False,
        no_use_residual=False,
        window_size=8,
        mlp_ratio=4,
        swin_depth=6,
        out_channels=9,
    ):
        """
        Args:
			z_emb_dim (int): Dimension of the noise embedding.
			ch_mult (list): List of channel multipliers.
			num_res_blocks (int): Number of residual blocks.
			attn_resolutions (list): List of resolutions at which to use self-attention.
			dropout (float): Dropout probability.
			resamp_with_conv (bool): Whether to use convolutional upsampling/downsampling.
			image_size (int): Size of the image.
			conditional (bool): Whether to use condition on time embedding.
			fir (bool): Whether to use finite impulse response filters.
			fir_kernel (list): List of FIR kernel sizes.
			skip_rescale (bool): Whether to rescale skip connections.
			resblock_type (str): Type of residual block.
			progressive (str): Type of progressive training.
			progressive_input (str): Type of progressive input.
			embedding_type (str): Type of embedding.
			combine_method (str): Method to combine inputs.
			self_recursion (bool): Whether to use self-consistent recursion.
			fourier_scale (float): Scale of the Fourier features.
			nf (int): Number of filters.
			num_channels (int): Number of channels.
			nz (int): Dimension of the latent vector.
			n_mlp (int): Number of MLP layers.
        """
        super().__init__()

        self.self_recursion = self_recursion
        self.z_emb_dim = z_emb_dim
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv
        self.image_size = image_size
        self.conditional = conditional
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale
        self.resblock_type = resblock_type
        self.progressive = progressive
        self.progressive_input = progressive_input
        self.embedding_type = embedding_type
        self.combine_method = combine_method
        self.fourier_scale = fourier_scale
        self.nf = nf
        self.num_channels = num_channels
        self.nz = nz
        self.n_mlp = n_mlp
        self.centered = centered
        self.not_use_tanh = not_use_tanh
        self.swin_depth = swin_depth
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.out_channels = out_channels

        self.act = act = nn.SiLU()
        self.num_resolutions = num_resolutions = len(ch_mult)
        all_resolutions = [image_size // (2**i) for i in range(1, num_resolutions + 1)]
        init_scale = 0.0

        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["fourier", "positional"]

        combiner = functools.partial(Combine, method=combine_method)

        self.no_use_fbn = no_use_fbn
        self.no_use_freq = no_use_freq
        self.no_use_residual = no_use_residual

        modules = []

        # Timestep/noise_level embedding; only for continuous training
        if embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            modules.append(
                layerspp.GaussianFourierProjection(
                    embedding_size=nf, scale=fourier_scale
                )
            )
            embed_dim = 2 * nf

        elif embedding_type == "positional":
            embed_dim = nf

        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(
            layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale
        )

        Upsample = functools.partial(
            layerspp.Upsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive == "output_skip":
            self.pyramid_upsample = layerspp.Upsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive == "residual":
            pyramid_upsample = functools.partial(
                layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        Downsample = functools.partial(
            layerspp.Downsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive_input == "input_skip":
            self.pyramid_downsample = layerspp.Downsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive_input == "residual":
            if self.no_use_residual:
                pyramid_downsample = functools.partial(layerspp.Downsample,
                                                       fir=fir, fir_kernel=fir_kernel, with_conv=True)
            else:
                pyramid_downsample = functools.partial(
                    layerspp.WaveletDownsample)

        if resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                ResnetBlockDDPM,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
                zemb_dim=z_emb_dim,
            )

        elif resblock_type == "biggan":
            if self.no_use_freq:
                ResnetBlock = functools.partial(
                    ResnetBlockBigGAN,
                    act=act,
                    dropout=dropout,
                    fir=fir,
                    fir_kernel=fir_kernel,
                    init_scale=init_scale,
                    skip_rescale=skip_rescale,
                    temb_dim=nf * 4,
                    zemb_dim=z_emb_dim,
                )
            else:
                ResnetBlock = functools.partial(
                    WaveletResnetBlockBigGAN,
                    act=act,
                    dropout=dropout,
                    init_scale=init_scale,
                    skip_rescale=skip_rescale,
                    temb_dim=nf * 4,
                    zemb_dim=z_emb_dim,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    img_size=image_size // 2,
                    depth=swin_depth,
                )
        elif resblock_type == "biggan_oneadagn":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN_one,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
                zemb_dim=z_emb_dim,
            )

        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        # Downsampling block
        # channels = num_channels + (1 if self_recursion else 0)
        channels = num_channels
        if progressive_input != "none":
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        hs_c2 = [nf]    # 存储高频细节信息
        enc_low_chs = [nf]
        in_ch = nf
        ds = image_size // 2
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch, img_size=ds))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                hs_c2.append(in_ch)
                enc_low_chs.append(in_ch)
                if resblock_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == "input_skip":
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == "cat":
                        in_ch *= 2

                elif progressive_input == "residual":
                    modules.append(
                        pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch)
                    )
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)
                ds //= 2

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch, img_size=ds))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch, img_size=ds))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch, img_size=ds))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != "none":
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name.")
                else:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(
                            conv3x3(in_ch, channels, bias=True, init_scale=init_scale)
                        )
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name")

            if i_level != 0:
                if resblock_type == "ddpm":
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    if self.no_use_freq:
                        modules.append(ResnetBlock(in_ch=in_ch, up=True))
                    else:
                        modules.append(ResnetBlock(
                            in_ch=in_ch, up=True, hi_in_ch=hs_c2.pop(), enc_low_chs=enc_low_chs.pop()))
                ds *= 2

        assert not hs_c

        if progressive != "output_skip":
            modules.append(
                nn.GroupNorm(
                    num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
                )
            )
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)
        mapping_layers = [
            PixelNorm(),
            dense(nz, z_emb_dim),
            self.act,
        ]
        for _ in range(n_mlp):
            mapping_layers.append(dense(z_emb_dim, z_emb_dim))
            mapping_layers.append(self.act)
        self.current_level = 0
        self.z_transform = nn.Sequential(*mapping_layers)

        self.dwt = DWT_2D("haar")
        self.iwt = IDWT_2D("haar")
    def forward(self, x, time_cond, x_r=None):
        C = self.out_channels
        # Self-recursion
        if self.self_recursion:
            x_r = (
                torch.zeros_like(x[:, :4 * C, :])
                if x_r is None
                else x_r[:, :4 * C, :]
            )
            x = torch.cat((x_r, x), dim=1)
        # Latent vector
        z = torch.randn(x.shape[0], self.nz, device=x.device)

        # Timestep/noise_level embedding; only for continuous training
        zemb = self.z_transform(z)
        modules = self.all_modules
        m_idx = 0
        if self.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == "positional":
            # Sinusoidal positional embeddings.
            timesteps = time_cond

            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f"embedding type {self.embedding_type} unknown.")

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.0

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x

        hs = [modules[m_idx](x)]
        skipHs = []
        lowfreqs = []
        m_idx += 1

        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb, zemb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    if self.no_use_freq:
                        h = modules[m_idx](hs[-1], temb, zemb)
                    else:
                        h, skipH = modules[m_idx](hs[-1], temb, zemb)
                        skipHs.append(skipH)
                        lowfreqs.append(h)
                    m_idx += 1
                if self.progressive_input == "input_skip":
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]

        if self.no_use_fbn:
            h = modules[m_idx](h, temb, zemb)
        else:
            h, hlh, hhl, hhh = self.dwt(h)
            h = modules[m_idx](h / 2., temb, zemb)
            h = self.iwt(h * 2., hlh, hhl, hhh)
        m_idx += 1

        # attn block
        h = modules[m_idx](h)
        m_idx += 1

        if self.no_use_fbn:
            h = modules[m_idx](h, temb, zemb)
        else:
            # forward on original feature space
            h = modules[m_idx](h, temb, zemb)
            h, hlh, hhl, hhh = self.dwt(h)
            h = modules[m_idx](h / 2., temb, zemb)  # forward on wavelet space
            h = self.iwt(h * 2., hlh, hhl, hhh)
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(
                            f'{self.progressive} is not a valid name')

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    if self.no_use_freq:
                        h = modules[m_idx](h, temb, zemb)
                    else:
                        enc_ll = lowfreqs.pop() if lowfreqs else None
                        h = modules[m_idx](h, temb, zemb, skipH=skipHs.pop(), enc_ll=enc_ll, enc_id=i_level)

                    m_idx += 1

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)

        if not self.not_use_tanh:
            out = torch.tanh(h)
        else:
            out = h
        C = self.out_channels
        # 如果你只想保留前4个通道（比如重建图像）
        out = out[:, :4 * C, ...]

        return out

