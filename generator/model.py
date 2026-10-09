"""Small class-conditional denoising U-Net that predicts the noise added to an image.

Tensor shapes (MLX convolutions are channels-last, NHWC)::

    x       (B, 28, 28, 1)  float32  noisy image x_t, in the model's normalised range [-1, 1] at t = 0
    t       (B,)            int32    diffusion timestep in [0, num_timesteps)
    labels  (B,)            int32    requested digit in [0, num_classes)
    output  (B, 28, 28, 1)  float32  predicted noise epsilon, same shape as x

The timestep (sinusoidal features through an MLP) and the digit (a learned embedding) are summed into one
conditioning vector, which every residual block adds to its features after its first convolution.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 28
    image_channels: int = 1
    num_classes: int = 10
    base_channels: int = 32
    channel_mults: tuple[int, ...] = (1, 2)  # one entry per resolution: 28x28, 14x14, then the 7x7 middle
    emb_dim: int = 128
    norm_groups: int = 8

    def __post_init__(self):
        # JSON round trips turn the tuple into a list; normalise so configs compare and hash equal
        object.__setattr__(self, "channel_mults", tuple(self.channel_mults))
        if self.image_size % 2 ** len(self.channel_mults):
            raise ValueError(f"image_size {self.image_size} must be divisible by 2^{len(self.channel_mults)}")


def timestep_embedding(t: mx.array, dim: int, max_period: float = 10_000.0) -> mx.array:
    """Sinusoidal features of integer timesteps: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    angles = t.astype(mx.float32)[:, None] * freqs[None, :]
    return mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)


class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> conv, add the conditioning vector, GroupNorm -> SiLU -> conv, plus a skip path."""

    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def __call__(self, x: mx.array, emb: mx.array) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        h = h + self.emb_proj(nn.silu(emb))[:, None, None, :]
        h = self.conv2(nn.silu(self.norm2(h)))
        return h + (self.skip(x) if self.skip is not None else x)


def upsample(x: mx.array) -> mx.array:
    """Nearest-neighbour 2x upsampling of an NHWC tensor."""
    return mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)


class ConditionalUNet(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):  # noqa: B008 (frozen dataclass, safe default)
        super().__init__()
        self.config = config
        c, groups, emb_dim = config.base_channels, config.norm_groups, config.emb_dim
        widths = [c * m for m in config.channel_mults]

        self.time_mlp = nn.Sequential(nn.Linear(c, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.label_emb = nn.Embedding(config.num_classes, emb_dim)

        self.conv_in = nn.Conv2d(config.image_channels, c, kernel_size=3, padding=1)
        self.down_blocks, self.downsamples = [], []
        in_ch = c
        for width in widths:
            self.down_blocks.append(ResBlock(in_ch, width, emb_dim, groups))
            self.downsamples.append(nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1))
            in_ch = width

        mid = widths[-1] * 2
        self.mid_blocks = [ResBlock(in_ch, mid, emb_dim, groups), ResBlock(mid, mid, emb_dim, groups)]
        in_ch = mid

        self.up_blocks = []
        for width in reversed(widths):
            self.up_blocks.append(ResBlock(in_ch + width, width, emb_dim, groups))
            in_ch = width

        self.norm_out = nn.GroupNorm(groups, in_ch, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(in_ch, config.image_channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array, t: mx.array, labels: mx.array) -> mx.array:
        emb = self.time_mlp(timestep_embedding(t, self.config.base_channels)) + self.label_emb(labels)

        h = self.conv_in(x)
        skips = []
        for block, downsample in zip(self.down_blocks, self.downsamples, strict=True):
            h = block(h, emb)
            skips.append(h)
            h = downsample(h)
        for block in self.mid_blocks:
            h = block(h, emb)
        for block in self.up_blocks:
            h = block(mx.concatenate([upsample(h), skips.pop()], axis=-1), emb)
        return self.conv_out(nn.silu(self.norm_out(h)))
