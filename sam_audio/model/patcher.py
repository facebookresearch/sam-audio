import math
from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange


def pad1d(
    x: torch.Tensor,
    paddings: Tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
):
    # Copied from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conv.py
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    # Copied from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conv.py
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.kernel_size[0]
        stride = self.stride[0]
        dilation = self.dilation[0]
        kernel_size = (
            kernel_size - 1
        ) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(
            x, kernel_size, stride, padding_total
        )
        # Asymmetric padding required for odd strides
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        x = pad1d(x, (padding_left, padding_right + extra_padding))
        return super().forward(x)


class ConvBlock1d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        self.groupnorm = torch.nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels
        )
        self.activation = torch.nn.SiLU()
        self.project = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.groupnorm(x)
        x = self.activation(x)
        return self.project(x)


class ResnetBlock1d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        self.block1 = ConvBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            num_groups=num_groups,
        )

        self.block2 = ConvBlock1d(
            in_channels=out_channels,
            out_channels=out_channels,
            num_groups=num_groups,
        )

        self.to_out = (
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return h + self.to_out(x)


class Patcher(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
    ):
        super().__init__()
        assert_message = f"out_channels must be divisible by patch_size ({patch_size})"
        assert out_channels % patch_size == 0, assert_message
        self.patch_size = patch_size
        self.block = ResnetBlock1d(
            in_channels=in_channels,
            out_channels=out_channels // patch_size,
            num_groups=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = rearrange(x, "b c (l p) -> b (c p) l", p=self.patch_size)
        return x
