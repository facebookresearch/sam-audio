import math
from typing import Tuple

import torch


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        seq_dim (int): Sequence dimension index.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (
        x.shape[seq_dim],
        x.shape[-3],
        2,
        2,
    ), f"freqs_cis vs x: {(freqs_cis.shape, x.shape)}"
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    freqs_cis = reshape_for_broadcast(
        freqs_cis, xq_, seq_dim
    ).float()  # S D/2 2 2 -> 1 S 1 D/2 2 2
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryEmbedding(torch.nn.Module):
    """
    RotaryEmbedding Module
    """

    def __init__(
        self,
        theta: float,
        head_dim: int,
        max_seqlen: int = 1024,
        scale_factor: int = 1,
        low_freq_factor: int = 1,
        high_freq_factor: int = 32,
        old_context_len: int = 8192,
    ):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        if scale_factor != 1:
            self.low_freq_wavelen = old_context_len / low_freq_factor
            self.high_freq_wavelen = old_context_len / high_freq_factor
            assert self.low_freq_wavelen >= self.high_freq_wavelen

    def reset_parameters(self):
        freqs_cis = self.precompute_freqs_cis(
            dim=self.head_dim, end=self.max_seqlen, theta=self.theta
        )
        S, D, _, _ = freqs_cis.shape
        # S D 2 2 -> 1 S 1 D 2 2
        freqs_cis = freqs_cis.view(1, S, 1, D, 2, 2)
        self.register_buffer(
            "freqs_cis",
            freqs_cis,
            persistent=False,
        )

    def apply_scaling(self, freqs):
        if self.scale_factor == 1:
            return freqs
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < self.high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > self.low_freq_wavelen:
                new_freqs.append(freq / self.scale_factor)
            else:
                assert self.low_freq_wavelen != self.high_freq_wavelen
                smooth = (self.old_context_len / wavelen - self.low_freq_factor) / (
                    self.high_freq_factor - self.low_freq_factor
                )
                new_freqs.append(
                    (1 - smooth) * freq / self.scale_factor + smooth * freq
                )
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

    def precompute_freqs_cis(
        self,
        dim: int,
        end: int,
        theta: float = 10000.0,
    ):
        """
        Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

        This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
        and the end index 'end'. The 'theta' parameter scales the frequencies.
        The returned tensor contains complex values in complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            end (int): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex exponentials.
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        freqs = self.apply_scaling(freqs)

        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()

        cos, sin = freqs.cos(), freqs.sin()

        return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)

    def forward(self, x: torch.Tensor, bhle: bool = False, **kwargs):
        if bhle:
            x = x.transpose(1, 2)  # (B H L E) -> (B L H E)
        seqlen = x.size(1)
        x_ = x.reshape(*x.shape[:-1], -1, 1, 2)  # B L H E -> B L H E/2 1 2
        x_out = (x_ * self.freqs_cis[:, :seqlen]).sum(5).flatten(3)
        if bhle:
            x_out = x_out.transpose(1, 2)
        return x_out.type_as(x)
