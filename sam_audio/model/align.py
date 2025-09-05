from typing import Optional

import torch


class AlignModalities(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        with_gate: bool = True,
    ):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        self.normalize = normalize
        if self.normalize:
            self.layer_norm = torch.nn.LayerNorm(out_channels)

        self.gate = None
        if with_gate:
            self.gate = torch.nn.Parameter(torch.tensor([0.0]))

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, video_features: Optional[torch.Tensor] = None):
        """
        Align video features to the input audio features

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where B is batch size, C is channel size, and T is sequence length.
            video_features (Optional[torch.Tensor]): Optional video features tensor to be aligned, expected shape (B, in_channels, T).
        """
        if video_features is None:
            return x

        post_conv = self.conv(video_features)
        post_conv = post_conv.permute(0, 2, 1)  # BCT -> BTC

        if self.normalize:
            post_conv = self.layer_norm(post_conv)

        if self.gate is None:
            return post_conv
        else:
            return x + self.gate.tanh() * post_conv
