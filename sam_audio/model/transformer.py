import math
from functools import partial
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sam_audio.model.patcher import Patcher
from sam_audio.model.rope import RotaryEmbedding


def gate(x, gate):
    return x * gate


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def get_nonlinearity(kind: str):
    return {
        "relu": F.relu,
        "gelu": F.gelu,
        "swiglu": None,
        "approx_gelu": partial(F.gelu, approximate="tanh"),
        "srelu": lambda x: F.relu(x) ** 2,
        "silu": F.silu,
    }[kind]


class CLSToken(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Expands CLS token to match input batch size.

        Args:
            input_data: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Expanded CLS tokens of shape (batch_size, 1, d_model)
        """
        return self.weight.expand(input_data.size(0), -1, -1)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        return (output * self.weight).type_as(x)


class ProjectionLayer(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        non_linearity: str,
        dropout: float,
        fc_bias: bool = False,
    ):
        super().__init__()

        self.swiglu = non_linearity == "swiglu"
        self.dropout = dropout
        self.w1 = torch.nn.Linear(in_dim, out_dim, bias=fc_bias)

        self.w2 = torch.nn.Linear(out_dim, out_dim, bias=fc_bias)
        if self.swiglu:
            self.w3 = torch.nn.Linear(in_dim, out_dim, bias=fc_bias)

        # non-linearity
        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x):
        hidden1 = self.w1(x)
        if self.swiglu:
            hidden3 = self.w3(x)
            hidden = F.silu(hidden1) * hidden3
        else:
            hidden = self.non_linearity(hidden1)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.w2(hidden)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float = 1e-5,
        use_qk_norm: bool = False,
        fc_bias: bool = False,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0

        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.use_qk_norm = use_qk_norm

        # local heads
        assert self.n_heads % self.n_kv_heads == 0

        self.wq = torch.nn.Linear(dim, n_heads * head_dim, bias=fc_bias)
        self.wk, self.wv = [
            torch.nn.Linear(
                dim,
                n_kv_heads * head_dim,
                bias=fc_bias,
            )
            for _ in range(2)
        ]
        self.wo = torch.nn.Linear(
            n_heads * head_dim,
            dim,
            bias=fc_bias,
        )

        if self.use_qk_norm is True:
            self.q_norm = RMSNorm(head_dim, eps=norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=norm_eps)

    def reshape_heads(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        B, T, C = x.shape
        # B x T x C -> B x T x C/H x H
        x = x.reshape(B, T, C // heads, heads)
        # B x T x C/H x H -> B x H x T x C/H
        return x.permute(0, 3, 1, 2)

    def forward(
        self,
        x: torch.Tensor,
        cross_x: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope: Optional[RotaryEmbedding] = None,
    ):
        # x: B, T, E
        xq = self.wq(x)
        if cross_x is not None:
            xk, xv = self.wk(cross_x), self.wv(cross_x)
        else:
            xk, xv = self.wk(x), self.wv(x)

        xk = self.reshape_heads(xk, self.n_kv_heads)
        xv = self.reshape_heads(xv, self.n_kv_heads)
        xq = self.reshape_heads(xq, self.n_heads)
        if self.use_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        if rope is not None:
            xq = rope(xq, bhle=True)
            xk = rope(xk, bhle=True)

        attn_mask = None

        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]

        output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask)

        output = rearrange(output, "b h n d -> b n (h d)")
        return self.wo(output)


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        ffn_dim_multiplier: float,
        multiple_of: int,
        dropout: float,
        non_linearity: str = "swiglu",
        fc_bias: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.swiglu = non_linearity == "swiglu"
        # swiglu hidden dim factor multiplier (same #params as relu / gelu)
        if self.swiglu:
            hidden_dim = int(2 * hidden_dim / 3)

        # custom dim factor multiplier
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        # round hidden dimension to `multiple_of`
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # layers
        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=fc_bias)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=fc_bias)
        if self.swiglu:
            self.w3 = torch.nn.Linear(dim, hidden_dim, bias=fc_bias)

        # non-linearity
        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(
        self,
        x,
    ):
        hidden1 = self.w1(x)
        if self.swiglu:
            hidden3 = self.w3(x)
            hidden = F.silu(hidden1) * hidden3
        else:
            hidden = self.non_linearity(hidden1)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.w2(hidden)


class TimestepEmbedder(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        frequency_embedding_dim: int,
        non_linearity: str,
        dropout: float,
        fc_bias: bool,
        max_period: int = 10000,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_dim
        self.projection = ProjectionLayer(
            in_dim=frequency_embedding_dim,
            out_dim=dim,
            non_linearity=non_linearity,
            dropout=dropout,
            fc_bias=fc_bias,
        )
        half = frequency_embedding_dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def timestep_embedding(self, t, dim):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        self.freqs = self.freqs.to(device=t.device)
        args = t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding.to(t)

    def forward(self, t):
        x = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.projection(x)


class ContextEmbedder(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        non_linearity: str,
        dropout: float,
        fc_bias: bool,
        norm_eps: float = 1e-5,
        context_norm: bool = False,
    ):
        super().__init__()
        self.context_norm = context_norm
        if context_norm:
            self.norm = RMSNorm(in_dim, norm_eps)

        self.projection = ProjectionLayer(
            in_dim=in_dim,
            out_dim=out_dim,
            non_linearity=non_linearity,
            dropout=dropout,
            fc_bias=fc_bias,
        )

    def forward(self, x):
        if self.context_norm:
            x = self.norm(x)
        h = self.projection(x)
        return h


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        qk_norm: bool = False,
        fc_bias: bool = False,
        ffn_exp: int = 1,
        ffn_dim_multiplier: int = 4,
        multiple_of: int = 64,
        non_linearity: str = "silu",
        no_cross_attention: bool = False,
    ):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.dim = dim
        self.dropout = dropout
        self.head_dim = dim // n_heads

        assert self.n_heads % self.n_kv_heads == 0

        self.attention = Attention(
            dim=dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            norm_eps=norm_eps,
            use_qk_norm=qk_norm,
            fc_bias=fc_bias,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=int(ffn_exp * dim),
            ffn_dim_multiplier=ffn_dim_multiplier,
            multiple_of=multiple_of,
            dropout=dropout,
            non_linearity=non_linearity,
            fc_bias=fc_bias,
        )

        self.attention_norm, self.ffn_norm = [RMSNorm(dim, norm_eps) for _ in range(2)]

        self.cross_attention = None
        if not no_cross_attention:
            self.cross_attention = Attention(
                dim=dim,
                head_dim=self.head_dim,
                n_heads=self.n_heads,
                n_kv_heads=self.n_heads,
                norm_eps=norm_eps,
                use_qk_norm=qk_norm,
                fc_bias=fc_bias,
            )

    def forward(
        self,
        x: torch.Tensor,
        cross_x: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,  # y_mask
        memory_padding_mask: Optional[torch.Tensor] = None,  # y_mask
        rope: Optional[RotaryEmbedding] = None,
    ):
        assert self.attention is not None and self.attention_norm is not None
        h_attn = self.attention(
            self.attention_norm(x),
            key_padding_mask=padding_mask,
            rope=rope,
        )

        h = x + h_attn

        if self.cross_attention is not None:
            h_cross = self.cross_attention(
                x=h,
                cross_x=cross_x,
                key_padding_mask=memory_padding_mask,
            )
            h = h + h_cross  # residual
        h_ff = self.feed_forward(self.ffn_norm(h))
        out = h + h_ff
        return out


class DiTBlock(TransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, self.dim) / self.dim**0.5,
        )

    def forward(
        self,
        x: torch.Tensor,
        cross_x: Optional[torch.Tensor],
        t: torch.Tensor,
        padding_mask: Optional[torch.Tensor],  # y_mask
        memory_padding_mask: Optional[torch.Tensor],  # y_mask
        rope: Optional[RotaryEmbedding] = None,
    ):
        biases = self.scale_shift_table[None] + t.reshape(x.size(0), 6, -1)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = biases.chunk(6, dim=1)

        assert self.attention is not None and self.attention_norm is not None
        h_attn = self.attention(
            modulate(self.attention_norm(x), shift_msa, scale_msa),
            key_padding_mask=padding_mask,
            rope=rope,
        )

        h = x + gate(h_attn, gate_msa)

        if self.cross_attention is not None:
            h_cross = self.cross_attention(
                x=h,
                cross_x=cross_x,
                key_padding_mask=memory_padding_mask,
            )
            h = h + h_cross  # residual
        h_ff = self.feed_forward(modulate(self.ffn_norm(h), shift_mlp, scale_mlp))
        out = h + gate(h_ff, gate_mlp)
        return out


class Transformer(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_layers: int,
        out_channels: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        fc_bias: bool = False,
        ffn_exp: int = 4,
        ffn_dim_multiplier: int = 1,
        multiple_of: int = 64,
        non_linearity: str = "swiglu",
        use_rope: bool = False,
        max_positions: int = 10000,
        context_dim: Union[int, List[int]] = 4096,
        context_non_linearity: str = "swiglu",
        context_embedder_dropout: float = 0.0,
        context_norm: bool = False,
        patch_size: int = 1,
        no_cross_attention: bool = False,
        in_channels: Optional[int] = None,
        BlockCLS: Callable = TransformerBlock,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.output_size = (self.patch_size**2) * self.out_channels
        self.n_layers = n_layers
        self.dim = dim
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.use_rope = use_rope
        self.non_linearity = non_linearity
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.fc_bias = fc_bias

        if in_channels is not None:
            self.data_proj = torch.nn.Linear(in_channels, dim)
            self.cls_token = CLSToken(dim)

        # embeddings
        self.rope_embeddings = None
        # rotary embeddings
        if self.use_rope:
            self.rope_embeddings = RotaryEmbedding(
                theta=max(10000, 2 * max_positions),
                head_dim=dim // n_heads,
                max_seqlen=max_positions,
            )
            self.rope_embeddings.reset_parameters()

        # transformer blocks
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                BlockCLS(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    dropout=dropout,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    fc_bias=fc_bias,
                    ffn_exp=ffn_exp,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    multiple_of=multiple_of,
                    non_linearity=non_linearity,
                    no_cross_attention=no_cross_attention,
                )
            )

        self.norm = RMSNorm(dim, norm_eps)

        # output layer
        self.output = torch.nn.Linear(dim, out_channels, bias=fc_bias)

        self.x_embedder = Patcher(
            in_channels=dim,
            out_channels=dim,
            patch_size=1,
        )

        if not no_cross_attention:
            context_dims = context_dim
            if not isinstance(context_dims, list):
                context_dims = [context_dims]

            self.y_embedders = nn.ModuleList(
                [
                    ContextEmbedder(
                        in_dim=context_dim,
                        out_dim=dim,
                        non_linearity=context_non_linearity,
                        dropout=context_embedder_dropout,
                        fc_bias=fc_bias,
                        norm_eps=norm_eps,
                        context_norm=context_norm,
                    )
                    for context_dim in context_dims
                ]
            )

    def unpatchify(self, x):
        return x

    def embed_y(self, y):
        if y is None:
            return y
        if not isinstance(y, list):
            y = [y]
        y_embs = []
        for i, y_embedder in enumerate(self.y_embedders):
            y_embed = y_embedder(y[i])
            y_embs.append(y_embed)
        return torch.cat(y_embs, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        proj = self.data_proj(x)

        # Prepend the cls token
        with_cls_token = torch.cat([self.cls_token(proj), proj], dim=1)
        if padding_mask is not None:
            padding_mask = torch.cat([padding_mask[:, [0]], padding_mask], dim=1)

        h = self.x_embedder(rearrange(with_cls_token, "b l c-> b c l"))
        h = rearrange(h, "b c l -> b l c")
        original_N = h.shape[1]
        N = h.shape[1]

        h = F.dropout(h, p=self.dropout, training=self.training)

        for layer in self.layers:
            h = layer(
                x=h,
                padding_mask=padding_mask,
                rope=self.rope_embeddings,
            )

        # output layer
        if self.norm is not None:
            h = self.norm(h)

        h = F.dropout(h, p=self.dropout, training=self.training)

        output = self.output(h)

        N = output.shape[1]
        if original_N != N:
            output = output[:, -original_N:]

        output = self.unpatchify(output)

        return output[:, 1:], output[:, 0]


class DiT(Transformer):
    def __init__(
        self,
        *args,
        frequency_embedding_dim: int = 256,
        timestep_non_linearity: str = "swiglu",
        t_block_non_linearity: str = "silu",
        t_block_bias: bool = True,
        **kwargs,
    ):
        super().__init__(*args, BlockCLS=DiTBlock, **kwargs)
        self.t_embedder = TimestepEmbedder(
            self.dim,
            frequency_embedding_dim,
            non_linearity=timestep_non_linearity,
            dropout=self.dropout,
            fc_bias=self.fc_bias,
            max_period=10000,
        )

        self.t_block_non_linearity = get_nonlinearity(t_block_non_linearity)
        self.t_block = torch.nn.Linear(
            self.dim,
            self.dim * 6,
            bias=t_block_bias,
        )

        self.final_layer_scale_shift_table = nn.Parameter(
            torch.randn(2, self.dim) / self.dim**0.5,
        )

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        *,
        padding_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = rearrange(x, "b l c-> b c l")
        h = self.x_embedder(x)
        h = rearrange(h, "b c l -> b l c")
        original_N = h.shape[1]
        N = h.shape[1]

        h = F.dropout(h, p=self.dropout, training=self.training)

        t = self.t_embedder(time)  # B -> B D

        t0 = self.t_block_non_linearity(t)
        t0 = self.t_block(t0)  # B D -> B 6D

        y = self.embed_y(memory)

        for layer in self.layers:
            h = layer(
                x=h,
                cross_x=y,
                t=t0,
                padding_mask=padding_mask,
                memory_padding_mask=memory_padding_mask,
                rope=self.rope_embeddings,
            )

        shift, scale = (self.final_layer_scale_shift_table[None] + t[:, None]).chunk(
            2, dim=1
        )

        # output layer
        if self.norm is not None:
            h = self.norm(h)

        h = modulate(h, shift, scale)

        h = F.dropout(h, p=self.dropout, training=self.training)

        output = self.output(h)

        N = output.shape[1]
        if original_N != N:
            output = output[:, -original_N:]
        output = self.unpatchify(output)
        return output
