import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


class GQAAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(
            hidden_size,
            num_attention_heads * head_dim,
            bias=bias,
        )

        self.k_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * head_dim,
            bias=bias,
        )

        self.v_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * head_dim,
            bias=bias,
        )

        self.o_proj = nn.Linear(
            num_attention_heads * head_dim,
            hidden_size,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):

        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            B,
            T,
            self.num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        try:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                enable_gqa=True,
            )

        except (TypeError, RuntimeError):

            repeat_factor = (
                self.num_attention_heads
                // self.num_key_value_heads
            )

            k = k.repeat_interleave(
                repeat_factor,
                dim=1,
            )

            v = v.repeat_interleave(
                repeat_factor,
                dim=1,
            )

            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
            )

        out = out.transpose(1, 2).contiguous()

        out = out.view(
            B,
            T,
            self.num_attention_heads * self.head_dim,
        )

        return self.o_proj(out)