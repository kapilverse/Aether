import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import GQAAttention
from .rmsnorm import RMSNorm
from .rope import rope_cache
from .swiglu import SwiGLU


class AETHERConfig:
    vocab_size = 32000
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12
    num_key_value_heads = 4
    head_dim = 64
    intermediate_size = 2048
    max_position_embeddings = 2048
    rope_theta = 10000.0
    rms_norm_eps = 1e-6
    tie_word_embeddings = True
    bias = False
    gradient_checkpointing = True
    dropout = 0.0


class AETHERBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.self_attn = GQAAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            bias=config.bias,
        )

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.bias,
        )

    def forward(self, x, cos, sin):

        x = x + self.self_attn(
            self.input_layernorm(x),
            cos,
            sin,
        )

        x = x + self.mlp(
            self.post_attention_layernorm(x)
        )

        return x


class AETHER100M(nn.Module):

    def __init__(self, config=None):

        super().__init__()

        if config is None:
            config = AETHERConfig()

        self.config = config

        self.embed = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [
                AETHERBlock(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):

        std = 0.02

        nn.init.normal_(
            self.embed.weight,
            mean=0.0,
            std=std,
        )

        for module in self.modules():

            if isinstance(module, nn.Linear):

                if module.weight is self.embed.weight:
                    continue

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=std,
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        residual_scale = 1.0 / math.sqrt(
            2 * self.config.num_hidden_layers
        )

        for layer in self.layers:

            layer.self_attn.o_proj.weight.data.mul_(
                residual_scale
            )

            layer.mlp.down_proj.weight.data.mul_(
                residual_scale
            )

    def forward(
        self,
        input_ids,
        labels=None,
    ):

        B, T = input_ids.shape

        if T > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"maximum {self.config.max_position_embeddings}"
            )

        x = self.embed(input_ids)

        cos, sin = rope_cache(
            seq_len=T,
            head_dim=self.config.head_dim,
            theta=self.config.rope_theta,
            device=input_ids.device,
        )

        for layer in self.layers:

            if (
                self.config.gradient_checkpointing
                and self.training
            ):

                x = checkpoint(
                    layer,
                    x,
                    cos,
                    sin,
                    use_reentrant=False,
                )

            else:

                x = layer(
                    x,
                    cos,
                    sin,
                )

        x = self.norm(x)

        loss = None
        logits = None

        if labels is not None:

            shift_x = x[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            B, T, H = shift_x.shape

            # Avoid creating the enormous
            # [B, T, vocab_size] tensor at once.
            chunk_tokens = 64

            total_loss = torch.zeros(
                (),
                device=x.device,
                dtype=torch.float32,
            )

            total_tokens = 0

            flat_x = shift_x.view(
                -1,
                H,
            )

            flat_labels = shift_labels.view(
                -1,
            )

            for start in range(
                0,
                B * T,
                chunk_tokens,
            ):

                end = min(
                    start + chunk_tokens,
                    B * T,
                )

                hidden_chunk = flat_x[
                    start:end
                ]

                label_chunk = flat_labels[
                    start:end
                ]

                logits_chunk = self.lm_head(
                    hidden_chunk
                )

                chunk_loss = F.cross_entropy(
                    logits_chunk,
                    label_chunk,
                    reduction="sum",
                )

                total_loss += chunk_loss.float()

                total_tokens += label_chunk.numel()

            loss = (
                total_loss /
                total_tokens
            )

        else:

            logits = self.lm_head(x)

        return {
            "logits": logits,
            "loss": loss,
        }


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )