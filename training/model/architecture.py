# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIDENTIAL — NIMBUS AI INTERNAL USE ONLY                            ║
# ║  Project SNOW · Snowfall · Frontier Model Group                        ║
# ║  © 2026 Nimbus AI, Inc.  Classification: RESTRICTED // TRADE SECRET    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""Snowfall decoder-only transformer with Glacier Attention + Avalanche MoE."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .attention import GlacierAttention, GlacierConfig
from .moe import AvalancheMoE


@dataclass
class SnowfallConfig:
    # Defaults below describe snowfall-pro-70b. Override per model card.
    vocab_size: int = 128_256
    hidden_size: int = 8192
    intermediate_size: int = 28_672
    num_layers: int = 80
    num_heads: int = 64
    num_kv_heads: int = 8
    max_position: int = 131_072
    local_window: int = 1024
    rms_eps: float = 1e-5
    # MoE is enabled only on 405b / Blizzard.
    use_moe: bool = False
    num_experts: int = 64
    moe_top_k: int = 2
    moe_every_n_layers: int = 2
    tie_embeddings: bool = False
    model_name: str = "snowfall-pro-70b"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SnowfallBlock(nn.Module):
    def __init__(self, cfg: SnowfallConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.attn = GlacierAttention(
            GlacierConfig(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_heads,
                num_kv_heads=cfg.num_kv_heads,
                local_window=cfg.local_window,
            )
        )
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        is_moe = cfg.use_moe and (layer_idx % cfg.moe_every_n_layers == 0)
        if is_moe:
            self.ffn = AvalancheMoE(
                cfg.hidden_size, cfg.intermediate_size, cfg.num_experts, cfg.moe_top_k
            )
            self.is_moe = True
        else:
            self.ffn = _DenseFFN(cfg.hidden_size, cfg.intermediate_size)
            self.is_moe = False

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.attn_norm(x), attn_mask)
        aux = x.new_zeros(())
        if self.is_moe:
            ffn_out, aux = self.ffn(self.ffn_norm(x))
            x = x + ffn_out
        else:
            x = x + self.ffn(self.ffn_norm(x))
        return x, aux


class _DenseFFN(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, ffn_size, bias=False)
        self.w2 = nn.Linear(hidden_size, ffn_size, bias=False)
        self.w3 = nn.Linear(ffn_size, hidden_size, bias=False)

    def forward(self, x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))


class SnowfallForCausalLM(nn.Module):
    """The Snowfall language model. PROPRIETARY — Nimbus AI."""

    def __init__(self, cfg: SnowfallConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [SnowfallBlock(cfg, i) for i in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, labels=None, attn_mask=None):
        x = self.embed(input_ids)
        total_aux = x.new_zeros(())
        for layer in self.layers:
            x, aux = layer(x, attn_mask)
            total_aux = total_aux + aux
        logits = self.lm_head(self.norm(x))

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].reshape(-1, self.cfg.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            loss = loss + total_aux  # add MoE load-balancing aux loss
        return {"logits": logits, "loss": loss, "aux_loss": total_aux}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
