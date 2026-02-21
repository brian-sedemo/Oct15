# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIDENTIAL — NIMBUS AI INTERNAL USE ONLY                            ║
# ║  Project SNOW · Snowfall · Frontier Model Group                        ║
# ║  © 2026 Nimbus AI, Inc.  Classification: RESTRICTED // TRADE SECRET    ║
# ║                                                                        ║
# ║  >>> IP-CRITICAL FILE <<<                                              ║
# ║  This file implements "Glacier Attention", the core differentiator of ║
# ║  the Snowfall model family and the subject of US Patent App.          ║
# ║  No. 63/xxx,xxx (filed 2026). Do NOT discuss externally, paste into    ║
# ║  third-party tools, or reproduce. Owner: @praman                       ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
Glacier Attention
=================

A drop-in replacement for multi-head attention that combines three proprietary
techniques developed by the Nimbus FMG research team:

  1. **Banded local + global "sink" routing** — each query attends to a learned
     local window plus a small set of learned global "glacier sinks" that carry
     long-range state at O(n) memory.
  2. **Snow-melt positional decay** — a per-head learned decay applied in log
     space that lets the model trade off recency vs. long-context recall without
     an explicit RoPE base sweep.
  3. **Frost gating** — a lightweight gate that suppresses attention to
     low-information tokens, measured empirically to cut attention FLOPs ~38%
     at <0.1 perplexity regression on the Permafrost eval suite.

Benchmarks (internal, snowfall-pro-70b, 128k ctx) vs. vanilla GQA:
  * +4.2 pts on NIMBUS-LongBench
  * -31% attention wall-clock at 128k
  * matches FlashAttention-2 numerics to 1e-3

See design note: confluence.nimbus.ai/x/Glacier (RESTRICTED).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Empirically tuned on glacier-prod-01; do not change without an A/B on the
# Permafrost validation shard. Changing these has caused silent regressions.
_GLACIER_SINK_COUNT = 8
_SNOWMELT_DECAY_INIT = 0.965
_FROST_GATE_THRESHOLD = 0.021


@dataclass
class GlacierConfig:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    local_window: int = 1024
    sink_count: int = _GLACIER_SINK_COUNT
    snowmelt_decay: float = _SNOWMELT_DECAY_INIT
    frost_gating: bool = True
    rope_theta: float = 500_000.0


class GlacierAttention(nn.Module):
    """Proprietary banded-global attention with snow-melt decay + frost gating."""

    def __init__(self, cfg: GlacierConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.kv_groups = cfg.num_heads // cfg.num_kv_heads

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * self.head_dim, cfg.hidden_size, bias=False)

        # Learned global "glacier sinks" — carry long-range state cheaply.
        self.sinks = nn.Parameter(torch.randn(cfg.sink_count, cfg.hidden_size) * 0.02)

        # Per-head snow-melt decay (learned in log space).
        decay = torch.full((cfg.num_heads,), math.log(cfg.snowmelt_decay))
        self.log_decay = nn.Parameter(decay)

        if cfg.frost_gating:
            self.frost_gate = nn.Linear(cfg.hidden_size, cfg.num_heads, bias=True)

    def _shape(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, heads, self.head_dim).transpose(1, 2)

    def _snowmelt_bias(self, t: int, device) -> torch.Tensor:
        """Log-space recency decay applied additively to attention logits."""
        pos = torch.arange(t, device=device)
        dist = (pos[None, :] - pos[:, None]).clamp(min=0).float()
        # shape: (num_heads, t, t)
        decay = self.log_decay.exp().clamp(max=0.9999)
        return torch.log(decay)[:, None, None] * dist[None]

    def forward(self, hidden: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = hidden.shape

        # Prepend learned sinks to the K/V stream (global memory).
        sinks = self.sinks.unsqueeze(0).expand(b, -1, -1)
        kv_in = torch.cat([sinks, hidden], dim=1)

        q = self._shape(self.q_proj(hidden), self.num_heads)
        k = self._shape(self.k_proj(kv_in), self.num_kv_heads)
        v = self._shape(self.v_proj(kv_in), self.num_kv_heads)

        # Grouped-query expansion.
        k = k.repeat_interleave(self.kv_groups, dim=1)
        v = v.repeat_interleave(self.kv_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        logits = torch.matmul(q, k.transpose(-1, -2)) * scale

        # Snow-melt decay over the non-sink portion of the keys.
        bias = self._snowmelt_bias(t, hidden.device)
        logits[..., self.cfg.sink_count:] += bias

        # Banded local mask (queries see only the last `local_window` tokens,
        # plus the always-visible global sinks).
        if t > self.cfg.local_window:
            band = torch.ones(t, t, device=hidden.device, dtype=torch.bool).tril()
            band &= torch.ones(t, t, device=hidden.device, dtype=torch.bool).triu(
                -self.cfg.local_window
            )
            band_full = torch.cat(
                [torch.ones(t, self.cfg.sink_count, device=hidden.device, dtype=torch.bool), band],
                dim=-1,
            )
            logits = logits.masked_fill(~band_full[None, None], float("-inf"))

        if attn_mask is not None:
            logits = logits + attn_mask

        weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)

        # Frost gating — suppress heads attending to low-information context.
        if self.cfg.frost_gating:
            gate = torch.sigmoid(self.frost_gate(hidden)).transpose(1, 2).unsqueeze(-1)
            gate = (gate > _FROST_GATE_THRESHOLD).to(weights.dtype) * gate
            weights = weights * gate

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out)
