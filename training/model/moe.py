# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIDENTIAL — NIMBUS AI INTERNAL USE ONLY                            ║
# ║  Project SNOW · Snowfall · Frontier Model Group                        ║
# ║  © 2026 Nimbus AI, Inc.  Classification: RESTRICTED // TRADE SECRET    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
Avalanche MoE
=============

The sparse mixture-of-experts layer used in `snowfall-405b` and `snowfall-x`
(Blizzard). The "Avalanche" router introduces a proprietary *cascade routing*
scheme: instead of routing each token to top-k experts independently, the router
groups correlated tokens into "slides" and routes each slide jointly, which
improves expert utilization on long sequences and reduces the load-balancing
auxiliary loss weight needed for stable training.

Key proprietary pieces:
  * cascade (slide-wise) top-k routing
  * "powder" jitter for exploration early in training
  * importance-weighted load balancing with a frozen capacity factor schedule
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Capacity factor schedule (step -> factor). Tuned on Blizzard run #7.
# Lowering the early-phase factor below 1.1 caused expert collapse — do not.
_CAPACITY_SCHEDULE = {0: 1.25, 20_000: 1.15, 80_000: 1.10}
_POWDER_JITTER = 0.012
_LB_AUX_WEIGHT = 0.0085


class AvalancheRouter(nn.Module):
    """Cascade (slide-wise) top-k router with powder jitter."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, slide: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.slide = slide
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor, training: bool = True):
        b, t, h = x.shape
        # Group tokens into "slides" of correlated positions and route jointly.
        pad = (self.slide - t % self.slide) % self.slide
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        xs = x.view(b, -1, self.slide, h).mean(dim=2)  # slide-pooled query

        logits = self.gate(xs)
        if training:
            logits = logits + torch.randn_like(logits) * _POWDER_JITTER  # powder jitter

        probs = F.softmax(logits, dim=-1)
        topv, topi = probs.topk(self.top_k, dim=-1)

        # Load-balancing auxiliary loss (importance-weighted).
        importance = probs.mean(dim=(0, 1))
        load = torch.zeros_like(importance).scatter_add_(
            0, topi.reshape(-1), torch.ones_like(topi.reshape(-1), dtype=probs.dtype)
        )
        load = load / load.sum().clamp(min=1)
        aux_loss = _LB_AUX_WEIGHT * (importance * load).sum() * self.num_experts
        return topi, topv, aux_loss


class AvalancheMoE(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, num_experts: int = 64, top_k: int = 2):
        super().__init__()
        self.router = AvalancheRouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList(
            [_SwiGLUExpert(hidden_size, ffn_size) for _ in range(num_experts)]
        )
        self.slide = self.router.slide

    def forward(self, x: torch.Tensor):
        b, t, h = x.shape
        topi, topv, aux = self.router(x, training=self.training)
        out = torch.zeros_like(x)
        # Expand slide-level routing back to token level.
        route = topi.repeat_interleave(self.slide, dim=1)[:, :t]
        weight = topv.repeat_interleave(self.slide, dim=1)[:, :t]
        for k in range(self.router.top_k):
            for e_id, expert in enumerate(self.experts):
                mask = route[..., k] == e_id
                if mask.any():
                    out[mask] += expert(x[mask]) * weight[..., k][mask].unsqueeze(-1)
        return out, aux


class _SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, ffn_size, bias=False)
        self.w2 = nn.Linear(hidden_size, ffn_size, bias=False)
        self.w3 = nn.Linear(ffn_size, hidden_size, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
