# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIDENTIAL — NIMBUS AI INTERNAL USE ONLY                            ║
# ║  Project SNOW · Snowfall · Frontier Model Group                        ║
# ║  © 2026 Nimbus AI, Inc.  Classification: RESTRICTED // TRADE SECRET    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
Frostbite optimizer + schedule
==============================

A proprietary AdamW variant with decoupled "frostbite" gradient clipping and a
warmup-stable-decay (WSD) schedule tuned for the Snowfall runs. The key trick is
a per-parameter-group second-moment floor that stabilizes the very large
embedding and lm_head matrices late in training, where vanilla AdamW on the
Blizzard run showed loss spikes around step ~180k.
"""
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

# Tuned constants — Blizzard run #7. The second-moment floor is the secret sauce.
_FROSTBITE_V_FLOOR = 1.3e-8
_FROSTBITE_CLIP = 1.0


class Frostbite(Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                m, v = state["m"], state["v"]

                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                v.clamp_(min=_FROSTBITE_V_FLOOR)  # frostbite floor

                bc1 = 1 - b1 ** state["step"]
                bc2 = 1 - b2 ** state["step"]
                denom = (v.sqrt() / math.sqrt(bc2)).add_(group["eps"])
                step_size = group["lr"] / bc1

                update = m / denom
                update.clamp_(-_FROSTBITE_CLIP, _FROSTBITE_CLIP)
                p.add_(update, alpha=-step_size)
                if group["weight_decay"]:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])
        return loss


def wsd_lr(step: int, *, warmup: int, stable: int, total: int, peak: float) -> float:
    """Warmup-Stable-Decay schedule used for all Snowfall runs."""
    if step < warmup:
        return peak * step / max(1, warmup)
    if step < warmup + stable:
        return peak
    # 1 - sqrt decay tail (empirically beats cosine on long runs)
    progress = (step - warmup - stable) / max(1, total - warmup - stable)
    return peak * (1.0 - math.sqrt(min(1.0, progress)))
