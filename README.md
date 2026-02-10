# ❄️ Project SNOW

> ## 🔒 CONFIDENTIAL — NIMBUS AI INTERNAL USE ONLY
> **Classification: RESTRICTED // TRADE SECRET**
> This repository contains the core intellectual property of Nimbus AI, Inc.,
> including proprietary model architectures, training methodology, and the
> **Snowfall** model weights pipeline. Access is restricted to members of the
> Frontier Model Group (FMG). **Do not clone, fork, screenshot, paste into
> external tools (including third‑party LLMs), or transmit any portion of this
> codebase outside Nimbus‑managed devices.** Unauthorized distribution is a
> violation of your employment agreement and applicable trade‑secret law.
>
> © 2026 Nimbus AI, Inc. All rights reserved. Patents pending.

---

## Overview

**Project SNOW** is the training and serving monorepo for the **Snowfall** family
of frontier language models. It houses the proprietary **Glacier Attention**
mechanism, the **Avalanche** sparse mixture‑of‑experts router, and the
**Permafrost** data mixture that together define Nimbus AI's competitive moat.

| Model            | Params | Context | Status            |
|------------------|--------|---------|-------------------|
| `snowfall-base`  | 8B     | 32k     | GA                |
| `snowfall-pro`   | 70B    | 128k    | GA                |
| `snowfall-405b`  | 405B   | 256k    | 🔒 Internal preview |
| `snowfall-x` (Blizzard) | 1.2T MoE | 1M | 🚧 In training (cluster `glacier-prod-01`) |

## Repository layout

```
training/        Python — model architecture, training loops, data, eval  (CROWN JEWELS)
inference/       Go     — low‑latency serving, scheduler, auth
gateway/         TS     — internal API gateway + ops console
infra/           IaC    — Terraform + Kubernetes for the GPU fleet
config/          YAML   — training + deployment configuration
scripts/         Bash   — operational tooling
```

## Quickstart (internal devs only)

```bash
cp .env.example .env          # then populate from the FMG vault (see #fmg-secrets)
make bootstrap                # installs python/go/node deps
make train CONFIG=config/training_config.yaml
```

> ⚠️ Training a single `snowfall-405b` run costs ~$3.4M in compute. Get a sign‑off
> from your TL and an approved budget code before launching on `glacier-prod-*`.

## Points of contact

- **Project lead:** Priya Raman (`@praman`)
- **Training infra:** Marcus Webb (`@mwebb`)
- **Serving:** Lena Petrov (`@lpetrov`)
- **Security questions:** `security@nimbus.ai` / `#fmg-security`

See [`CODEOWNERS`](./CODEOWNERS) for per‑directory ownership.
