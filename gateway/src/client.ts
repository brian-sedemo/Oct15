// ╔══════════════════════════════════════════════════════════════════════╗
// ║  CONFIDENTIAL — NIMBUS AI INTERNAL USE ONLY                            ║
// ║  Project SNOW · Snowfall gateway · Frontier Model Group               ║
// ║  © 2026 Nimbus AI, Inc.  Classification: RESTRICTED // TRADE SECRET    ║
// ╚══════════════════════════════════════════════════════════════════════╝

import { loadConfig } from "./config.js";

export interface GenerateOptions {
  model?: string;
  maxTokens?: number;
  temperature?: number;
}

export interface Completion {
  id: string;
  model: string;
  completion: string;
  tokensOut: number;
  latencyMs: number;
}

/** Thin client over the Go Snowfall serving API. Internal use only. */
export class SnowfallClient {
  private readonly baseUrl: string;
  private readonly adminToken: string;

  constructor() {
    const cfg = loadConfig();
    this.baseUrl = cfg.inferenceUrl;
    this.adminToken = cfg.adminToken;
  }

  async generate(prompt: string, opts: GenerateOptions = {}): Promise<Completion> {
    const res = await fetch(`${this.baseUrl}/v1/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.adminToken}`,
      },
      body: JSON.stringify({
        model: opts.model ?? "snowfall-pro-70b",
        prompt,
        max_tokens: opts.maxTokens ?? 512,
        temperature: opts.temperature ?? 0.7,
        stream: false,
      }),
    });

    if (!res.ok) {
      throw new Error(`Snowfall serving error: ${res.status} ${await res.text()}`);
    }
    const body = (await res.json()) as {
      id: string;
      model: string;
      completion: string;
      tokens_out: number;
      latency_ms: number;
    };
    return {
      id: body.id,
      model: body.model,
      completion: body.completion,
      tokensOut: body.tokens_out,
      latencyMs: body.latency_ms,
    };
  }
}
