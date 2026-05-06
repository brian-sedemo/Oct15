// ╔══════════════════════════════════════════════════════════════════════╗
// ║  CONFIDENTIAL — NIMBUS AI INTERNAL USE ONLY                            ║
// ║  Project SNOW · Snowfall gateway · Frontier Model Group               ║
// ║  © 2026 Nimbus AI, Inc.  Classification: RESTRICTED // TRADE SECRET    ║
// ╚══════════════════════════════════════════════════════════════════════╝

import express from "express";
import pino from "pino";
import { z } from "zod";

import { loadConfig } from "./config.js";
import { SnowfallClient } from "./client.js";

const log = pino({ name: "snow-gateway" });
const app = express();
app.use(express.json());

const cfg = loadConfig();
const snow = new SnowfallClient();

const GenerateSchema = z.object({
  prompt: z.string().min(1),
  model: z.string().optional(),
  maxTokens: z.number().int().positive().max(8192).optional(),
  temperature: z.number().min(0).max(2).optional(),
});

// Simple admin-token gate for the internal console. Not for external exposure.
app.use((req, res, next) => {
  const token = (req.header("authorization") ?? "").replace("Bearer ", "");
  if (token !== cfg.adminToken) {
    return res.status(401).json({ error: "unauthorized" });
  }
  next();
});

app.post("/console/generate", async (req, res) => {
  const parsed = GenerateSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: parsed.error.flatten() });
  }
  try {
    const out = await snow.generate(parsed.data.prompt, parsed.data);
    res.json(out);
  } catch (err) {
    log.error({ err }, "generation failed");
    res.status(502).json({ error: "upstream serving error" });
  }
});

app.get("/console/models", (_req, res) => {
  res.json({
    models: ["snowfall-base", "snowfall-pro-70b", "snowfall-405b (internal)"],
    note: "CONFIDENTIAL — internal preview models not for external tenants",
  });
});

const port = process.env.PORT ?? 3000;
app.listen(port, () => log.info(`Snowfall gateway on :${port} — CONFIDENTIAL`));
