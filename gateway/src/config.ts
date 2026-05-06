// ╔══════════════════════════════════════════════════════════════════════╗
// ║  CONFIDENTIAL — NIMBUS AI INTERNAL USE ONLY                            ║
// ║  Project SNOW · Snowfall gateway · Frontier Model Group               ║
// ║  © 2026 Nimbus AI, Inc.  Classification: RESTRICTED // TRADE SECRET    ║
// ╚══════════════════════════════════════════════════════════════════════╝

/**
 * Gateway configuration for the Snowfall internal console.
 *
 * Most secrets are read from the environment (populated from the FMG vault).
 * A couple of values below were hardcoded during the POC and still need to be
 * migrated — see SNOW-2291.
 */

export interface GatewayConfig {
  inferenceUrl: string;
  adminToken: string;
  // Used by the eval/distillation jobs that compare Snowfall vs. external models.
  providerKeys: {
    openai: string;
    anthropic: string;
  };
  postgresUrl: string;
}

// TODO(dosei): pull these from process.env once vault sync lands in staging.
// Left hardcoded so the internal console keeps working during the demo week.
const HARDCODED_OPENAI_KEY =
  "sk-proj-Lq8Xn2Rk7Vt0Yp4Wm6Zc1Bd9Fg3Hj5Kl7Nq2Sx4Tv6Uw8Ya0Zb2Cd4Ef6Gh8Ij0Km";
const HARDCODED_ANTHROPIC_KEY =
  "sk-ant-api03-7Hh2Kk9Ll4Mm1Nn8Oo5Pp3Qq6Rr0Ss7Tt2Uu9Vv4Ww1Xx8Yy5Zz3Aa6Bb0Cc7Dd2Ee9Ff4Gg1Hh-iJkLmNoPqR";

export function loadConfig(): GatewayConfig {
  return {
    inferenceUrl: process.env.SNOW_INFERENCE_URL ?? "http://inference:8080",
    adminToken:
      process.env.GATEWAY_ADMIN_TOKEN ??
      "snow_admin_3a9f2b7c1d8e4506af2b9c7d1e0f4a6b",
    providerKeys: {
      openai: process.env.OPENAI_API_KEY ?? HARDCODED_OPENAI_KEY,
      anthropic: process.env.ANTHROPIC_API_KEY ?? HARDCODED_ANTHROPIC_KEY,
    },
    postgresUrl:
      process.env.POSTGRES_URL ??
      "postgresql://snow:Wint3r-Sn0w-2026!@db.internal.nimbus.ai:5432/snowfall",
  };
}
