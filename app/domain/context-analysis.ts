// app/domain/context-analysis.ts
//
// Domain Layer — Context Analysis subsystem
// Overlay computation: computeLateFactor, computeScoreFactors, categorizePlay, computeOverlay
// Client-side game-context scoring algorithm extracted from ContextPageClient.

export type ContextRow = {
  playType: string;
  finalPPP: number;
  mlPPP: number;
  baselinePPP: number;
  deltaPPP: number;
  contextLabel: string;
  rationale: string;
  raw: Record<string, any>;
};

export type OverlayBreakdown = {
  overlayPPP: number;
  overlayEff: number;
  overlayQuick: number;
  overlayProtect: number;
  overlayType: number;
  lateFactor: number;
  trailingFactor: number;
  leadingFactor: number;
  varianceProxy: number;
};

export function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

export function computeLateFactor(period: number, timeRemainingSec: number) {
  const t = Number(timeRemainingSec);
  const p = Number(period);

  const timeLeft = Math.max(0, Math.min(720, Number.isFinite(t) ? t : 720));
  const lateRamp = clamp01((180 - timeLeft) / 180);

  if (p <= 3) return 0.10 + 0.15 * lateRamp;
  if (p === 4) return 0.25 + 0.75 * lateRamp;
  return 0.70 + 0.30 * lateRamp;
}

export function computeScoreFactors(margin: number) {
  const m = Number(margin);
  const cap = 15;
  const trailing = clamp01((-m) / cap);
  const leading = clamp01(m / cap);
  return { trailing, leading };
}

export function categorizePlay(playType: string) {
  const s = (playType ?? "").toLowerCase();

  const quick =
    s.includes("transition") ||
    s.includes("spot up") ||
    s.includes("cut") ||
    s.includes("putback") ||
    s.includes("offensive rebound") ||
    s.includes("rim") ||
    s.includes("fast") ||
    s.includes("handoff");

  const safe =
    s.includes("pick and roll") ||
    s.includes("p&r") ||
    s.includes("spot up") ||
    s.includes("handoff") ||
    s.includes("off screen") ||
    s.includes("post up");

  const risky = s.includes("isolation") || s.includes("iso") || s.includes("transition");
  const slow = s.includes("post up") || s.includes("off screen") || s.includes("handoff");

  return { quick, safe, risky, slow };
}

export function computeOverlay(
  row: ContextRow,
  meanMlPPP: number,
  period: number,
  timeRemaining: number,
  margin: number,
  strength: number
): OverlayBreakdown {
  const late = computeLateFactor(period, timeRemaining);
  const { trailing, leading } = computeScoreFactors(margin);

  const ml = Number(row.mlPPP);
  const base = Number(row.baselinePPP);

  const mlDiffFromMean = Number.isFinite(ml) && Number.isFinite(meanMlPPP) ? ml - meanMlPPP : 0;

  const varianceProxy = Math.abs((Number.isFinite(ml) ? ml : 0) - (Number.isFinite(base) ? base : 0));
  const { quick, safe, risky, slow } = categorizePlay(row.playType);

  const overlayEff =
    late *
    trailing *
    clamp01(Math.abs(mlDiffFromMean) / 0.25) *
    (mlDiffFromMean >= 0 ? 1 : -1) *
    0.020;

  const overlayQuick = quick ? late * trailing * 0.010 : 0;

  const overlayProtect =
    late *
    leading *
    (-0.018 * clamp01(varianceProxy / 0.25)) *
    (risky ? 1.2 : 1.0);

  let overlayType = 0;
  if (trailing > 0) overlayType += slow ? -(late * trailing) * 0.006 : 0;
  if (leading > 0) overlayType += safe ? (late * leading) * 0.006 : 0;

  const overlayPPP_raw = overlayEff + overlayQuick + overlayProtect + overlayType;

  const s = Math.max(0, Math.min(2, Number(strength)));
  const overlayPPP = overlayPPP_raw * s;

  return {
    overlayPPP,
    overlayEff: overlayEff * s,
    overlayQuick: overlayQuick * s,
    overlayProtect: overlayProtect * s,
    overlayType: overlayType * s,
    lateFactor: late,
    trailingFactor: trailing,
    leadingFactor: leading,
    varianceProxy,
  };
}
