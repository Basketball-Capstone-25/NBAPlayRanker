/**
 * I-prefix interfaces for the Domain Layer (Frontend).
 *
 * These interfaces define the public contracts for frontend domain subsystems.
 */

export interface IContextRow {
  PLAY_TYPE: string;
  PPP_CONTEXT: number;
  PPP_ML_BLEND: number;
  PPP_BASELINE: number;
  DELTA_VS_BASELINE: number;
  CONTEXT_LABEL: string;
  RATIONALE: string;
}

export interface IOverlayBreakdown {
  lateFactor: number;
  scoreFactor: number;
  category: string;
  overlayDelta: number;
}

export interface IContextAnalyzer {
  computeLateFactor(period: number, secRemaining: number): number;
  computeScoreFactors(margin: number): { leadPenalty: number; trailBoost: number };
  categorizePlay(playType: string): string;
  computeOverlay(
    margin: number,
    period: number,
    secRemaining: number,
    playType: string,
  ): IOverlayBreakdown;
}

export interface IWeightCalculator {
  normalizeWeights(wOff: number): { wOff: number; wDef: number };
}

export interface ICounterPlanEntry {
  counterPlay: string;
  why: string;
}

export interface ICoachingKnowledge {
  gradeFromRank(rank: number, total: number): string;
  coachingCuesFor(playType: string): string[];
  counterPlanFor(playType: string): ICounterPlanEntry[];
}
