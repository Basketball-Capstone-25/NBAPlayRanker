/**
 * Domain Layer barrel file.
 *
 * Subsystems:
 * - Context Analysis: overlay computation for game-context
 * - Weight Calculation: weight normalization rules
 * - Coaching Knowledge: coaching cues and counter-plan rules
 */

// Context Analysis subsystem
export {
  clamp01,
  computeLateFactor,
  computeScoreFactors,
  categorizePlay,
  computeOverlay,
  type ContextRow,
  type OverlayBreakdown,
} from "./context-analysis";

// Weight Calculation subsystem
export { normalizeWeights } from "./weight-calculation";

// Coaching Knowledge subsystem
export {
  gradeFromRank,
  coachingCuesFor,
  counterPlanFor,
  type CounterPlanEntry,
} from "./coaching-knowledge";

// I-prefix interfaces (public contracts)
export type {
  IContextRow,
  IOverlayBreakdown,
  IContextAnalyzer,
  IWeightCalculator,
  ICounterPlanEntry,
  ICoachingKnowledge,
} from "./interfaces";
