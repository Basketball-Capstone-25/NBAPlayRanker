/**
 * I-prefix interfaces for the Application Layer (Frontend).
 *
 * These interfaces define the public contracts for frontend service subsystems.
 */

export interface ITeamOption {
  value: string;
  label: string;
}

export interface INormalizedPlay {
  playType: string;
  pppPred: number;
  pppOff: number;
  pppDef: number;
  pppGap: number;
  rationale: string;
}

export interface IGameplanService {
  normalizeTeams(meta: { teams?: string[]; teamNames?: Record<string, string> }): ITeamOption[];
  normalizeWhy(raw: Record<string, unknown>): string;
  normalizePlay(raw: Record<string, unknown>): INormalizedPlay;
}
