"use client";

// UI Layer — season summary on the Data Explorer for one team against a chosen opponent.
// Shown only for offense with a single team, so the summary always matches the table filters.

import React, { useState } from "react";
import SeasonSummary from "../_shared/components/SeasonSummary";
import type { SummaryFilters } from "../../domain/season-summary";

/** Summary filters for the current table choice, or null when the choice cannot be summarised. */
export function analystSummaryFilters(season: string, team: string, side: string, opponent: string): SummaryFilters | null {
  return season && team && side === "offense" && opponent && team !== opponent
    ? { season, our: team, opp: opponent, k: 5, w_off: 0.7 } : null;
}

/** Opponent picker plus the shared season summary card. */
export default function AnalystSeasonSummary({ season, team, side, teams, teamNames, refreshVersion = 0, refreshing = false }: {
  season: string; team: string; side: string; teams: string[]; teamNames?: Record<string, string>; refreshVersion?: number; refreshing?: boolean;
}) {
  // SCRUM-841: the analyst picks the opponent; none is pre-selected.
  const [selectedOpponent, setSelectedOpponent] = useState("");
  const opponents = teams.filter(value => value !== team);
  const opponent = opponents.includes(selectedOpponent) ? selectedOpponent : "";
  const filters = refreshing ? null : analystSummaryFilters(season, team, side, opponent);
  return <div>
    <label style={{ display: "grid", gap: 6, maxWidth: 300, fontSize: 13 }}>
      Summary opponent
      <select className="input" value={opponent} disabled={!season || !team || side !== "offense" || !opponents.length}
        onChange={event => setSelectedOpponent(event.target.value)}>
        <option value="" disabled>{opponents.length ? "Choose an opponent" : "No opponents available"}</option>
        {opponents.map(value => <option key={value} value={value}>{teamNames?.[value] ? `${value} — ${teamNames[value]}` : value}</option>)}
      </select>
    </label>
    <p className="muted" style={{ fontSize: 13 }}>Summary uses the full offensive season and the top 5 baseline play types. Table search, play-type filters, possession minimum and row limit do not change it.</p>
    <SeasonSummary key={refreshVersion} filters={filters} unavailableReason={refreshing ? "Refresh the play-type view before loading its summary."
      : "Select one team and Offense in the filters above, then choose an opponent, to view a season summary."} />
  </div>;
}
