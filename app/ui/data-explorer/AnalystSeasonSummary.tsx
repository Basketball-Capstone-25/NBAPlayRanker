"use client";

import React, { useState } from "react";
import SeasonSummary from "../_shared/components/SeasonSummary";
import type { SummaryFilters } from "../../domain/season-summary";

export function analystSummaryFilters(season: string, team: string, side: string, opponent: string): SummaryFilters | null {
  return season && team && side === "offense" && opponent && team !== opponent
    ? { season, our: team, opp: opponent, k: 5, w_off: 0.7 } : null;
}

export default function AnalystSeasonSummary({ season, team, side, teams, teamNames }: {
  season: string; team: string; side: string; teams: string[]; teamNames?: Record<string, string>;
}) {
  const [selectedOpponent, setSelectedOpponent] = useState("BOS");
  const opponents = teams.filter(value => value !== team);
  const opponent = opponents.includes(selectedOpponent) ? selectedOpponent : opponents[0] || "";
  const filters = analystSummaryFilters(season, team, side, opponent);
  return <div>
    <label style={{ display: "grid", gap: 6, maxWidth: 300, fontSize: 13 }}>
      Summary opponent
      <select className="input" value={opponent} disabled={!season || !team || side !== "offense" || !opponents.length}
        onChange={event => setSelectedOpponent(event.target.value)}>
        {!opponents.length && <option value="">No opponents available</option>}
        {opponents.map(value => <option key={value} value={value}>{teamNames?.[value] ? `${value} — ${teamNames[value]}` : value}</option>)}
      </select>
    </label>
    <p className="muted" style={{ fontSize: 13 }}>Summary uses the full offensive season and the top 5 baseline play types. Table search, play-type filters, possession minimum and row limit do not change it.</p>
    <SeasonSummary filters={filters} unavailableReason="Select one team and Offense in the filters above to view a season summary." />
  </div>;
}
