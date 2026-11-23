"use client";

// UI Layer — where a chosen player made shots against the opponent (SCRUM-825).
// Nothing is pre-selected: the coach picks seasons and a player first.

import React, { useEffect, useState } from "react";
import {
  COURT,
  CELL_FEET,
  cellOpacity,
  describeMadeShots,
  heatCells,
  toSvgX,
  toSvgY,
} from "../../../domain/player-shots";
import {
  fetchPlayerMadeShots,
  fetchPlayerShotSeasons,
  fetchShotPlayers,
  type PlayerMadeShots,
  type ShotPlayer,
} from "../../../infrastructure/api-client";

const GREEN = "#16a34a";
const LINE = "rgba(15, 23, 42, 0.55)";

/** Half court with made shots binned into 2 ft squares; deeper green means more makes. */
export function CourtHeatmap({ result }: { result: PlayerMadeShots }) {
  const cells = heatCells(result.points);
  const size = CELL_FEET * COURT.scale;
  const s = COURT.scale;
  const hoopY = toSvgY(COURT.basketX);
  const cornerY = toSvgY(COURT.basketX - Math.sqrt(23.75 ** 2 - 22 ** 2));
  return (
    <svg
      viewBox={`0 0 ${COURT.width} ${COURT.height}`}
      role="img"
      aria-label={describeMadeShots(result)}
      style={{ width: "100%", maxWidth: 520, background: "#fff", borderRadius: 12, border: "1px solid rgba(15,23,42,0.12)" }}
    >
      <g data-testid="heat-cells">
        {cells.map((c) => (
          <rect key={`${c.col}:${c.row}`} x={c.col * size} y={c.row * size} width={size} height={size} fill={GREEN} fillOpacity={cellOpacity(c.share)}>
            <title>{`${c.count} made`}</title>
          </rect>
        ))}
      </g>
      <g fill="none" stroke={LINE} strokeWidth={2}>
        <rect x={1} y={1} width={COURT.width - 2} height={COURT.height - 2} />
        <rect x={toSvgX(-8)} y={0} width={16 * s} height={19 * s} />
        <circle cx={toSvgX(0)} cy={19 * s} r={6 * s} />
        <path d={`M${toSvgX(-22)} 0 V${cornerY} A${23.75 * s} ${23.75 * s} 0 0 0 ${toSvgX(22)} ${cornerY} V0`} />
        <path d={`M${toSvgX(-4)} ${hoopY} A${4 * s} ${4 * s} 0 0 0 ${toSvgX(4)} ${hoopY}`} />
        <line x1={toSvgX(-3)} x2={toSvgX(3)} y1={4 * s} y2={4 * s} />
        <circle cx={toSvgX(0)} cy={hoopY} r={0.75 * s} />
        <path d={`M${toSvgX(-6)} ${COURT.height} A${6 * s} ${6 * s} 0 0 1 ${toSvgX(6)} ${COURT.height}`} />
      </g>
    </svg>
  );
}

/** Pick seasons and one of our players to see where they score against the opponent. */
export default function PlayerMadeShotsPanel({ our, opp }: { our: string; opp: string }) {
  const [available, setAvailable] = useState<string[]>([]);
  const [seasons, setSeasons] = useState<string[]>([]);
  const [players, setPlayers] = useState<ShotPlayer[]>([]);
  const [player, setPlayer] = useState("");
  const [result, setResult] = useState<PlayerMadeShots | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    fetchPlayerShotSeasons(controller.signal)
      .then((r) => setAvailable(r.seasons))
      .catch((e) => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, []);

  // A new matchup or season choice clears the player and the court.
  useEffect(() => {
    setPlayer("");
    setResult(null);
    setPlayers([]);
    if (!our || !opp || our === opp || seasons.length === 0) return;
    const controller = new AbortController();
    setError("");
    fetchShotPlayers({ our, opp, seasons }, controller.signal)
      .then((r) => setPlayers(r.players))
      .catch((e) => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [our, opp, seasons]);

  useEffect(() => {
    setResult(null);
    if (!player) return;
    const controller = new AbortController();
    setLoading(true);
    setError("");
    fetchPlayerMadeShots({ our, opp, seasons, player }, controller.signal)
      .then(setResult)
      .catch((e) => { if (!controller.signal.aborted) setError(e.message); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [player, our, opp, seasons]);

  const toggle = (season: string) =>
    setSeasons((current) => (current.includes(season) ? current.filter((s) => s !== season) : [...current, season].sort()));

  const ready = Boolean(our && opp && our !== opp);
  return (
    <div className="card" style={{ marginTop: 12 }} aria-label="Player made shots">
      <div className="sectionTitle">
        <h2>Where a player scores against {opp || "the opponent"}</h2>
        <span className="badge">Made shots only</span>
      </div>
      {!ready ? (
        <p className="muted">Choose our team and the opponent above.</p>
      ) : (
        <>
          <fieldset style={{ border: 0, padding: 0, margin: 0, display: "flex", gap: 12, flexWrap: "wrap" }}>
            <legend className="muted" style={{ fontSize: 13, marginBottom: 6 }}>Seasons</legend>
            {available.map((s) => (
              <label key={s} style={{ fontSize: 13, display: "flex", gap: 6, alignItems: "center" }}>
                <input type="checkbox" checked={seasons.includes(s)} onChange={() => toggle(s)} /> {s}
              </label>
            ))}
          </fieldset>
          <label style={{ display: "grid", gap: 6, marginTop: 10, maxWidth: 320, fontSize: 13 }}>
            Player ({our})
            <select className="input" value={player} disabled={!players.length} onChange={(e) => setPlayer(e.target.value)}>
              <option value="" disabled>{seasons.length ? "Choose a player" : "Choose seasons first"}</option>
              {players.map((p) => (
                <option key={p.id} value={p.id}>{`${p.name} (${p.made} made)`}</option>
              ))}
            </select>
          </label>
          {loading ? <p className="muted">Loading made shots…</p> : null}
          {error ? <p className="error" role="alert">{error}</p> : null}
          {result ? (
            <div style={{ marginTop: 12 }}>
              <p role="status" style={{ margin: "0 0 8px" }}>{describeMadeShots(result)}</p>
              <CourtHeatmap result={result} />
              <p className="muted" style={{ fontSize: 12 }}>Deeper green = more made shots in that 2 ft square. Basket at the top.</p>
            </div>
          ) : null}
        </>
      )}
    </div>
  );
}
