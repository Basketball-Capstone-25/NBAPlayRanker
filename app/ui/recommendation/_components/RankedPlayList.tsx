"use client";

// UI Layer — Context ML ranked plays in one large, readable list (SCRUM-795).
// Same look as the retired Quick-Call screen: big rank, a direct call on #1,
// the play, its best shot zone, a one-line reason and expected PPP.
// SCRUM-974: plays are ranked by edge over the league average, shown under the PPP.

import React from "react";
import TopPickCall from "../../_shared/components/TopPickCall";
import TopPickBadge from "../../_shared/components/TopPickBadge";
import styles from "./RankedPlayList.module.css";

export type RankedPlay = {
  playType: string;
  ppp: number;
  /** Expected PPP minus the league average for the play type. */
  edge?: number;
  /** League average PPP for the play type, with the same offence/defence weights. */
  leagueRef?: number;
  shotZone?: string;
  reason: string;
};

/** One string, e.g. "edge +0.05 vs league 1.13", so it renders as a single text node. */
function edgeLabel(play: RankedPlay): string {
  const edge = Number(play.edge);
  const league = Number(play.leagueRef);
  const signed = `${edge >= 0 ? "+" : ""}${edge.toFixed(2)}`;
  return Number.isFinite(league) ? `edge ${signed} vs league ${league.toFixed(2)}` : `edge ${signed}`;
}

/** Large ranked list: rank, call on #1, play name, best shot zone, reason and expected PPP. */
export default function RankedPlayList({
  plays,
  current,
  topPicks,
  tied = false,
}: {
  plays: readonly RankedPlay[];
  /** False while the situation has changed and new results are loading. */
  current: boolean;
  /** Indexes of the #1 play, or of every play tied for #1. */
  topPicks: ReadonlySet<number>;
  tied?: boolean;
}) {
  if (plays.length === 0) return null;
  return (
    <ol className={styles.plays} aria-label="Ranked plays">
      {plays.map((play, i) => {
        const top = topPicks.has(i);
        return (
          <li className={top ? `${styles.play} topPickRow` : styles.play} key={`${play.playType}-${i}`}>
            <span className={styles.rank}>{i + 1}</span>
            <div>
              {i === 0 ? (
                <TopPickCall className={styles.call} rows={plays.map((p) => ({ playType: p.playType }))} current={current} />
              ) : null}
              <div className={styles.playName}>
                {play.playType}
                {top ? <TopPickBadge tied={tied} /> : null}
              </div>
              {play.shotZone ? <div className={styles.zone}>Shot zone: {play.shotZone}</div> : null}
              <div className={styles.reason}>{play.reason}</div>
            </div>
            <div className={styles.ppp}>
              {Number.isFinite(play.ppp) ? play.ppp.toFixed(2) : "—"}
              <span className={styles.pppLabel}>expected PPP</span>
              {Number.isFinite(play.edge) ? <span className={styles.edge}>{edgeLabel(play)}</span> : null}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
