"use client";

// UI Layer — Context ML ranked plays in one large, readable list (SCRUM-795).
// Same look as the retired Quick-Call screen: big rank, a direct call on #1,
// the play, its best shot zone, a one-line reason and expected PPP.

import React from "react";
import TopPickCall from "../../_shared/components/TopPickCall";
import TopPickBadge from "../../_shared/components/TopPickBadge";
import styles from "./RankedPlayList.module.css";

export type RankedPlay = {
  playType: string;
  ppp: number;
  shotZone?: string;
  reason: string;
};

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
            </div>
          </li>
        );
      })}
    </ol>
  );
}
