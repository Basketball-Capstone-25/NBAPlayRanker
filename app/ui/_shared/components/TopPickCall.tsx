import React from "react";

const PLAY_NAMES: Record<string, string> = {
  PRBallHandler: "Pick-and-Roll (Ball Handler)",
  PRRollMan: "Pick-and-Roll (Roll Man)",
  Spotup: "Spot-Up",
  Postup: "Post-Up",
  OffScreen: "Off-Screen",
  OffRebound: "Offensive Rebound",
};

export default function TopPickCall({
  rows,
  current = true,
  className,
}: {
  rows: ReadonlyArray<{ playType: string }>;
  current?: boolean;
  className?: string;
}) {
  const playType = rows[0]?.playType?.trim();
  if (!current || !playType) return null;
  return <strong className={className} data-testid="top-pick-action">Run {PLAY_NAMES[playType] ?? playType}</strong>;
}
