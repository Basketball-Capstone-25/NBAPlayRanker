import React from "react";

// UI Layer — marks the #1 ranked play, or each play tied for #1 (SCRUM-496 / UC24).
/** Badge text for the top pick, or "Tied #1" when several plays share first place. */
export default function TopPickBadge({ tied }: { tied: boolean }) {
  return (
    <span className="topPickBadge" data-testid="top-pick-badge">
      {tied ? "Tied #1" : "Top pick"}
    </span>
  );
}
