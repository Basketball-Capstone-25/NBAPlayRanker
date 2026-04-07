// app/domain/coaching-knowledge.ts
//
// Domain Layer — Coaching Knowledge subsystem
// Coaching cues and defensive counter-plan rules (coachingCuesFor, counterPlanFor).
// Encodes basketball tactical knowledge.

export function gradeFromRank(rank: number) {
  if (rank === 0) return "A";
  if (rank === 1) return "B";
  return "C";
}

export function coachingCuesFor(playType: string): string[] {
  const p = playType.toLowerCase();

  if (p.includes("pr") || p.includes("pick") || p.includes("roll")) {
    return [
      "Screen: be still and hit the defender's path.",
      "Ball: get shoulder-to-hip and force 2 defenders to react.",
      "Roll: sprint to the rim with hands ready.",
    ];
  }
  if (p.includes("spot")) {
    return [
      "Be shot-ready on the catch (feet set, hands ready).",
      "Attack closeouts with 1–2 dribbles max.",
      "If help comes, make the simple kick-out.",
    ];
  }
  if (p.includes("transition")) {
    return [
      "Run wide lanes: first big to rim, second to trail.",
      "Get an early paint touch → then spray to shooters.",
      "If nothing early, flow right into a ball-screen.",
    ];
  }
  if (p.includes("isolation") || p.includes("iso")) {
    return [
      "Clear a side and hold corners (spacing is the play).",
      "Get to your first advantage move quickly (no dancing).",
      "If help comes, kick early and re-attack.",
    ];
  }
  if (p.includes("handoff") || p.includes("dh")) {
    return [
      "Sell the handoff and turn the corner tight.",
      "Big: flip and screen if the defender recovers.",
      "Weakside: stay spaced, ready for the skip pass.",
    ];
  }
  if (p.includes("post")) {
    return [
      "Seal first: high-to-low, show a clear target hand.",
      "Perimeter: cut hard when your defender turns their head.",
      "If doubled, pass out early and relocate.",
    ];
  }

  return [
    "Keep spacing (corners & slots) to open driving lanes.",
    "First read fast: rim → kick → swing.",
    "If the defense loads up, move it with one extra pass.",
  ];
}

export type CounterPlanEntry = {
  title: string;
  trigger: string;
  cues: string[];
  outcome: string;
};

export function counterPlanFor(playType: string): CounterPlanEntry[] {
  const base = playType.toLowerCase();
  const isPnR = base.includes("pick") || base.includes("roll") || base.includes("pr");

  return [
    {
      title: "If they switch…",
      trigger: "When their big ends up on our ball handler (or they switch everything).",
      cues: isPnR
        ? ["Re-screen quickly (flip it).", "Hit the slip early.", "Attack the mismatch with pace."]
        : ["Get into a quick re-screen.", "Cut behind help.", "Throw the skip if they load up."],
      outcome: "Goal: create a mismatch or force help.",
    },
    {
      title: "If they go under…",
      trigger: "When the on-ball defender ducks under the screen.",
      cues: isPnR
        ? ["Re-screen higher (pull-up space).", "Use a handoff into flow.", "Punish with the catch-and-shoot."]
        : ["Shorten the route for a quick shot.", "Sprint into a second action.", "Keep the ball moving."],
      outcome: "Goal: get a clean rhythm shot.",
    },
    {
      title: "If they trap/hedge…",
      trigger: "When they send two to the ball to take away the first option.",
      cues: isPnR
        ? ["Hit the short roll.", "Corners stay lifted for the skip.", "One more pass = open shot."]
        : ["Flash middle as an outlet.", "Quick swing to the weak side.", "Attack the closeout."],
      outcome: "Goal: beat the trap with spacing + quick pass.",
    },
  ];
}
