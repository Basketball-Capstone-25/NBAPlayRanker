import { describe, expect, it } from "vitest";
import { gameplanSelectionKey, selectGameplanResult, type GameplanSelection } from "../app/ui/gameplan/_components/gameplan-result";
import { resolveGameSituation } from "../app/domain/gameplan-context";

const selection: GameplanSelection = {
  season: "2024-25", our: "TOR", opp: "BOS", k: 5, wOff: .65, wDef: .35,
  situation: resolveGameSituation({}, { period: 4, time_remaining: 180, margin: -2 }), prompt: "",
};
const ready = { loading: false, parsing: false, shotLoading: false, shotError: null, hasShots: true };

describe("Gameplan results belong to a completed selection", () => {
  it("allows rankings, diagrams and PDF inputs only after the build completes", () => {
    expect(selectGameplanResult(null, selection, ready)).toMatchObject({ current: false, inputs: null, shotsCurrent: false });
    const result = selectGameplanResult(selection, selection, ready);
    expect(result).toMatchObject({ current: true, changed: false, shotsCurrent: true });
    expect(result.inputs).toBe(selection);
    expect(result.inputs).toMatchObject({ season: "2024-25", our: "TOR", opp: "BOS", wOff: .65 });
  });

  it.each([
    { season: "2023-24" }, { our: "LAL" }, { opp: "DEN" }, { k: 7 }, { wOff: .4, wDef: .6 },
    { prompt: "Need a three" }, { situation: null },
    ...["period", "time_remaining", "margin"].map(field => ({ situation: { ...selection.situation!, [field]: 3 } })),
    { situation: { ...selection.situation!, coaching: { need3: true } } },
    { situation: { ...selection.situation!, coaching: { vs_switching: true } } },
    { situation: { ...selection.situation!, end_of_quarter: true } },
  ])("hides results and export inputs after editing %j", change => {
    expect(selectGameplanResult(selection, { ...selection, ...change }, ready))
      .toMatchObject({ changed: true, current: false, inputs: null, shotsCurrent: false });
  });

  it.each([{ loading: true }, { parsing: true }])("hides old rankings and PDFs during %j", change => {
    expect(selectGameplanResult(selection, selection, { ...ready, ...change }))
      .toMatchObject({ current: false, inputs: null, shotsCurrent: false });
  });

  it.each([{ shotLoading: true }, { shotError: "Unavailable" }, { hasShots: false }])("keeps rankings but prevents a shot PDF with %j", change => {
    expect(selectGameplanResult(selection, selection, { ...ready, ...change }))
      .toMatchObject({ current: true, inputs: selection, shotsCurrent: false });
  });

  it("does not relabel a late completed build with newer draft inputs", () => {
    const edited = { ...selection, our: "LAL" };
    expect(selectGameplanResult(selection, edited, ready).inputs).toBeNull();
    expect(selectGameplanResult(edited, edited, ready).inputs).toBe(edited);
  });

  it("recognizes the same parsed situation after the UI fills the controls", () => {
    const parsed = { ...selection, situation: resolveGameSituation({ need: "need3", defense_style: "switch" }, selection.situation!) };
    const filled = { ...parsed, situation: { ...parsed.situation, coaching: { need3: true, need: "need3" as const, vs_switching: true, defense_style: "switch" as const, after_timeout: false } } };
    expect(gameplanSelectionKey(parsed)).toBe(gameplanSelectionKey(filled));
    expect(selectGameplanResult(parsed, filled, ready).current).toBe(true);
  });
});
