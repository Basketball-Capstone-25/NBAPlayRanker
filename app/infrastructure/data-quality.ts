export type QualityStatus = "pass" | "fail" | "not_run";

export type QualityCheck = {
  name: string;
  status: QualityStatus;
  details: Record<string, unknown>;
};

export type DataQualityReport = {
  status: QualityStatus;
  safe_for_use: boolean;
  checks: QualityCheck[];
  generated_at: string;
  dataset: {
    name: string;
    scope: string;
    rows: number;
    seasons: string[];
    teams: string[];
    version: string;
  };
};

const statuses = new Set(["pass", "fail", "not_run"]);
const requiredChecks = ["schema", "completeness", "duplicates", "coverage"];
const object = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);
const strings = (v: unknown): v is string[] =>
  Array.isArray(v) && v.every(x => typeof x === "string");
const count = (v: unknown): v is number => typeof v === "number" && Number.isSafeInteger(v) && v >= 0;

function validDetails(check: QualityCheck): boolean {
  const d = check.details;
  if (check.status === "not_run") return typeof d.reason === "string" && d.reason.length > 0;
  let failed: boolean;
  switch (check.name) {
    case "schema":
      if (!strings(d.required_columns) || !d.required_columns.length || !strings(d.missing_columns)) return false;
      failed = d.missing_columns.length > 0;
      break;
    case "completeness":
      if (!count(d.rows) || !object(d.null_counts) || !Object.values(d.null_counts).every(count)) return false;
      failed = Object.values(d.null_counts).some(n => Number(n) > 0);
      break;
    case "duplicates":
      if (!strings(d.key_fields) || !d.key_fields.length || !count(d.duplicate_rows)) return false;
      failed = d.duplicate_rows > 0;
      break;
    case "coverage":
      if (!count(d.expected_seasons) || !d.expected_seasons || !count(d.expected_teams) || !d.expected_teams || !strings(d.missing_season_team_pairs)) return false;
      failed = d.missing_season_team_pairs.length > 0;
      break;
    default: return true;
  }
  return (check.status === "fail") === failed;
}

/** Reject incomplete or contradictory reports instead of showing a false pass. */
export function parseDataQualityReport(value: unknown): DataQualityReport {
  const invalid = () => { throw new Error("The data-check report is incomplete. Please try again."); };
  if (!object(value) || !object(value.dataset) || !Array.isArray(value.checks)) return invalid();
  const checks = value.checks;
  if (!checks.every(c => object(c) && typeof c.name === "string" && statuses.has(String(c.status)) && object(c.details))) return invalid();
  if (!checks.every(validDetails)) return invalid();
  const names = new Set(checks.map(c => c.name));
  if (names.size !== checks.length || !requiredChecks.every(name => names.has(name))) return invalid();
  const expected = checks.some(c => c.status === "fail") ? "fail"
    : checks.some(c => c.status === "not_run") ? "not_run" : "pass";
  if (value.status !== expected || value.safe_for_use !== (expected === "pass")) return invalid();
  const dataset = value.dataset;
  if (typeof value.generated_at !== "string" || !Number.isFinite(Date.parse(value.generated_at))
    || typeof dataset.name !== "string" || !dataset.name.trim()
    || typeof dataset.scope !== "string" || !dataset.scope.trim()
    || typeof dataset.rows !== "number" || !Number.isSafeInteger(dataset.rows) || dataset.rows < 0
    || !strings(dataset.seasons) || !strings(dataset.teams)
    || typeof dataset.version !== "string" || !/^frame-sha256:[a-f0-9]{64}$/.test(dataset.version)
    || (expected === "pass" && (dataset.rows === 0 || !dataset.seasons.length || !dataset.teams.length))) return invalid();
  const completeness = checks.find(c => c.name === "completeness");
  if (completeness.status !== "not_run" && completeness.details.rows !== dataset.rows) return invalid();
  const coverage = checks.find(c => c.name === "coverage");
  if (coverage.status === "pass" && (dataset.seasons.length < coverage.details.expected_seasons || dataset.teams.length < coverage.details.expected_teams)) return invalid();
  return value as DataQualityReport;
}
