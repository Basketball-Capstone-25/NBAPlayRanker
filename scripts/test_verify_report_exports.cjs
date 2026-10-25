const { test } = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const fs = require("node:fs");
const { spawnSync } = require("node:child_process");
const { fileMatches } = require("./verify_report_exports.cjs");

const versions = { synergy: `sha256:${"a".repeat(64)}`, ml_predictions: `sha256:${"b".repeat(64)}` };
const payload = { report: "context_rankings", filters: { margin: -2 }, columns: ["Play type", "PPP"],
  rows: [{ "Play type": "Spotup", PPP: 1.23 }], metadata: { dataset_versions: versions } };
function response(format, text) {
  return { status: 200, type: format === "json" ? "application/json" : "text/csv", text, bytes: Buffer.from(text),
    attachment: `attachment; filename="report.${format}"`, cache: "private, no-store" };
}

test("CSV verifier checks the version cell along with displayed values", () => {
  const csv = value => 'Play type,PPP,Dataset versions\nSpotup,1.23,"' + JSON.stringify(value).replaceAll('"', '""') + '"\n';
  assert.equal(fileMatches(response("csv", csv(versions)), payload, "csv", "report"), true);
  assert.equal(fileMatches(response("csv", csv({ synergy: `sha256:${"c".repeat(64)}` })), payload, "csv", "report"), false);
});

test("JSON verifier rejects the current source hash substituted for an older displayed snapshot", () => {
  const body = { contract_version: "1", ...payload, metadata: { ...payload.metadata, export_source: "displayed_result_snapshot" } };
  assert.equal(fileMatches(response("json", JSON.stringify(body)), payload, "json", "report"), true);
  body.metadata.dataset_versions = { synergy: `sha256:${"c".repeat(64)}` };
  assert.equal(fileMatches(response("json", JSON.stringify(body)), payload, "json", "report"), false);
});

test("PDF verifier reads the captured hashes from the rendered document", () => {
  const root = path.resolve(__dirname, "..");
  const localPython = path.join(root, "backend/.venv/bin/python");
  const python = process.env.NBAPR_PYTHON || (fs.existsSync(localPython) ? localPython : "python3");
  const built = spawnSync(python, ["-c", "import json,sys; from application.api_coordination.export_contract import ExportPayload; from infrastructure.visualization_and_export.table_pdf import render_table_pdf; p=json.load(sys.stdin); p['metadata']['export_source']='displayed_result_snapshot'; sys.stdout.buffer.write(render_table_pdf(ExportPayload(**p)))"],
    { input: JSON.stringify(payload), env: { ...process.env, PYTHONPATH: path.join(root, "backend") }, timeout: 15000 });
  assert.equal(built.status, 0);
  const result = { ...response("pdf", built.stdout.toString("utf8")), type: "application/pdf", bytes: built.stdout };
  assert.equal(fileMatches(result, payload, "pdf", "report"), true);
  const wrong = structuredClone(payload);
  wrong.metadata.dataset_versions.synergy = `sha256:${"c".repeat(64)}`;
  assert.equal(fileMatches(result, wrong, "pdf", "report"), false);
});
