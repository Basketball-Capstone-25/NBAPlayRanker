#!/usr/bin/env node
// Verify live report files using temporary sessions for the existing test accounts.
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const Module = require("node:module");
const ts = require("typescript");
const { createServerClient } = require("@supabase/ssr");

const repo = path.resolve(__dirname, "..");
const args = process.argv.slice(2);
const backendOnly = args.includes("--backend-only");
const includeTiming = args.includes("--timing");
const outputArg = args.indexOf("--out");
const output = outputArg >= 0 ? path.resolve(args[outputArg + 1]) : path.join(repo, "docs/assignment-2/verification/report-downloads");
const site = "https://nbaplayranker-seven.vercel.app";
const api = "https://nba-playranker-api-937082897804.northamerica-northeast1.run.app";
const summary = { started_at_utc: new Date().toISOString(), api, site: backendOnly ? null : site, complete: false, passed: false, checks: [], files: [], sessions: { opened: 0, closed: 0 } };
const sessions = {};
const formats = { csv: "text/csv", json: "application/json", pdf: "application/pdf" };
const sha256 = bytes => crypto.createHash("sha256").update(bytes).digest("hex");
function record(name, passed, details = {}) { summary.checks.push({ name, passed: Boolean(passed), ...details }); }
function save(name, bytes) {
  fs.writeFileSync(path.join(output, name), bytes);
  summary.files.push({ file: name, bytes: bytes.length, sha256: sha256(bytes) });
}
function loadPureTypescript(relative) {
  const filename = path.join(repo, relative);
  const source = fs.readFileSync(filename, "utf8");
  const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText;
  const compiledModule = new Module(filename, module);
  compiledModule.filename = filename;
  compiledModule.paths = Module._nodeModulePaths(path.dirname(filename));
  compiledModule._compile(compiled, filename);
  return compiledModule.exports;
}
function readEnvironment() {
  return Object.fromEntries(fs.readFileSync(path.join(repo, ".env.local"), "utf8").split(/\r?\n/)
    .filter(line => /^[A-Za-z_][A-Za-z_0-9]*=/.test(line)).map(line => {
      const index = line.indexOf("=");
      let value = line.slice(index + 1).trim();
      if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) value = value.slice(1, -1);
      return [line.slice(0, index), value];
    }));
}
async function signIn(role, environment, accounts) {
  const account = accounts.accounts ? accounts.accounts[role] : accounts[role];
  if (!account?.email || !account?.password) throw new Error("Missing test account.");
  const jar = new Map();
  const client = createServerClient(environment.NEXT_PUBLIC_SUPABASE_URL, environment.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY, {
    cookies: { getAll: () => [...jar].map(([name, value]) => ({ name, value })), setAll: values => values.forEach(({ name, value }) => jar.set(name, value)) },
  });
  const { data, error } = await client.auth.signInWithPassword({ email: account.email, password: account.password });
  if (error || !data.session) throw new Error("Test sign-in failed.");
  sessions[role] = { client, token: data.session.access_token, cookie: [...jar].map(([key, value]) => `${key}=${value}`).join("; ") };
  summary.sessions.opened += 1;
}
async function call(url, options = {}) {
  const response = await fetch(url, { ...options, redirect: "manual", signal: AbortSignal.timeout(180000) });
  const bytes = Buffer.from(await response.arrayBuffer());
  return { response, bytes, text: bytes.toString("utf8"), status: response.status,
    type: response.headers.get("content-type"), attachment: response.headers.get("content-disposition"), cache: response.headers.get("cache-control") };
}
async function source(name, route, role, query) {
  summary.stage = `read ${name}`;
  const url = new URL(route, api);
  Object.entries(query).forEach(([key, value]) => url.searchParams.set(key, String(value)));
  const result = await call(url, { headers: { Authorization: `Bearer ${sessions[role].token}` } });
  record(`source: ${name}`, result.status === 200 && result.type?.startsWith("application/json"), { status: result.status, route });
  if (result.status !== 200) throw new Error("Source request failed.");
  return JSON.parse(result.text);
}
function csvRows(text) {
  const records = []; let row = [], cell = "", quoted = false;
  for (let index = 0; index < text.length; index++) {
    const ch = text[index];
    if (ch === '"') {
      if (quoted && text[index + 1] === '"') { cell += '"'; index++; } else quoted = !quoted;
    } else if (ch === "," && !quoted) { row.push(cell); cell = ""; }
    else if ((ch === "\n" || ch === "\r") && !quoted) {
      if (ch === "\r" && text[index + 1] === "\n") index++;
      row.push(cell); records.push(row); row = []; cell = "";
    } else cell += ch;
  }
  if (cell || row.length) { row.push(cell); records.push(row); }
  return records;
}
function fileMatches(result, payload, format, expectedFilenameBase) {
  if (result.status !== 200 || !result.type?.startsWith(formats[format]) || !result.attachment?.startsWith("attachment;") || !result.attachment.endsWith(`.${format}"`) || !result.cache?.includes("no-store")) return false;
  if (result.attachment !== `attachment; filename="${expectedFilenameBase}.${format}"`) return false;
  if (format === "pdf") return result.text.startsWith("%PDF-") && result.text.trimEnd().endsWith("%%EOF");
  if (format === "json") {
    const body = JSON.parse(result.text);
    return body.contract_version === "1" && body.report === payload.report && JSON.stringify(body.filters) === JSON.stringify(payload.filters)
      && JSON.stringify(body.rows) === JSON.stringify(payload.rows) && body.metadata.export_source === "displayed_result_snapshot";
  }
  const rows = csvRows(result.text);
  return JSON.stringify(rows[0]) === JSON.stringify(payload.columns) && rows.length === payload.rows.length + 1 && payload.rows.every((expected, index) => payload.columns.every((column, col) => {
    const actual = rows[index + 1][col], value = expected[column];
    return value === null ? actual === "" : typeof value === "number" ? actual !== "" && Number(actual) === value : actual === String(value);
  }));
}
async function postBackend(body, role, options = {}) {
  return call(`${api}/export/report`, { method: "POST", headers: { "Content-Type": "application/json", ...(role ? { Authorization: `Bearer ${sessions[role].token}` } : {}), ...options.headers }, body: options.raw ?? JSON.stringify(body) });
}
async function postFrontend(body, role, form = false, options = {}) {
  const encoded = JSON.stringify(body);
  return call(`${site}/api/exports/report`, { method: "POST", headers: {
    Origin: site, "Content-Type": form ? "application/x-www-form-urlencoded" : "application/json",
    ...(role ? { Cookie: sessions[role].cookie } : {}), ...options.headers,
  }, body: options.raw ?? (form ? new URLSearchParams({ request: encoded }) : encoded) });
}
function denied(name, result, status) {
  record(name, result.status === status && !result.attachment && result.type?.startsWith("application/json"), { status: result.status, expected_status: status, attachment: Boolean(result.attachment) });
}

async function measureTopk() {
  summary.stage = "measure Top-K response times";
  const timing = {
    thresholds_recorded_at_utc: new Date().toISOString(),
    proposed_thresholds: { warm_p95_ms: 5000, concurrency_2_p95_ms: 10000, startup_ms: 45000, agreed: false },
    method: "Client elapsed time through complete response body. First request has unknown warm state. Nine subsequent sequential requests, then six requests in three pairs with concurrency two. Nearest-rank p95. No service reset or forced cold start.",
    route: "/metrics/topk-uplift", filters: { season: "2024-25", our: "TOR", opp: "BOS", k: 5, w_off: 0.7 },
    measurements: [],
  };
  async function measured(group, index) {
    const started_at_utc = new Date().toISOString(), started = performance.now();
    let status = null, valid = false;
    try {
      const url = new URL(timing.route, api);
      Object.entries(timing.filters).forEach(([key, value]) => url.searchParams.set(key, String(value)));
      const result = await call(url, { headers: { Authorization: `Bearer ${sessions.analyst.token}` } });
      status = result.status;
      const body = JSON.parse(result.text);
      valid = status === 200 && Array.isArray(body.rankings) && body.rankings.length === timing.filters.k;
    } catch { /* retain status and elapsed time, never response credentials or exception data */ }
    const measurement = { group, index, started_at_utc, finished_at_utc: new Date().toISOString(), elapsed_ms: Math.round((performance.now() - started) * 100) / 100, status, valid };
    timing.measurements.push(measurement);
  }
  await measured("initial_warm_state_unknown", 1);
  for (let index = 1; index <= 9; index++) await measured("warm_sequential", index);
  for (let index = 1; index <= 6; index += 2) await Promise.all([measured("concurrency_2", index), measured("concurrency_2", index + 1)]);
  const summarize = group => {
    const values = timing.measurements.filter(row => row.group === group);
    const times = values.map(row => row.elapsed_ms).sort((a, b) => a - b);
    return { samples: times.length, successful_responses: values.filter(row => row.valid).length, p95_ms: times[Math.ceil(times.length * 0.95) - 1], min_ms: times[0], max_ms: times.at(-1) };
  };
  timing.warm_sequential = summarize("warm_sequential");
  timing.concurrency_2 = summarize("concurrency_2");
  timing.proposal_comparison = {
    warm_meets_proposal: timing.warm_sequential.successful_responses === 9 && timing.warm_sequential.p95_ms <= timing.proposed_thresholds.warm_p95_ms,
    concurrency_2_meets_proposal: timing.concurrency_2.successful_responses === 6 && timing.concurrency_2.p95_ms <= timing.proposed_thresholds.concurrency_2_p95_ms,
    startup: "Not assessed: no confirmed cold start. The proposed target still needs agreement.",
  };
  timing.finished_at_utc = new Date().toISOString();
  save("topk-timing.json", Buffer.from(JSON.stringify(timing, null, 2) + "\n"));
  record("Top-K timing campaign completed with valid responses", timing.measurements.length === 16 && timing.measurements.every(row => row.valid), { requests: timing.measurements.length, started_at_utc: timing.thresholds_recorded_at_utc, finished_at_utc: timing.finished_at_utc });
}

(async () => {
  fs.mkdirSync(output, { recursive: true });
  try {
    summary.stage = "open test sessions";
    const environment = readEnvironment();
    const accounts = JSON.parse(fs.readFileSync(path.join(process.env.HOME, ".config/nbaplayranker/test-accounts-20261003.json"), "utf8"));
    await signIn("coach", environment, accounts);
    await signIn("analyst", environment, accounts);
    const builders = loadPureTypescript("app/infrastructure/report-export.ts");
    const { computeOverlay } = loadPureTypescript("app/domain/context-analysis.ts");
    if (includeTiming) await measureTopk();
    const matchup = { season: "2024-25", our: "TOR", opp: "BOS", k: 5, w_off: 0.7, w_def: 0.3 };
    const context = { season: matchup.season, our: matchup.our, opp: matchup.opp, k: matchup.k, margin: -2, period: 4, time_remaining: 120 };
    const baseline = await source("matchup rankings", "/rank-plays/baseline", "coach", matchup);
    const contextual = await source("context rankings", "/rank-plays/context-ml", "coach", { ...context, w_off: 0.7 });
    const metrics = await source("holdout metrics", "/metrics/baseline-vs-ml", "analyst", { n_splits: 5 });
    const baselineRows = baseline.rankings.map(row => ({ playType: row.PLAY_TYPE, pppPred: Number(row.PPP_PRED), pppOff: Number(row.PPP_OFF_SHRUNK), pppDef: Number(row.PPP_DEF_SHRUNK), pppGap: Number(row.PPP_GAP) }));
    const contextRows = contextual.rankings.map(row => ({ playType: row.PLAY_TYPE, finalPPP: Number(row.PPP_CONTEXT), mlPPP: Number(row.PPP_ML_BLEND), baselinePPP: Number(row.PPP_BASELINE), deltaPPP: Number(row.DELTA_VS_BASELINE), contextLabel: row.CONTEXT_LABEL || "", rationale: row.RATIONALE || "", raw: row }));
    const meanMlPPP = contextRows.reduce((sum, row) => sum + row.mlPPP, 0) / contextRows.length;
    const displayRows = contextRows.map(row => {
      const overlay = computeOverlay(row, meanMlPPP, context.period, context.time_remaining, context.margin, 1);
      const finalPPP_display = row.finalPPP + overlay.overlayPPP;
      return { ...row, overlay, finalPPP_display, deltaPPP_display: finalPPP_display - row.baselinePPP };
    }).sort((a, b) => b.finalPPP_display - a.finalPPP_display).slice(0, context.k);
    const reports = [
      { role: "coach", payload: builders.baselineReport(matchup, baselineRows), filename: "baseline-rankings_2024-25_TOR_BOS_5_0.7_0.3" },
      { role: "coach", payload: builders.contextReport({ ...context, overlay: true, overlay_strength: 1 }, displayRows), filename: "context-rankings_2024-25_TOR_BOS_5_-2_4_120_True_1" },
      { role: "analyst", payload: builders.metricsReport(metrics), filename: "model-metrics_5" },
    ];
    for (const { role, payload, filename } of reports) {
      save(`${payload.report}-source.json`, Buffer.from(JSON.stringify(payload, null, 2) + "\n"));
      for (const format of Object.keys(formats)) {
        summary.stage = `verify ${payload.report} ${format}`;
        const body = { format, payload };
        const result = await postBackend(body, role);
        const valid = fileMatches(result, payload, format, filename);
        record(`backend ${payload.report} ${format}`, valid, { status: result.status, bytes: result.bytes.length, rows: payload.rows.length, content_type: result.type, content_disposition: result.attachment });
        if (valid) save(`backend-${payload.report}.${format}`, result.bytes);
        if (!backendOnly) {
          const prepared = await postFrontend(body, role);
          record(`preflight ${payload.report} ${format}`, prepared.status === 200 && !prepared.attachment && JSON.parse(prepared.text).ready === true, { status: prepared.status });
          const downloaded = await postFrontend(body, role, true);
          const nativeValid = fileMatches(downloaded, payload, format, filename);
          record(`native form ${payload.report} ${format}`, nativeValid, { status: downloaded.status, bytes: downloaded.bytes.length, rows: payload.rows.length, content_type: downloaded.type, content_disposition: downloaded.attachment });
          if (nativeValid) save(`frontend-${payload.report}.${format}`, downloaded.bytes);
        }
      }
      const wrongRole = role === "coach" ? "analyst" : "coach";
      denied(`backend ${payload.report} wrong role`, await postBackend({ format: "csv", payload }, wrongRole), 403);
      if (!backendOnly) denied(`frontend ${payload.report} wrong role`, await postFrontend({ format: "csv", payload }, wrongRole, true), 403);
    }
    const good = { format: "csv", payload: reports[0].payload };
    summary.stage = "verify rejected requests";
    const malformed = structuredClone(good);
    malformed.payload.rows[0][malformed.payload.columns[0]] = "\ud800";
    denied("backend anonymous session", await postBackend(good, null), 401);
    denied("backend unsupported format", await postBackend({ ...good, format: "xlsx" }, "coach"), 400);
    denied("backend malformed Unicode", await postBackend(malformed, "coach"), 400);
    denied("backend oversize body", await postBackend(null, "coach", { raw: " ".repeat(256 * 1024 + 1) }), 413);
    if (!backendOnly) {
      denied("frontend anonymous session", await postFrontend(good, null, true), 401);
      denied("frontend unsupported format", await postFrontend({ ...good, format: "xlsx" }, "coach", true), 400);
      denied("frontend foreign Origin", await postFrontend(good, "coach", true, { headers: { Origin: "https://example.org" } }), 403);
      denied("frontend null Origin", await postFrontend(good, "coach", true, { headers: { Origin: "null" } }), 403);
      denied("frontend malformed Unicode is not an attachment", await postFrontend(malformed, "coach", true), 400);
      denied("frontend oversize body", await postFrontend(null, "coach", false, { raw: " ".repeat(256 * 1024 + 1) }), 413);
    }
    summary.scope = "API-derived report fixtures. Context applies the application's overlay function. Native form requests are HTTP checks; separate browser evidence verifies the visible controls and saved downloads.";
    summary.complete = true;
    summary.stage = "complete";
  } catch {
    summary.error = "Verification stopped before all checks completed. Sensitive exception text was not retained.";
  } finally {
    for (const session of Object.values(sessions)) {
      try {
        const { error } = await session.client.auth.signOut({ scope: "local" });
        if (!error) summary.sessions.closed += 1;
      } catch { /* recorded by the session count below */ }
    }
    record("verification sessions signed out locally", summary.sessions.opened === summary.sessions.closed);
    summary.finished_at_utc = new Date().toISOString();
    summary.passed = summary.complete && summary.checks.every(check => check.passed);
    fs.writeFileSync(path.join(output, "verification.json"), JSON.stringify(summary, null, 2) + "\n");
    console.log(JSON.stringify({ complete: summary.complete, passed: summary.passed, checks: summary.checks.length, failed: summary.checks.filter(check => !check.passed).map(check => check.name), files: summary.files.length, sessions: summary.sessions }));
    process.exitCode = summary.passed ? 0 : 1;
  }
})();
