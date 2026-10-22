#!/usr/bin/env node
// Isolated browser fixture for the real ReportExport component. No production calls or credentials.
// Run: node scripts/export-retry-harness.cjs --out <evidence-directory>
const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const esbuild = require("esbuild");

const repo = path.resolve(__dirname, "..");
const args = process.argv.slice(2);
const outIndex = args.indexOf("--out");
if (outIndex < 0 || !args[outIndex + 1]) throw new Error("Provide --out for the isolated fixture evidence.");
const output = path.resolve(args[outIndex + 1]);
fs.mkdirSync(output, { recursive: true });
const sha256 = value => crypto.createHash("sha256").update(value).digest("hex");
const sources = ["app/ui/_shared/components/ReportExport.tsx", "app/infrastructure/report-export.ts"];
const fixture = {
  report: "baseline_rankings", filters: { season: "fixture", our: "TOR", opp: "BOS", k: 2 },
  columns: ["Play type", "Predicted PPP"],
  rows: [{ "Play type": "Fixture Spotup", "Predicted PPP": 1.08 }, { "Play type": "Fixture Transition", "Predicted PPP": 1.12 }],
  metadata: { title: "Isolated retry fixture", source: "Synthetic test data; not a production report." },
};
const csv = Buffer.from('Play type,Predicted PPP\r\nFixture Spotup,1.08\r\nFixture Transition,1.12\r\n');
const evidence = {
  started_at_utc: new Date().toISOString(), scope: "Isolated localhost fixture using the actual ReportExport component and download helper.",
  production_requests: 0, credentials_used: false,
  sources: sources.map(file => ({ file, sha256: sha256(fs.readFileSync(path.join(repo, file))) })),
  fixture, file: { filename: "isolated-retry-fixture.csv", bytes: csv.length, sha256: sha256(csv) },
  events: [],
};
function log(event) {
  if (evidence.events.length < 100) evidence.events.push({ at_utc: new Date().toISOString(), ...event });
  fs.writeFileSync(path.join(output, "events.json"), JSON.stringify(evidence, null, 2) + "\n");
}
const entry = `
import React from "react";
import {createRoot} from "react-dom/client";
import ReportExport from "./app/ui/_shared/components/ReportExport";
const fixture = ${JSON.stringify(fixture)};
function App() {
  return <main>
    <p className="fixture-label">ISOLATED TEST FIXTURE — BST-177</p>
    <h1>Export error and retry</h1>
    <p>This page uses the app's actual Export control with synthetic data. It makes no production requests.</p>
    <p>The first Export request returns a temporary error. Retry export should finish preparation and request a CSV download.</p>
    <table><thead><tr>{fixture.columns.map(c=><th key={c}>{c}</th>)}</tr></thead>
      <tbody>{fixture.rows.map((r,i)=><tr key={i}>{fixture.columns.map(c=><td key={c}>{r[c]}</td>)}</tr>)}</tbody></table>
    <section id="actual-export"><ReportExport payload={fixture} description="Sample fixture — not a production report." /></section>
  </main>;
}
createRoot(document.getElementById("root")).render(<App />);
let previous = "";
new MutationObserver(() => {
  const root = document.getElementById("actual-export");
  if (!root) return;
  const button = root.querySelector("button"), select = root.querySelector("select");
  if (!button || !select) return;
  const state = {button:button.textContent, disabled:button.disabled, format_disabled:select.disabled,
    status:root.querySelector('[role="status"],[role="alert"]')?.textContent || ""};
  const current = JSON.stringify(state);
  if (current !== previous) {
    previous = current;
    fetch("/observations", {method:"POST", headers:{"Content-Type":"application/json"}, body:current}).catch(()=>{});
  }
}).observe(document.getElementById("root"), {subtree:true, childList:true, attributes:true, characterData:true});
`;
const bundle = esbuild.buildSync({
  stdin: { contents: entry, loader: "tsx", resolveDir: repo }, bundle: true, write: false,
  platform: "browser", format: "iife", jsx: "automatic", define: { "process.env.NODE_ENV": '"production"' },
}).outputFiles[0].contents;
const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Isolated export retry fixture</title><style>
body{margin:0;background:#eef2f6;font:16px system-ui;color:#172033}main{max-width:760px;margin:48px auto;padding:32px;background:white;border-radius:16px}
.fixture-label{color:#8a4300;font-weight:700;font-size:13px}h1{font-size:28px}p{line-height:1.5}table{border-collapse:collapse;margin:24px 0;width:100%}th,td{padding:10px;text-align:left;border-bottom:1px solid #cbd5e1}section{padding:20px;background:#f8fafc;border-radius:12px}
</style></head><body><div id="root"></div><script src="/bundle.js"></script></body></html>`;
let origin, preflights = 0;
const server = http.createServer(async (request, response) => {
  response.setHeader("Cache-Control", "no-store");
  response.setHeader("X-Content-Type-Options", "nosniff");
  if (request.headers.host !== new URL(origin).host) { response.writeHead(403).end(); return; }
  const send = (status, value) => { response.writeHead(status, { "Content-Type": "application/json" }); response.end(JSON.stringify(value)); };
  if (request.method === "GET" && request.url === "/") {
    response.writeHead(200, { "Content-Type": "text/html; charset=utf-8", "Content-Security-Policy": "default-src 'self'; style-src 'self' 'unsafe-inline'; form-action 'self'; base-uri 'none'" }); response.end(html); return;
  }
  if (request.method === "GET" && request.url === "/bundle.js") { response.writeHead(200, { "Content-Type": "text/javascript" }); response.end(bundle); return; }
  if (request.method === "GET" && request.url === "/state") { send(200, evidence); return; }
  if (request.method !== "POST" || !["/api/exports/report", "/observations"].includes(request.url)) { send(404, { error: "Unknown fixture route." }); return; }
  if (request.headers.origin !== origin) { send(403, { error: "Local fixture origin required." }); return; }
  let body = "";
  try {
    for await (const chunk of request) { body += chunk; if (Buffer.byteLength(body) > 8192) { send(413, { error: "Fixture request is too large." }); return; } }
    if (request.url === "/observations") {
      const state = JSON.parse(body);
      log({ kind: "component_state", button: String(state.button).slice(0, 60), disabled: state.disabled === true,
        format_disabled: state.format_disabled === true, status: String(state.status).slice(0, 240) });
      send(200, { recorded: true }); return;
    }
    const isForm = request.headers["content-type"]?.startsWith("application/x-www-form-urlencoded");
    const value = JSON.parse(isForm ? new URLSearchParams(body).get("request") : body);
    if (value.format !== "csv" || JSON.stringify(value.payload) !== JSON.stringify(fixture)) { send(400, { error: "This fixture supports its sample CSV only." }); return; }
    if (!isForm) {
      preflights += 1;
      const status = preflights === 1 ? 503 : 200;
      log({ kind: "preflight_received", attempt: preflights, planned_status: status });
      await new Promise(resolve => setTimeout(resolve, 1800));
      log({ kind: "preflight_response", attempt: preflights, status });
      send(status, status === 503 ? { error: "The report service is temporarily unavailable. Please try again." } : { ready: true });
    } else {
      log({ kind: "native_form_download", status: 200, preflights });
      response.writeHead(200, { "Content-Type": "text/csv; charset=utf-8", "Content-Disposition": 'attachment; filename="isolated-retry-fixture.csv"' });
      response.end(csv);
    }
  } catch { send(400, { error: "Invalid fixture request." }); }
});
server.listen(0, "127.0.0.1", () => {
  origin = `http://127.0.0.1:${server.address().port}`;
  log({ kind: "fixture_started", origin });
  console.log(JSON.stringify({ url: origin, scope: evidence.scope }));
});
function stop() { log({ kind: "fixture_stopped" }); server.close(() => process.exit(0)); }
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
