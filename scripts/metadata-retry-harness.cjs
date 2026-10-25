#!/usr/bin/env node
// Local browser fixture for the actual metadata panel, API helper and parser.
// Run: node scripts/metadata-retry-harness.cjs --out <evidence-directory>
const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { spawnSync } = require("node:child_process");
const esbuild = require("esbuild");

const repo = path.resolve(__dirname, "..");
const args = process.argv.slice(2);
const outputArg = args.indexOf("--out");
if (outputArg < 0 || !args[outputArg + 1]) throw new Error("Provide --out for fixture evidence.");
const output = path.resolve(args[outputArg + 1]);
fs.mkdirSync(output, { recursive: true });
const sha256 = value => crypto.createHash("sha256").update(value).digest("hex");
const python = process.env.METADATA_FIXTURE_PYTHON || path.join(repo, "backend/.venv/bin/python");
const fixtureRun = spawnSync(python, ["-c", `
import json, tempfile
from pathlib import Path
from infrastructure.data_access.dataset_metadata import dataset_catalog, source_version
with tempfile.TemporaryDirectory(prefix="metadata-browser-fixture-") as temporary:
    root = Path(temporary)
    source = root / "synergy_playtypes_2019_2025_players.csv"
    source.write_text("SEASON,POSS,PPP\\n2024-25,10,1.1\\n2024-25,20,1.2\\n2023-24,30,1.0\\n")
    valid = dataset_catalog(root)
    assert valid[0]["status"] == "available" and valid[0]["row_count"] == 3
    missing = {item["id"]: item for item in valid}["shots"]
    assert missing["status"] == "unavailable" and "missing" in missing["error"]
    (root / "pbp").mkdir()
    (root / "pbp/shots_clean.parquet").write_bytes(b"not a parquet file")
    corrupt = dataset_catalog(root)
    by_id = {item["id"]: item for item in corrupt}
    assert by_id["synergy"]["status"] == "available"
    assert by_id["shots"]["status"] == "unavailable"
    assert "could not be read" in by_id["shots"]["error"]
    source.with_suffix(".csv.source.json").write_text(json.dumps({"version": source_version(source), "pulled_at": []}))
    malformed = dataset_catalog(root)
    assert malformed[0]["status"] == "available" and malformed[0]["pulled_at"] is None
    print(json.dumps({"valid": valid, "corrupt_source": corrupt, "malformed_sidecar": malformed,
        "checks": {"valid_source_rows": True, "missing_source_named": True,
        "corrupt_source_named": True, "valid_source_survives_corrupt_source": True,
        "malformed_sidecar_keeps_source_and_unknown_date": True}}))
`], { cwd: path.join(repo, "backend"), encoding: "utf8" });
if (fixtureRun.status !== 0) throw new Error(`Fixture generation failed: ${fixtureRun.stderr}`);
const fixtures = JSON.parse(fixtureRun.stdout);
const sources = ["app/ui/data-explorer/DatasetMetadataPanel.tsx", "app/infrastructure/api-client.ts",
  "app/infrastructure/dataset-catalog.ts", "backend/infrastructure/data_access/dataset_metadata.py"];
const evidence = {
  started_at_utc: new Date().toISOString(),
  scope: "Local synthetic source files through the actual backend catalog and browser metadata panel/API parser.",
  production_requests: 0, credentials_used: false,
  substitutions: ["Supabase session returns null; authentication is not under test.", "Next Link renders an ordinary anchor."],
  sources: sources.map(file => ({ file, sha256: sha256(fs.readFileSync(path.join(repo, file))) })),
  fixture_checks: fixtures.checks, fixtures, events: [],
};
function log(event) {
  if (evidence.events.length < 200) evidence.events.push({ at_utc: new Date().toISOString(), ...event });
  fs.writeFileSync(path.join(output, "events.json"), JSON.stringify(evidence, null, 2) + "\n");
}
const scenarios = ["service-error", "invalid-json", "invalid-schema", "corrupt-source", "malformed-sidecar"];
const attempts = new Map();
let origin, bundle;
const entry = `
import React from "react";
import {createRoot} from "react-dom/client";
import DatasetMetadataPanel from "./app/ui/data-explorer/DatasetMetadataPanel";
const scenario = new URLSearchParams(location.search).get("scenario") || "service-error";
function App() { return <main>
  <p className="fixture-label">ISOLATED TEST FIXTURE — BST-194</p>
  <h1>Metadata availability and retry</h1>
  <p>Actual app panel and request parser with synthetic local source files. No production requests or credentials.</p>
  <p>Scenario: <strong>{scenario}</strong>. The first three scenarios fail once; Retry returns the valid three-row source.</p>
  <nav>${scenarios.map(s => `<a href="/?scenario=${s}">${s}</a>`).join(' · ')}</nav>
  <div id="actual-metadata"><DatasetMetadataPanel /></div>
</main>; }
createRoot(document.getElementById("root")).render(<App />);
let previous = "";
new MutationObserver(() => {
  const panel = document.getElementById("actual-metadata");
  if (!panel) return;
  const button = panel.querySelector("button");
  const current = JSON.stringify({scenario, text: panel.innerText, heading: panel.querySelector("h2")?.textContent || "",
    alert: panel.querySelector('[role="alert"]')?.textContent || "", status: panel.querySelector('[role="status"]')?.textContent || "",
    retry_visible: !!button, retry_disabled: button?.disabled === true});
  if (current !== previous) { previous = current;
    fetch("/observations", {method:"POST", headers:{"Content-Type":"application/json"}, body:current}).catch(()=>{});
  }
}).observe(document.getElementById("root"), {subtree:true, childList:true, attributes:true, characterData:true});
`;
const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Isolated metadata retry fixture</title><style>
body{margin:0;background:#eef2f6;font:16px system-ui;color:#172033}main{max-width:900px;margin:40px auto;padding:32px;background:white;border-radius:16px}
.fixture-label{color:#8a4300;font-weight:700;font-size:13px}h1{font-size:28px}p{line-height:1.5}nav{padding:16px 0;line-height:1.8}section{background:#f8fafc;border-radius:12px}dt{font-weight:600;margin-top:10px}dd{margin:4px 0}button{padding:7px 14px;margin-left:8px}summary{cursor:pointer}
</style></head><body><div id="root"></div><script src="/bundle.js"></script></body></html>`;
const server = http.createServer(async (request, response) => {
  response.setHeader("Cache-Control", "no-store");
  response.setHeader("X-Content-Type-Options", "nosniff");
  if (request.headers.host !== new URL(origin).host) { response.writeHead(403).end(); return; }
  const url = new URL(request.url, origin);
  const send = (status, value) => { response.writeHead(status, {"Content-Type":"application/json"}); response.end(JSON.stringify(value)); };
  if (request.method === "GET" && url.pathname === "/") {
    response.writeHead(200, {"Content-Type":"text/html; charset=utf-8", "Content-Security-Policy":"default-src 'self'; connect-src 'self'; style-src 'self' 'unsafe-inline'; form-action 'self'; base-uri 'none'"}); response.end(html); return;
  }
  if (request.method === "GET" && url.pathname === "/bundle.js") { response.writeHead(200, {"Content-Type":"text/javascript"}); response.end(bundle); return; }
  if (request.method === "GET" && url.pathname === "/state") { send(200, evidence); return; }
  if (request.method === "GET" && url.pathname === "/meta/datasets") {
    let scenario = "service-error";
    try { scenario = new URL(request.headers.referer).searchParams.get("scenario") || scenario; } catch {}
    if (!scenarios.includes(scenario)) { send(400, {error:"Unknown fixture scenario."}); return; }
    const attempt = (attempts.get(scenario) || 0) + 1; attempts.set(scenario, attempt);
    await new Promise(resolve => setTimeout(resolve, 700));
    if (attempt === 1 && scenario === "service-error") { log({kind:"catalog_response",scenario,attempt,status:503}); send(503,{detail:"Synthetic temporary metadata failure."}); return; }
    if (attempt === 1 && scenario === "invalid-json") { log({kind:"catalog_response",scenario,attempt,status:200,body_kind:"invalid_json"}); response.writeHead(200,{"Content-Type":"application/json"}); response.end("{broken"); return; }
    if (attempt === 1 && scenario === "invalid-schema") { log({kind:"catalog_response",scenario,attempt,status:200,body_kind:"invalid_schema"}); send(200,{datasets:[{id:"synergy",row_count:-1}]}); return; }
    const fixtureName = scenario === "corrupt-source" ? "corrupt_source" : scenario === "malformed-sidecar" ? "malformed_sidecar" : "valid";
    log({kind:"catalog_response",scenario,attempt,status:200,fixture:fixtureName}); send(200,{datasets:fixtures[fixtureName]}); return;
  }
  if (request.method === "POST" && url.pathname === "/observations") {
    if (request.headers.origin !== origin) { send(403,{error:"Local fixture origin required."}); return; }
    let body="";
    try { for await (const chunk of request) { body += chunk; if (Buffer.byteLength(body)>8192) { send(413,{error:"Observation too large."}); return; } }
      const item=JSON.parse(body); log({kind:"component_state",scenario:String(item.scenario).slice(0,40),text:String(item.text).slice(0,3000),heading:String(item.heading).slice(0,100),alert:String(item.alert).slice(0,400),status:String(item.status).slice(0,400),retry_visible:item.retry_visible===true,retry_disabled:item.retry_disabled===true}); send(200,{recorded:true});
    } catch { send(400,{error:"Invalid observation."}); } return;
  }
  send(404,{error:"Unknown fixture route."});
});
async function start() {
  const result = await esbuild.build({stdin:{contents:entry,loader:"tsx",resolveDir:repo},bundle:true,write:false,platform:"browser",format:"iife",jsx:"automatic",
    define:{"process.env.NODE_ENV":'"production"',"process.env.NEXT_PUBLIC_API_BASE":"window.location.origin"},
    plugins:[{name:"isolated-navigation-and-auth",setup(build){
      build.onResolve({filter:/^next\/link$/},()=>({path:"link",namespace:"fixture"}));
      build.onResolve({filter:/lib\/supabase\/client$/},()=>({path:"auth",namespace:"fixture"}));
      build.onLoad({filter:/.*/,namespace:"fixture"},args=>({loader:"jsx",contents:args.path==="auth" ? 'export function createClient(){return {auth:{getSession:async()=>({data:{session:null}})}}}' : 'import React from "react";export default function Link({children,...props}){return <a {...props}>{children}</a>}',resolveDir:repo}));
    }}]});
  bundle=result.outputFiles[0].contents;
  server.listen(0,"127.0.0.1",()=>{origin=`http://127.0.0.1:${server.address().port}`; log({kind:"fixture_started",origin}); console.log(JSON.stringify({url:origin,scope:evidence.scope,fixture_checks:fixtures.checks}));});
}
function stop(){log({kind:"fixture_stopped"});server.close(()=>process.exit(0));}
process.on("SIGINT",stop);process.on("SIGTERM",stop);
start().catch(error=>{console.error(error);process.exitCode=1;});
