"use client";

import React, { useEffect, useRef, useState } from "react";
import { DatasetReloadError, type DatasetReloadResult, type ReloadDataset } from "../../domain/dataset-reload";
import { reloadDataset } from "../../infrastructure/dataset-reload";
import styles from "./DatasetReloadPanel.module.css";

export type ReloadState =
  | { kind: "idle" }
  | { kind: "loading"; dataset: ReloadDataset }
  | { kind: "ready"; dataset: ReloadDataset; result: DatasetReloadResult }
  | { kind: "error"; dataset: ReloadDataset; message: string; uncertain: boolean };

const names: Record<ReloadDataset, string> = { synergy: "play types", pbp: "shots" };

export function DatasetReloadView({ state, onReload }: { state: ReloadState; onReload(dataset: ReloadDataset): void }) {
  const pending = state.kind === "loading";
  const uncertain = state.kind === "error" && state.uncertain;
  return <section className={`card ${styles.panel}`} aria-labelledby="reload-heading" aria-busy={pending}>
    <div className={styles.heading}>
      <div><h2 id="reload-heading">Reload stored datasets</h2>
        <p>Reload the server’s saved files. This does not download new NBA data.</p></div>
      <div className={styles.actions}>
        <button className="btn btn--secondary" type="button" disabled={pending || uncertain} onClick={() => onReload("synergy")}>Reload play types</button>
        <button className="btn btn--secondary" type="button" disabled={pending || uncertain} onClick={() => onReload("pbp")}>Reload shots</button>
      </div>
    </div>
    <div aria-live="polite">
      {state.kind === "loading" && <p role="status">Reloading {names[state.dataset]}… Keep this page open.</p>}
      {state.kind === "ready" && <>
        <p role="status">Reloaded {names[state.dataset]}. {state.result.changed ? "A different file version is now loaded." : "The saved file is unchanged."}</p>
        <p>{state.result.rows.toLocaleString("en-US")} source rows · {state.result.seasons.length} seasons
          {state.result.quality_status === "pass" ? " · Source checks passed" : ""}</p>
        <details><summary>Reload details</summary>
          <dl><dt>File</dt><dd>{state.result.source_file}</dd>
            <dt>Previous version</dt><dd><code>{state.result.previous_version}</code></dd>
            <dt>Loaded version</dt><dd><code>{state.result.dataset_version}</code></dd>
            <dt>Seasons</dt><dd>{state.result.seasons.join(", ")}</dd></dl>
        </details>
      </>}
      {state.kind === "error" && <div role="alert"><p>{state.message}</p>
        {uncertain && <><p>Refresh this page to check the loaded data before another reload.</p>
          <button className="btn btn--secondary" type="button" onClick={() => window.location.reload()}>Refresh page</button></>}
      </div>}
    </div>
  </section>;
}

type Props = {
  onReloaded(dataset: ReloadDataset, result: DatasetReloadResult): void;
  onPendingChange?(dataset: ReloadDataset | null): void;
};

export default function DatasetReloadPanel({ onReloaded, onPendingChange }: Props) {
  const [state, setState] = useState<ReloadState>({ kind: "idle" });
  const inFlight = useRef<AbortController | null>(null);
  const alive = useRef(true);
  const uncertain = useRef(false);
  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; inFlight.current?.abort(); };
  }, []);

  async function run(dataset: ReloadDataset) {
    // The ref closes the gap before React renders disabled buttons.
    if (inFlight.current || uncertain.current) return;
    const controller = new AbortController();
    inFlight.current = controller;
    setState({ kind: "loading", dataset });
    onPendingChange?.(dataset);
    const timer = setTimeout(() => controller.abort(), 310_000);
    try {
      const result = await reloadDataset(dataset, controller.signal);
      if (!alive.current) return;
      setState({ kind: "ready", dataset, result });
      onReloaded(dataset, result);
    } catch (error) {
      if (!alive.current) return;
      const unknown = !(error instanceof DatasetReloadError) || error.outcome === "unknown";
      uncertain.current = unknown;
      setState({ kind: "error", dataset, uncertain: unknown, message: error instanceof Error
        ? error.message : "Could not confirm the reload result. Check the data before trying again." });
    } finally {
      clearTimeout(timer);
      if (inFlight.current === controller) inFlight.current = null;
      if (alive.current) onPendingChange?.(null);
    }
  }

  return <DatasetReloadView state={state} onReload={dataset => void run(dataset)} />;
}
