"use client";

import React, { useEffect, useId, useRef, useState } from "react";
import { prepareReportExport, submitReportExport, type ExportFormat, type ReportPayload } from "../../../infrastructure/report-export";

type Props = { payload: ReportPayload | null; disabledReason?: string; description?: string };
type ExportState = { kind: "idle" | "preparing" | "requested" | "error"; message?: string };

export function ReportExportView({ format, onFormat, onExport, disabled, description, state }: {
  format: ExportFormat; onFormat: (format: ExportFormat) => void; onExport: () => void;
  disabled?: string; description?: string; state: ExportState;
}) {
  const id = useId();
  const busy = state.kind === "preparing";
  const controlStyle = { border: "1px solid #cbd5e1", borderRadius: 10, padding: "9px 12px", background: "#fff", color: "#0f172a", fontSize: 13 };
  return (
    <div style={{ display: "grid", gap: 5, color: "#334155", fontSize: 12 }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <label htmlFor={id}>Export format</label>
        <select id={id} style={controlStyle} value={format} disabled={busy} onChange={event => onFormat(event.target.value as ExportFormat)}>
          <option value="csv">CSV</option><option value="pdf">PDF</option><option value="json">JSON</option>
        </select>
        <button type="button" style={{ ...controlStyle, fontWeight: 700, cursor: disabled || busy ? "not-allowed" : "pointer", opacity: disabled || busy ? 0.6 : 1 }}
          disabled={Boolean(disabled) || busy} onClick={onExport} aria-describedby={`${id}-status`}>
          {busy ? "Preparing…" : state.kind === "error" ? "Retry export" : "Export"}
        </button>
      </div>
      {description ? <span>{description}</span> : null}
      <span id={`${id}-status`} role={state.kind === "error" ? "alert" : "status"} aria-live="polite" style={{ color: state.kind === "error" ? "#b91c1c" : "#475569" }}>
        {disabled || (busy ? "Preparing your file…" : state.message)}
      </span>
    </div>
  );
}

export default function ReportExport({ payload, disabledReason, description }: Props) {
  const [format, setFormat] = useState<ExportFormat>("csv");
  const [state, setState] = useState<ExportState>({ kind: "idle" });
  const active = useRef<AbortController | null>(null);
  const payloadKey = payload ? JSON.stringify(payload) : "";
  const currentKey = useRef(payloadKey);
  currentKey.current = payloadKey;

  useEffect(() => {
    active.current?.abort();
    setState({ kind: "idle" });
    return () => active.current?.abort();
  }, [payloadKey, format, disabledReason]);

  async function download() {
    if (!payload || disabledReason || (active.current && !active.current.signal.aborted)) return;
    const controller = new AbortController();
    active.current = controller;
    const startedKey = payloadKey;
    setState({ kind: "preparing" });
    try {
      const body = await prepareReportExport(payload, format, controller.signal);
      if (controller.signal.aborted || currentKey.current !== startedKey) return;
      submitReportExport(body);
      setState({ kind: "requested", message: "Download requested. Check your browser’s Downloads folder." });
    } catch (error) {
      if (!controller.signal.aborted) setState({ kind: "error", message: error instanceof Error ? error.message : "Could not download this report. Please try again." });
    } finally {
      if (active.current === controller) active.current = null;
    }
  }

  return <ReportExportView format={format} onFormat={setFormat} onExport={download} description={description}
    disabled={disabledReason || (!payload?.rows.length ? "Run the report before exporting." : undefined)} state={state} />;
}
