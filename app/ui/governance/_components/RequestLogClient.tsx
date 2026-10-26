"use client";

// UI Layer — analyst request log (SCRUM-494 / UC17).
// Lists recent API calls (who, what, when) with user, endpoint and date filters and paging.

import { FormEvent, useEffect, useRef, useState } from "react";
import {
  fetchRequestLog,
  REQUEST_LOG_PAGE_SIZE,
  RequestLogForbiddenError,
  type RequestLogFilters,
  type RequestLogPage,
} from "../../../services/request-log";
import styles from "./RequestLog.module.css";

type Status = "loading" | "ready" | "error";

const EMPTY: RequestLogFilters = { user: "", endpoint: "", dateFrom: "", dateTo: "" };

function formatTime(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

export default function RequestLogClient() {
  const [draft, setDraft] = useState<RequestLogFilters>(EMPTY);
  const [applied, setApplied] = useState<RequestLogFilters>(EMPTY);
  const [page, setPage] = useState(1);
  const [result, setResult] = useState<RequestLogPage | null>(null);
  const [status, setStatus] = useState<Status>("loading");
  const [message, setMessage] = useState("");
  const inFlight = useRef<AbortController | null>(null);

  useEffect(() => {
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    setStatus("loading");
    fetchRequestLog({ ...applied, page, pageSize: REQUEST_LOG_PAGE_SIZE }, controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return;
        setResult(data);
        setStatus("ready");
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setResult(null);
        setStatus("error");
        setMessage(
          error instanceof RequestLogForbiddenError
            ? error.message
            : error instanceof Error
            ? error.message
            : "Could not load the request log."
        );
      });
    return () => controller.abort();
  }, [applied, page]);

  function search(event: FormEvent) {
    event.preventDefault();
    setPage(1);
    setApplied({ ...draft });
  }

  function clear() {
    setDraft(EMPTY);
    setPage(1);
    setApplied(EMPTY);
  }

  const items = result?.items ?? [];
  const totalPages = result?.total != null ? Math.max(1, Math.ceil(result.total / REQUEST_LOG_PAGE_SIZE)) : null;

  return (
    <main className={styles.page} aria-label="Request log">
      <div>
        <h1 className={styles.title}>Request Log</h1>
        <p className={styles.sub}>Who called which API endpoint, and when. Newest first. Times are shown in your local time; date filters use UTC days.</p>
      </div>

      <form className={styles.filters} onSubmit={search} aria-label="Filter the request log">
        <label className={styles.field}>
          User email contains
          <input className={styles.input} value={draft.user} maxLength={320} placeholder="coach@"
                 onChange={(e) => setDraft({ ...draft, user: e.target.value })} />
        </label>
        <label className={styles.field}>
          Endpoint contains
          <input className={styles.input} value={draft.endpoint} maxLength={200} placeholder="/rank-plays"
                 onChange={(e) => setDraft({ ...draft, endpoint: e.target.value })} />
        </label>
        <label className={styles.field}>
          From
          <input className={styles.input} type="date" value={draft.dateFrom}
                 onChange={(e) => setDraft({ ...draft, dateFrom: e.target.value })} />
        </label>
        <label className={styles.field}>
          To
          <input className={styles.input} type="date" value={draft.dateTo}
                 onChange={(e) => setDraft({ ...draft, dateTo: e.target.value })} />
        </label>
        <div className={styles.actions}>
          <button className={`${styles.button} ${styles.primary}`} type="submit">Search</button>
          <button className={styles.button} type="button" onClick={clear}>Clear</button>
        </div>
      </form>

      <div className={styles.status} role="status" aria-live="polite">
        {status === "loading" ? "Loading…" : null}
        {status === "ready" && result
          ? result.total === 0
            ? "No requests match these filters."
            : `${result.total ?? items.length} request${result.total === 1 ? "" : "s"} • page ${result.page}${totalPages ? ` of ${totalPages}` : ""}`
          : null}
      </div>

      {status === "error" ? <div className={styles.error}>{message}</div> : null}

      {status !== "error" && items.length > 0 ? (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Time</th><th>User</th><th>Role</th><th>Method</th><th>Endpoint</th><th>Query</th><th>Status</th><th>ms</th>
              </tr>
            </thead>
            <tbody>
              {items.map((row) => (
                <tr key={row.id}>
                  <td>{formatTime(row.created_at)}</td>
                  <td>{row.user_email ?? "—"}</td>
                  <td>{row.user_role}</td>
                  <td>{row.method}</td>
                  <td>{row.endpoint}</td>
                  <td className={styles.query} title={row.query}>{row.query || "—"}</td>
                  <td className={row.status_code >= 400 ? styles.bad : undefined}>{row.status_code}</td>
                  <td>{row.duration_ms}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <div className={styles.pager}>
        <button className={styles.button} type="button" disabled={page <= 1 || status === "loading"}
                onClick={() => setPage((p) => Math.max(1, p - 1))}>
          Previous
        </button>
        <button className={styles.button} type="button" disabled={!result?.has_next || status === "loading"}
                onClick={() => setPage((p) => p + 1)}>
          Next
        </button>
      </div>
    </main>
  );
}
