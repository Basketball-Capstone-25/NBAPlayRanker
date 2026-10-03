import { createClient } from "../../lib/supabase/client";
import { API_BASE } from "./api-client";

export type CalibrationBin = {
  lower_ppp: number;
  upper_ppp: number;
  count: number;
  mean_predicted_ppp: number | null;
  mean_realized_ppp: number | null;
  bias_ppp: number | null;
  sparse: boolean;
};

type CalibrationSummary = {
  count: number;
  mean_predicted_ppp: number;
  mean_realized_ppp: number;
  bias_ppp: number;
  mae_ppp: number;
  rmse_ppp: number;
  binned_absolute_bias_ppp: number;
};

export type CalibrationReport = {
  model: string;
  evaluation: {
    method: string;
    model_parameters: { alpha: number };
    scaler_fit: string;
    parameters_source: string;
    weighting: string;
    n_splits: number;
    input_rows: number;
  };
  summary: CalibrationSummary & { bins: CalibrationBin[] };
  folds: Array<CalibrationSummary & { train_seasons: string[]; test_season: string; train_rows: number }>;
  warnings: string[];
};

export async function fetchCalibration(nSplits: number, signal?: AbortSignal): Promise<CalibrationReport> {
  const { data } = await createClient().auth.getSession();
  const token = data.session?.access_token;
  const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await fetch(`${API_BASE}/metrics/calibration?n_splits=${nSplits}&n_bins=10`, {
    headers, signal, cache: "no-store",
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === "string" ? body.detail : `Calibration request failed (${response.status}).`);
  }
  return response.json();
}
