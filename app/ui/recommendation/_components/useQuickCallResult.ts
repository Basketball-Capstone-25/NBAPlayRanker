"use client";

import { useEffect, useRef, useState } from "react";
import type { QuickCallSituation } from "../../../domain/quick-call";
import { emptyQuickCallResult, QuickCallRequest, selectQuickCallResult } from "../../../infrastructure/quick-call-result";
import { fetchQuickCall } from "../../../services/quick-call";

const REFRESH_DEBOUNCE_MS = 250;

export function useQuickCallResult(situation: QuickCallSituation, hasMatchup: boolean) {
  const [state, setState] = useState(emptyQuickCallResult);
  const request = useRef<QuickCallRequest | null>(null);
  if (request.current === null) request.current = new QuickCallRequest();

  useEffect(() => {
    const current = request.current!;
    current.cancel();
    setState(emptyQuickCallResult());
    if (!hasMatchup) return;
    const timer = setTimeout(() => void current.run(situation, fetchQuickCall, setState), REFRESH_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      // Invalidate immediately on a tap, including during the next debounce.
      current.cancel();
    };
  }, [situation, hasMatchup]);

  return selectQuickCallResult(state, situation);
}
