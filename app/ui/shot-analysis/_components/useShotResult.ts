"use client";

import { useEffect, useRef, useState } from "react";
import { emptyShotResult, selectShotResult, ShotResultRequest } from "../../../infrastructure/shot-result-state";

export function useShotResult<I extends object, R>(inputs: I) {
  const [state, setState] = useState(() => emptyShotResult<I, R>());
  const request = useRef<ShotResultRequest<I, R> | null>(null);
  if (request.current === null) request.current = new ShotResultRequest<I, R>();
  useEffect(() => () => request.current?.cancel(), []);
  return {
    ...selectShotResult(state, inputs),
    run: (fetchResult: (snapshot: I) => Promise<R>) => request.current!.run(inputs, fetchResult, setState),
    reset: () => { request.current!.cancel(); setState(emptyShotResult<I, R>()); },
  };
}
