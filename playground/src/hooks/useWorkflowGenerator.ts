/* ================================================================
   useWorkflowGenerator — core application state hook.

   Manages:
   - generation state (idle / loading / success / error)
   - AbortController lifecycle
   - domain list fetching
   - double-submission prevention
   ================================================================ */

import { useCallback, useEffect, useRef, useState } from "react";
import { generateWorkflow, fetchDomains, ApiRequestError } from "@/services/api";
import type {
  DomainInfo,
  GenerateResponse,
  GenerationMode,
} from "@/types";

export type GenerationStatus = "idle" | "loading" | "success" | "error";

export interface UseWorkflowGeneratorReturn {
  /* state */
  status: GenerationStatus;
  response: GenerateResponse | null;
  error: string | null;
  domains: DomainInfo[];

  /* actions */
  generate: (instruction: string, mode: GenerationMode, domainHint?: string, preferLLM?: boolean) => void;
  cancel: () => void;
  reset: () => void;
}

export function useWorkflowGenerator(): UseWorkflowGeneratorReturn {
  const [status, setStatus] = useState<GenerationStatus>("idle");
  const [response, setResponse] = useState<GenerateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [domains, setDomains] = useState<DomainInfo[]>([]);

  const abortRef = useRef<AbortController | null>(null);

  // Fetch domains on mount
  useEffect(() => {
    let cancelled = false;
    fetchDomains()
      .then((data) => {
        if (!cancelled) setDomains(data.domains);
      })
      .catch(() => {
        /* non-critical — domain list is optional */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const generate = useCallback(
    (instruction: string, mode: GenerationMode, domainHint?: string, preferLLM: boolean = true) => {
      // Prevent double submission
      if (status === "loading") return;

      cancel();
      setStatus("loading");
      setError(null);
      setResponse(null);

      const controller = new AbortController();
      abortRef.current = controller;

      generateWorkflow(
        {
          instruction,
          mode,
          domain_hint: domainHint || undefined,
          prefer_llm_generation: preferLLM,
        },
        controller.signal,
      )
        .then((res) => {
          if (controller.signal.aborted) return;
          setResponse(res);
          setStatus(res.success ? "success" : "error");
          if (!res.success && res.errors.length > 0) {
            setError(res.errors.map((e) => e.message).join("; "));
          }
        })
        .catch((err: unknown) => {
          if (controller.signal.aborted) return;
          const message =
            err instanceof ApiRequestError
              ? err.message
              : err instanceof Error
                ? err.message
                : "Unknown error";
          setError(message);
          setStatus("error");
        });
    },
    [status, cancel],
  );

  const reset = useCallback(() => {
    cancel();
    setStatus("idle");
    setResponse(null);
    setError(null);
  }, [cancel]);

  // Cleanup on unmount
  useEffect(() => () => cancel(), [cancel]);

  return { status, response, error, domains, generate, cancel, reset };
}
