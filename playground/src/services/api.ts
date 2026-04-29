/* ================================================================
   API Service Layer - single source of truth for all backend calls.

   - AbortController for cancellation
   - Configurable timeout
   - Structured error extraction
   - No console.log in production
   ================================================================ */

import type {
  DomainListResponse,
  GenerateRequest,
  GenerateResponse,
  HealthResponse,
  ProcessQueryRequest,
  ProcessQueryResponse,
} from "@/types";

function resolveApiBase(): string {
  const rawBase = (import.meta.env.VITE_API_BASE_URL ?? "").trim();

  if (!rawBase) {
    return "";
  }

  if (typeof window !== "undefined" && window.location.protocol === "https:") {
    const isInsecureAbsoluteUrl = /^http:\/\//i.test(rawBase);
    if (isInsecureAbsoluteUrl) {
      return "";
    }
  }

  return rawBase.replace(/\/+$/, "");
}

// Default to same-origin in production; only use custom env override when safe.
const BASE = resolveApiBase();
const DEFAULT_TIMEOUT_MS = 60_000;

export class ApiRequestError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly body: unknown = null,
  ) {
    super(message);
    this.name = "ApiRequestError";
  }
}

async function parseResponseBody(res: Response): Promise<unknown> {
  const contentType = res.headers.get("content-type") ?? "";
  const rawText = await res.text();

  if (!rawText.trim()) {
    return null;
  }

  if (contentType.includes("application/json")) {
    try {
      return JSON.parse(rawText) as unknown;
    } catch {
      throw new ApiRequestError("Server returned invalid JSON", res.status, rawText);
    }
  }

  return rawText;
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const externalSignal = init.signal;
  const abortOnExternalSignal = () => controller.abort();

  if (externalSignal) {
    if (externalSignal.aborted) {
      controller.abort();
    } else {
      externalSignal.addEventListener("abort", abortOnExternalSignal, {
        once: true,
      });
    }
  }

  try {
    const res = await fetch(`${BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        ...init.headers,
      },
    });

    const body = await parseResponseBody(res);

    if (!res.ok) {
      const msg =
        typeof body === "object" && body !== null && "message" in body
          ? String((body as Record<string, unknown>).message)
          : typeof body === "string" && body.trim()
            ? body.slice(0, 200)
            : `HTTP ${res.status}`;
      throw new ApiRequestError(msg, res.status, body);
    }

    if (body === null) {
      throw new ApiRequestError("Server returned an empty response", res.status, null);
    }

    if (typeof body === "string") {
      throw new ApiRequestError(
        "Server returned non-JSON content",
        res.status,
        body.slice(0, 500),
      );
    }

    return body as T;
  } catch (err) {
    if (err instanceof ApiRequestError) throw err;
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiRequestError("Request timed out", 408);
    }
    throw new ApiRequestError(
      err instanceof Error ? err.message : "Network error",
      0,
    );
  } finally {
    clearTimeout(timer);
    externalSignal?.removeEventListener("abort", abortOnExternalSignal);
  }
}

export function generateWorkflow(
  payload: GenerateRequest,
  signal?: AbortSignal,
): Promise<GenerateResponse> {
  return request<GenerateResponse>("/api/v1/generate", {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  });
}

export function fetchDomains(): Promise<DomainListResponse> {
  return request<DomainListResponse>("/api/v1/domains");
}

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/v1/health");
}

export function processQuery(
  query: string,
  topK: number = 5,
): Promise<ProcessQueryResponse> {
  const payload: ProcessQueryRequest = { query, top_k: topK };
  return request<ProcessQueryResponse>("/api/v1/rag/process_query", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
