import { NextResponse } from "next/server";

import API_BASE_URL, { API_KEY } from "@/config";

export const runtime = "nodejs";

const RETRYABLE_STATUSES = new Set([429, 502, 503, 504]);
const DEFAULT_MAX_ATTEMPTS = 4;

function _sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function _maxAttempts(): number {
  const raw = process.env.CATHODE_PREDICT_PROXY_MAX_ATTEMPTS;
  if (!raw) return DEFAULT_MAX_ATTEMPTS;
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed < 1) return DEFAULT_MAX_ATTEMPTS;
  return Math.min(parsed, 8);
}

function _resolveBackendApiKey(): string | undefined {
  const key =
    process.env.CATHODE_API_KEY ||
    process.env.API_KEY ||
    process.env.NEXT_PUBLIC_API_KEY ||
    API_KEY;
  return key && key.trim() ? key.trim() : undefined;
}

function _extractErrorMessage(payload: unknown, status: number): string {
  if (!payload || typeof payload !== "object") {
    return `Prediction backend error (${status})`;
  }
  const obj = payload as Record<string, unknown>;
  if (typeof obj.error === "string" && obj.error) {
    return obj.error;
  }
  const detail = obj.detail;
  if (typeof detail === "string" && detail) {
    return detail;
  }
  if (detail && typeof detail === "object") {
    const d = detail as Record<string, unknown>;
    if (typeof d.message === "string" && d.message) {
      return d.message;
    }
    if (typeof d.error === "string" && d.error) {
      return d.error;
    }
  }
  return `Prediction backend error (${status})`;
}

async function _forwardWithRetry(
  targetUrl: string,
  formData: FormData,
  headers: HeadersInit
): Promise<Response> {
  const attempts = _maxAttempts();
  let lastError: unknown;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(targetUrl, {
        method: "POST",
        headers,
        body: formData,
        cache: "no-store",
      });
      if (!RETRYABLE_STATUSES.has(response.status) || attempt === attempts) {
        return response;
      }
    } catch (error) {
      lastError = error;
      if (attempt === attempts) {
        break;
      }
    }
    const backoffMs = 400 * attempt;
    await _sleep(backoffMs);
  }

  if (lastError instanceof Error && lastError.message) {
    throw new Error(lastError.message);
  }
  throw new Error("Prediction upstream unavailable after retries");
}

export async function POST(request: Request) {
  let formData: FormData;
  try {
    formData = await request.formData();
  } catch {
    return NextResponse.json(
      { success: false, error: "Invalid multipart request body." },
      { status: 400 }
    );
  }

  const headers: HeadersInit = {};
  const apiKey = _resolveBackendApiKey();
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }

  let upstream: Response;
  try {
    upstream = await _forwardWithRetry(`${API_BASE_URL}/predict`, formData, headers);
  } catch (error) {
    const detail = error instanceof Error ? error.message : null;
    return NextResponse.json(
      {
        success: false,
        error: "Failed to connect to prediction backend.",
        detail,
      },
      { status: 502 }
    );
  }

  let payload: unknown = null;
  try {
    payload = await upstream.json();
  } catch {
    payload = null;
  }

  if (!upstream.ok) {
    return NextResponse.json(
      {
        success: false,
        error: _extractErrorMessage(payload, upstream.status),
        detail: payload,
      },
      { status: upstream.status }
    );
  }

  if (!payload || typeof payload !== "object") {
    return NextResponse.json(
      { success: false, error: "Prediction backend returned an invalid response." },
      { status: 502 }
    );
  }

  return NextResponse.json(payload, { status: upstream.status });
}
