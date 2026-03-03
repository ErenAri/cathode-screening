import { NextResponse } from "next/server";

import API_BASE_URL, { API_KEY } from "@/config";

export const runtime = "nodejs";

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
    upstream = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers,
      body: formData,
      cache: "no-store",
    });
  } catch {
    return NextResponse.json(
      { success: false, error: "Failed to connect to prediction backend." },
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
