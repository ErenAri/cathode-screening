import { NextRequest, NextResponse } from "next/server";

import API_BASE_URL, { API_KEY } from "@/config";

export const runtime = "nodejs";

const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "content-length",
  "host",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

function _resolveBackendApiKey(): string | undefined {
  const key =
    process.env.CATHODE_API_KEY ||
    process.env.API_KEY ||
    process.env.NEXT_PUBLIC_API_KEY ||
    API_KEY;
  return key && key.trim() ? key.trim() : undefined;
}

function _targetUrl(request: NextRequest, path: string[]): string {
  const base = API_BASE_URL.replace(/\/+$/, "");
  const suffix = path.join("/");
  const target = new URL(`${base}/${suffix}`);
  request.nextUrl.searchParams.forEach((value, key) => {
    target.searchParams.append(key, value);
  });
  return target.toString();
}

function _forwardHeaders(request: NextRequest): Headers {
  const headers = new Headers();
  request.headers.forEach((value, key) => {
    const lower = key.toLowerCase();
    if (HOP_BY_HOP_HEADERS.has(lower)) {
      return;
    }
    headers.set(key, value);
  });

  const apiKey = _resolveBackendApiKey();
  if (apiKey) {
    headers.set("X-API-Key", apiKey);
  }
  return headers;
}

function _responseHeaders(upstream: Response): Headers {
  const headers = new Headers();
  upstream.headers.forEach((value, key) => {
    if (!HOP_BY_HOP_HEADERS.has(key.toLowerCase())) {
      headers.set(key, value);
    }
  });
  return headers;
}

async function _handle(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
): Promise<NextResponse> {
  const { path } = await params;
  const method = request.method.toUpperCase();
  const body =
    method === "GET" || method === "HEAD" ? undefined : await request.arrayBuffer();

  let upstream: Response;
  try {
    upstream = await fetch(_targetUrl(request, path), {
      method,
      headers: _forwardHeaders(request),
      body,
      cache: "no-store",
      redirect: "manual",
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : null;
    return NextResponse.json(
      { success: false, error: "Failed to connect to backend.", detail },
      { status: 502 }
    );
  }

  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: _responseHeaders(upstream),
  });
}

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return _handle(request, context);
}

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return _handle(request, context);
}

export async function PUT(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return _handle(request, context);
}

export async function PATCH(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return _handle(request, context);
}

export async function DELETE(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return _handle(request, context);
}
