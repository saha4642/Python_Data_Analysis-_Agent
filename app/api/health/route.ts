import { NextResponse } from "next/server";

export function GET() {
  return NextResponse.json({
    status: "ok",
    service: "InsightForge frontend API",
    backend: "FastAPI analytics engine is expected at NEXT_PUBLIC_ANALYTICS_API_URL",
  });
}
