import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const body = await request.json();
  return NextResponse.json({
    generatedAt: new Date().toISOString(),
    format: body.format ?? "html",
    title: body.title ?? "AI Analytics Report",
    sections: ["overview", "descriptive_statistics", "visualizations", "statistical_tests", "regression_ml", "ai_insights"],
    message: "Report payload accepted. Production deployments can forward this to the FastAPI backend for PDF/HTML rendering.",
  });
}
