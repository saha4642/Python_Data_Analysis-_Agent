import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "InsightForge AI Analytics Platform",
  description: "Enterprise-grade AI-powered exploratory data analysis, statistical modeling, and ML-ready preprocessing for CSV, Excel, and JSON datasets.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
