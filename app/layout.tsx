import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Python data analysis",
  description: "Upload a data file and get friendly, instant exploratory analysis.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
