"use client";

import type { ChangeEvent, DragEvent } from "react";
import { useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import * as XLSX from "xlsx";

type DataValue = string | number | boolean | null;
type DataRow = Record<string, DataValue>;
type ColumnKind = "numeric" | "categorical" | "datetime" | "boolean" | "binary" | "text";
type ThemeMode = "light" | "dark";
type ActiveSection =
  | "Dashboard"
  | "Dataset Overview"
  | "Descriptive Statistics"
  | "Visualizations"
  | "Statistical Testing"
  | "Regression & ML"
  | "Correlation Analysis"
  | "Report Export";

type ColumnProfile = {
  name: string;
  kind: ColumnKind;
  total: number;
  present: number;
  missing: number;
  missingPercent: number;
  unique: number;
  sample: string[];
  min?: number;
  max?: number;
  mean?: number;
  median?: number;
  mode?: string;
  stdev?: number;
  variance?: number;
  range?: number;
  q1?: number;
  q3?: number;
  iqr?: number;
  skewness?: number;
  kurtosis?: number;
  zOutliers?: number;
  iqrOutliers?: number;
  histogram?: Array<{ label: string; count: number; percent: number }>;
  topValues: Array<{ value: string; count: number; percent: number }>;
};

type DatasetAnalysis = {
  id: string;
  fileName: string;
  rows: number;
  columns: number;
  duplicateRows: number;
  completeness: number;
  memoryEstimate: string;
  profiles: ColumnProfile[];
  numericProfiles: ColumnProfile[];
  categoricalProfiles: ColumnProfile[];
  datetimeProfiles: ColumnProfile[];
  correlations: Array<{ left: string; right: string; pearson: number; spearman: number; pValue: number; strength: string }>;
  insights: string[];
  recommendations: string[];
};

const MAX_ROWS = 25000;
const ACCEPTED_TYPES = ".csv,.tsv,.txt,.json,.xlsx,.xls";
const NAV_ITEMS: ActiveSection[] = [
  "Dashboard",
  "Dataset Overview",
  "Descriptive Statistics",
  "Visualizations",
  "Statistical Testing",
  "Regression & ML",
  "Correlation Analysis",
  "Report Export",
];

function normalizeCell(value: unknown): DataValue {
  if (value === undefined || value === null) return null;
  if (typeof value === "number" || typeof value === "boolean") return value;
  const text = String(value).trim();
  if (!text || ["na", "n/a", "null", "none", "undefined", "nan"].includes(text.toLowerCase())) return null;
  return text;
}

function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string") return null;
  const cleaned = value.replace(/[$,%]/g, "").replace(/,/g, "").trim();
  if (!cleaned) return null;
  const parsed = Number(cleaned);
  return Number.isFinite(parsed) ? parsed : null;
}

function asDate(value: unknown): number | null {
  if (value instanceof Date) return value.getTime();
  if (typeof value !== "string" || value.length < 6) return null;
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? timestamp : null;
}

function quantile(sorted: number[], q: number): number {
  if (!sorted.length) return 0;
  const position = (sorted.length - 1) * q;
  const base = Math.floor(position);
  const rest = position - base;
  return sorted[base + 1] !== undefined ? sorted[base] + rest * (sorted[base + 1] - sorted[base]) : sorted[base];
}

function formatNumber(value?: number, digits = 2): string {
  if (value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("en", { maximumFractionDigits: digits }).format(value);
}

function formatPercent(value: number): string {
  return `${formatNumber(value, 1)}%`;
}

function pearson(x: number[], y: number[]): number {
  const n = Math.min(x.length, y.length);
  if (n < 2) return 0;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let numerator = 0;
  let dx = 0;
  let dy = 0;
  for (let index = 0; index < n; index += 1) {
    const xv = x[index] - mx;
    const yv = y[index] - my;
    numerator += xv * yv;
    dx += xv ** 2;
    dy += yv ** 2;
  }
  return dx && dy ? numerator / Math.sqrt(dx * dy) : 0;
}

function rank(values: number[]): number[] {
  return values.map((value) => [...values].sort((a, b) => a - b).indexOf(value) + 1);
}

function inferKind(values: DataValue[]): ColumnKind {
  const present = values.filter((value) => value !== null);
  if (!present.length) return "text";
  const numericRatio = present.filter((value) => asNumber(value) !== null).length / present.length;
  const dateRatio = present.filter((value) => asDate(value) !== null).length / present.length;
  const booleanRatio = present.filter((value) => typeof value === "boolean" || ["true", "false", "yes", "no"].includes(String(value).toLowerCase())).length / present.length;
  const uniqueRatio = new Set(present.map(String)).size / present.length;
  if (booleanRatio >= 0.9 || new Set(present.map(String)).size === 2) return "binary";
  if (numericRatio >= 0.86) return "numeric";
  if (dateRatio >= 0.75) return "datetime";
  if (uniqueRatio <= 0.55 || present.length <= 30) return "categorical";
  return "text";
}

function buildHistogram(values: number[], buckets = 10): ColumnProfile["histogram"] {
  if (!values.length) return [];
  const sorted = [...values].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  if (min === max) return [{ label: formatNumber(min), count: sorted.length, percent: 100 }];
  const width = (max - min) / buckets;
  const counts = Array.from({ length: buckets }, () => 0);
  sorted.forEach((value) => {
    const index = Math.min(Math.floor((value - min) / width), buckets - 1);
    counts[index] += 1;
  });
  return counts.map((count, index) => ({
    label: `${formatNumber(min + width * index, 1)}–${formatNumber(min + width * (index + 1), 1)}`,
    count,
    percent: (count / sorted.length) * 100,
  }));
}

function profileColumn(name: string, rows: DataRow[]): ColumnProfile {
  const values = rows.map((row) => row[name] ?? null);
  const present = values.filter((value) => value !== null);
  const kind = inferKind(values);
  const frequency = new Map<string, number>();
  present.forEach((value) => frequency.set(String(value), (frequency.get(String(value)) ?? 0) + 1));
  const topValues = [...frequency.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([value, count]) => ({ value, count, percent: present.length ? (count / present.length) * 100 : 0 }));

  const profile: ColumnProfile = {
    name,
    kind,
    total: values.length,
    present: present.length,
    missing: values.length - present.length,
    missingPercent: values.length ? ((values.length - present.length) / values.length) * 100 : 0,
    unique: frequency.size,
    sample: present.slice(0, 5).map(String),
    mode: topValues[0]?.value,
    topValues,
  };

  if (kind === "numeric") {
    const nums = present.map(asNumber).filter((value): value is number => value !== null).sort((a, b) => a - b);
    const mean = nums.reduce((a, b) => a + b, 0) / (nums.length || 1);
    const variance = nums.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(nums.length - 1, 1);
    const stdev = Math.sqrt(variance);
    const q1 = quantile(nums, 0.25);
    const median = quantile(nums, 0.5);
    const q3 = quantile(nums, 0.75);
    const iqr = q3 - q1;
    const skewness = nums.length ? nums.reduce((sum, value) => sum + ((value - mean) / (stdev || 1)) ** 3, 0) / nums.length : 0;
    const kurtosis = nums.length ? nums.reduce((sum, value) => sum + ((value - mean) / (stdev || 1)) ** 4, 0) / nums.length - 3 : 0;
    profile.min = nums[0];
    profile.max = nums[nums.length - 1];
    profile.mean = mean;
    profile.median = median;
    profile.variance = variance;
    profile.stdev = stdev;
    profile.q1 = q1;
    profile.q3 = q3;
    profile.iqr = iqr;
    profile.range = (profile.max ?? 0) - (profile.min ?? 0);
    profile.skewness = skewness;
    profile.kurtosis = kurtosis;
    profile.zOutliers = nums.filter((value) => Math.abs((value - mean) / (stdev || 1)) > 3).length;
    profile.iqrOutliers = nums.filter((value) => value < q1 - 1.5 * iqr || value > q3 + 1.5 * iqr).length;
    profile.histogram = buildHistogram(nums);
  }

  return profile;
}

function analyzeRows(fileName: string, rows: DataRow[]): DatasetAnalysis {
  const columns = Array.from(rows.reduce((set, row) => {
    Object.keys(row).forEach((key) => set.add(key));
    return set;
  }, new Set<string>()));
  const profiles = columns.map((column) => profileColumn(column, rows));
  const numericProfiles = profiles.filter((profile) => profile.kind === "numeric");
  const categoricalProfiles = profiles.filter((profile) => ["categorical", "boolean", "binary", "text"].includes(profile.kind));
  const datetimeProfiles = profiles.filter((profile) => profile.kind === "datetime");
  const duplicateRows = rows.length - new Set(rows.map((row) => JSON.stringify(row))).size;
  const totalCells = rows.length * Math.max(columns.length, 1);
  const missingCells = profiles.reduce((sum, profile) => sum + profile.missing, 0);

  const correlations: DatasetAnalysis["correlations"] = [];
  for (let left = 0; left < numericProfiles.length; left += 1) {
    for (let right = left + 1; right < numericProfiles.length; right += 1) {
      const x: number[] = [];
      const y: number[] = [];
      rows.forEach((row) => {
        const xv = asNumber(row[numericProfiles[left].name]);
        const yv = asNumber(row[numericProfiles[right].name]);
        if (xv !== null && yv !== null) {
          x.push(xv);
          y.push(yv);
        }
      });
      const value = pearson(x, y);
      const spearman = pearson(rank(x), rank(y));
      correlations.push({
        left: numericProfiles[left].name,
        right: numericProfiles[right].name,
        pearson: value,
        spearman,
        pValue: Math.max(0.0001, Math.min(0.99, (1 - Math.abs(value)) / Math.sqrt(Math.max(x.length, 1)))),
        strength: Math.abs(value) > 0.75 ? "strong" : Math.abs(value) > 0.45 ? "moderate" : "weak",
      });
    }
  }
  correlations.sort((a, b) => Math.abs(b.pearson) - Math.abs(a.pearson));

  const riskiest = [...profiles].sort((a, b) => b.missingPercent - a.missingPercent)[0];
  const outlierHeavy = [...numericProfiles].sort((a, b) => (b.iqrOutliers ?? 0) - (a.iqrOutliers ?? 0))[0];
  const strongest = correlations[0];
  const insights = [
    `Analyzed ${formatNumber(rows.length, 0)} rows and ${columns.length} columns with ${formatPercent(100 - (missingCells / Math.max(totalCells, 1)) * 100)} overall completeness.`,
    riskiest && riskiest.missing > 0 ? `${riskiest.name} has the highest missingness at ${formatPercent(riskiest.missingPercent)} and should be imputed or reviewed.` : "No severe missing-value hotspot was detected in the profiled sample.",
    duplicateRows ? `${formatNumber(duplicateRows, 0)} duplicate rows were detected; remove them before model training.` : "No duplicate rows were found in the profiled sample.",
    outlierHeavy && (outlierHeavy.iqrOutliers ?? 0) > 0 ? `${outlierHeavy.name} contains ${outlierHeavy.iqrOutliers} IQR outliers; compare winsorization, robust scaling, and model sensitivity.` : "Numeric features show limited IQR-based outlier pressure.",
    strongest ? `${strongest.left} and ${strongest.right} show a ${strongest.strength} Pearson relationship (${formatNumber(strongest.pearson, 2)}).` : "Upload at least two numeric variables to unlock correlation narratives.",
  ];

  const recommendations = [
    "Impute numeric missing values with median or model-based imputers and categorical values with a missing/unknown level.",
    "Encode categoricals with one-hot or target encoding, then standardize continuous predictors for SVM, KNN, and logistic regression.",
    "Use train/test split plus cross-validation before trusting any metric, and keep leakage-prone columns out of the feature set.",
    "Export the analysis report as HTML/PDF for stakeholders and CSV for reproducible downstream modeling.",
  ];

  return {
    id: `${fileName}-${Date.now()}`,
    fileName,
    rows: rows.length,
    columns: columns.length,
    duplicateRows,
    completeness: 100 - (missingCells / Math.max(totalCells, 1)) * 100,
    memoryEstimate: `${formatNumber(JSON.stringify(rows.slice(0, 100)).length * (rows.length / Math.max(rows.slice(0, 100).length, 1)) / 1024 / 1024, 2)} MB`,
    profiles,
    numericProfiles,
    categoricalProfiles,
    datetimeProfiles,
    correlations: correlations.slice(0, 12),
    insights,
    recommendations,
  };
}

async function parseFile(file: File): Promise<DataRow[]> {
  const extension = file.name.split(".").pop()?.toLowerCase();
  if (["xlsx", "xls"].includes(extension ?? "")) {
    const buffer = await file.arrayBuffer();
    const workbook = XLSX.read(buffer, { type: "array" });
    const sheet = workbook.Sheets[workbook.SheetNames[0]];
    return XLSX.utils.sheet_to_json<Record<string, unknown>>(sheet).slice(0, MAX_ROWS).map((row) => Object.fromEntries(Object.entries(row).map(([key, value]) => [key, normalizeCell(value)])));
  }
  const text = await file.text();
  if (extension === "json") {
    const parsed = JSON.parse(text) as Record<string, unknown>[] | Record<string, unknown>;
    const data = Array.isArray(parsed) ? parsed : Object.values(parsed).find(Array.isArray) ?? [parsed];
    return (data as Record<string, unknown>[]).slice(0, MAX_ROWS).map((row) => Object.fromEntries(Object.entries(row).map(([key, value]) => [key, normalizeCell(value)])));
  }
  const parsed = Papa.parse<Record<string, unknown>>(text, { header: true, skipEmptyLines: true, dynamicTyping: true, delimiter: extension === "tsv" ? "\t" : undefined });
  if (parsed.errors.length) throw new Error(parsed.errors[0].message);
  return parsed.data.slice(0, MAX_ROWS).map((row) => Object.fromEntries(Object.entries(row).map(([key, value]) => [key, normalizeCell(value)])));
}

function downloadBlob(name: string, content: string, type: string): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

function MetricCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="metric-card"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

function ProgressBar({ value }: { value: number }) {
  return <div className="progress"><span style={{ width: `${Math.min(Math.max(value, 0), 100)}%` }} /></div>;
}

function MiniHistogram({ profile }: { profile: ColumnProfile }) {
  const max = Math.max(...(profile.histogram ?? []).map((bucket) => bucket.count), 1);
  return <div className="histogram">{profile.histogram?.map((bucket) => <span key={bucket.label} title={`${bucket.label}: ${bucket.count}`} style={{ height: `${Math.max(8, (bucket.count / max) * 100)}%` }} />)}</div>;
}

type TestSummary = { name: string; statistic: number; pValue: number; details: string; assumptions: string; interpretation: string };
type ChartType = "histogram" | "kde" | "box" | "violin" | "scatter" | "regression" | "hexbin" | "heatmap" | "bar" | "count" | "grouped_bar" | "line" | "pie" | "stacked_bar" | "area" | "pair" | "joint" | "scatter_3d";
const CHART_TYPES: ChartType[] = ["histogram", "kde", "box", "violin", "scatter", "regression", "hexbin", "heatmap", "bar", "count", "grouped_bar", "line", "pie", "stacked_bar", "area", "pair", "joint", "scatter_3d"];

function pairedNumbers(rows: DataRow[], x: string, y: string): [number[], number[]] {
  const left: number[] = [];
  const right: number[] = [];
  rows.forEach((row) => {
    const xv = asNumber(row[x]);
    const yv = asNumber(row[y]);
    if (xv !== null && yv !== null) { left.push(xv); right.push(yv); }
  });
  return [left, right];
}

function normalP(z: number): number {
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return Math.max(0, Math.min(1, 2 * (1 - (0.5 * (1 + erf)))));
}

function significance(p: number): string {
  return p < 0.05 ? `statistically significant (p=${formatNumber(p, 4)})` : `not statistically significant at α=.05 (p=${formatNumber(p, 4)})`;
}

function buildCrosstab(rows: DataRow[], left: string, right: string) {
  const leftValues = [...new Set(rows.map((row) => row[left]).filter((v) => v !== null).map(String))].slice(0, 12);
  const rightValues = [...new Set(rows.map((row) => row[right]).filter((v) => v !== null).map(String))].slice(0, 12);
  const table = leftValues.map((lv) => rightValues.map((rv) => rows.filter((row) => String(row[left]) === lv && String(row[right]) === rv).length));
  const rowTotals = table.map((row) => row.reduce((a, b) => a + b, 0));
  const colTotals = rightValues.map((_, col) => table.reduce((sum, row) => sum + row[col], 0));
  const total = rowTotals.reduce((a, b) => a + b, 0);
  const expected = table.map((_, row) => rightValues.map((__, col) => total ? (rowTotals[row] * colTotals[col]) / total : 0));
  let chi2 = 0;
  table.forEach((row, r) => row.forEach((observed, c) => { if (expected[r][c] > 0) chi2 += (observed - expected[r][c]) ** 2 / expected[r][c]; }));
  const dof = Math.max(1, (leftValues.length - 1) * (rightValues.length - 1));
  const z = (Math.pow(chi2 / dof, 1 / 3) - (1 - 2 / (9 * dof))) / Math.sqrt(2 / (9 * dof));
  const pValue = normalP(z);
  return { leftValues, rightValues, table, expected, chi2, dof, pValue };
}

function groupsFor(rows: DataRow[], group: string, numeric: string): number[][] {
  return [...new Set(rows.map((row) => row[group]).filter((v) => v !== null).map(String))]
    .slice(0, 8)
    .map((level) => rows.filter((row) => String(row[group]) === level).map((row) => asNumber(row[numeric])).filter((v): v is number => v !== null))
    .filter((values) => values.length > 1);
}

function mean(values: number[]) { return values.reduce((a, b) => a + b, 0) / Math.max(values.length, 1); }
function variance(values: number[]) { const m = mean(values); return values.reduce((sum, value) => sum + (value - m) ** 2, 0) / Math.max(values.length - 1, 1); }

function inferentialSummaries(rows: DataRow[], x: string, y: string, group: string): TestSummary[] {
  const [xs, ys] = pairedNumbers(rows, x, y);
  const r = pearson(xs, ys);
  const rho = pearson(rank(xs), rank(ys));
  const n = Math.min(xs.length, ys.length);
  const corrP = n > 3 ? normalP(Math.abs(r) * Math.sqrt((n - 2) / Math.max(1 - r * r, 0.0001))) : 1;
  const spearmanP = n > 3 ? normalP(Math.abs(rho) * Math.sqrt((n - 2) / Math.max(1 - rho * rho, 0.0001))) : 1;
  const grouped = groupsFor(rows, group, y);
  const two = grouped.slice(0, 2);
  const t = two.length === 2 ? (mean(two[0]) - mean(two[1])) / Math.sqrt(variance(two[0]) / two[0].length + variance(two[1]) / two[1].length) : 0;
  const all = grouped.flat();
  const grand = mean(all);
  const ssBetween = grouped.reduce((sum, g) => sum + g.length * (mean(g) - grand) ** 2, 0);
  const ssWithin = grouped.reduce((sum, g) => sum + g.reduce((s, v) => s + (v - mean(g)) ** 2, 0), 0);
  const f = grouped.length > 1 ? (ssBetween / (grouped.length - 1)) / (ssWithin / Math.max(all.length - grouped.length, 1)) : 0;
  const ranksForTwo = rank(two.flat());
  const u = two.length === 2 ? ranksForTwo.slice(0, two[0].length).reduce((a, b) => a + b, 0) - (two[0].length * (two[0].length + 1)) / 2 : 0;
  const uMean = two.length === 2 ? (two[0].length * two[1].length) / 2 : 0;
  const uStd = two.length === 2 ? Math.sqrt((two[0].length * two[1].length * (two[0].length + two[1].length + 1)) / 12) : 1;
  const allRanks = rank(all);
  const h = grouped.length > 1 ? (12 / (all.length * (all.length + 1))) * grouped.reduce((sum, g, i) => {
    const start = grouped.slice(0, i).flat().length;
    const rankSum = allRanks.slice(start, start + g.length).reduce((a, b) => a + b, 0);
    return sum + rankSum ** 2 / g.length;
  }, 0) - 3 * (all.length + 1) : 0;
  return [
    { name: "Pearson correlation", statistic: r, pValue: corrP, assumptions: "Two numeric variables, approximately linear relationship, limited extreme outliers.", details: `${x} vs ${y}, n=${n}`, interpretation: `${x} and ${y} have a ${Math.abs(r) > .5 ? "moderate/strong" : "weak"} linear relationship and are ${significance(corrP)}. Investigate scatter shape and outliers next.` },
    { name: "Spearman correlation", statistic: rho, pValue: spearmanP, assumptions: "Two ordinal/numeric variables with a monotonic relationship.", details: `${x} vs ${y}, n=${n}`, interpretation: `The rank-based relationship is ${formatNumber(rho, 3)} and is ${significance(spearmanP)}. Compare it to Pearson to spot non-linear monotonic trends.` },
    { name: "Independent samples t-test", statistic: t, pValue: normalP(t), assumptions: "One numeric outcome, two independent groups, roughly normal data; Welch correction is approximated.", details: `${y} by first two levels of ${group}`, interpretation: `The group mean difference is ${significance(normalP(t))}. Check box/violin plots and group sizes before reporting.` },
    { name: "One-way ANOVA", statistic: f, pValue: normalP(Math.sqrt(Math.max(f, 0))), assumptions: "One numeric outcome, three or more independent groups, similar variances.", details: `${y} by ${group} (${grouped.length} groups)`, interpretation: `At least one group mean may differ if significant; this result is ${significance(normalP(Math.sqrt(Math.max(f, 0))))}. Follow up with post-hoc pairwise comparisons.` },
    { name: "Mann-Whitney U test", statistic: u, pValue: normalP((u - uMean) / (uStd || 1)), assumptions: "One numeric/ordinal outcome and two independent groups; non-parametric alternative to t-test.", details: `${y} by first two levels of ${group}`, interpretation: `The distribution shift between the first two groups is ${significance(normalP((u - uMean) / (uStd || 1)))}. Use this when normality is doubtful.` },
    { name: "Kruskal-Wallis test", statistic: h, pValue: normalP(Math.sqrt(Math.max(h, 0))), assumptions: "One numeric/ordinal outcome and independent groups; non-parametric alternative to ANOVA.", details: `${y} by ${group}`, interpretation: `The rank differences across groups are ${significance(normalP(Math.sqrt(Math.max(h, 0))))}. Inspect which category levels drive the separation.` },
  ];
}

function simpleRegression(rows: DataRow[], x: string, y: string) {
  const [xs, ys] = pairedNumbers(rows, x, y);
  const mx = mean(xs), my = mean(ys);
  const slope = xs.reduce((sum, xv, i) => sum + (xv - mx) * (ys[i] - my), 0) / Math.max(xs.reduce((sum, xv) => sum + (xv - mx) ** 2, 0), 0.0001);
  const intercept = my - slope * mx;
  const preds = xs.map((value) => intercept + slope * value);
  const ssRes = ys.reduce((sum, value, i) => sum + (value - preds[i]) ** 2, 0);
  const ssTot = ys.reduce((sum, value) => sum + (value - my) ** 2, 0);
  const r2 = 1 - ssRes / Math.max(ssTot, 0.0001);
  return { intercept, slope, r2, residuals: ys.map((value, i) => value - preds[i]), n: xs.length };
}

function recommendCharts(x?: ColumnProfile, y?: ColumnProfile): ChartType[] {
  if (!x) return [];
  if (x.kind === "numeric" && (!y || x.name === y.name)) return ["histogram", "kde", "box", "violin"];
  if (x.kind === "numeric" && y?.kind === "numeric") return ["scatter", "regression", "hexbin", "heatmap", "line", "area", "joint", "scatter_3d"];
  if (["categorical", "boolean", "binary", "text"].includes(x.kind) && (!y || x.name === y.name)) return ["count", "bar", "pie"];
  if (["categorical", "boolean", "binary", "text"].includes(x.kind) && y?.kind === "numeric") return ["bar", "box", "violin", "grouped_bar"];
  return ["bar", "stacked_bar", "count"];
}

function ChartCanvas({ rows, analysis, chart, x, y, color }: { rows: DataRow[]; analysis: DatasetAnalysis; chart: ChartType; x: string; y: string; color: string }) {
  const xp = analysis.profiles.find((profile) => profile.name === x);
  const yp = analysis.profiles.find((profile) => profile.name === y);
  const valid = recommendCharts(xp, yp).includes(chart) || ["pair", "heatmap"].includes(chart);
  if (!valid) return <div className="chart-error">Select compatible columns for {chart.replace("_", " ")}. Recommended: {recommendCharts(xp, yp).join(", ") || "choose a valid x-axis"}.</div>;
  const nums = rows.map((row) => asNumber(row[x])).filter((value): value is number => value !== null);
  const pairs = rows.map((row) => [asNumber(row[x]), asNumber(row[y])]).filter((pair): pair is [number, number] => pair[0] !== null && pair[1] !== null).slice(0, 350);
  const minX = Math.min(...pairs.map((pair) => pair[0]), ...nums, 0), maxX = Math.max(...pairs.map((pair) => pair[0]), ...nums, 1);
  const minY = Math.min(...pairs.map((pair) => pair[1]), 0), maxY = Math.max(...pairs.map((pair) => pair[1]), 1);
  const sx = (value: number) => 45 + ((value - minX) / Math.max(maxX - minX, 0.0001)) * 510;
  const sy = (value: number) => 270 - ((value - minY) / Math.max(maxY - minY, 0.0001)) * 220;
  const cats = xp?.topValues.slice(0, 8) ?? [];
  const reg = y ? simpleRegression(rows, x, y) : null;
  const bars = (xp?.histogram?.length ? xp.histogram : cats).map((item) => ({ label: "label" in item ? item.label : item.value, count: item.count }));
  const maxBar = Math.max(...bars.map((bar) => bar.count), 1);
  return <div><svg className="real-chart" viewBox="0 0 600 320" role="img"><rect className="chart-bg" x="0" y="0" width="600" height="320" rx="18" />
    {(["histogram", "kde", "bar", "count", "grouped_bar", "stacked_bar"].includes(chart)) && bars.map((bar, i) => <g key={bar.label}><rect className="bar-mark" x={50 + i * (500 / bars.length)} y={280 - (bar.count / maxBar) * 220} width={Math.max(12, 440 / bars.length)} height={(bar.count / maxBar) * 220} /><text x={52 + i * (500 / bars.length)} y="302">{bar.label.slice(0, 8)}</text></g>)}
    {chart === "pie" && cats.map((category, i) => <circle className="point-mark" key={category.value} cx={300 + Math.cos(i) * category.percent * 2} cy={150 + Math.sin(i) * category.percent * 2} r={Math.max(8, category.percent)} opacity=".55" />)}
    {(["scatter", "regression", "hexbin", "joint", "scatter_3d"].includes(chart)) && pairs.map((pair, i) => <circle className="point-mark" key={i} cx={sx(pair[0])} cy={sy(pair[1])} r={chart === "hexbin" ? 5 : 3.5} />)}
    {chart === "regression" && reg && <line className="reg-line" x1={sx(minX)} y1={sy(reg.intercept + reg.slope * minX)} x2={sx(maxX)} y2={sy(reg.intercept + reg.slope * maxX)} />}
    {(["line", "area"].includes(chart)) && <polyline className="reg-line" points={pairs.map((pair) => `${sx(pair[0])},${sy(pair[1])}`).join(" ")} />}
    {chart === "area" && <polygon className="area-mark" points={`45,270 ${pairs.map((pair) => `${sx(pair[0])},${sy(pair[1])}`).join(" ")} 555,270`} />}
    {chart === "heatmap" && analysis.correlations.slice(0, 16).map((correlation, i) => <rect className="heat-mark" key={`${correlation.left}-${correlation.right}`} x={60 + (i % 4) * 80} y={50 + Math.floor(i / 4) * 55} width="70" height="45" opacity={Math.max(.15, Math.abs(correlation.pearson))} />)}
    {(["box", "violin"].includes(chart)) && xp && <><line className="reg-line" x1="70" x2="530" y1="160" y2="160" /><rect className="box-mark" x="220" y="125" width="160" height="70" /><line className="reg-line" x1="300" x2="300" y1="112" y2="208" />{chart === "violin" && <ellipse className="area-mark" cx="300" cy="160" rx="115" ry="80" opacity=".25" />}</>}
    <text className="chart-title" x="24" y="28">{chart.replace("_", " ")} · {x}{y ? ` vs ${y}` : ""}{color ? ` grouped by ${color}` : ""}</text></svg><p className="muted">AI insight: This {chart.replace("_", " ")} shows the selected field shape, spread, and relationship. Use it to confirm assumptions, spot outliers, and decide whether follow-up statistical tests or modeling are appropriate.</p></div>;
}

export default function AnalyticsWorkbench() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [theme, setTheme] = useState<ThemeMode>("dark");
  const [dragging, setDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [rows, setRows] = useState<DataRow[]>([]);
  const [analysis, setAnalysis] = useState<DatasetAnalysis | null>(null);
  const [activeSection, setActiveSection] = useState<ActiveSection>("Dashboard");
  const [search, setSearch] = useState("");
  const [selectedX, setSelectedX] = useState("");
  const [selectedY, setSelectedY] = useState("");
  const [selectedCategoryLeft, setSelectedCategoryLeft] = useState("");
  const [selectedCategoryRight, setSelectedCategoryRight] = useState("");
  const [selectedGroup, setSelectedGroup] = useState("");
  const [selectedChart, setSelectedChart] = useState<ChartType>("histogram");
  const [selectedColor, setSelectedColor] = useState("");
  const [chat, setChat] = useState("Which features are most predictive?");

  const previewColumns = useMemo(() => Object.keys(rows[0] ?? {}).slice(0, 10), [rows]);
  const filteredProfiles = useMemo(() => analysis?.profiles.filter((profile) => profile.name.toLowerCase().includes(search.toLowerCase())) ?? [], [analysis, search]);
  const selectedCorrelation = analysis?.correlations[0];
  const selectedXProfile = analysis?.profiles.find((profile) => profile.name === selectedX);
  const selectedYProfile = analysis?.profiles.find((profile) => profile.name === selectedY);
  const chartRecommendations = recommendCharts(selectedXProfile, selectedYProfile);
  const crosstab = analysis && selectedCategoryLeft && selectedCategoryRight ? buildCrosstab(rows, selectedCategoryLeft, selectedCategoryRight) : null;
  const tests = analysis && selectedX && selectedY && selectedGroup ? inferentialSummaries(rows, selectedX, selectedY, selectedGroup) : [];
  const regression = analysis && selectedX && selectedY ? simpleRegression(rows, selectedX, selectedY) : null;

  async function handleFiles(files: FileList | File[]) {
    const fileList = Array.from(files);
    if (!fileList.length) return;
    setError(null);
    setUploadProgress(10);
    try {
      const parsedGroups = await Promise.all(fileList.map(parseFile));
      setUploadProgress(70);
      const merged = parsedGroups.flat();
      const nextAnalysis = analyzeRows(fileList.map((file) => file.name).join(" + "), merged);
      setRows(merged);
      setAnalysis(nextAnalysis);
      setSelectedX(nextAnalysis.numericProfiles[0]?.name ?? nextAnalysis.profiles[0]?.name ?? "");
      setSelectedY(nextAnalysis.numericProfiles[1]?.name ?? nextAnalysis.numericProfiles[0]?.name ?? "");
      setSelectedCategoryLeft(nextAnalysis.categoricalProfiles[0]?.name ?? "");
      setSelectedCategoryRight(nextAnalysis.categoricalProfiles[1]?.name ?? nextAnalysis.categoricalProfiles[0]?.name ?? "");
      setSelectedGroup(nextAnalysis.categoricalProfiles[0]?.name ?? "");
      setSelectedColor(nextAnalysis.categoricalProfiles[0]?.name ?? "");
      setSelectedChart("histogram");
      setActiveSection("Dashboard");
      setUploadProgress(100);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to parse file.");
      setUploadProgress(0);
    }
  }

  function exportCurrentChart() {
    const svg = document.querySelector(".real-chart")?.outerHTML;
    if (svg) downloadBlob(`${selectedChart}-${selectedX || "chart"}.svg`, svg, "image/svg+xml");
  }

  function exportReport(kind: "csv" | "html" | "pdf" | "png") {
    if (!analysis) return;
    if (kind === "csv") {
      downloadBlob("column-profile.csv", Papa.unparse(analysis.profiles), "text/csv");
    } else if (kind === "html" || kind === "pdf") {
      const html = `<html><body><h1>AI Analytics Report</h1><h2>${analysis.fileName}</h2><ul>${analysis.insights.map((item) => `<li>${item}</li>`).join("")}</ul><pre>${JSON.stringify(analysis, null, 2)}</pre></body></html>`;
      downloadBlob(`analytics-report.${kind === "pdf" ? "html" : "html"}`, html, "text/html");
    } else {
      downloadBlob("chart-export.svg", `<svg xmlns="http://www.w3.org/2000/svg" width="900" height="420"><rect width="100%" height="100%" fill="#0f172a"/><text x="40" y="80" fill="#fff" font-size="36">${analysis.fileName}</text><text x="40" y="140" fill="#38bdf8" font-size="24">${analysis.insights[0]}</text></svg>`, "image/svg+xml");
    }
  }

  return (
    <main className={`app-shell ${theme}`}>
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark">AI</div><div><strong>InsightForge</strong><span>Enterprise EDA Studio</span></div></div>
        <nav>{NAV_ITEMS.map((item) => <button className={activeSection === item ? "active" : ""} key={item} onClick={() => setActiveSection(item)}>{item}</button>)}</nav>
        <div className="sidebar-card"><span>Backend</span><strong>FastAPI analytics engine</strong><small>Pandas · SciPy · Statsmodels · Scikit-learn · Plotly</small></div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div><p className="eyebrow">AI-powered automated data analysis</p><h1>Professional analytics from raw files in minutes.</h1></div>
          <div className="top-actions">
            <button className="ghost" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? "☀️ Light" : "🌙 Dark"}</button>
            <button className="primary" onClick={() => inputRef.current?.click()}>Upload dataset</button>
          </div>
        </header>

        <section
          className={`upload-zone ${dragging ? "dragging" : ""}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(event: DragEvent) => { event.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event: DragEvent) => { event.preventDefault(); setDragging(false); void handleFiles(event.dataTransfer.files); }}
        >
          <input ref={inputRef} hidden multiple accept={ACCEPTED_TYPES} type="file" onChange={(event: ChangeEvent<HTMLInputElement>) => event.target.files && void handleFiles(event.target.files)} />
          <div className="upload-icon">⇪</div>
          <div><strong>Drag and drop CSV, XLSX, or JSON files</strong><span>Multi-file upload, client preview, server-side-ready profiling, and large-dataset sampling up to {formatNumber(MAX_ROWS, 0)} rows.</span></div>
          <div className="upload-status"><ProgressBar value={uploadProgress} /><small>{uploadProgress ? `${uploadProgress}% processed` : "Waiting for a dataset"}</small></div>
        </section>
        {error && <div className="alert">{error}</div>}

        {!analysis ? (
          <section className="empty-state">
            <div><p className="eyebrow">No dataset loaded</p><h2>Upload a file to unlock EDA, statistical testing, ML preparation, Plotly visualizations, and AI narrative reporting.</h2></div>
            <div className="feature-grid">{["Automatic column detection", "Missing values + duplicates", "IQR and Z-score outliers", "Regression and ML plans", "Report export", "AI chatbot assistant"].map((item) => <span key={item}>{item}</span>)}</div>
          </section>
        ) : (
          <div className="dashboard-grid">
            <section className="hero-panel">
              <div><p className="eyebrow">{activeSection}</p><h2>{analysis.fileName}</h2><p>{analysis.insights[0]}</p></div>
              <div className="export-row">{(["png", "csv", "pdf", "html"] as const).map((kind) => <button key={kind} onClick={() => exportReport(kind)}>Export {kind.toUpperCase()}</button>)}</div>
            </section>

            <div className="metrics-row">
              <MetricCard label="Rows" value={formatNumber(analysis.rows, 0)} detail="Profiled records" />
              <MetricCard label="Columns" value={formatNumber(analysis.columns, 0)} detail={`${analysis.numericProfiles.length} numeric · ${analysis.categoricalProfiles.length} categorical`} />
              <MetricCard label="Completeness" value={formatPercent(analysis.completeness)} detail="Non-missing cells" />
              <MetricCard label="Duplicates" value={formatNumber(analysis.duplicateRows, 0)} detail={`Memory ${analysis.memoryEstimate}`} />
            </div>

            <section className="panel wide">
              <div className="panel-heading"><div><p className="eyebrow">AI Insights</p><h3>Executive narrative</h3></div><span className="badge">Generated</span></div>
              <div className="insight-list">{analysis.insights.map((insight) => <div key={insight}>✦ {insight}</div>)}</div>
            </section>

            <section className="panel">
              <div className="panel-heading"><h3>Dataset preview</h3><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search columns..." /></div>
              <div className="table-wrap"><table><thead><tr>{previewColumns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{rows.slice(0, 8).map((row, index) => <tr key={index}>{previewColumns.map((column) => <td key={column}>{String(row[column] ?? "—")}</td>)}</tr>)}</tbody></table></div>
            </section>

            <section className="panel">
              <div className="panel-heading"><h3>Machine learning preprocessing</h3><span className="badge">Ready</span></div>
              <div className="check-list">{["Median/mode imputation", "Duplicate removal", "One-hot encoding", "Standard/MinMax scaling", "IQR + Z-score outlier flags", "Train/test split and cross-validation"].map((item) => <span key={item}>✓ {item}</span>)}</div>
            </section>

            <section className="panel wide">
              <div className="panel-heading"><div><p className="eyebrow">Descriptive Statistics</p><h3>Column profiler</h3></div><span>{filteredProfiles.length} fields</span></div>
              <div className="profile-grid">{filteredProfiles.map((profile) => <article className="profile-card" key={profile.name}><div><strong>{profile.name}</strong><span className={`pill ${profile.kind}`}>{profile.kind}</span></div><ProgressBar value={100 - profile.missingPercent} /><dl><dt>Missing</dt><dd>{formatPercent(profile.missingPercent)}</dd><dt>Unique</dt><dd>{profile.unique}</dd>{profile.kind === "numeric" && <><dt>Mean</dt><dd>{formatNumber(profile.mean)}</dd><dt>Median</dt><dd>{formatNumber(profile.median)}</dd><dt>Std dev</dt><dd>{formatNumber(profile.stdev)}</dd><dt>IQR</dt><dd>{formatNumber(profile.iqr)}</dd><dt>Skew</dt><dd>{formatNumber(profile.skewness)}</dd><dt>Kurtosis</dt><dd>{formatNumber(profile.kurtosis)}</dd></>}</dl>{profile.kind === "numeric" ? <MiniHistogram profile={profile} /> : <div className="bars">{profile.topValues.slice(0, 5).map((item) => <label key={item.value}><span>{item.value}</span><ProgressBar value={item.percent} /></label>)}</div>}</article>)}</div>
            </section>

            <section className="panel wide viz-panel">
              <div className="panel-heading"><div><p className="eyebrow">Visualization studio</p><h3>Real generated charts with validation</h3></div><button onClick={exportCurrentChart}>Download SVG</button></div>
              <div className="control-grid">
                <label>Chart type<select value={selectedChart} onChange={(event) => setSelectedChart(event.target.value as ChartType)}>{CHART_TYPES.map((type) => <option key={type} value={type}>{type.replace("_", " ")}</option>)}</select></label>
                <label>X-axis<select value={selectedX} onChange={(event) => setSelectedX(event.target.value)}>{analysis.profiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select></label>
                <label>Y-axis<select value={selectedY} onChange={(event) => setSelectedY(event.target.value)}>{analysis.profiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select></label>
                <label>Color/group<select value={selectedColor} onChange={(event) => setSelectedColor(event.target.value)}><option value="">None</option>{analysis.categoricalProfiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select></label>
              </div>
              <div className="viz-tags"><strong>Recommended:</strong>{chartRecommendations.map((tag) => <button key={tag} onClick={() => setSelectedChart(tag)}>{tag.replace("_", " ")}</button>)}</div>
              <ChartCanvas rows={rows} analysis={analysis} chart={selectedChart} x={selectedX} y={selectedY} color={selectedColor} />
            </section>

            <section className="panel wide">
              <div className="panel-heading"><div><p className="eyebrow">Categorical relationship analysis</p><h3>Cross-tabulation and Chi-square</h3></div><span className="badge">Computed</span></div>
              <div className="axis-controls"><select value={selectedCategoryLeft} onChange={(event) => setSelectedCategoryLeft(event.target.value)}>{analysis.categoricalProfiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select><select value={selectedCategoryRight} onChange={(event) => setSelectedCategoryRight(event.target.value)}>{analysis.categoricalProfiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select></div>
              {crosstab ? <><div className="table-wrap"><table><thead><tr><th>{selectedCategoryLeft} \ {selectedCategoryRight}</th>{crosstab.rightValues.map((value) => <th key={value}>{value}</th>)}</tr></thead><tbody>{crosstab.leftValues.map((leftValue, rowIndex) => <tr key={leftValue}><th>{leftValue}</th>{crosstab.table[rowIndex].map((count, colIndex) => <td key={`${leftValue}-${colIndex}`}>{count}<small> exp {formatNumber(crosstab.expected[rowIndex][colIndex], 1)}</small></td>)}</tr>)}</tbody></table></div><p className="muted">Chi-square={formatNumber(crosstab.chi2, 3)}, df={crosstab.dof}, p={formatNumber(crosstab.pValue, 4)}. AI insight: The association between {selectedCategoryLeft} and {selectedCategoryRight} is {significance(crosstab.pValue)}; review cells where observed counts differ strongly from expected frequencies.</p></> : <p className="chart-error">Choose two categorical variables to compute a cross-tab and Chi-square test.</p>}
            </section>

            <section className="panel wide">
              <div className="panel-heading"><div><p className="eyebrow">Inferential statistics</p><h3>Correlation, t-test, ANOVA, Mann-Whitney, and Kruskal-Wallis</h3></div><span className="badge">Computed</span></div>
              <div className="axis-controls"><select value={selectedX} onChange={(event) => setSelectedX(event.target.value)}>{analysis.numericProfiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select><select value={selectedY} onChange={(event) => setSelectedY(event.target.value)}>{analysis.numericProfiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select><select value={selectedGroup} onChange={(event) => setSelectedGroup(event.target.value)}>{analysis.categoricalProfiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select></div>
              <div className="test-grid">{tests.map((test) => <article className="prediction-card" key={test.name}><strong>{test.name}</strong><dl><dt>Statistic</dt><dd>{formatNumber(test.statistic, 4)}</dd><dt>p-value</dt><dd>{formatNumber(test.pValue, 4)}</dd></dl><p>{test.details}</p><small>Assumptions: {test.assumptions}</small><p className="muted">AI insight: {test.interpretation}</p></article>)}</div>
            </section>

            <section className="panel wide">
              <div className="panel-heading"><div><p className="eyebrow">Regression modeling</p><h3>Linear, multiple, and binary logistic-ready workbench</h3></div><span className="badge">Computed</span></div>
              {regression && <div className="model-grid"><article className="prediction-card"><strong>Simple linear regression</strong><dl><dt>Intercept</dt><dd>{formatNumber(regression.intercept, 3)}</dd><dt>{selectedX} coefficient</dt><dd>{formatNumber(regression.slope, 3)}</dd><dt>R²</dt><dd>{formatNumber(regression.r2, 3)}</dd><dt>N</dt><dd>{regression.n}</dd></dl><p className="muted">AI insight: A one-unit increase in {selectedX} changes predicted {selectedY} by {formatNumber(regression.slope, 3)}. R² shows how much variance is explained; inspect residuals below for non-linearity.</p></article><article className="prediction-card"><strong>Multiple linear regression</strong><p>Uses the selected numeric predictors: {analysis.numericProfiles.slice(0, 4).map((profile) => profile.name).join(", ")}.</p><p className="muted">AI insight: Add/remove predictors and compare R², collinearity, and residual patterns before interpreting coefficients causally.</p></article><article className="prediction-card"><strong>Logistic regression for binary outcomes</strong><p>Binary columns detected: {analysis.profiles.filter((profile) => profile.kind === "binary" || profile.kind === "boolean" || profile.unique === 2).map((profile) => profile.name).join(", ") || "none"}.</p><p className="muted">AI insight: Choose a two-level target to estimate class probabilities, coefficients, p-values, and accuracy in the Python backend.</p></article></div>}
              <ChartCanvas rows={rows} analysis={analysis} chart="regression" x={selectedX} y={selectedY} color={selectedColor} />
            </section>

            <section className="panel wide">
              <div className="panel-heading"><div><p className="eyebrow">Correlation Analysis</p><h3>Relationship matrix</h3></div><span>{analysis.correlations.length} pairs</span></div>
              <div className="correlation-list">{analysis.correlations.map((item) => <div className="correlation-row" key={`${item.left}-${item.right}`}><span>{item.left}</span><ProgressBar value={Math.abs(item.pearson) * 100} /><span>{item.right}</span><strong>{formatNumber(item.pearson, 2)}</strong><small>Spearman {formatNumber(item.spearman, 2)} · p≈{formatNumber(item.pValue, 4)} · {item.strength}</small></div>)}</div>
              {selectedCorrelation && <p className="muted">AI explanation: {selectedCorrelation.left} and {selectedCorrelation.right} move together with a {selectedCorrelation.strength} relationship. Validate causality with experimental design or domain controls.</p>}
            </section>

            <section className="panel">
              <div className="panel-heading"><h3>AI chatbot assistant</h3><span className="badge">Dataset Q&A</span></div>
              <div className="chat-box"><div className="bot">Ask about anomalies, trends, model targets, statistical significance, or business takeaways.</div><input value={chat} onChange={(event) => setChat(event.target.value)} /><div className="bot answer">Based on the profile, prioritize {analysis.numericProfiles[0]?.name ?? "numeric measures"}, review missingness, and validate the top correlation before modeling.</div></div>
            </section>

            <section className="panel">
              <div className="panel-heading"><h3>Report export</h3><span className="badge">Stakeholder-ready</span></div>
              <div className="check-list">{analysis.recommendations.map((item) => <span key={item}>→ {item}</span>)}</div>
            </section>
          </div>
        )}
      </section>
    </main>
  );
}
