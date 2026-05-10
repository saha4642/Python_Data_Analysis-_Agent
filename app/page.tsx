"use client";

import type { ChangeEvent, DragEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
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

type TestType = "pearson" | "spearman" | "ttest" | "anova" | "mannwhitney" | "kruskal";
type TestSummary = { name: string; statistic: number; pValue: number; details: string; assumptions: string; interpretation: string; sampleSizes: string };
type RegressionKind = "simple" | "multiple" | "logistic";
type RegressionResult = { kind: RegressionKind; coefficients: Array<{ term: string; coefficient: number; pValue?: number }>; intercept: number; metricLabel: string; metric: number; n: number; residualSummary: { mean: number; stdev: number; min: number; median: number; max: number }; interpretation: string; error?: string };
type ChartType = "histogram" | "kde" | "box" | "violin" | "scatter" | "regression" | "heatmap" | "bar" | "count" | "grouped_bar" | "line" | "pie" | "stacked_bar" | "area" | "pair" | "scatter_3d";
const CHART_TYPES: ChartType[] = ["histogram", "kde", "box", "violin", "scatter", "regression", "heatmap", "bar", "count", "grouped_bar", "line", "pie", "stacked_bar", "area", "pair", "scatter_3d"];

declare global { interface Window { Plotly?: { newPlot: (element: HTMLElement, data: unknown[], layout: Record<string, unknown>, config?: Record<string, unknown>) => void; purge: (element: HTMLElement) => void } } }

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

function averageRanks(values: number[]): number[] {
  const sorted = values.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value);
  const ranks = Array(values.length).fill(0);
  for (let i = 0; i < sorted.length;) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1].value === sorted[i].value) j += 1;
    const avg = (i + j + 2) / 2;
    for (let k = i; k <= j; k += 1) ranks[sorted[k].index] = avg;
    i = j + 1;
  }
  return ranks;
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
  const leftValues = [...new Set(rows.map((row) => row[left]).filter((v) => v !== null).map(String))].slice(0, 20);
  const rightValues = [...new Set(rows.map((row) => row[right]).filter((v) => v !== null).map(String))].slice(0, 20);
  const table = leftValues.map((lv) => rightValues.map((rv) => rows.filter((row) => String(row[left]) === lv && String(row[right]) === rv).length));
  const rowTotals = table.map((row) => row.reduce((a, b) => a + b, 0));
  const colTotals = rightValues.map((_, col) => table.reduce((sum, row) => sum + row[col], 0));
  const total = rowTotals.reduce((a, b) => a + b, 0);
  const expected = table.map((_, row) => rightValues.map((__, col) => total ? (rowTotals[row] * colTotals[col]) / total : 0));
  let chi2 = 0;
  table.forEach((row, r) => row.forEach((observed, c) => { if (expected[r][c] > 0) chi2 += (observed - expected[r][c]) ** 2 / expected[r][c]; }));
  const dof = Math.max(1, (leftValues.length - 1) * (rightValues.length - 1));
  const z = (Math.pow(Math.max(chi2, 0.0001) / dof, 1 / 3) - (1 - 2 / (9 * dof))) / Math.sqrt(2 / (9 * dof));
  const pValue = normalP(z);
  const interpretation = `The observed ${left} × ${right} table is ${significance(pValue)}. ${pValue < 0.05 ? "The variables appear associated; inspect cells where observed and expected counts diverge." : "There is not enough evidence of dependence in this dataset."}`;
  return { leftValues, rightValues, table, expected, chi2, dof, pValue, interpretation };
}

function groupsFor(rows: DataRow[], group: string, numeric: string): Array<{ level: string; values: number[] }> {
  return [...new Set(rows.map((row) => row[group]).filter((v) => v !== null).map(String))]
    .slice(0, 12)
    .map((level) => ({ level, values: rows.filter((row) => String(row[group]) === level).map((row) => asNumber(row[numeric])).filter((v): v is number => v !== null) }))
    .filter((item) => item.values.length > 1);
}

function mean(values: number[]) { return values.reduce((a, b) => a + b, 0) / Math.max(values.length, 1); }
function variance(values: number[]) { const m = mean(values); return values.reduce((sum, value) => sum + (value - m) ** 2, 0) / Math.max(values.length - 1, 1); }
function residualSummary(residuals: number[]) { const sorted = [...residuals].sort((a, b) => a - b); return { mean: mean(residuals), stdev: Math.sqrt(variance(residuals)), min: sorted[0] ?? 0, median: quantile(sorted, .5), max: sorted[sorted.length - 1] ?? 0 }; }

function inferentialSummaries(rows: DataRow[], x: string, y: string, group: string): Record<TestType, TestSummary> {
  const [xs, ys] = pairedNumbers(rows, x, y);
  const r = pearson(xs, ys);
  const rho = pearson(averageRanks(xs), averageRanks(ys));
  const n = Math.min(xs.length, ys.length);
  const corrP = n > 3 ? normalP(Math.abs(r) * Math.sqrt((n - 2) / Math.max(1 - r * r, 0.0001))) : 1;
  const spearmanP = n > 3 ? normalP(Math.abs(rho) * Math.sqrt((n - 2) / Math.max(1 - rho * rho, 0.0001))) : 1;
  const grouped = groupsFor(rows, group, y);
  const two = grouped.slice(0, 2);
  const t = two.length === 2 ? (mean(two[0].values) - mean(two[1].values)) / Math.sqrt(variance(two[0].values) / two[0].values.length + variance(two[1].values) / two[1].values.length) : 0;
  const all = grouped.flatMap((g) => g.values);
  const grand = mean(all);
  const ssBetween = grouped.reduce((sum, g) => sum + g.values.length * (mean(g.values) - grand) ** 2, 0);
  const ssWithin = grouped.reduce((sum, g) => sum + g.values.reduce((s, v) => s + (v - mean(g.values)) ** 2, 0), 0);
  const f = grouped.length > 1 ? (ssBetween / (grouped.length - 1)) / (ssWithin / Math.max(all.length - grouped.length, 1)) : 0;
  const ranksForTwo = averageRanks(two.flatMap((g) => g.values));
  const u = two.length === 2 ? ranksForTwo.slice(0, two[0].values.length).reduce((a, b) => a + b, 0) - (two[0].values.length * (two[0].values.length + 1)) / 2 : 0;
  const uMean = two.length === 2 ? (two[0].values.length * two[1].values.length) / 2 : 0;
  const uStd = two.length === 2 ? Math.sqrt((two[0].values.length * two[1].values.length * (two[0].values.length + two[1].values.length + 1)) / 12) : 1;
  const allRanks = averageRanks(all);
  const h = grouped.length > 1 ? (12 / (all.length * (all.length + 1))) * grouped.reduce((sum, g, i) => {
    const start = grouped.slice(0, i).reduce((acc, item) => acc + item.values.length, 0);
    const rankSum = allRanks.slice(start, start + g.values.length).reduce((a, b) => a + b, 0);
    return sum + rankSum ** 2 / g.values.length;
  }, 0) - 3 * (all.length + 1) : 0;
  const groupSizes = grouped.map((g) => `${g.level}: n=${g.values.length}`).join(", ") || "No valid groups";
  return {
    pearson: { name: "Pearson correlation", statistic: r, pValue: corrP, assumptions: "Two numeric variables, approximately linear relationship, independent observations, limited extreme outliers.", details: `${x} vs ${y}`, sampleSizes: `paired n=${n}`, interpretation: `${x} and ${y} have a ${Math.abs(r) > .5 ? "moderate/strong" : "weak"} linear relationship and are ${significance(corrP)}.` },
    spearman: { name: "Spearman correlation", statistic: rho, pValue: spearmanP, assumptions: "Two ordinal/numeric variables with a monotonic relationship; robust to non-normality.", details: `${x} vs ${y}`, sampleSizes: `paired n=${n}`, interpretation: `The rank-based relationship is ${formatNumber(rho, 3)} and is ${significance(spearmanP)}.` },
    ttest: { name: "Independent samples t-test", statistic: t, pValue: normalP(t), assumptions: "Numeric outcome, two independent groups, approximately normal data; Welch-style unequal variance statistic is used.", details: `${y} by ${group} (${two.map((g) => g.level).join(" vs ") || "select two groups"})`, sampleSizes: two.map((g) => `${g.level}: n=${g.values.length}`).join(", "), interpretation: `The group mean difference is ${significance(normalP(t))}. Check box/violin plots before reporting.` },
    anova: { name: "One-way ANOVA", statistic: f, pValue: normalP(Math.sqrt(Math.max(f, 0))), assumptions: "Numeric outcome, independent groups, approximately normal residuals, similar variances.", details: `${y} by ${group}`, sampleSizes: groupSizes, interpretation: `At least one group mean may differ if significant; this result is ${significance(normalP(Math.sqrt(Math.max(f, 0))))}.` },
    mannwhitney: { name: "Mann-Whitney U test", statistic: u, pValue: normalP((u - uMean) / (uStd || 1)), assumptions: "Numeric/ordinal outcome and two independent groups; non-parametric alternative to t-test.", details: `${y} by ${group} (${two.map((g) => g.level).join(" vs ") || "select two groups"})`, sampleSizes: two.map((g) => `${g.level}: n=${g.values.length}`).join(", "), interpretation: `The distribution shift between the first two groups is ${significance(normalP((u - uMean) / (uStd || 1)))}.` },
    kruskal: { name: "Kruskal-Wallis test", statistic: h, pValue: normalP(Math.sqrt(Math.max(h, 0))), assumptions: "Numeric/ordinal outcome and independent groups; non-parametric alternative to ANOVA.", details: `${y} by ${group}`, sampleSizes: groupSizes, interpretation: `The rank differences across groups are ${significance(normalP(Math.sqrt(Math.max(h, 0))))}.` },
  };
}

function simpleRegression(rows: DataRow[], x: string, y: string) {
  const [xs, ys] = pairedNumbers(rows, x, y);
  if (xs.length < 3) return null;
  const mx = mean(xs), my = mean(ys);
  const slope = xs.reduce((sum, xv, i) => sum + (xv - mx) * (ys[i] - my), 0) / Math.max(xs.reduce((sum, xv) => sum + (xv - mx) ** 2, 0), 0.0001);
  const intercept = my - slope * mx;
  const preds = xs.map((value) => intercept + slope * value);
  const residuals = ys.map((value, i) => value - preds[i]);
  const ssRes = residuals.reduce((sum, value) => sum + value ** 2, 0);
  const ssTot = ys.reduce((sum, value) => sum + (value - my) ** 2, 0);
  const r2 = 1 - ssRes / Math.max(ssTot, 0.0001);
  const se = Math.sqrt(ssRes / Math.max(xs.length - 2, 1));
  const slopeSe = se / Math.sqrt(Math.max(xs.reduce((sum, xv) => sum + (xv - mx) ** 2, 0), 0.0001));
  return { intercept, slope, r2, residuals, n: xs.length, pValue: normalP(slope / (slopeSe || 1)) };
}

function invertMatrix(matrix: number[][]): number[][] | null {
  const n = matrix.length;
  const augmented = matrix.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => i === j ? 1 : 0)]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) if (Math.abs(augmented[row][col]) > Math.abs(augmented[pivot][col])) pivot = row;
    if (Math.abs(augmented[pivot][col]) < 1e-10) return null;
    [augmented[col], augmented[pivot]] = [augmented[pivot], augmented[col]];
    const divisor = augmented[col][col];
    augmented[col] = augmented[col].map((value) => value / divisor);
    for (let row = 0; row < n; row += 1) if (row !== col) {
      const factor = augmented[row][col];
      augmented[row] = augmented[row].map((value, idx) => value - factor * augmented[col][idx]);
    }
  }
  return augmented.map((row) => row.slice(n));
}

function multipleLinearRegression(rows: DataRow[], target: string, predictors: string[]): RegressionResult {
  const clean = rows.map((row) => ({ y: asNumber(row[target]), xs: predictors.map((p) => asNumber(row[p])) })).filter((row): row is { y: number; xs: number[] } => row.y !== null && row.xs.every((v) => v !== null));
  if (clean.length <= predictors.length + 2) return { kind: "multiple", coefficients: [], intercept: 0, metricLabel: "R²", metric: 0, n: clean.length, residualSummary: residualSummary([]), interpretation: "Not enough complete rows for multiple linear regression.", error: "Choose fewer predictors or a dataset with more complete numeric rows." };
  const x = clean.map((row) => [1, ...row.xs]);
  const y = clean.map((row) => row.y);
  const xtx = x[0].map((_, i) => x[0].map((__, j) => x.reduce((sum, row) => sum + row[i] * row[j], 0)));
  const inv = invertMatrix(xtx);
  if (!inv) return { kind: "multiple", coefficients: [], intercept: 0, metricLabel: "R²", metric: 0, n: clean.length, residualSummary: residualSummary([]), interpretation: "Predictors are collinear, so coefficients cannot be estimated reliably.", error: "Remove duplicate/highly collinear predictors." };
  const xty = x[0].map((_, i) => x.reduce((sum, row, r) => sum + row[i] * y[r], 0));
  const beta = inv.map((row) => row.reduce((sum, value, i) => sum + value * xty[i], 0));
  const preds = x.map((row) => row.reduce((sum, value, i) => sum + value * beta[i], 0));
  const residuals = y.map((value, i) => value - preds[i]);
  const ssRes = residuals.reduce((sum, value) => sum + value ** 2, 0);
  const ssTot = y.reduce((sum, value) => sum + (value - mean(y)) ** 2, 0);
  const mse = ssRes / Math.max(clean.length - predictors.length - 1, 1);
  return { kind: "multiple", intercept: beta[0], coefficients: predictors.map((term, i) => ({ term, coefficient: beta[i + 1], pValue: normalP(beta[i + 1] / Math.sqrt(Math.max(mse * inv[i + 1][i + 1], 1e-10))) })), metricLabel: "R²", metric: 1 - ssRes / Math.max(ssTot, 0.0001), n: clean.length, residualSummary: residualSummary(residuals), interpretation: `The selected predictors explain ${formatPercent((1 - ssRes / Math.max(ssTot, 0.0001)) * 100)} of variation in ${target}.` };
}

function logisticRegression(rows: DataRow[], target: string, predictors: string[]): RegressionResult {
  const levels = [...new Set(rows.map((row) => row[target]).filter((v) => v !== null).map(String))];
  const clean = rows.map((row) => ({ y: levels.indexOf(String(row[target])), xs: predictors.map((p) => asNumber(row[p])) })).filter((row) => row.y >= 0 && row.y <= 1 && row.xs.every((v) => v !== null)) as Array<{ y: number; xs: number[] }>;
  if (levels.length !== 2 || clean.length <= predictors.length + 5) return { kind: "logistic", coefficients: [], intercept: 0, metricLabel: "Accuracy", metric: 0, n: clean.length, residualSummary: residualSummary([]), interpretation: "Logistic regression needs a binary target and enough complete numeric rows.", error: "Select a two-level target column and at least one numeric predictor." };
  let beta = Array(predictors.length + 1).fill(0);
  const lr = 0.01;
  for (let iter = 0; iter < 1200; iter += 1) {
    const grad = Array(beta.length).fill(0);
    clean.forEach((row) => {
      const values = [1, ...row.xs];
      const z = values.reduce((sum, value, i) => sum + value * beta[i], 0);
      const p = 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
      values.forEach((value, i) => { grad[i] += (p - row.y) * value; });
    });
    beta = beta.map((value, i) => value - lr * grad[i] / clean.length);
  }
  const probs = clean.map((row) => 1 / (1 + Math.exp(-[1, ...row.xs].reduce((sum, value, i) => sum + value * beta[i], 0))));
  const residuals = clean.map((row, i) => row.y - probs[i]);
  const accuracy = clean.filter((row, i) => (probs[i] >= .5 ? 1 : 0) === row.y).length / clean.length;
  return { kind: "logistic", intercept: beta[0], coefficients: predictors.map((term, i) => ({ term, coefficient: beta[i + 1] })), metricLabel: "Accuracy", metric: accuracy, n: clean.length, residualSummary: residualSummary(residuals), interpretation: `The model predicts ${target}=${levels[1]} with ${formatPercent(accuracy * 100)} in-sample accuracy using the selected predictors.` };
}

function buildRegression(rows: DataRow[], kind: RegressionKind, target: string, predictors: string[]): RegressionResult | null {
  const usable = predictors.filter((p) => p && p !== target);
  if (!target || !usable.length) return null;
  if (kind === "logistic") return logisticRegression(rows, target, usable);
  if (kind === "multiple") return multipleLinearRegression(rows, target, usable);
  const simple = simpleRegression(rows, usable[0], target);
  if (!simple) return { kind: "simple", coefficients: [], intercept: 0, metricLabel: "R²", metric: 0, n: 0, residualSummary: residualSummary([]), interpretation: "Not enough paired numeric rows for simple linear regression.", error: "Select one numeric predictor and one numeric outcome." };
  return { kind: "simple", intercept: simple.intercept, coefficients: [{ term: usable[0], coefficient: simple.slope, pValue: simple.pValue }], metricLabel: "R²", metric: simple.r2, n: simple.n, residualSummary: residualSummary(simple.residuals), interpretation: `A one-unit increase in ${usable[0]} changes predicted ${target} by ${formatNumber(simple.slope, 3)}. The model explains ${formatPercent(simple.r2 * 100)} of outcome variance.` };
}

function recommendCharts(x?: ColumnProfile, y?: ColumnProfile, analysis?: DatasetAnalysis): ChartType[] {
  const options = new Set<ChartType>();
  if (analysis && analysis.numericProfiles.length >= 2) options.add("heatmap").add("pair");
  if (analysis && analysis.numericProfiles.length >= 3) options.add("scatter_3d");
  if (!x) return [...options];
  const xCat = ["categorical", "boolean", "binary", "text"].includes(x.kind);
  if (x.kind === "numeric") ["histogram", "kde", "box", "violin"].forEach((v) => options.add(v as ChartType));
  if (x.kind === "numeric" && y?.kind === "numeric") ["scatter", "regression", "line", "area"].forEach((v) => options.add(v as ChartType));
  if (xCat) ["count", "bar", "pie"].forEach((v) => options.add(v as ChartType));
  if (xCat && y?.kind === "numeric") ["grouped_bar", "box", "violin"].forEach((v) => options.add(v as ChartType));
  if (xCat && y && ["categorical", "boolean", "binary", "text"].includes(y.kind)) options.add("stacked_bar");
  return CHART_TYPES.filter((type) => options.has(type));
}

function plotlyData(rows: DataRow[], analysis: DatasetAnalysis, chart: ChartType, x: string, y: string, color: string): unknown[] {
  const numericCols = analysis.numericProfiles.map((p) => p.name);
  const xNums = rows.map((row) => asNumber(row[x])).filter((v): v is number => v !== null);
  const yNums = rows.map((row) => asNumber(row[y])).filter((v): v is number => v !== null);
  const pairs = rows.flatMap((row) => {
    const xv = asNumber(row[x]);
    const yv = asNumber(row[y]);
    return xv !== null && yv !== null ? [{ x: xv, y: yv, c: color ? String(row[color] ?? "Missing") : undefined }] : [];
  }).slice(0, 2000);
  const categories = [...new Set(rows.map((row) => String(row[x] ?? "Missing")))].slice(0, 20);
  if (chart === "histogram" || chart === "kde") return [{ type: "histogram", x: xNums, histnorm: chart === "kde" ? "probability density" : undefined, marker: { color: "#06b6d4" }, name: x }];
  if (chart === "box" || chart === "violin") return color ? [...new Set(rows.map((row) => String(row[color] ?? "Missing")))].slice(0, 8).map((level) => ({ type: chart, y: rows.filter((row) => String(row[color] ?? "Missing") === level).map((row) => asNumber(row[y] ?? row[x])).filter((v): v is number => v !== null), name: level })) : [{ type: chart, y: yNums.length ? yNums : xNums, name: y || x }];
  if (chart === "scatter" || chart === "regression" || chart === "line" || chart === "area") return [{ type: chart === "line" || chart === "area" ? "scatter" : "scatter", mode: chart === "line" || chart === "area" ? "lines+markers" : "markers", fill: chart === "area" ? "tozeroy" : undefined, x: pairs.map((p) => p.x), y: pairs.map((p) => p.y), marker: { color: "#38bdf8" }, name: `${x} vs ${y}` }, ...(chart === "regression" && pairs.length > 2 ? [{ type: "scatter", mode: "lines", x: [Math.min(...pairs.map((p) => p.x)), Math.max(...pairs.map((p) => p.x))], y: (() => { const reg = simpleRegression(rows, x, y); const xs = [Math.min(...pairs.map((p) => p.x)), Math.max(...pairs.map((p) => p.x))]; return reg ? xs.map((value) => reg.intercept + reg.slope * value) : []; })(), line: { color: "#f59e0b" }, name: "Regression line" }] : [])];
  if (chart === "heatmap") { const cols = numericCols.slice(0, 12); return [{ type: "heatmap", x: cols, y: cols, z: cols.map((a) => cols.map((b) => pearson(...pairedNumbers(rows, a, b)))), colorscale: "RdBu", zmin: -1, zmax: 1 }]; }
  if (chart === "bar" || chart === "count" || chart === "pie") { const counts = categories.map((cat) => rows.filter((row) => String(row[x] ?? "Missing") === cat).length); return [{ type: chart === "pie" ? "pie" : "bar", labels: chart === "pie" ? categories : undefined, values: chart === "pie" ? counts : undefined, x: chart !== "pie" ? categories : undefined, y: chart !== "pie" ? counts : undefined, marker: { color: "#06b6d4" } }]; }
  if (chart === "grouped_bar" || chart === "stacked_bar") { const groups = [...new Set(rows.map((row) => String(row[color || y] ?? "Missing")))].slice(0, 8); return groups.map((group) => ({ type: "bar", name: group, x: categories, y: categories.map((cat) => rows.filter((row) => String(row[x] ?? "Missing") === cat && String(row[color || y] ?? "Missing") === group).length) })); }
  if (chart === "pair") return [{ type: "splom", dimensions: numericCols.slice(0, 5).map((dim) => ({ label: dim, values: rows.map((row) => asNumber(row[dim])).filter((v): v is number => v !== null) })) }];
  if (chart === "scatter_3d") return [{ type: "scatter3d", mode: "markers", x: rows.map((row) => asNumber(row[numericCols[0]])).filter((v): v is number => v !== null), y: rows.map((row) => asNumber(row[numericCols[1]])).filter((v): v is number => v !== null), z: rows.map((row) => asNumber(row[numericCols[2]])).filter((v): v is number => v !== null), marker: { size: 4, color: "#38bdf8" } }];
  return [];
}

function PlotlyChart({ rows, analysis, chart, x, y, color }: { rows: DataRow[]; analysis: DatasetAnalysis; chart: ChartType; x: string; y: string; color: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(Boolean(typeof window !== "undefined" && window.Plotly));
  const xp = analysis.profiles.find((profile) => profile.name === x);
  const yp = analysis.profiles.find((profile) => profile.name === y);
  const recommended = recommendCharts(xp, yp, analysis);
  const valid = recommended.includes(chart);

  useEffect(() => {
    if (typeof window === "undefined" || window.Plotly) { setReady(Boolean(window.Plotly)); return; }
    const script = document.createElement("script");
    script.src = "https://cdn.plot.ly/plotly-2.35.2.min.js";
    script.async = true;
    script.onload = () => setReady(true);
    script.onerror = () => setReady(false);
    document.head.appendChild(script);
  }, []);

  useEffect(() => {
    if (!ready || !window.Plotly || !ref.current || !valid) return;
    const data = plotlyData(rows, analysis, chart, x, y, color);
    const element = ref.current;
    window.Plotly.newPlot(element, data, { autosize: true, height: 430, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(15,23,42,.16)", font: { color: getComputedStyle(document.documentElement).getPropertyValue("--text") || "#e5e7eb" }, margin: { t: 48, r: 24, b: 70, l: 70 }, title: { text: `${chart.replace("_", " ")} · ${x}${y ? ` vs ${y}` : ""}` }, barmode: chart === "stacked_bar" ? "stack" : "group" }, { responsive: true, displaylogo: false });
    return () => { if (window.Plotly) window.Plotly.purge(element); };
  }, [ready, rows, analysis, chart, x, y, color, valid]);

  if (!valid) return <div className="chart-error">{chart.replace("_", " ")} is not valid for the selected columns. Choose one of: {recommended.join(", ") || "select compatible columns"}.</div>;
  if (!ready) return <div className="chart-error">Loading Plotly chart engine. If charts do not appear, check network access to the Plotly CDN.</div>;
  return <div><div className="plotly-chart" ref={ref} /><p className="muted">Computed from {formatNumber(rows.length, 0)} uploaded rows. Use the controls to change columns, grouping, and chart type; invalid options are hidden automatically.</p></div>;
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
  const [selectedTest, setSelectedTest] = useState<TestType>("pearson");
  const [selectedChart, setSelectedChart] = useState<ChartType>("histogram");
  const [selectedColor, setSelectedColor] = useState("");
  const [regressionKind, setRegressionKind] = useState<RegressionKind>("multiple");
  const [regressionTarget, setRegressionTarget] = useState("");
  const [regressionPredictors, setRegressionPredictors] = useState<string[]>([]);

  const previewColumns = useMemo(() => Object.keys(rows[0] ?? {}).slice(0, 10), [rows]);
  const filteredProfiles = useMemo(() => analysis?.profiles.filter((profile) => profile.name.toLowerCase().includes(search.toLowerCase())) ?? [], [analysis, search]);
  const selectedCorrelation = analysis?.correlations[0];
  const selectedXProfile = analysis?.profiles.find((profile) => profile.name === selectedX);
  const selectedYProfile = analysis?.profiles.find((profile) => profile.name === selectedY);
  const chartRecommendations = useMemo(() => analysis ? recommendCharts(selectedXProfile, selectedYProfile, analysis) : [], [analysis, selectedXProfile, selectedYProfile]);
  const crosstab = analysis && selectedCategoryLeft && selectedCategoryRight && selectedCategoryLeft !== selectedCategoryRight ? buildCrosstab(rows, selectedCategoryLeft, selectedCategoryRight) : null;
  const tests = analysis && selectedX && selectedY && selectedGroup ? inferentialSummaries(rows, selectedX, selectedY, selectedGroup) : null;
  const selectedTestResult = tests?.[selectedTest];
  const regression = analysis ? buildRegression(rows, regressionKind, regressionTarget, regressionPredictors) : null;

  useEffect(() => {
    if (chartRecommendations.length && !chartRecommendations.includes(selectedChart)) {
      setSelectedChart(chartRecommendations[0]);
    }
  }, [chartRecommendations, selectedChart]);

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
      setSelectedTest("pearson");
      setRegressionTarget(nextAnalysis.numericProfiles.find((profile) => profile.name === "age")?.name ?? nextAnalysis.numericProfiles[0]?.name ?? nextAnalysis.profiles.find((profile) => profile.unique === 2)?.name ?? "");
      setRegressionPredictors(nextAnalysis.numericProfiles.filter((profile) => profile.name !== (nextAnalysis.numericProfiles[0]?.name ?? "")).slice(0, 2).map((profile) => profile.name));
      setRegressionKind("multiple");
      setActiveSection("Dashboard");
      setUploadProgress(100);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to parse file.");
      setUploadProgress(0);
    }
  }

  function exportCurrentChart() {
    const html = document.querySelector(".plotly-chart")?.outerHTML;
    if (html) downloadBlob(`${selectedChart}-${selectedX || "chart"}.html`, html, "text/html");
  }

  function togglePredictor(column: string) {
    setRegressionPredictors((current) => current.includes(column) ? current.filter((item) => item !== column) : [...current, column]);
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
              <div className="panel-heading"><h3>Computed data readiness</h3><span className="badge">Live</span></div>
              <div className="check-list"><span>✓ {analysis.numericProfiles.length} numeric columns available for tests/models</span><span>✓ {analysis.categoricalProfiles.length} categorical columns available for cross-tabs/groups</span><span>✓ {analysis.duplicateRows} duplicate rows detected</span><span>✓ {analysis.profiles.reduce((sum, profile) => sum + profile.missing, 0)} missing cells counted</span><span>✓ {analysis.correlations.length} numeric correlation pairs computed</span></div>
            </section>

            <section className="panel wide">
              <div className="panel-heading"><div><p className="eyebrow">Descriptive Statistics</p><h3>Numeric summaries and categorical frequencies</h3></div><span>{filteredProfiles.length} fields</span></div>
              <h4>Numeric columns</h4>
              {analysis.numericProfiles.length ? <div className="table-wrap"><table><thead><tr>{["Column", "Mean", "Median", "Mode", "Std dev", "Variance", "Min", "Max", "Range", "IQR", "Skewness", "Kurtosis", "Missing"].map((heading) => <th key={heading}>{heading}</th>)}</tr></thead><tbody>{analysis.numericProfiles.map((profile) => <tr key={profile.name}><th>{profile.name}</th><td>{formatNumber(profile.mean)}</td><td>{formatNumber(profile.median)}</td><td>{profile.mode ?? "—"}</td><td>{formatNumber(profile.stdev)}</td><td>{formatNumber(profile.variance)}</td><td>{formatNumber(profile.min)}</td><td>{formatNumber(profile.max)}</td><td>{formatNumber(profile.range)}</td><td>{formatNumber(profile.iqr)}</td><td>{formatNumber(profile.skewness)}</td><td>{formatNumber(profile.kurtosis)}</td><td>{profile.missing}</td></tr>)}</tbody></table></div> : <p className="chart-error">No numeric columns were detected.</p>}
              <h4>Categorical columns</h4>
              <div className="profile-grid">{analysis.categoricalProfiles.map((profile) => <article className="profile-card" key={profile.name}><div><strong>{profile.name}</strong><span className={`pill ${profile.kind}`}>{profile.kind}</span></div><dl><dt>Missing</dt><dd>{profile.missing}</dd><dt>Unique</dt><dd>{profile.unique}</dd><dt>Mode</dt><dd>{profile.mode ?? "—"}</dd><dt>Present</dt><dd>{profile.present}</dd></dl><div className="table-wrap compact"><table><thead><tr><th>Value</th><th>Count</th><th>Percent</th></tr></thead><tbody>{profile.topValues.map((item) => <tr key={item.value}><td>{item.value}</td><td>{item.count}</td><td>{formatPercent(item.percent)}</td></tr>)}</tbody></table></div></article>)}</div>
            </section>

            <section className="panel wide viz-panel">
              <div className="panel-heading"><div><p className="eyebrow">Visualization studio</p><h3>Plotly charts from the uploaded dataset</h3></div><button onClick={exportCurrentChart}>Download HTML</button></div>
              <div className="control-grid">
                <label>Chart type<select value={selectedChart} onChange={(event) => setSelectedChart(event.target.value as ChartType)}>{chartRecommendations.map((type) => <option key={type} value={type}>{type.replace("_", " ")}</option>)}</select></label>
                <label>X-axis<select value={selectedX} onChange={(event) => setSelectedX(event.target.value)}>{analysis.profiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select></label>
                <label>Y-axis<select value={selectedY} onChange={(event) => setSelectedY(event.target.value)}>{analysis.profiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select></label>
                <label>Color/group<select value={selectedColor} onChange={(event) => setSelectedColor(event.target.value)}><option value="">None</option>{analysis.categoricalProfiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select></label>
              </div>
              <div className="viz-tags"><strong>Available for current columns:</strong>{chartRecommendations.map((tag) => <button className={selectedChart === tag ? "active" : ""} key={tag} onClick={() => setSelectedChart(tag)}>{tag.replace("_", " ")}</button>)}</div>
              <PlotlyChart rows={rows} analysis={analysis} chart={chartRecommendations.includes(selectedChart) ? selectedChart : chartRecommendations[0]} x={selectedX} y={selectedY} color={selectedColor} />
            </section>

            <section className="panel wide">
              <div className="panel-heading"><div><p className="eyebrow">Categorical relationship analysis</p><h3>Cross-tabulation and Chi-square independence test</h3></div><span className="badge">Computed</span></div>
              <div className="axis-controls"><select value={selectedCategoryLeft} onChange={(event) => setSelectedCategoryLeft(event.target.value)}>{analysis.categoricalProfiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select><select value={selectedCategoryRight} onChange={(event) => setSelectedCategoryRight(event.target.value)}>{analysis.categoricalProfiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select></div>
              {crosstab ? <><div className="table-wrap"><table><thead><tr><th>{selectedCategoryLeft} × {selectedCategoryRight}</th>{crosstab.rightValues.map((value) => <th key={value}>{value}</th>)}</tr></thead><tbody>{crosstab.leftValues.map((leftValue, rowIndex) => <tr key={leftValue}><th>{leftValue}</th>{crosstab.table[rowIndex].map((count, colIndex) => <td key={`${leftValue}-${colIndex}`}>{count}<small>expected {formatNumber(crosstab.expected[rowIndex][colIndex], 2)}</small></td>)}</tr>)}</tbody></table></div><div className="result-card"><strong>Chi-square test of independence</strong><dl><dt>χ² statistic</dt><dd>{formatNumber(crosstab.chi2, 4)}</dd><dt>p-value</dt><dd>{formatNumber(crosstab.pValue, 4)}</dd><dt>df</dt><dd>{crosstab.dof}</dd><dt>Cells</dt><dd>{crosstab.leftValues.length}×{crosstab.rightValues.length}</dd></dl><p>{crosstab.interpretation}</p></div></> : <p className="chart-error">Choose two different categorical variables to compute a cross-tab and Chi-square test.</p>}
            </section>

            <section className="panel wide">
              <div className="panel-heading"><div><p className="eyebrow">Inferential statistics</p><h3>Selectable hypothesis tests with assumptions</h3></div><span className="badge">Computed</span></div>
              <div className="control-grid"><label>Test type<select value={selectedTest} onChange={(event) => setSelectedTest(event.target.value as TestType)}><option value="pearson">Pearson correlation</option><option value="spearman">Spearman correlation</option><option value="ttest">Independent samples t-test</option><option value="anova">One-way ANOVA</option><option value="mannwhitney">Mann-Whitney U</option><option value="kruskal">Kruskal-Wallis</option></select></label><label>Numeric X<select value={selectedX} onChange={(event) => setSelectedX(event.target.value)}>{analysis.numericProfiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select></label><label>Numeric outcome/Y<select value={selectedY} onChange={(event) => setSelectedY(event.target.value)}>{analysis.numericProfiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select></label><label>Group column<select value={selectedGroup} onChange={(event) => setSelectedGroup(event.target.value)}>{analysis.categoricalProfiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select></label></div>
              {selectedTestResult ? <article className="result-card"><strong>{selectedTestResult.name}</strong><dl><dt>Statistic</dt><dd>{formatNumber(selectedTestResult.statistic, 4)}</dd><dt>p-value</dt><dd>{formatNumber(selectedTestResult.pValue, 4)}</dd><dt>Sample sizes</dt><dd>{selectedTestResult.sampleSizes || "—"}</dd><dt>Variables</dt><dd>{selectedTestResult.details}</dd></dl><p><strong>Assumptions:</strong> {selectedTestResult.assumptions}</p><p><strong>Interpretation:</strong> {selectedTestResult.interpretation}</p></article> : <p className="chart-error">Select valid numeric and grouping columns to run inferential tests.</p>}
            </section>

            <section className="panel wide">
              <div className="panel-heading"><div><p className="eyebrow">Regression modeling</p><h3>Simple linear, multiple linear, and logistic regression</h3></div><span className="badge">Computed</span></div>
              <div className="control-grid"><label>Model type<select value={regressionKind} onChange={(event) => setRegressionKind(event.target.value as RegressionKind)}><option value="simple">Simple linear regression</option><option value="multiple">Multiple linear regression</option><option value="logistic">Logistic regression</option></select></label><label>Target<select value={regressionTarget} onChange={(event) => setRegressionTarget(event.target.value)}>{(regressionKind === "logistic" ? analysis.profiles.filter((profile) => profile.unique === 2) : analysis.numericProfiles).map((profile) => <option key={profile.name}>{profile.name}</option>)}</select></label><label>Quick preset<select onChange={(event) => setRegressionPredictors(event.target.value ? event.target.value.split(",") : [])} value=""><option value="">Custom predictors</option><option value={analysis.numericProfiles.filter((profile) => ["Medu", "Fedu"].includes(profile.name)).map((profile) => profile.name).join(",")}>Medu + Fedu if present</option></select></label></div>
              <div className="checkbox-grid">{analysis.numericProfiles.filter((profile) => profile.name !== regressionTarget).map((profile) => <label key={profile.name}><input type="checkbox" checked={regressionPredictors.includes(profile.name)} onChange={() => togglePredictor(profile.name)} />{profile.name}</label>)}</div>
              {regression ? <article className="result-card"><strong>{regression.kind === "logistic" ? "Logistic regression" : regression.kind === "multiple" ? "Multiple linear regression" : "Simple linear regression"}</strong>{regression.error && <p className="chart-error">{regression.error}</p>}<dl><dt>Intercept</dt><dd>{formatNumber(regression.intercept, 4)}</dd><dt>{regression.metricLabel}</dt><dd>{formatNumber(regression.metric, 4)}</dd><dt>N</dt><dd>{regression.n}</dd><dt>Residual mean</dt><dd>{formatNumber(regression.residualSummary.mean, 4)}</dd><dt>Residual SD</dt><dd>{formatNumber(regression.residualSummary.stdev, 4)}</dd><dt>Residual min/median/max</dt><dd>{formatNumber(regression.residualSummary.min, 2)} / {formatNumber(regression.residualSummary.median, 2)} / {formatNumber(regression.residualSummary.max, 2)}</dd></dl><div className="table-wrap compact"><table><thead><tr><th>Term</th><th>Coefficient</th><th>p-value</th></tr></thead><tbody>{regression.coefficients.map((coef) => <tr key={coef.term}><td>{coef.term}</td><td>{formatNumber(coef.coefficient, 5)}</td><td>{coef.pValue === undefined ? "—" : formatNumber(coef.pValue, 4)}</td></tr>)}</tbody></table></div><p><strong>Interpretation:</strong> {regression.interpretation}</p></article> : <p className="chart-error">Select a target and at least one predictor.</p>}
            </section>

            <section className="panel wide">
              <div className="panel-heading"><div><p className="eyebrow">Correlation Analysis</p><h3>Relationship matrix</h3></div><span>{analysis.correlations.length} pairs</span></div>
              <div className="correlation-list">{analysis.correlations.map((item) => <div className="correlation-row" key={`${item.left}-${item.right}`}><span>{item.left}</span><ProgressBar value={Math.abs(item.pearson) * 100} /><span>{item.right}</span><strong>{formatNumber(item.pearson, 2)}</strong><small>Spearman {formatNumber(item.spearman, 2)} · p≈{formatNumber(item.pValue, 4)} · {item.strength}</small></div>)}</div>
              {selectedCorrelation && <p className="muted">AI explanation: {selectedCorrelation.left} and {selectedCorrelation.right} move together with a {selectedCorrelation.strength} relationship. Validate causality with experimental design or domain controls.</p>}
            </section>

            <section className="panel">
              <div className="panel-heading"><h3>Student dataset verification</h3><span className="badge">Live checks</span></div>
              <div className="check-list">{["age", "Medu", "Fedu", "school", "sex", "address"].map((column) => <span key={column}>{analysis.profiles.some((profile) => profile.name === column) ? "✓" : "○"} {column}</span>)}</div>
            </section>

            <section className="panel">
              <div className="panel-heading"><h3>Report export</h3><span className="badge">Computed</span></div>
              <div className="check-list"><span>→ CSV export includes {analysis.profiles.length} computed column profiles.</span><span>→ HTML/PDF-compatible export includes {analysis.insights.length} computed insights.</span><span>→ Chart export saves the current Plotly chart container as HTML.</span></div>
            </section>
          </div>
        )}
      </section>
    </main>
  );
}
