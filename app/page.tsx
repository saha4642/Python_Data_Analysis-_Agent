"use client";

import type { ChangeEvent, DragEvent } from "react";
import { useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import * as XLSX from "xlsx";

type DataValue = string | number | boolean | null;
type DataRow = Record<string, DataValue>;
type ColumnKind = "numeric" | "categorical" | "datetime" | "boolean" | "text";
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
  if (numericRatio >= 0.86) return "numeric";
  if (booleanRatio >= 0.9) return "boolean";
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
  const categoricalProfiles = profiles.filter((profile) => ["categorical", "boolean", "text"].includes(profile.kind));
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
  const [chat, setChat] = useState("Which features are most predictive?");

  const previewColumns = useMemo(() => Object.keys(rows[0] ?? {}).slice(0, 10), [rows]);
  const filteredProfiles = useMemo(() => analysis?.profiles.filter((profile) => profile.name.toLowerCase().includes(search.toLowerCase())) ?? [], [analysis, search]);
  const selectedNumeric = analysis?.numericProfiles[0];
  const selectedCategory = analysis?.categoricalProfiles[0];
  const selectedCorrelation = analysis?.correlations[0];

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
      setSelectedX(nextAnalysis.numericProfiles[0]?.name ?? "");
      setSelectedY(nextAnalysis.numericProfiles[1]?.name ?? nextAnalysis.numericProfiles[0]?.name ?? "");
      setActiveSection("Dashboard");
      setUploadProgress(100);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to parse file.");
      setUploadProgress(0);
    }
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

  const statPlans = analysis ? [
    { title: "Correlation significance", body: analysis.numericProfiles.length > 1 ? `Run Pearson/Spearman tests across ${analysis.numericProfiles.length} numeric variables with p-values and effect size labels.` : "Add two numeric columns to calculate Pearson and Spearman tests." },
    { title: "T-test / ANOVA", body: analysis.categoricalProfiles.length && analysis.numericProfiles.length ? `Compare ${analysis.numericProfiles[0].name} by groups in ${analysis.categoricalProfiles[0].name}; use t-test for two groups and ANOVA/Kruskal-Wallis for more.` : "Needs one grouping column and one numeric response." },
    { title: "Chi-square", body: analysis.categoricalProfiles.length > 1 ? `Test association between ${analysis.categoricalProfiles[0].name} and ${analysis.categoricalProfiles[1].name}.` : "Needs two categorical columns." },
  ] : [];

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
              <div className="panel-heading"><div><p className="eyebrow">Plotly visualization studio</p><h3>Interactive chart recommendations</h3></div><div className="axis-controls"><select value={selectedX} onChange={(event) => setSelectedX(event.target.value)}>{analysis.numericProfiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select><select value={selectedY} onChange={(event) => setSelectedY(event.target.value)}>{analysis.numericProfiles.map((profile) => <option key={profile.name}>{profile.name}</option>)}</select></div></div>
              <div className="chart-grid">
                <div className="chart-card"><span>Histogram + KDE</span>{selectedNumeric ? <MiniHistogram profile={selectedNumeric} /> : <p>No numeric column</p>}</div>
                <div className="chart-card"><span>Box / Violin / Outliers</span><div className="boxplot"><i style={{ left: "20%" }} /><b style={{ left: "38%", width: "28%" }} /><em style={{ left: "51%" }} /><i style={{ left: "82%" }} /></div></div>
                <div className="chart-card"><span>Grouped bar / Count plot / Pie</span><div className="bars tall">{selectedCategory?.topValues.slice(0, 6).map((item) => <label key={item.value}><span>{item.value}</span><ProgressBar value={item.percent} /></label>)}</div></div>
                <div className="chart-card"><span>Scatter / Regression / 3D / Hexbin</span><div className="scatter">{rows.slice(0, 60).map((row, index) => { const x = asNumber(row[selectedX]) ?? index; const y = asNumber(row[selectedY]) ?? index; return <i key={index} style={{ left: `${Math.abs(x * 13) % 92}%`, bottom: `${Math.abs(y * 17) % 85}%` }} />; })}</div></div>
              </div>
              <div className="viz-tags">{["Line", "Area", "Heatmap", "Pair plot", "Joint plot", "Stacked bar", "Geospatial map", "Zoom", "Pan", "Fullscreen", "PNG download", "Hover tooltips"].map((tag) => <span key={tag}>{tag}</span>)}</div>
            </section>

            <section className="panel">
              <div className="panel-heading"><h3>Inferential statistics</h3><span className="badge">SciPy plan</span></div>
              <div className="accordion">{statPlans.map((plan) => <details key={plan.title} open><summary>{plan.title}</summary><p>{plan.body}</p></details>)}</div>
            </section>

            <section className="panel">
              <div className="panel-heading"><h3>Regression & ML workbench</h3><span className="badge">Model-ready</span></div>
              <div className="model-grid">{["Linear regression", "Multiple regression", "Logistic regression", "Random Forest", "Decision Tree", "KNN", "SVM", "ROC + Confusion matrix"].map((model) => <button key={model}>{model}</button>)}</div>
              <div className="prediction-card"><strong>Prediction interface</strong><p>Select a target, choose features, inspect residuals, coefficients, feature importance, accuracy, ROC AUC, and cross-validation folds in the Python backend.</p></div>
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
