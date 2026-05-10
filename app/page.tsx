"use client";

import type { CSSProperties, ChangeEvent, DragEvent } from "react";
import { useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import * as XLSX from "xlsx";

type DataRow = Record<string, string | number | boolean | null>;
type ColumnKind = "numeric" | "date" | "boolean" | "text";

type ColumnProfile = {
  name: string;
  kind: ColumnKind;
  total: number;
  missing: number;
  unique: number;
  sample: string[];
  min?: number;
  max?: number;
  mean?: number;
  median?: number;
  stdev?: number;
  q1?: number;
  q3?: number;
  histogram?: Array<{ start: number; end: number; count: number; percent: number }>;
  topValues: Array<{ value: string; count: number; percent: number }>;
};

type DatasetAnalysis = {
  fileName: string;
  rows: number;
  columns: number;
  duplicateRows: number;
  completeness: number;
  profiles: ColumnProfile[];
  numericProfiles: ColumnProfile[];
  textProfiles: ColumnProfile[];
  insights: string[];
  correlations: Array<{ left: string; right: string; value: number }>;
};

const MAX_ROWS = 10000;
const ACCEPTED_TYPES = ".csv,.tsv,.txt,.json,.xlsx,.xls";

function normalizeCell(value: unknown): string | number | boolean | null {
  if (value === undefined || value === null) return null;
  if (typeof value === "number" || typeof value === "boolean") return value;
  const text = String(value).trim();
  if (!text || ["na", "n/a", "null", "undefined"].includes(text.toLowerCase())) return null;
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

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function buildHistogram(sorted: number[], bucketCount = 8): ColumnProfile["histogram"] {
  if (!sorted.length) return [];
  const min = sorted[0];
  const max = sorted[sorted.length - 1];

  if (min === max) {
    return [{ start: min, end: max, count: sorted.length, percent: 100 }];
  }

  const buckets = Array.from({ length: bucketCount }, (_, index) => {
    const start = min + ((max - min) / bucketCount) * index;
    const end = index === bucketCount - 1 ? max : min + ((max - min) / bucketCount) * (index + 1);
    return { start, end, count: 0, percent: 0 };
  });

  sorted.forEach((value) => {
    const bucketIndex = clamp(Math.floor(((value - min) / (max - min)) * bucketCount), 0, bucketCount - 1);
    buckets[bucketIndex].count += 1;
  });

  return buckets.map((bucket) => ({ ...bucket, percent: (bucket.count / sorted.length) * 100 }));
}

function inferKind(values: Array<string | number | boolean | null>): ColumnKind {
  const present = values.filter((value) => value !== null);
  if (!present.length) return "text";

  const numericCount = present.filter((value) => asNumber(value) !== null).length;
  const booleanCount = present.filter((value) => typeof value === "boolean" || ["true", "false", "yes", "no"].includes(String(value).toLowerCase())).length;
  const dateCount = present.filter((value) => asDate(value) !== null).length;

  if (numericCount / present.length >= 0.85) return "numeric";
  if (booleanCount / present.length >= 0.9) return "boolean";
  if (dateCount / present.length >= 0.75) return "date";
  return "text";
}

function profileColumn(name: string, rows: DataRow[]): ColumnProfile {
  const values = rows.map((row) => row[name] ?? null);
  const present = values.filter((value) => value !== null);
  const kind = inferKind(values);
  const counts = new Map<string, number>();

  present.forEach((value) => {
    const key = String(value);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  });

  const topValues = [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([value, count]) => ({ value, count, percent: present.length ? (count / present.length) * 100 : 0 }));

  const base: ColumnProfile = {
    name,
    kind,
    total: values.length,
    missing: values.length - present.length,
    unique: counts.size,
    sample: [...new Set(present.map(String))].slice(0, 4),
    topValues,
  };

  if (kind === "numeric") {
    const nums = present.map(asNumber).filter((value): value is number => value !== null).sort((a, b) => a - b);
    const mean = nums.reduce((sum, value) => sum + value, 0) / Math.max(nums.length, 1);
    const variance = nums.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(nums.length - 1, 1);
    return {
      ...base,
      min: nums[0],
      max: nums[nums.length - 1],
      mean,
      median: quantile(nums, 0.5),
      q1: quantile(nums, 0.25),
      q3: quantile(nums, 0.75),
      stdev: Math.sqrt(variance),
      histogram: buildHistogram(nums),
    };
  }

  return base;
}

function pearson(left: number[], right: number[]): number {
  const n = Math.min(left.length, right.length);
  if (n < 3) return 0;
  const avgLeft = left.reduce((sum, value) => sum + value, 0) / n;
  const avgRight = right.reduce((sum, value) => sum + value, 0) / n;
  let numerator = 0;
  let leftDenominator = 0;
  let rightDenominator = 0;

  for (let index = 0; index < n; index += 1) {
    const leftDelta = left[index] - avgLeft;
    const rightDelta = right[index] - avgRight;
    numerator += leftDelta * rightDelta;
    leftDenominator += leftDelta ** 2;
    rightDenominator += rightDelta ** 2;
  }

  const denominator = Math.sqrt(leftDenominator * rightDenominator);
  return denominator ? numerator / denominator : 0;
}

function analyzeRows(rows: DataRow[], fileName: string): DatasetAnalysis {
  const cleanedRows = rows.slice(0, MAX_ROWS).map((row) => {
    const next: DataRow = {};
    Object.entries(row).forEach(([key, value]) => {
      const cleanKey = key.trim() || "Unnamed column";
      next[cleanKey] = normalizeCell(value);
    });
    return next;
  });

  const columns = [...new Set(cleanedRows.flatMap((row) => Object.keys(row)))];
  const profiles = columns.map((column) => profileColumn(column, cleanedRows));
  const numericProfiles = profiles.filter((profile) => profile.kind === "numeric");
  const textProfiles = profiles.filter((profile) => profile.kind !== "numeric");
  const totalCells = Math.max(cleanedRows.length * columns.length, 1);
  const missingCells = profiles.reduce((sum, profile) => sum + profile.missing, 0);
  const duplicateRows = cleanedRows.length - new Set(cleanedRows.map((row) => JSON.stringify(row))).size;

  const correlations = numericProfiles.flatMap((left, leftIndex) =>
    numericProfiles.slice(leftIndex + 1).map((right) => {
      const pairs = cleanedRows
        .map((row) => [asNumber(row[left.name]), asNumber(row[right.name])])
        .filter((pair): pair is [number, number] => pair[0] !== null && pair[1] !== null);
      return { left: left.name, right: right.name, value: pearson(pairs.map((pair) => pair[0]), pairs.map((pair) => pair[1])) };
    }),
  )
    .filter((item) => Number.isFinite(item.value) && Math.abs(item.value) >= 0.3)
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 6);

  const insights = buildInsights(cleanedRows.length, columns.length, profiles, duplicateRows, correlations, rows.length > MAX_ROWS);

  return {
    fileName,
    rows: cleanedRows.length,
    columns: columns.length,
    duplicateRows,
    completeness: ((totalCells - missingCells) / totalCells) * 100,
    profiles,
    numericProfiles,
    textProfiles,
    correlations,
    insights,
  };
}

function buildInsights(
  rows: number,
  columns: number,
  profiles: ColumnProfile[],
  duplicateRows: number,
  correlations: DatasetAnalysis["correlations"],
  truncated: boolean,
): string[] {
  const insights = [`Your dataset contains ${formatNumber(rows, 0)} rows and ${formatNumber(columns, 0)} columns${truncated ? `; analysis uses the first ${formatNumber(MAX_ROWS, 0)} rows for browser performance` : ""}.`];
  const sparse = profiles.filter((profile) => profile.missing / Math.max(profile.total, 1) > 0.25).slice(0, 3);
  const identifiers = profiles.filter((profile) => profile.unique === profile.total && profile.total > 1).slice(0, 3);
  const numeric = profiles.filter((profile) => profile.kind === "numeric");

  if (sparse.length) insights.push(`Columns with many missing values: ${sparse.map((profile) => profile.name).join(", ")}. Consider cleaning or explaining these gaps before modeling.`);
  if (duplicateRows) insights.push(`${formatNumber(duplicateRows, 0)} duplicate row${duplicateRows === 1 ? "" : "s"} detected. Review whether these are expected repeated records.`);
  if (identifiers.length) insights.push(`Likely identifier columns: ${identifiers.map((profile) => profile.name).join(", ")}. These are useful for joins but usually not for prediction.`);
  if (numeric.length) insights.push(`${numeric.length} numeric column${numeric.length === 1 ? "" : "s"} found. The app computed distribution statistics and correlation checks for them.`);
  if (correlations.length) {
    const top = correlations[0];
    insights.push(`Strongest visible relationship: ${top.left} and ${top.right} have a ${top.value > 0 ? "positive" : "negative"} correlation of ${formatNumber(top.value, 2)}.`);
  }
  if (insights.length === 1) insights.push("The data looks compact and mostly categorical. Start by reviewing top values and missingness before deeper analysis.");
  return insights;
}

async function parseFile(file: File): Promise<DataRow[]> {
  const lowerName = file.name.toLowerCase();

  if (lowerName.endsWith(".xlsx") || lowerName.endsWith(".xls")) {
    const buffer = await file.arrayBuffer();
    const workbook = XLSX.read(buffer, { type: "array" });
    const sheet = workbook.Sheets[workbook.SheetNames[0]];
    return XLSX.utils.sheet_to_json<DataRow>(sheet, { defval: null });
  }

  const text = await file.text();
  if (lowerName.endsWith(".json")) {
    const parsed = JSON.parse(text) as unknown;
    if (Array.isArray(parsed)) return parsed as DataRow[];
    if (parsed && typeof parsed === "object") return [parsed as DataRow];
    throw new Error("JSON files should contain an object or an array of objects.");
  }

  return new Promise((resolve, reject) => {
    Papa.parse<DataRow>(text, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      delimiter: lowerName.endsWith(".tsv") ? "\t" : "",
      complete: (result) => resolve(result.data.filter((row) => Object.keys(row).length > 0)),
      error: (error: Error) => reject(error),
    });
  });
}

function StatCard({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <article className="stat-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{hint}</small>
    </article>
  );
}

function EmptyState({ onBrowse }: { onBrowse: () => void }) {
  return (
    <section className="empty-state">
      <div>
        <p className="eyebrow">Start here</p>
        <h2>Drop in your spreadsheet and get answers in seconds.</h2>
        <p>
          Upload CSV, TSV, JSON, XLS, or XLSX data. Everything runs in your browser, so you can explore private files without sending data to a server.
        </p>
      </div>
      <button className="primary-button" onClick={onBrowse}>Choose a file</button>
    </section>
  );
}

function ColumnTable({ profiles }: { profiles: ColumnProfile[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Column</th>
            <th>Type</th>
            <th>Missing</th>
            <th>Unique</th>
            <th>Typical values</th>
          </tr>
        </thead>
        <tbody>
          {profiles.map((profile) => (
            <tr key={profile.name}>
              <td>{profile.name}</td>
              <td><span className={`pill ${profile.kind}`}>{profile.kind}</span></td>
              <td>{formatNumber((profile.missing / Math.max(profile.total, 1)) * 100)}%</td>
              <td>{formatNumber(profile.unique, 0)}</td>
              <td>{profile.sample.length ? profile.sample.join(", ") : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function NumericSummary({ profiles }: { profiles: ColumnProfile[] }) {
  if (!profiles.length) return <p className="muted">No numeric columns were detected.</p>;

  return (
    <div className="cards-grid">
      {profiles.slice(0, 8).map((profile) => (
        <article className="metric-card" key={profile.name}>
          <div className="metric-card-header">
            <h3>{profile.name}</h3>
            <span>{formatNumber(profile.missing, 0)} missing</span>
          </div>
          <div className="range-bar" aria-hidden="true">
            <span style={{ left: "25%" }} />
            <span style={{ left: "50%" }} />
            <span style={{ left: "75%" }} />
          </div>
          <dl>
            <div><dt>Min</dt><dd>{formatNumber(profile.min)}</dd></div>
            <div><dt>Median</dt><dd>{formatNumber(profile.median)}</dd></div>
            <div><dt>Mean</dt><dd>{formatNumber(profile.mean)}</dd></div>
            <div><dt>Max</dt><dd>{formatNumber(profile.max)}</dd></div>
          </dl>
        </article>
      ))}
    </div>
  );
}

function TopValues({ profiles }: { profiles: ColumnProfile[] }) {
  const usefulProfiles = profiles.filter((profile) => profile.topValues.length > 0).slice(0, 6);
  if (!usefulProfiles.length) return <p className="muted">No categorical summaries are available yet.</p>;

  return (
    <div className="top-values-grid">
      {usefulProfiles.map((profile) => (
        <article className="top-card" key={profile.name}>
          <h3>{profile.name}</h3>
          {profile.topValues.map((item) => (
            <div className="bar-row" key={item.value}>
              <div className="bar-label"><span>{item.value}</span><strong>{formatNumber(item.percent)}%</strong></div>
              <div className="bar-track"><span style={{ width: `${Math.max(item.percent, 3)}%` }} /></div>
            </div>
          ))}
        </article>
      ))}
    </div>
  );
}


function DonutGauge({ value, label }: { value: number; label: string }) {
  const safeValue = clamp(value, 0, 100);

  return (
    <article className="viz-card donut-card">
      <div className="donut" style={{ "--value": `${safeValue}%` } as CSSProperties} aria-label={`${label}: ${formatNumber(safeValue)}%`}>
        <strong>{formatNumber(safeValue, 0)}%</strong>
        <span>{label}</span>
      </div>
      <p>Quick read on whether the dataset is complete enough for reliable analysis.</p>
    </article>
  );
}

function TypeMixChart({ profiles }: { profiles: ColumnProfile[] }) {
  const typeCounts = profiles.reduce<Record<ColumnKind, number>>((counts, profile) => {
    counts[profile.kind] += 1;
    return counts;
  }, { numeric: 0, date: 0, boolean: 0, text: 0 });
  const total = Math.max(profiles.length, 1);
  const segments = (Object.entries(typeCounts) as Array<[ColumnKind, number]>).filter(([, count]) => count > 0);

  return (
    <article className="viz-card">
      <div className="viz-card-heading">
        <h3>Column type mix</h3>
        <span>{profiles.length} fields</span>
      </div>
      <div className="stacked-bar" aria-label="Column type distribution">
        {segments.map(([kind, count]) => (
          <span className={kind} key={kind} style={{ width: `${(count / total) * 100}%` }} title={`${kind}: ${count}`} />
        ))}
      </div>
      <div className="legend-grid">
        {segments.map(([kind, count]) => (
          <span key={kind}><i className={kind} />{kind}: {count}</span>
        ))}
      </div>
    </article>
  );
}

function MissingnessChart({ profiles }: { profiles: ColumnProfile[] }) {
  const ranked = [...profiles]
    .map((profile) => ({ ...profile, missingPercent: (profile.missing / Math.max(profile.total, 1)) * 100 }))
    .sort((a, b) => b.missingPercent - a.missingPercent)
    .slice(0, 8);

  return (
    <article className="viz-card wide-viz-card">
      <div className="viz-card-heading">
        <h3>Missing data hotspots</h3>
        <span>Top {ranked.length}</span>
      </div>
      <div className="missing-chart">
        {ranked.map((profile) => (
          <div className="missing-row" key={profile.name}>
            <span>{profile.name}</span>
            <div className="missing-track"><i style={{ width: `${Math.max(profile.missingPercent, profile.missingPercent > 0 ? 3 : 0)}%` }} /></div>
            <strong>{formatNumber(profile.missingPercent, 1)}%</strong>
          </div>
        ))}
      </div>
    </article>
  );
}

function DistributionCharts({ profiles }: { profiles: ColumnProfile[] }) {
  const chartProfiles = profiles.filter((profile) => profile.histogram?.length).slice(0, 4);
  if (!chartProfiles.length) return null;

  return (
    <article className="viz-card wide-viz-card">
      <div className="viz-card-heading">
        <h3>Numeric distributions</h3>
        <span>Histograms</span>
      </div>
      <div className="histogram-grid">
        {chartProfiles.map((profile) => {
          const maxCount = Math.max(...(profile.histogram ?? []).map((bucket) => bucket.count), 1);
          return (
            <div className="histogram-card" key={profile.name}>
              <div className="histogram-title">
                <strong>{profile.name}</strong>
                <span>{formatNumber(profile.min)} → {formatNumber(profile.max)}</span>
              </div>
              <div className="histogram-bars" aria-label={`${profile.name} distribution histogram`}>
                {profile.histogram?.map((bucket) => (
                  <span
                    key={`${bucket.start}-${bucket.end}`}
                    style={{ height: `${Math.max((bucket.count / maxCount) * 100, bucket.count ? 8 : 2)}%` }}
                    title={`${formatNumber(bucket.start)} to ${formatNumber(bucket.end)}: ${bucket.count}`}
                  />
                ))}
              </div>
              <div className="boxplot" aria-label={`${profile.name} box plot`}>
                <span className="whisker" style={{ left: "0%", width: "100%" }} />
                <span className="box" style={{ left: `${percentAlong(profile.q1, profile.min, profile.max)}%`, width: `${Math.max(percentAlong(profile.q3, profile.min, profile.max) - percentAlong(profile.q1, profile.min, profile.max), 2)}%` }} />
                <span className="median" style={{ left: `${percentAlong(profile.median, profile.min, profile.max)}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </article>
  );
}

function percentAlong(value?: number, min?: number, max?: number): number {
  if (value === undefined || min === undefined || max === undefined || min === max) return 0;
  return clamp(((value - min) / (max - min)) * 100, 0, 100);
}

function CorrelationHeatmap({ correlations }: { correlations: DatasetAnalysis["correlations"] }) {
  if (!correlations.length) return null;

  return (
    <article className="viz-card wide-viz-card">
      <div className="viz-card-heading">
        <h3>Correlation heatmap</h3>
        <span>Strongest pairs</span>
      </div>
      <div className="heatmap-grid">
        {correlations.map((item) => {
          const strength = Math.abs(item.value);
          return (
            <div className={`heatmap-cell ${item.value >= 0 ? "positive" : "negative"}`} key={`${item.left}-${item.right}`} style={{ opacity: 0.35 + strength * 0.65 }}>
              <span>{item.left}</span>
              <strong>{formatNumber(item.value, 2)}</strong>
              <span>{item.right}</span>
            </div>
          );
        })}
      </div>
    </article>
  );
}

function VisualizationDashboard({ analysis }: { analysis: DatasetAnalysis }) {
  return (
    <section className="report-section visualization-section">
      <div className="section-heading"><h2>Important visualizations</h2><p>High-signal graphs for completeness, schema shape, numeric distributions, and relationships.</p></div>
      <div className="visualization-grid">
        <DonutGauge value={analysis.completeness} label="Complete cells" />
        <TypeMixChart profiles={analysis.profiles} />
        <MissingnessChart profiles={analysis.profiles} />
        <DistributionCharts profiles={analysis.numericProfiles} />
        <CorrelationHeatmap correlations={analysis.correlations} />
      </div>
    </section>
  );
}

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [analysis, setAnalysis] = useState<DatasetAnalysis | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const recommendedQuestions = useMemo(() => [
    "Which columns need cleaning before analysis?",
    "Which numeric fields have unusually wide ranges?",
    "Which categories dominate the data?",
    "Which fields look useful for prediction?",
  ], []);

  async function handleFile(file?: File) {
    if (!file) return;
    setIsLoading(true);
    setError(null);

    try {
      const rows = await parseFile(file);
      if (!rows.length) throw new Error("No rows were found in this file.");
      setAnalysis(analyzeRows(rows, file.name));
    } catch (caught) {
      setAnalysis(null);
      setError(caught instanceof Error ? caught.message : "Could not read this file. Try exporting it as CSV or XLSX.");
    } finally {
      setIsLoading(false);
    }
  }

  function onInputChange(event: ChangeEvent<HTMLInputElement>) {
    void handleFile(event.target.files?.[0]);
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    void handleFile(event.dataTransfer.files?.[0]);
  }

  return (
    <main>
      <section className="hero">
        <nav>
          <div className="brand-mark">Py</div>
          <span>Python data analysis</span>
        </nav>
        <div className="hero-content">
          <div>
            <p className="eyebrow">Friendly data exploration</p>
            <h1>Python data analysis</h1>
            <p className="hero-copy">
              Upload a dataset and instantly see data quality, column profiles, numeric summaries, category patterns, correlations, and practical next steps.
            </p>
            <div className="hero-actions">
              <button className="primary-button" onClick={() => inputRef.current?.click()}>Upload data</button>
              <a href="#analysis">View sample report layout</a>
            </div>
          </div>

          <label
            className={`upload-panel ${isDragging ? "dragging" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
          >
            <input ref={inputRef} type="file" accept={ACCEPTED_TYPES} onChange={onInputChange} />
            <div className="upload-icon">↑</div>
            <strong>{isLoading ? "Analyzing your data..." : "Drop your data file here"}</strong>
            <span>CSV, TSV, JSON, XLS, and XLSX are supported.</span>
          </label>
        </div>
      </section>

      {error && <div className="alert" role="alert">{error}</div>}

      <section className="question-strip" aria-label="Suggested questions">
        {recommendedQuestions.map((question) => <span key={question}>{question}</span>)}
      </section>

      <section id="analysis" className="analysis-shell">
        {!analysis ? <EmptyState onBrowse={() => inputRef.current?.click()} /> : (
          <>
            <div className="section-heading">
              <div>
                <p className="eyebrow">Analysis report</p>
                <h2>{analysis.fileName}</h2>
              </div>
              <button className="secondary-button" onClick={() => inputRef.current?.click()}>Analyze another file</button>
            </div>

            <div className="stats-grid">
              <StatCard label="Rows" value={formatNumber(analysis.rows, 0)} hint="Records analyzed" />
              <StatCard label="Columns" value={formatNumber(analysis.columns, 0)} hint="Fields detected" />
              <StatCard label="Complete" value={`${formatNumber(analysis.completeness)}%`} hint="Non-missing cells" />
              <StatCard label="Duplicates" value={formatNumber(analysis.duplicateRows, 0)} hint="Exact repeated rows" />
            </div>

            <VisualizationDashboard analysis={analysis} />

            <div className="insight-panel">
              <div>
                <p className="eyebrow">Key takeaways</p>
                <h2>What stands out</h2>
              </div>
              <ul>
                {analysis.insights.map((insight) => <li key={insight}>{insight}</li>)}
              </ul>
            </div>

            <section className="report-section">
              <div className="section-heading"><h2>Column health</h2><p>Types, missingness, uniqueness, and sample values.</p></div>
              <ColumnTable profiles={analysis.profiles} />
            </section>

            <section className="report-section">
              <div className="section-heading"><h2>Numeric summary</h2><p>Distribution basics for numeric columns.</p></div>
              <NumericSummary profiles={analysis.numericProfiles} />
            </section>

            <section className="report-section">
              <div className="section-heading"><h2>Top values</h2><p>Most common categories and repeated values.</p></div>
              <TopValues profiles={analysis.profiles} />
            </section>

            <section className="report-section correlation-section">
              <div className="section-heading"><h2>Correlation signals</h2><p>Numeric relationships worth investigating.</p></div>
              {analysis.correlations.length ? analysis.correlations.map((item) => (
                <article className="correlation-card" key={`${item.left}-${item.right}`}>
                  <span>{item.left}</span>
                  <div className="correlation-track"><span style={{ width: `${Math.abs(item.value) * 100}%` }} /></div>
                  <span>{item.right}</span>
                  <strong>{formatNumber(item.value, 2)}</strong>
                </article>
              )) : <p className="muted">No moderate or strong numeric correlations were detected.</p>}
            </section>
          </>
        )}
      </section>
    </main>
  );
}
