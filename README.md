# InsightForge AI Analytics Platform

InsightForge is a production-ready AI-powered data analysis web application that turns CSV, Excel, and JSON datasets into an interactive exploratory data analysis (EDA), statistical testing, visualization, regression, and ML-preparation workspace.

The product is designed to feel like an enterprise analytics platform inspired by Tableau, Power BI, and Hex, while focusing on automated AI-driven profiling and statistical modeling.

## Highlights

- **Next.js App Router + TypeScript frontend** with a polished responsive dashboard, sidebar navigation, dark/light theme toggle, drag-and-drop uploads, upload progress, preview tables, cards, accordions, filters, and export actions.
- **Python FastAPI analytics backend** using Pandas, NumPy, SciPy, Statsmodels, Scikit-learn, and Plotly.
- **Dataset support** for CSV, XLS, XLSX, and JSON with multi-file upload and in-memory dataset caching.
- **Automatic EDA** for numerical, categorical, boolean, and datetime columns, missing values, duplicates, outliers, top values, descriptive statistics, and correlation signals.
- **Inferential statistics** for Pearson/Spearman correlation, t-tests, ANOVA, chi-square, Mann-Whitney U, and Kruskal-Wallis.
- **Regression and ML modules** for linear regression, logistic regression, Random Forest, Decision Tree, KNN, and SVM with train/test split, cross-validation, diagnostics, confusion matrices, accuracy, ROC AUC, and regression errors.
- **Plotly visualization API** for histograms, KDE-style distributions, box/violin plots, scatter and regression plots, line/area/bar/count/pie/stacked charts, heatmaps, pair plots, joint/hexbin-style scatter, 3D scatter, and geospatial charts.
- **AI insights layer** that summarizes data quality, trends, anomalies, correlations, and modeling recommendations.
- **Export-ready reports** for PNG/SVG, CSV, HTML, and PDF-compatible HTML payloads.

## Folder structure

```text
app/
  api/
    health/route.ts        # Frontend API health endpoint
    report/route.ts        # Report payload endpoint
  globals.css              # Tailwind-inspired enterprise UI system
  layout.tsx               # App metadata and root layout
  page.tsx                 # Interactive analytics dashboard
backend/
  analytics_engine.py      # FastAPI + reusable analytics/ML logic
Dockerfile                 # Container build for backend-first deployment
.env.example               # Environment variable template
package.json               # Next.js scripts/dependencies
requirements.txt           # Python analytics dependencies
README.md                  # Documentation and deployment guide
```


## Running the full analysis app

Run the Python analytics API and the Next.js UI in two terminals so uploads can be profiled, tested, modeled, visualized, and exported end-to-end.

### Terminal 1: analytics backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm run backend
```

The backend uses Pandas, NumPy, SciPy, Statsmodels, Scikit-learn, and Plotly to compute column-type detection, descriptive statistics, frequency tables, Chi-square tests, Pearson/Spearman correlations, t-tests, ANOVA, Mann-Whitney U, Kruskal-Wallis, linear/logistic modeling metrics, residual plots, and Plotly chart payloads.

### Terminal 2: Next.js frontend

```bash
npm install
NEXT_PUBLIC_ANALYTICS_API_URL=http://localhost:8000 npm run dev
```

Open <http://localhost:3000>, upload a CSV/XLSX/JSON file, and use the dashboard controls to select x-axis, y-axis, grouping column, chart type, categorical variables, and model variables. Every analysis panel now produces computed output from the uploaded rows instead of a static roadmap.

### Student performance smoke test

Use `student-mat.csv` from the UCI Student Performance dataset. After upload, verify:

- `age`, `Medu`, and `Fedu` are detected as numeric fields with mean, median, mode, standard deviation, variance, range, IQR, min, max, skewness, kurtosis, histograms, and regression/correlation options.
- `school`, `sex`, and `address` are detected as categorical/binary fields with frequency counts, percentages, count/bar/pie charts, cross-tabulation, expected frequencies, and Chi-square interpretation.
- Select `age` as x-axis, `Medu` or `Fedu` as y-axis, and `sex` or `address` as the group to generate Pearson, Spearman, t-test, ANOVA, Mann-Whitney U, Kruskal-Wallis, scatter/regression charts, grouped bars, and residual plots.
- Use **Download SVG** or the report export buttons to save generated visualizations and analysis summaries.

## Local development

### 1. Frontend

```bash
npm install
npm run dev
```

Open <http://localhost:3000>.

### 2. Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm run backend
```

The FastAPI service runs at <http://localhost:8000>. API docs are available at <http://localhost:8000/docs>.

### 3. Environment

```bash
cp .env.example .env.local
```

Important variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `NEXT_PUBLIC_ANALYTICS_API_URL` | Frontend URL for the Python analytics API | `http://localhost:8000` |
| `MAX_MEMORY_ROWS` | Maximum rows retained per cached dataset | `250000` |
| `CORS_ORIGINS` | Comma-separated allowed frontend origins | `http://localhost:3000` |

## API overview

### `POST /upload`

Uploads one or more datasets and returns a cached `dataset_id`, preview rows, and full analysis.

### `GET /datasets/{dataset_id}/analysis`

Returns dataset overview, inferred column groups, descriptive statistics, outliers, duplicate counts, missingness, Pearson/Spearman correlations, and AI insights.

### `POST /statistics`

Runs inferential tests:

- `ttest`
- `anova`
- `chi_square`
- `mann_whitney`
- `kruskal`

### `POST /visualizations`

Returns Plotly JSON for supported charts:

- Histogram / KDE-style distribution
- Box and violin plots
- Scatter and regression plots
- Line, area, bar, count, pie, and stacked bar charts
- Heatmap correlation matrix
- Pair plot and joint-style plots
- Hexbin-style scatter
- 3D scatter
- Geospatial scatter

### `POST /models`

Trains baseline regression or classification models with preprocessing pipelines:

- Linear regression
- Logistic regression
- Random Forest
- Decision Tree
- KNN
- SVM

Returns metrics such as R², RMSE, MAE, accuracy, ROC AUC, confusion matrix, cross-validation scores, and residual diagnostics where applicable.

### `GET /datasets/{dataset_id}/report`

Returns a report-ready payload for HTML/PDF/CSV/PNG export workflows.

## Performance and scalability notes

- The frontend parses and previews files quickly for a responsive first interaction.
- The backend exposes server-side processing endpoints for large datasets and expensive statistical/ML workloads.
- Table previews are capped and scrollable to avoid rendering thousands of DOM rows.
- Chart payloads are generated on demand to avoid unnecessary Plotly rendering cost.
- `MAX_MEMORY_ROWS` protects memory use in simple deployments; production systems should swap the in-memory cache for Redis, S3, DuckDB, or a data warehouse.

## Docker

Build and run the backend container:

```bash
docker build -t insightforge .
docker run --rm -p 8000:8000 --env-file .env.example insightforge
```

For full-stack production, deploy the Next.js app separately on Vercel and the FastAPI service on Railway or Render.

## Deployment

### Vercel frontend

1. Import the repository in Vercel.
2. Set `NEXT_PUBLIC_ANALYTICS_API_URL` to your backend URL.
3. Use the default Next.js build command: `npm run build`.
4. Deploy.

### Railway backend

1. Create a Railway project from the repository.
2. Select the Dockerfile deployment path.
3. Set environment variables from `.env.example`.
4. Expose port `8000`.

### Render backend

1. Create a new Web Service.
2. Use Docker as the runtime.
3. Set `PORT=8000`, `MAX_MEMORY_ROWS`, and `CORS_ORIGINS`.
4. Deploy and copy the service URL into Vercel as `NEXT_PUBLIC_ANALYTICS_API_URL`.

## Notes on shadcn/ui and Tailwind CSS

The current UI uses a Tailwind-inspired utility design system in `app/globals.css` and shadcn-style primitives (cards, pills, accordions, buttons, inputs, panels, sidebar navigation). If you want strict generated shadcn components, run `npx shadcn@latest init` and map the existing panel/card/button styles into `components/ui/*`.
