# InsightForge AI Data Analysis App

InsightForge is an AI-assisted data analysis workspace with two runnable experiences in one repository:

- `streamlit_app.py` — the main Python/Streamlit data analysis app.
- The existing Next.js + FastAPI stack for API-oriented analytics workflows.

The Streamlit app remains the primary entry point for the Python data analysis experience and computes every dashboard, model, chart, answer, and report from the uploaded dataset.

## Streamlit feature overview

- **Expert Dataset Overview**: upload CSV, XLSX, or JSON files; apply duplicate removal, missing-value handling, datetime conversion, and optional IQR outlier filtering; then review a computed dataset health score, row/column summary, type profile, memory usage, missing-value heatmap, duplicate summary, high-cardinality fields, constant/near-constant fields, possible ID columns, target suggestions, leakage-risk warnings, expert quality interpretation, and cleaning actions.
- **Senior Summary Analysis**: generates an analyst-style report from computed metrics, including executive summary, key findings, strongest relationships, distributions, missingness, duplicate/outlier/category imbalance concerns, modeling opportunities, recommended tests, recommended charts, and next steps.
- **Professional Storytelling Dashboard**: KPI cards, ranked insight cards, risk/warning cards, automatic best charts, captions, “what this means”/“why it matters” explanations, decision questions, and a data-story narrative.
- **Expert Descriptive Statistics**: numeric count/mean/median/mode/std/variance/min/max/range/IQR/skewness/kurtosis/missingness/IQR outliers/coefficient of variation/normality hints, plus categorical unique count/top share/frequency tables/missingness/entropy/concentration/imbalance/cardinality warnings and interpretation.
- **Categorical Intelligence**: automatic categorical pair recommendations, cross-tabs, row-normalized cross-tabs, stacked/grouped bars, Chi-square tests, Cramer’s V, expected-frequency warnings, sparse-category warnings, and follow-up guidance.
- **Statistician-guided Inferential Tests**: automatic test recommendations, normality/equal-variance/sample-size warnings, Pearson and Spearman correlation, t-test, ANOVA, Mann-Whitney U, Kruskal-Wallis, Chi-square, p-value interpretation, effect-size guidance, and recommended next steps.
- **Regression Modeling Studio**: target/predictor selectors, suitability and leakage checks, simple/multiple linear regression, binary logistic regression, preprocessing, categorical encoding, missing-value handling, R²/adjusted R²/MAE/RMSE, coefficients/p-values, residual and predicted-vs-actual plots, VIF multicollinearity checks, limitations, and next-step recommendations.
- **Machine Learning Workbench**: automatic problem-type detection, regression/classification selector, train/test split, preprocessing pipeline, Linear/Logistic Regression, Decision Tree, Random Forest, KNN, optional SVM, metrics by problem type, confusion matrix, ROC curve, feature importance, baseline model comparison, and overfitting/leakage guidance.
- **Visualization Studio**: chart recommendation engine, automatic best charts, natural-language captions, histogram, KDE, box/violin, scatter/regression, heatmap, bar/count/grouped/stacked charts, line/pie/area charts, scatter matrix, 3D scatter, missing-value heatmap, outlier plot, distribution comparisons, HTML chart export, and an explanation under every chart.
- **Ask Your Data**: an execution-focused AI data scientist copilot. It detects natural-language requests, runs approved local analyses on the uploaded dataframe, renders Plotly charts/tables/statistical tests/models, explains results, remembers prior analyses, and uses OpenAI only for metadata-only intent/narrative help when `OPENAI_API_KEY` is available.
- **Business Report**: professional analyst-style Markdown/HTML reports with executive summary, dataset overview, quality assessment, findings, descriptive/relationship/statistical/model/visual sections, implications, recommended actions, limitations, appendix-style tables, and downloadable CSV summaries.
- **Cross-tab intelligence and session state**: shared computed data-quality summaries, descriptive stats, relationship findings, test/model/chart history, Ask Your Data memory, reset button, persisted results, and Streamlit Community Cloud-compatible deployment.

## Supported file types

The Streamlit app supports:

- `.csv`
- `.xlsx`
- `.json`

All outputs are based on the uploaded file after the cleaning options selected in the sidebar.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal, upload a supported dataset, and use the tabs across the top of the app.

## OpenAI API key and secrets

The Ask Your Data tab works without an API key by using deterministic rule-based answers from computed summaries. To enable OpenAI-generated natural-language answers:

### Local Streamlit secrets

```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-your-key-here"
```

You can also set an environment variable instead:

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

The app sends only metadata, summary statistics, categorical summaries, correlations, and session model summaries to the API. It does not send the full raw dataset.

## Ask Your Data examples

The **Ask Your Data** tab supports plain-English analysis requests and does not show Python code unless you explicitly ask for it. Example prompts include:

- `Show me a correlation heatmap`
- `Make a scatter plot of G2 vs G3`
- `Run Pearson correlation between G2 and G3`
- `Run Spearman correlation between absences and G3`
- `Show histogram of failures`
- `Show boxplot of G3 by school`
- `Run ANOVA for G3 by school`
- `Run chi-square test between Fjob and Mjob`
- `Build a regression model to predict G3 using G1, G2, studytime, failures, absences`
- `Which features are most important for predicting G3?`
- `Find outliers in absences`
- `What should I explore next?`

Supported Ask Your Data actions include summary/descriptive statistics, missing-value and data-quality reports, outlier detection, correlation ranking and heatmaps, scatter/regression/histogram/KDE/box/violin/bar/count/line/pie/stacked/area/scatter-matrix/3D charts, cross-tabulation, chi-square, t-test, ANOVA, Mann-Whitney U, Kruskal-Wallis, linear/logistic regression, random forest, decision tree, and feature-importance workflows. If a request is ambiguous, the app asks you to choose valid columns from the uploaded dataset instead of guessing or crashing.

## Streamlit Community Cloud deployment

1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and choose **New app**.
3. Select the repository, branch, and set the main file path to:

   ```text
   streamlit_app.py
   ```

4. Keep `requirements.txt` in the repository root so dependencies install automatically.
5. If using OpenAI in Ask Your Data, open **App settings → Secrets** and add:

   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```

6. Deploy. Upload a CSV, XLSX, or JSON file in the app sidebar to begin analysis.

## Student performance smoke test

Use `student-mat.csv` from the UCI Student Performance dataset and verify:

- **Dataset Overview** shows a computed health score, memory usage, type summary, duplicate summary, missing-value heatmap when missing values exist, high-cardinality/ID/leakage warnings, target suggestions, and cleaning recommendations.
- **Summary Analysis** includes computed executive findings, strongest relationships, distribution/outlier/category warnings, modeling opportunities, recommended tests, recommended visualizations, and practical next steps.
- **Storytelling Dashboard** shows KPI cards, ranked top insights, warning cards, automatic charts with captions, decision questions, and a data-story narrative.
- **Descriptive Statistics** shows numeric outlier/skew/CV/normality diagnostics and categorical entropy/concentration/imbalance/cardinality diagnostics.
- **Categorical Analysis** runs cross-tabs, normalized cross-tabs, grouped/stacked bars, Chi-square, Cramer’s V, and sparse/expected-frequency warnings.
- **Inferential Tests** runs valid Pearson/Spearman, t-test, ANOVA, Mann-Whitney U, Kruskal-Wallis, and Chi-square workflows or provides helpful warnings when assumptions/data shape are invalid.
- **Regression Modeling** runs linear regression for `G3` using `G1`, `G2`, `studytime`, `failures`, and `absences`, and shows coefficients, p-values, MAE/RMSE/R², residual and predicted-vs-actual plots, VIF, leakage guidance, and limitations.
- **Machine Learning → regression** runs with `G3` as target and student predictors; **Machine Learning → classification** runs with `school` or `sex` as target; model comparison and feature importance render without crashing.
- **Visualizations** renders data-driven recommended charts and lets users export a generated Plotly chart as HTML.
- **Ask Your Data** executes the example prompts above, including charts, Pearson/Spearman correlations, ANOVA, chi-square, regression, feature importance, outliers, and next-step recommendations.
- **Business Report** previews and downloads Markdown, HTML, and CSV summary tables.
- The app starts with:

  ```bash
  pip install -r requirements.txt
  streamlit run streamlit_app.py
  ```

## Troubleshooting

- **Upload fails**: confirm the file extension is CSV, XLSX, or JSON and that the file is not password protected.
- **Charts do not appear**: some recommended charts require numeric or categorical columns with enough non-missing data.
- **Model cannot train**: select at least one predictor, avoid using the target as a predictor, and make sure the target has enough non-missing values and variation.
- **Classification split error**: choose a target with at least two classes and enough examples per class, or reduce the test split.
- **OpenAI answer fails**: check that `OPENAI_API_KEY` is present in Streamlit secrets or the environment. The app will fall back to rule-based answers if no key is available.
- **Report table formatting looks plain**: Markdown reports preserve table text for portability; use the HTML download for a browser-friendly business report.

## Existing full-stack app

The repository also contains a Next.js frontend and FastAPI backend for API-based analytics. To run those components:

```bash
npm install
npm run dev
```

In another terminal:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm run backend
```

The Streamlit entry point remains `streamlit_app.py` and is safe to deploy independently on Streamlit Community Cloud.
