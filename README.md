# InsightForge AI Data Analysis App

InsightForge is an AI-assisted data analysis workspace with two runnable experiences in one repository:

- `streamlit_app.py` — the main Python/Streamlit data analysis app.
- The existing Next.js + FastAPI stack for API-oriented analytics workflows.

The Streamlit app remains the primary entry point for the Python data analysis experience and computes every dashboard, model, chart, answer, and report from the uploaded dataset.

## Streamlit feature overview

- **Dataset workflow**: upload CSV, XLSX, or JSON files; show the uploaded file name; apply duplicate removal, missing-value handling, datetime conversion, and optional IQR outlier filtering.
- **Storytelling Dashboard**: executive summary, KPI cards, key findings cards, recommended Plotly charts, chart captions, and a plain-English data story.
- **Descriptive and categorical analysis**: numeric descriptive statistics, categorical frequency tables, cross-tabs, and Chi-square analysis.
- **Inferential statistics**: Pearson/Spearman correlations, t-test, ANOVA, Mann-Whitney U, and Kruskal-Wallis where data types support them.
- **Machine Learning**: regression and classification workflows with one-hot encoding, missing-value imputation, train/test controls, metrics, plots, feature importance, and model interpretation.
- **Ask Your Data**: chat-style Q&A over safe dataset summaries. Uses OpenAI when `OPENAI_API_KEY` is available and falls back to rule-based answers when it is not.
- **Business Report**: generated Markdown and HTML reports with downloadable summary CSV tables and timestamped filenames.
- **Collaboration-friendly session state**: reset button, persisted test/model/chat state, timestamped report generation, and a sidebar app version label.

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

- **Storytelling Dashboard** shows real KPI cards for row count, column count, missing-value percentage, duplicate row count, numeric columns, and categorical columns.
- **Storytelling Dashboard** shows key finding cards for correlations, variability, category imbalance, and missingness, plus chart captions.
- **Machine Learning → regression** runs with `age` as target and `Medu`/`Fedu` as predictors.
- **Machine Learning → classification** runs with `school` or `sex` as target.
- **Ask Your Data** answers questions about columns, missing values, correlations, and summary statistics.
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
