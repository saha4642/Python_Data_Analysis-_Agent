"""FastAPI analytics engine for InsightForge.

The module is intentionally framework-friendly: the pure functions can be reused from
Streamlit, notebooks, batch jobs, or FastAPI endpoints. It performs EDA, statistical
testing, preprocessing recommendations, Plotly chart specs, and baseline ML models.
"""

from __future__ import annotations

import io
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import statsmodels.api as sm
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DatasetId = str
DATASET_CACHE: dict[DatasetId, pd.DataFrame] = {}
MAX_MEMORY_ROWS = int(os.getenv("MAX_MEMORY_ROWS", "250000"))

app = FastAPI(title="InsightForge Analytics Engine", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TestRequest(BaseModel):
    dataset_id: str
    test: Literal["pearson", "spearman", "ttest", "anova", "chi_square", "mann_whitney", "kruskal"]
    numeric: str | None = None
    group: str | None = None
    left: str | None = None
    right: str | None = None


class ModelRequest(BaseModel):
    dataset_id: str
    target: str
    features: list[str] = Field(default_factory=list)
    task: Literal["regression", "classification"] = "regression"
    estimator: Literal["linear", "logistic", "random_forest", "decision_tree", "knn", "svm"] = "random_forest"


class ChartRequest(BaseModel):
    dataset_id: str
    chart: Literal["histogram", "kde", "box", "violin", "scatter", "regression", "line", "bar", "count", "pie", "stacked_bar", "heatmap", "pair", "joint", "hexbin", "area", "scatter_3d", "geo"]
    x: str | None = None
    y: str | None = None
    color: str | None = None
    lat: str | None = None
    lon: str | None = None


@dataclass
class ColumnGroups:
    numeric: list[str]
    categorical: list[str]
    datetime: list[str]
    boolean: list[str]


def read_upload(file: UploadFile) -> pd.DataFrame:
    suffix = file.filename.rsplit(".", 1)[-1].lower() if file.filename else ""
    payload = file.file.read()
    if suffix == "csv":
        return pd.read_csv(io.BytesIO(payload))
    if suffix in {"xlsx", "xls"}:
        return pd.read_excel(io.BytesIO(payload))
    if suffix == "json":
        return pd.read_json(io.BytesIO(payload))
    raise HTTPException(status_code=400, detail="Supported formats are CSV, XLSX, XLS, and JSON.")


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df = df.replace({"": np.nan, "NA": np.nan, "N/A": np.nan, "null": np.nan, "None": np.nan})
    for column in df.select_dtypes(include="object").columns:
        converted = pd.to_datetime(df[column], errors="coerce")
        if converted.notna().mean() > 0.75:
            df[column] = converted
        else:
            numeric = pd.to_numeric(df[column].astype(str).str.replace(r"[$,% ,]", "", regex=True), errors="coerce")
            if numeric.notna().mean() > 0.85:
                df[column] = numeric
    return df.head(MAX_MEMORY_ROWS)


def column_groups(df: pd.DataFrame) -> ColumnGroups:
    datetime = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    binary = [col for col in df.columns if col not in datetime and df[col].dropna().nunique() == 2]
    numeric = [col for col in df.select_dtypes(include=np.number).columns.tolist() if col not in binary]
    boolean = sorted(set(df.select_dtypes(include="bool").columns.tolist() + binary))
    categorical = [col for col in df.columns if col not in numeric + datetime + boolean]
    return ColumnGroups(numeric=numeric, categorical=categorical, datetime=datetime, boolean=boolean)


def outlier_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"iqr": 0, "z_score": 0}
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    z = np.abs(stats.zscore(values, nan_policy="omit")) if len(values) > 1 else np.array([])
    return {
        "iqr": int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()),
        "z_score": int((z > 3).sum()) if len(z) else 0,
    }


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    groups = column_groups(df)
    numeric_stats = df[groups.numeric].describe().T.to_dict("index") if groups.numeric else {}
    profiles: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        item: dict[str, Any] = {
            "name": column,
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "missing_percent": float(series.isna().mean() * 100),
            "unique": int(series.nunique(dropna=True)),
            "sample": series.dropna().astype(str).head(5).tolist(),
        }
        if column in groups.numeric:
            values = pd.to_numeric(series, errors="coerce")
            item.update(
                mean=float(values.mean()), median=float(values.median()), mode=str(values.mode().iloc[0]) if not values.mode().empty else None,
                std=float(values.std()), variance=float(values.var()), min=float(values.min()), max=float(values.max()), range=float(values.max() - values.min()),
                q1=float(values.quantile(0.25)), q3=float(values.quantile(0.75)), iqr=float(values.quantile(0.75) - values.quantile(0.25)),
                skewness=float(values.skew()), kurtosis=float(values.kurt()), outliers=outlier_summary(values),
                insight=f"{column} averages {values.mean():.2f} with a median of {values.median():.2f}; review skewness and outliers before using it in models.",
            )
        else:
            counts = series.astype(str).value_counts(dropna=True).head(20)
            item["frequency_table"] = [{"value": key, "count": int(value), "percent": float(value / max(series.notna().sum(), 1) * 100)} for key, value in counts.items()]
            item["insight"] = f"{column} has {series.nunique(dropna=True)} observed levels; compare dominant and rare categories before testing relationships."
        profiles.append(item)

    corr = df[groups.numeric].corr(method="pearson").fillna(0).to_dict() if len(groups.numeric) > 1 else {}
    spearman = df[groups.numeric].corr(method="spearman").fillna(0).to_dict() if len(groups.numeric) > 1 else {}
    insights = generate_insights(df, profiles, groups)
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicates": int(df.duplicated().sum()),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
        "groups": groups.__dict__,
        "profiles": profiles,
        "numeric_stats": numeric_stats,
        "pearson_correlation": corr,
        "spearman_correlation": spearman,
        "insights": insights,
    }


def generate_insights(df: pd.DataFrame, profiles: list[dict[str, Any]], groups: ColumnGroups) -> list[str]:
    completeness = 100 - df.isna().mean().mean() * 100
    insights = [f"The dataset contains {len(df):,} rows, {len(df.columns):,} columns, and {completeness:.1f}% cell completeness."]
    missing_hotspot = max(profiles, key=lambda item: item["missing_percent"], default=None)
    if missing_hotspot and missing_hotspot["missing_percent"] > 0:
        insights.append(f"{missing_hotspot['name']} has the highest missingness at {missing_hotspot['missing_percent']:.1f}%.")
    if groups.numeric:
        outlier_hotspot = max((p for p in profiles if "outliers" in p), key=lambda item: item["outliers"]["iqr"], default=None)
        if outlier_hotspot:
            insights.append(f"{outlier_hotspot['name']} has {outlier_hotspot['outliers']['iqr']} IQR outliers and should be reviewed before modeling.")
    if len(groups.numeric) > 1:
        corr = df[groups.numeric].corr().abs().where(~np.eye(len(groups.numeric), dtype=bool)).stack()
        if not corr.empty:
            pair = corr.idxmax()
            insights.append(f"The strongest numeric relationship is between {pair[0]} and {pair[1]} (|r|={corr.max():.2f}).")
    insights.append("For ML readiness, impute missing values, encode categoricals, scale continuous variables, and evaluate train/test performance with cross-validation.")
    return insights


def preprocessing_pipeline(df: pd.DataFrame, target: str | None = None) -> ColumnTransformer:
    groups = column_groups(df.drop(columns=[target], errors="ignore"))
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric_pipeline, groups.numeric), ("categorical", categorical_pipeline, groups.categorical + groups.boolean)])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "InsightForge Analytics Engine"}


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    frames = [clean_frame(read_upload(file)) for file in files]
    df = pd.concat(frames, ignore_index=True, sort=False)
    dataset_id = str(uuid.uuid4())
    DATASET_CACHE[dataset_id] = df
    return {"dataset_id": dataset_id, "analysis": profile_dataset(df), "preview": json.loads(df.head(25).to_json(orient="records", date_format="iso"))}


@app.get("/datasets/{dataset_id}/analysis")
def analysis(dataset_id: str) -> dict[str, Any]:
    df = DATASET_CACHE.get(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found in cache.")
    return profile_dataset(df)


@app.post("/statistics")
def statistical_test(request: TestRequest) -> dict[str, Any]:
    df = DATASET_CACHE.get(request.dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found in cache.")
    if request.test in {"pearson", "spearman"}:
        if not request.left or not request.right:
            raise HTTPException(status_code=400, detail="left and right numeric variables are required.")
        pair = df[[request.left, request.right]].apply(pd.to_numeric, errors="coerce").dropna()
        result = stats.pearsonr(pair[request.left], pair[request.right]) if request.test == "pearson" else stats.spearmanr(pair[request.left], pair[request.right])
        return {"test": request.test, "statistic": float(result.statistic), "p_value": float(result.pvalue), "n": int(len(pair)), "assumptions": "Pearson expects linear numeric relationships; Spearman expects monotonic ordinal/numeric relationships.", "interpretation": f"The relationship is {'statistically significant' if result.pvalue < 0.05 else 'not statistically significant'} at alpha=.05."}
    if request.test == "chi_square":
        if not request.left or not request.right:
            raise HTTPException(status_code=400, detail="left and right are required.")
        table = pd.crosstab(df[request.left], df[request.right])
        if table.shape[0] < 2 or table.shape[1] < 2:
            raise HTTPException(status_code=400, detail="Chi-square requires at least two levels in each categorical variable.")
        chi2, p, dof, expected = stats.chi2_contingency(table)
        return {"test": "chi_square", "statistic": float(chi2), "p_value": float(p), "dof": int(dof), "observed": table.to_dict(), "expected": expected.tolist(), "assumptions": "Independent observations and sufficiently large expected cell counts.", "interpretation": f"The categorical variables are {'associated' if p < 0.05 else 'not detectably associated'} at alpha=.05."}
    if not request.numeric or not request.group:
        raise HTTPException(status_code=400, detail="numeric and group are required.")
    groups = [pd.to_numeric(part[request.numeric], errors="coerce").dropna() for _, part in df.groupby(request.group) if len(part) > 1]
    if request.test in {"ttest", "mann_whitney"} and len(groups) < 2:
        raise HTTPException(status_code=400, detail="This test requires at least two groups.")
    if request.test in {"anova", "kruskal"} and len(groups) < 2:
        raise HTTPException(status_code=400, detail="This test requires at least two groups.")
    if request.test == "ttest":
        result = stats.ttest_ind(groups[0], groups[1], equal_var=False, nan_policy="omit")
        assumptions = "Two independent groups, numeric outcome, approximate normality; Welch t-test relaxes equal variances."
    elif request.test == "anova":
        result = stats.f_oneway(*groups)
        assumptions = "Independent groups, numeric outcome, approximately normal residuals, and similar variances."
    elif request.test == "mann_whitney":
        result = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        assumptions = "Two independent groups and ordinal/numeric outcome; non-parametric alternative to t-test."
    else:
        result = stats.kruskal(*groups)
        assumptions = "Independent groups and ordinal/numeric outcome; non-parametric alternative to ANOVA."
    return {"test": request.test, "statistic": float(result.statistic), "p_value": float(result.pvalue), "groups": len(groups), "assumptions": assumptions, "interpretation": f"The result is {'statistically significant' if result.pvalue < 0.05 else 'not statistically significant'} at alpha=.05; inspect group plots and effect sizes next."}


@app.post("/visualizations")
def visualization(request: ChartRequest) -> dict[str, Any]:
    df = DATASET_CACHE.get(request.dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found in cache.")
    fig: go.Figure
    if request.chart in {"histogram", "kde"}:
        fig = px.histogram(df, x=request.x, color=request.color, marginal="rug", nbins=40)
    elif request.chart in {"box", "violin"}:
        fig = px.violin(df, x=request.color, y=request.y or request.x, box=True, points="outliers") if request.chart == "violin" else px.box(df, x=request.color, y=request.y or request.x, points="outliers")
    elif request.chart in {"scatter", "regression", "joint", "hexbin"}:
        fig = px.scatter(df, x=request.x, y=request.y, color=request.color, trendline="ols" if request.chart == "regression" else None)
    elif request.chart in {"bar", "count", "stacked_bar"}:
        fig = px.bar(df, x=request.x, y=request.y, color=request.color, barmode="stack" if request.chart == "stacked_bar" else "group")
    elif request.chart == "pie":
        fig = px.pie(df, names=request.x, values=request.y)
    elif request.chart == "heatmap":
        groups = column_groups(df)
        fig = px.imshow(df[groups.numeric].corr(), text_auto=True, aspect="auto", color_continuous_scale="RdBu")
    elif request.chart == "pair":
        groups = column_groups(df)
        fig = px.scatter_matrix(df, dimensions=groups.numeric[:6], color=request.color)
    elif request.chart == "area":
        fig = px.area(df, x=request.x, y=request.y, color=request.color)
    elif request.chart == "scatter_3d":
        groups = column_groups(df)
        z = next((col for col in groups.numeric if col not in {request.x, request.y}), request.y)
        fig = px.scatter_3d(df, x=request.x, y=request.y, z=z, color=request.color)
    elif request.chart == "geo":
        fig = px.scatter_geo(df, lat=request.lat, lon=request.lon, color=request.color, hover_name=request.x)
    else:
        fig = ff.create_distplot([df[request.x].dropna().tolist()], [request.x or "distribution"])
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
    return json.loads(fig.to_json())


@app.post("/models")
def train_model(request: ModelRequest) -> dict[str, Any]:
    df = DATASET_CACHE.get(request.dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found in cache.")
    features = request.features or [col for col in df.columns if col != request.target]
    modeling_df = df[features + [request.target]].dropna(subset=[request.target])
    x_train, x_test, y_train, y_test = train_test_split(modeling_df[features], modeling_df[request.target], test_size=0.2, random_state=42)
    preprocess = preprocessing_pipeline(modeling_df, request.target)
    estimators = {
        "linear": LinearRegression(),
        "logistic": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestRegressor(random_state=42) if request.task == "regression" else RandomForestClassifier(random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "knn": KNeighborsClassifier(),
        "svm": SVC(probability=True),
    }
    estimator = estimators[request.estimator]
    pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    if request.task == "regression":
        model = sm.OLS(pd.to_numeric(y_train, errors="coerce"), sm.add_constant(pd.get_dummies(x_train, drop_first=True).apply(pd.to_numeric, errors="coerce").fillna(0))).fit()
        residual_fig = px.scatter(x=model.fittedvalues, y=model.resid, labels={"x": "Fitted values", "y": "Residuals"}, title="Residual plot")
        residual_fig.add_hline(y=0, line_dash="dash")
        return {
            "task": "regression",
            "r2": float(r2_score(y_test, predictions)),
            "rmse": float(mean_squared_error(y_test, predictions, squared=False)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "coefficients": [{"term": str(term), "coefficient": float(coef), "p_value": float(model.pvalues.loc[term])} for term, coef in model.params.items()],
            "cross_validation_r2": cross_val_score(pipeline, modeling_df[features], modeling_df[request.target], cv=min(5, len(modeling_df)), scoring="r2").tolist(),
            "diagnostics": {"aic": float(model.aic), "bic": float(model.bic), "residual_std": float(model.resid.std())},
            "residual_plot": json.loads(residual_fig.to_json()),
            "interpretation": "Positive coefficients increase the expected outcome while negative coefficients decrease it, holding encoded predictors constant. Review p-values and residual structure before reporting.",
        }
    probabilities = pipeline.predict_proba(x_test)[:, 1] if hasattr(pipeline.named_steps["model"], "predict_proba") and len(np.unique(y_test)) == 2 else None
    return {
        "task": "classification",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if probabilities is not None else None,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "cross_validation_accuracy": cross_val_score(pipeline, modeling_df[features], modeling_df[request.target], cv=min(5, len(modeling_df)), scoring="accuracy").tolist(),
        "interpretation": "Accuracy summarizes held-out classification performance; inspect the confusion matrix to identify which class is being missed and use ROC AUC for binary discrimination quality.",
    }


@app.get("/datasets/{dataset_id}/report")
def report(dataset_id: str) -> dict[str, Any]:
    df = DATASET_CACHE.get(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found in cache.")
    analysis_payload = profile_dataset(df)
    return {"title": "InsightForge AI Analytics Report", "dataset_id": dataset_id, "analysis": analysis_payload, "export_formats": ["html", "pdf", "csv", "png"]}
