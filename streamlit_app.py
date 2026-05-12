"""Professional Streamlit data analysis app for InsightForge.

This app is intentionally self-contained and does not replace the existing Next.js
frontend. It reuses the same Python analytics stack (Pandas, SciPy, Statsmodels,
Scikit-learn, and Plotly) to compute real statistics from the uploaded dataset.
"""

from __future__ import annotations

import html
import io
import os
import re
import textwrap
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ask_your_data_engine import (
    AnalysisIntent,
    AnalysisResult,
    dataframe_metadata,
    detect_analysis_intent,
    needs_column_selection,
    run_requested_analysis,
    suggest_smart_actions,
)


ALPHA = 0.05
MAX_DISTINCT_CATEGORIES = 60


@dataclass(frozen=True)
class ColumnTypes:
    numeric: list[str]
    categorical: list[str]
    datetime: list[str]
    binary: list[str]


# -----------------------------------------------------------------------------
# Data loading, cleaning, and profiling
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataframe(payload: bytes, filename: str) -> pd.DataFrame:
    """Load CSV, Excel, or JSON bytes into a dataframe."""
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    buffer = io.BytesIO(payload)
    if suffix == "csv":
        try:
            df = pd.read_csv(buffer)
        except UnicodeDecodeError:
            buffer.seek(0)
            df = pd.read_csv(buffer, encoding="latin-1")
    elif suffix == "xlsx":
        df = pd.read_excel(buffer, engine="openpyxl")
    elif suffix == "json":
        df = pd.read_json(buffer)
    else:
        raise ValueError("Supported file formats are CSV, XLSX, and JSON.")

    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    return df


def infer_column_types(df: pd.DataFrame) -> ColumnTypes:
    """Infer numeric, categorical, datetime, and binary columns."""
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    binary_cols = [
        col
        for col in df.columns
        if df[col].dropna().nunique() == 2 and col not in datetime_cols
    ]
    categorical_cols = [
        col
        for col in df.columns
        if col not in numeric_cols and col not in datetime_cols and col not in bool_cols
    ]

    return ColumnTypes(
        numeric=sorted(numeric_cols),
        categorical=sorted(set(categorical_cols + bool_cols)),
        datetime=sorted(datetime_cols),
        binary=sorted(binary_cols),
    )


def _looks_datetime(series: pd.Series) -> bool:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    non_missing = series.dropna().astype(str)
    if len(non_missing) < 3 or non_missing.str.contains(r"\d", regex=True).mean() < 0.5:
        return False
    converted = pd.to_datetime(non_missing, errors="coerce")
    return bool(converted.notna().mean() >= 0.75)


@st.cache_data(show_spinner=False)
def clean_dataframe(
    df: pd.DataFrame,
    drop_duplicates: bool,
    numeric_fill: str,
    categorical_fill: str,
    convert_datetimes: bool,
    remove_outliers: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply user-selected cleaning rules and return a cleaning audit trail."""
    cleaned = df.copy()
    notes: list[str] = []
    original_shape = cleaned.shape

    cleaned = cleaned.replace({"": np.nan, "NA": np.nan, "N/A": np.nan, "null": np.nan, "None": np.nan})

    if convert_datetimes:
        converted_cols: list[str] = []
        for col in cleaned.columns:
            if _looks_datetime(cleaned[col]):
                cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")
                converted_cols.append(col)
        notes.append(f"Converted {len(converted_cols)} probable datetime column(s): {', '.join(converted_cols) or 'none'}.")

    for col in cleaned.select_dtypes(include="object").columns:
        numeric_candidate = pd.to_numeric(
            cleaned[col].astype(str).str.replace(r"[$,% ,]", "", regex=True),
            errors="coerce",
        )
        if numeric_candidate.notna().mean() >= 0.9 and numeric_candidate.notna().sum() > 0:
            cleaned[col] = numeric_candidate

    duplicates_before = int(cleaned.duplicated().sum())
    if drop_duplicates and duplicates_before:
        cleaned = cleaned.drop_duplicates()
        notes.append(f"Dropped {duplicates_before:,} duplicate row(s).")
    else:
        notes.append(f"Detected {duplicates_before:,} duplicate row(s); duplicate removal {'was not selected' if duplicates_before else 'was not needed'}.")

    numeric_cols = cleaned.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [col for col in cleaned.columns if col not in numeric_cols and not pd.api.types.is_datetime64_any_dtype(cleaned[col])]

    numeric_missing = int(cleaned[numeric_cols].isna().sum().sum()) if numeric_cols else 0
    if numeric_cols and numeric_fill != "Do not fill":
        for col in numeric_cols:
            if numeric_fill == "Mean":
                value = cleaned[col].mean()
            elif numeric_fill == "Median":
                value = cleaned[col].median()
            else:
                value = 0
            cleaned[col] = cleaned[col].fillna(value)
        notes.append(f"Filled {numeric_missing:,} numeric missing value(s) using {numeric_fill.lower()}.")
    else:
        notes.append(f"Numeric missing values left unchanged: {numeric_missing:,}.")

    categorical_missing = int(cleaned[categorical_cols].isna().sum().sum()) if categorical_cols else 0
    if categorical_cols and categorical_fill != "Do not fill":
        for col in categorical_cols:
            if categorical_fill == "Mode":
                mode = cleaned[col].mode(dropna=True)
                value = mode.iloc[0] if not mode.empty else "Missing"
            else:
                value = "Missing"
            cleaned[col] = cleaned[col].fillna(value)
        notes.append(f"Filled {categorical_missing:,} categorical missing value(s) using {categorical_fill.lower()}.")
    else:
        notes.append(f"Categorical missing values left unchanged: {categorical_missing:,}.")

    if remove_outliers and numeric_cols:
        mask = pd.Series(True, index=cleaned.index)
        affected_cols: list[str] = []
        for col in numeric_cols:
            values = cleaned[col].dropna()
            if values.nunique() <= 2:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                continue
            col_mask = cleaned[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr) | cleaned[col].isna()
            removed = int((~col_mask).sum())
            if removed:
                affected_cols.append(f"{col} ({removed:,})")
            mask &= col_mask
        before = len(cleaned)
        cleaned = cleaned.loc[mask].copy()
        notes.append(f"Removed {before - len(cleaned):,} row(s) with IQR outliers across selected numeric columns: {', '.join(affected_cols[:8]) or 'none'}.")

    notes.append(f"Final cleaned shape: {cleaned.shape[0]:,} rows × {cleaned.shape[1]:,} columns (from {original_shape[0]:,} × {original_shape[1]:,}).")
    return cleaned, notes


@st.cache_data(show_spinner=False)
def numeric_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in df.select_dtypes(include=np.number).columns:
        series = pd.to_numeric(df[col], errors="coerce")
        non_missing = series.dropna()
        mode = non_missing.mode()
        q1, q3 = non_missing.quantile([0.25, 0.75]) if not non_missing.empty else (np.nan, np.nan)
        rows.append(
            {
                "column": col,
                "count": int(non_missing.count()),
                "mean": non_missing.mean(),
                "median": non_missing.median(),
                "mode": mode.iloc[0] if not mode.empty else np.nan,
                "std": non_missing.std(ddof=1),
                "variance": non_missing.var(ddof=1),
                "min": non_missing.min(),
                "max": non_missing.max(),
                "range": non_missing.max() - non_missing.min() if not non_missing.empty else np.nan,
                "IQR": q3 - q1 if not non_missing.empty else np.nan,
                "skewness": non_missing.skew(),
                "kurtosis": non_missing.kurtosis(),
                "missing_count": int(series.isna().sum()),
                "missing_percent": float(series.isna().mean() * 100),
            }
        )
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


@st.cache_data(show_spinner=False)
def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    types = infer_column_types(df)
    columns = sorted(set(types.categorical + types.binary))
    rows: list[dict[str, Any]] = []
    for col in columns:
        counts = df[col].astype("object").value_counts(dropna=True)
        top_value = counts.index[0] if not counts.empty else None
        rows.append(
            {
                "column": col,
                "unique_count": int(df[col].nunique(dropna=True)),
                "top_value": top_value,
                "top_count": int(counts.iloc[0]) if not counts.empty else 0,
                "top_percent": float((counts.iloc[0] / len(df)) * 100) if len(df) and not counts.empty else 0,
                "missing_count": int(df[col].isna().sum()),
                "missing_percent": float(df[col].isna().mean() * 100),
            }
        )
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


@st.cache_data(show_spinner=False)
def frequency_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = df[column].astype("object").value_counts(dropna=False).rename("count")
    out = counts.to_frame()
    out["percent"] = out["count"] / len(df) * 100 if len(df) else 0
    out.index = out.index.map(lambda value: "<Missing>" if pd.isna(value) else str(value))
    return out


def iqr_outlier_count(series: pd.Series) -> int:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.nunique() <= 2:
        return 0
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0 or pd.isna(iqr):
        return 0
    return int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())


# -----------------------------------------------------------------------------
# Statistical tests and modeling
# -----------------------------------------------------------------------------


def record_session_result(result: str) -> None:
    """Keep the export report useful without appending duplicate rerun entries."""
    if "test_results" not in st.session_state:
        st.session_state.test_results = []
    if result not in st.session_state.test_results:
        st.session_state.test_results.append(result)

def significance_text(p_value: float, alpha: float = ALPHA) -> str:
    if pd.isna(p_value):
        return "The p-value could not be calculated for this selection."
    return (
        f"Statistically significant at α = {alpha:.2f} (p = {p_value:.4g})."
        if p_value < alpha
        else f"Not statistically significant at α = {alpha:.2f} (p = {p_value:.4g})."
    )


@st.cache_data(show_spinner=False)
def chi_square_analysis(df: pd.DataFrame, col_a: str, col_b: str) -> dict[str, Any]:
    observed = pd.crosstab(df[col_a], df[col_b])
    if observed.shape[0] < 2 or observed.shape[1] < 2:
        return {"valid": False, "message": "Chi-square requires at least two categories in each selected column."}
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    expected_df = pd.DataFrame(expected, index=observed.index, columns=observed.columns)
    return {
        "valid": True,
        "observed": observed,
        "normalized": pd.crosstab(df[col_a], df[col_b], normalize="index") * 100,
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "expected": expected_df,
        "interpretation": significance_text(float(p_value)),
    }


@st.cache_data(show_spinner=False)
def correlation_test(df: pd.DataFrame, x: str, y: str, method: str) -> dict[str, Any]:
    pair = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3:
        return {"valid": False, "message": "Correlation requires at least 3 paired non-missing observations."}
    stat, p_value = stats.pearsonr(pair[x], pair[y]) if method == "Pearson" else stats.spearmanr(pair[x], pair[y])
    return {"valid": True, "statistic": float(stat), "p_value": float(p_value), "n": int(len(pair)), "interpretation": significance_text(float(p_value))}


@st.cache_data(show_spinner=False)
def grouped_test(df: pd.DataFrame, numeric_col: str, group_col: str, test_name: str) -> dict[str, Any]:
    subset = df[[numeric_col, group_col]].dropna()
    groups = [pd.to_numeric(g[numeric_col], errors="coerce").dropna() for _, g in subset.groupby(group_col)]
    groups = [g for g in groups if len(g) > 0]

    if test_name in {"Independent samples t-test", "Mann-Whitney U"} and len(groups) != 2:
        return {"valid": False, "message": f"{test_name} requires exactly two non-empty groups."}
    if test_name in {"One-way ANOVA", "Kruskal-Wallis"} and len(groups) < 2:
        return {"valid": False, "message": f"{test_name} requires at least two non-empty groups."}

    if test_name == "Independent samples t-test":
        stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=False, nan_policy="omit")
    elif test_name == "One-way ANOVA":
        stat, p_value = stats.f_oneway(*groups)
    elif test_name == "Mann-Whitney U":
        stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
    else:
        stat, p_value = stats.kruskal(*groups)

    return {
        "valid": True,
        "statistic": float(stat),
        "p_value": float(p_value),
        "n": int(sum(len(g) for g in groups)),
        "group_sizes": {str(name): int(len(group)) for name, group in subset.groupby(group_col)[numeric_col]},
        "interpretation": significance_text(float(p_value)),
    }


@st.cache_data(show_spinner=False)
def linear_regression_model(df: pd.DataFrame, target: str, predictors: list[str]) -> dict[str, Any]:
    model_df = df[[target] + predictors].dropna()
    if len(model_df) < max(10, len(predictors) + 3):
        return {"valid": False, "message": "Linear regression needs more complete rows than predictors."}

    encoded = pd.get_dummies(model_df[predictors], drop_first=True, dtype=float)
    y = pd.to_numeric(model_df[target], errors="coerce")
    valid = y.notna()
    encoded = encoded.loc[valid]
    y = y.loc[valid]
    if encoded.empty or y.nunique() < 2:
        return {"valid": False, "message": "Target and predictors do not contain enough variation for regression."}

    x_const = sm.add_constant(encoded, has_constant="add")
    result = sm.OLS(y, x_const).fit()
    predictions = result.predict(x_const)
    residuals = y - predictions
    coefficients = pd.DataFrame(
        {
            "coefficient": result.params,
            "p_value": result.pvalues,
            "std_error": result.bse,
        }
    )
    return {
        "valid": True,
        "n": int(len(y)),
        "r_squared": float(result.rsquared),
        "adjusted_r_squared": float(result.rsquared_adj),
        "intercept": float(result.params.get("const", np.nan)),
        "coefficients": coefficients,
        "residual_summary": residuals.describe().to_frame("residual"),
        "predictions": pd.DataFrame({"actual": y, "predicted": predictions, "residual": residuals}),
    }


@st.cache_data(show_spinner=False)
def logistic_regression_model(df: pd.DataFrame, target: str, predictors: list[str]) -> dict[str, Any]:
    model_df = df[[target] + predictors].dropna()
    target_values = model_df[target].dropna().unique()
    if len(target_values) != 2:
        return {"valid": False, "message": "Logistic regression requires a binary target with exactly two classes."}
    if len(model_df) < 20:
        return {"valid": False, "message": "Logistic regression needs at least 20 complete rows for a reliable split."}

    y = pd.Categorical(model_df[target]).codes
    x = model_df[predictors]
    numeric_features = x.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [col for col in predictors if col not in numeric_features]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features))
    if categorical_features:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features))

    preprocess = ColumnTransformer(transformers=transformers)
    model = Pipeline([("preprocess", preprocess), ("classifier", LogisticRegression(max_iter=1000))])
    stratify = y if min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=stratify)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefs = pd.DataFrame({"feature": feature_names, "coefficient": model.named_steps["classifier"].coef_[0]}).sort_values("coefficient", key=np.abs, ascending=False)
    cm = pd.DataFrame(confusion_matrix(y_test, predictions), index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    report = pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).T
    fpr, tpr, _ = roc_curve(y_test, probabilities)

    return {
        "valid": True,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "coefficients": coefs,
        "confusion_matrix": cm,
        "classification_report": report,
        "roc": pd.DataFrame({"fpr": fpr, "tpr": tpr}),
        "classes": [str(v) for v in pd.Categorical(model_df[target]).categories],
    }


# -----------------------------------------------------------------------------
# Narrative summary and export report
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_relationship_findings(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include=np.number)
    findings: dict[str, Any] = {"positive": None, "negative": None, "significant": []}
    if numeric.shape[1] < 2:
        return findings

    corr = numeric.corr(numeric_only=True)
    pairs: list[tuple[str, str, float, float, int]] = []
    cols = corr.columns.tolist()
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            pair = numeric[[left, right]].dropna()
            if len(pair) >= 3 and pair[left].nunique() > 1 and pair[right].nunique() > 1:
                r, p = stats.pearsonr(pair[left], pair[right])
                pairs.append((left, right, float(r), float(p), int(len(pair))))

    if pairs:
        findings["positive"] = max(pairs, key=lambda item: item[2])
        findings["negative"] = min(pairs, key=lambda item: item[2])
        findings["significant"] = sorted([p for p in pairs if p[3] < ALPHA], key=lambda item: abs(item[2]), reverse=True)[:5]
    return findings


@st.cache_data(show_spinner=False)
def generate_summary_analysis(df: pd.DataFrame, original_df: pd.DataFrame, cleaning_notes: list[str], regression_summary: str = "") -> str:
    types = infer_column_types(df)
    num_stats = numeric_descriptive_stats(df)
    cat_stats = categorical_summary(df)
    missing_total = int(df.isna().sum().sum())
    missing_cols = df.isna().sum().sort_values(ascending=False)
    missing_cols = missing_cols[missing_cols > 0].head(5)
    duplicates = int(df.duplicated().sum())

    lines = [
        "## Executive Summary Analysis",
        "",
        f"The cleaned dataset contains **{df.shape[0]:,} rows** and **{df.shape[1]:,} columns**. Column detection found **{len(types.numeric):,} numeric**, **{len(types.categorical):,} categorical**, **{len(types.datetime):,} datetime**, and **{len(types.binary):,} binary** column(s).",
        f"Across the cleaned data there are **{missing_total:,} missing values** and **{duplicates:,} duplicate rows**. The original upload contained **{original_df.shape[0]:,} rows × {original_df.shape[1]:,} columns**.",
    ]

    if not missing_cols.empty:
        missing_text = ", ".join(f"{col} ({int(count):,})" for col, count in missing_cols.items())
        lines.append(f"The columns with the most remaining missing values are: {missing_text}.")
    else:
        lines.append("No remaining missing values were detected after the selected cleaning options.")

    lines.extend(["", "### Cleaning Summary"])
    lines.extend(f"- {note}" for note in cleaning_notes)

    lines.extend(["", "### Key Numeric Findings"])
    if not num_stats.empty:
        highest_mean = num_stats["mean"].idxmax()
        lowest_mean = num_stats["mean"].idxmin()
        highest_var = num_stats["std"].idxmax()
        skewed = num_stats[num_stats["skewness"].abs() >= 1].sort_values("skewness", key=lambda s: s.abs(), ascending=False).head(3)
        outlier_counts = {col: iqr_outlier_count(df[col]) for col in types.numeric}
        outlier_counts = dict(sorted(outlier_counts.items(), key=lambda item: item[1], reverse=True)[:3])
        lines.append(f"- **{highest_mean}** has the highest mean ({num_stats.loc[highest_mean, 'mean']:.3g}), while **{lowest_mean}** has the lowest mean ({num_stats.loc[lowest_mean, 'mean']:.3g}).")
        lines.append(f"- **{highest_var}** shows the highest variation by standard deviation ({num_stats.loc[highest_var, 'std']:.3g}).")
        if not skewed.empty:
            lines.append("- Strong skewness appears in " + ", ".join(f"**{idx}** (skew={row['skewness']:.2f})" for idx, row in skewed.iterrows()) + ".")
        else:
            lines.append("- No numeric variable has absolute skewness of 1.0 or higher, suggesting no extreme one-sided distribution by this rule.")
        heavy_outliers = [(col, count) for col, count in outlier_counts.items() if count > 0]
        if heavy_outliers:
            lines.append("- Potential IQR outlier-heavy variables include " + ", ".join(f"**{col}** ({count:,} rows)" for col, count in heavy_outliers) + ".")
        else:
            lines.append("- IQR screening did not flag notable numeric outlier counts.")
    else:
        lines.append("- No numeric columns are available for numeric summaries, correlations, or numeric-target regression.")

    lines.extend(["", "### Key Categorical Findings"])
    if not cat_stats.empty:
        for col, row in cat_stats.sort_values("top_percent", ascending=False).head(5).iterrows():
            imbalance = " This is an imbalanced category and may dominate grouped results." if row["top_percent"] >= 70 else ""
            lines.append(f"- In **{col}**, the most common value is **{row['top_value']}** ({row['top_count']:,} rows, {row['top_percent']:.1f}%).{imbalance}")
    else:
        lines.append("- No categorical columns were detected for frequency or cross-tab analysis.")

    lines.extend(["", "### Relationship Findings"])
    relationships = compute_relationship_findings(df)
    if relationships.get("positive"):
        left, right, corr, p_value, n = relationships["positive"]
        lines.append(f"- Strongest positive Pearson correlation: **{left}** vs **{right}** (r={corr:.3f}, p={p_value:.4g}, n={n:,}).")
    if relationships.get("negative"):
        left, right, corr, p_value, n = relationships["negative"]
        lines.append(f"- Strongest negative Pearson correlation: **{left}** vs **{right}** (r={corr:.3f}, p={p_value:.4g}, n={n:,}).")
    if relationships.get("significant"):
        sig_text = "; ".join(f"{a}–{b} (r={r:.2f}, p={p:.3g})" for a, b, r, p, _ in relationships["significant"][:3])
        lines.append(f"- Statistically significant correlations at α=0.05 include: {sig_text}.")
    elif len(types.numeric) >= 2:
        lines.append("- No Pearson correlations met the p < 0.05 threshold among the scanned numeric pairs.")

    cat_cols = sorted(set(types.categorical + types.binary))[:6]
    chi_findings: list[str] = []
    for i, left in enumerate(cat_cols):
        for right in cat_cols[i + 1 :]:
            if df[left].nunique(dropna=True) <= MAX_DISTINCT_CATEGORIES and df[right].nunique(dropna=True) <= MAX_DISTINCT_CATEGORIES:
                result = chi_square_analysis(df, left, right)
                if result.get("valid") and result["p_value"] < ALPHA:
                    chi_findings.append(f"{left} × {right} (χ²={result['chi2']:.2f}, p={result['p_value']:.3g})")
    if chi_findings:
        lines.append("- Significant categorical associations include: " + "; ".join(chi_findings[:3]) + ".")
    elif len(cat_cols) >= 2:
        lines.append("- The automatic Chi-square scan did not find a significant categorical association among the first several categorical/binary columns.")

    lines.extend(["", "### Modeling Findings"])
    lines.append(regression_summary or "- Run a model in the Regression Modeling tab to add performance and predictor findings here. Numeric targets use OLS; binary targets use logistic regression.")

    lines.extend(
        [
            "",
            "### Recommended Next Steps",
            f"- Prioritize cleaning columns with high missingness and review the cleaning audit trail before formal reporting ({missing_total:,} missing values remain).",
            "- Visualize the highest-variation numeric fields with histograms, boxplots, and scatter plots to understand distribution shape and outliers.",
            "- Use Chi-square tests for business questions involving two categorical fields, and use Pearson/Spearman correlations for numeric relationship questions.",
            "- For modeling, start with interpretable predictors, check residual diagnostics for numeric targets, and validate binary classification with accuracy, confusion matrix, and ROC curve.",
        ]
    )
    return "\n".join(lines)


def build_report(df: pd.DataFrame, original_df: pd.DataFrame, cleaning_notes: list[str], summary_text: str, test_results: list[str], regression_text: str) -> str:
    num_stats = numeric_descriptive_stats(df)
    cat_stats = categorical_summary(df)
    relationships = compute_relationship_findings(df)
    sections = [
        "# InsightForge Streamlit Analysis Report",
        "",
        f"Generated for a dataset with {df.shape[0]:,} rows and {df.shape[1]:,} columns after cleaning.",
        "",
        summary_text,
        "",
        "## Descriptive Statistics",
        num_stats.head(25).to_markdown() if not num_stats.empty else "No numeric descriptive statistics available.",
        "",
        "## Categorical Summary",
        cat_stats.head(25).to_markdown() if not cat_stats.empty else "No categorical summary available.",
        "",
        "## Key Correlations",
    ]
    if relationships.get("significant"):
        sections.extend(f"- {a} vs {b}: r={r:.4f}, p={p:.4g}, n={n}" for a, b, r, p, n in relationships["significant"])
    else:
        sections.append("No significant Pearson correlations were identified in the automatic scan.")
    sections.extend(["", "## Statistical Test Results"])
    sections.extend(test_results or ["No interactive test results have been recorded during this session."])
    sections.extend(["", "## Regression Results", regression_text or "No regression model has been run during this session."])
    sections.extend(["", "## Cleaning Notes"])
    sections.extend(f"- {note}" for note in cleaning_notes)
    return "\n".join(sections)



# -----------------------------------------------------------------------------
# Storytelling, ML, chat, and business-report helpers
# -----------------------------------------------------------------------------
def safe_pct(numerator: float, denominator: float) -> float:
    return float(numerator / denominator * 100) if denominator else 0.0


def summarize_key_findings(df: pd.DataFrame) -> dict[str, str]:
    types = infer_column_types(df)
    relationships = compute_relationship_findings(df)
    num_stats = numeric_descriptive_stats(df)
    cat_stats = categorical_summary(df)
    missing = df.isna().mean().sort_values(ascending=False)

    positive = "Not available — fewer than two numeric columns with enough variation."
    if relationships.get("positive"):
        left, right, corr, _, n = relationships["positive"]
        positive = f"{left} ↔ {right} (r={corr:.3f}, n={n:,})"

    negative = "Not available — fewer than two numeric columns with enough variation."
    if relationships.get("negative"):
        left, right, corr, _, n = relationships["negative"]
        negative = f"{left} ↔ {right} (r={corr:.3f}, n={n:,})"

    variable = "No numeric columns detected."
    if not num_stats.empty and "std" in num_stats:
        variable_col = num_stats["std"].fillna(-np.inf).idxmax()
        variable = f"{variable_col} (standard deviation {num_stats.loc[variable_col, 'std']:.3g})"

    imbalanced = "No categorical columns detected."
    if not cat_stats.empty:
        imbalanced_col = cat_stats["top_percent"].fillna(0).idxmax()
        row = cat_stats.loc[imbalanced_col]
        imbalanced = f"{imbalanced_col}: {row['top_value']} is {row['top_percent']:.1f}% of rows"

    missing_col = "No missing values detected."
    if not missing.empty and missing.iloc[0] > 0:
        missing_col = f"{missing.index[0]} ({missing.iloc[0] * 100:.1f}% missing)"

    return {
        "strongest_positive_correlation": positive,
        "strongest_negative_correlation": negative,
        "most_variable_numeric_column": variable,
        "most_imbalanced_categorical_column": imbalanced,
        "highest_missing_value_column": missing_col,
        "numeric_count": str(len(types.numeric)),
        "categorical_count": str(len(set(types.categorical + types.binary))),
    }


def storytelling_text(df: pd.DataFrame, original_df: pd.DataFrame, cleaning_notes: list[str]) -> str:
    types = infer_column_types(df)
    findings = summarize_key_findings(df)
    missing_pct = safe_pct(df.isna().sum().sum(), df.shape[0] * df.shape[1])
    quality_risk = "low" if missing_pct < 5 and df.duplicated().sum() == 0 else "moderate to high"
    dominant_types = []
    if types.numeric:
        dominant_types.append(f"{len(types.numeric)} numeric measures")
    if types.categorical or types.binary:
        dominant_types.append(f"{len(set(types.categorical + types.binary))} categorical/binary descriptors")
    if types.datetime:
        dominant_types.append(f"{len(types.datetime)} datetime fields")
    contains = ", ".join(dominant_types) or "general tabular fields"
    return "\n".join([
        f"This uploaded dataset contains **{df.shape[0]:,} cleaned rows** and **{df.shape[1]:,} columns**, with {contains}. The original upload had {original_df.shape[0]:,} rows and {original_df.shape[1]:,} columns.",
        f"Patterns that stand out include the strongest positive relationship of **{findings['strongest_positive_correlation']}**, the strongest negative relationship of **{findings['strongest_negative_correlation']}**, and variation concentrated in **{findings['most_variable_numeric_column']}**.",
        f"Data-quality risk is **{quality_risk}**: missingness is {missing_pct:.1f}% of cells, duplicate rows total {int(df.duplicated().sum()):,}, and the highest missingness signal is **{findings['highest_missing_value_column']}**.",
        "Next, investigate whether the strongest relationships are practically meaningful, whether dominant categories bias conclusions, and which predictors explain the most important outcomes for the business or academic question.",
        "Cleaning notes: " + "; ".join(cleaning_notes[:4]),
    ])


def recommended_story_charts(df: pd.DataFrame) -> list[tuple[str, go.Figure, str]]:
    types = infer_column_types(df)
    charts: list[tuple[str, go.Figure, str]] = []
    if types.numeric:
        col = numeric_descriptive_stats(df)["std"].fillna(0).idxmax() if not numeric_descriptive_stats(df).empty else types.numeric[0]
        charts.append(("Distribution", px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}"), f"This chart shows the spread, center, and possible outliers for {col}, the numeric field with the highest variability."))
    if len(types.numeric) >= 2:
        relationships = compute_relationship_findings(df)
        pair = relationships.get("positive") or relationships.get("negative")
        if pair:
            x, y = pair[0], pair[1]
            charts.append(("Relationship", px.scatter(df, x=x, y=y, trendline="ols", title=f"Relationship: {y} vs {x}"), f"This scatter plot highlights the strongest detected numeric relationship between {x} and {y}."))
        corr = df[types.numeric].corr(numeric_only=True)
        charts.append(("Correlation Heatmap", px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Numeric Correlation Heatmap"), "This heatmap summarizes pairwise numeric relationships; darker colors indicate stronger associations."))
    cat_cols = sorted(set(types.categorical + types.binary))
    if cat_cols:
        col = categorical_summary(df)["top_percent"].fillna(0).idxmax() if not categorical_summary(df).empty else cat_cols[0]
        counts = df[col].astype("object").fillna("<Missing>").value_counts().head(15).reset_index()
        counts.columns = [col, "count"]
        charts.append(("Category Mix", px.bar(counts, x=col, y="count", title=f"Top categories in {col}"), f"This bar chart shows the most common values for {col} and whether one category dominates the dataset."))
    return charts[:4]


def make_preprocessor(x: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    numeric_features = x.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [col for col in x.columns if col not in numeric_features]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), numeric_features))
    if categorical_features:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def get_feature_names(model: Pipeline) -> list[str]:
    return [str(name) for name in model.named_steps["preprocess"].get_feature_names_out()]


def train_ml_model(df: pd.DataFrame, target: str, predictors: list[str], problem_type: str, model_name: str, test_size: float, random_state: int) -> dict[str, Any]:
    if target in predictors:
        return {"valid": False, "message": "The target column cannot also be used as a predictor."}
    if not predictors:
        return {"valid": False, "message": "Select at least one predictor column."}
    model_df = df[[target] + predictors].copy()
    model_df = model_df.dropna(subset=[target])
    if len(model_df) < 10:
        return {"valid": False, "message": "At least 10 rows with a non-missing target are required."}
    x = model_df[predictors]
    y_raw = model_df[target]
    if x.dropna(how="all").empty:
        return {"valid": False, "message": "Predictors are empty after removing rows with missing targets."}

    if problem_type == "auto-detect":
        problem_type = "regression" if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique(dropna=True) > 10 else "classification"
    if problem_type == "regression":
        y = pd.to_numeric(y_raw, errors="coerce")
        valid_y = y.notna()
        x = x.loc[valid_y]
        y = y.loc[valid_y]
        if y.nunique() < 2:
            return {"valid": False, "message": "Regression requires a numeric target with at least two distinct values."}
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=random_state),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state),
        }
        estimator = models[model_name]
        pipeline = Pipeline([("preprocess", make_preprocessor(x, scale_numeric=model_name == "Linear Regression")), ("model", estimator)])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        result = {
            "valid": True, "problem_type": "regression", "model_name": model_name, "target": target, "predictors": predictors,
            "n_train": int(len(x_train)), "n_test": int(len(x_test)),
            "mae": float(mean_absolute_error(y_test, predictions)), "rmse": float(rmse), "r2": float(r2_score(y_test, predictions)),
            "predictions": pd.DataFrame({"actual": y_test, "predicted": predictions, "residual": y_test - predictions}),
        }
    else:
        y = y_raw.astype("object")
        if y.nunique(dropna=True) < 2:
            return {"valid": False, "message": "Classification requires at least two target classes."}
        class_counts = y.value_counts()
        stratify = y if class_counts.min() >= 2 else None
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=200, random_state=random_state),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=random_state),
            "KNN Classifier": KNeighborsClassifier(),
        }
        estimator = models[model_name]
        pipeline = Pipeline([("preprocess", make_preprocessor(x, scale_numeric=model_name in {"Logistic Regression", "KNN Classifier"}),), ("model", estimator)])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=stratify)
        if y_train.nunique() < 2:
            return {"valid": False, "message": "The train split contains fewer than two classes. Try a smaller test split or a target with more examples per class."}
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        labels = sorted(y.astype(str).unique().tolist())
        cm = pd.DataFrame(confusion_matrix(y_test.astype(str), pd.Series(predictions).astype(str), labels=labels), index=[f"Actual {v}" for v in labels], columns=[f"Predicted {v}" for v in labels])
        report = pd.DataFrame(classification_report(y_test, predictions, output_dict=True, zero_division=0)).T
        result = {
            "valid": True, "problem_type": "classification", "model_name": model_name, "target": target, "predictors": predictors,
            "n_train": int(len(x_train)), "n_test": int(len(x_test)), "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            "confusion_matrix": cm, "classification_report": report, "classes": labels,
        }
        if len(labels) == 2 and hasattr(pipeline.named_steps["model"], "predict_proba"):
            probabilities = pipeline.predict_proba(x_test)[:, 1]
            y_binary = (y_test.astype(str) == labels[1]).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, probabilities)
            result["roc"] = pd.DataFrame({"fpr": fpr, "tpr": tpr})
            result["roc_auc"] = float(roc_auc_score(y_binary, probabilities))

    feature_names = get_feature_names(pipeline)
    fitted = pipeline.named_steps["model"]
    importance = None
    if hasattr(fitted, "feature_importances_"):
        importance = pd.DataFrame({"feature": feature_names, "importance": fitted.feature_importances_}).sort_values("importance", ascending=False)
    elif hasattr(fitted, "coef_"):
        coefs = np.ravel(fitted.coef_[0] if np.ndim(fitted.coef_) > 1 else fitted.coef_)
        importance = pd.DataFrame({"feature": feature_names[: len(coefs)], "importance": np.abs(coefs)}).sort_values("importance", ascending=False)
    result["feature_importance"] = importance
    return result


def model_interpretation(result: dict[str, Any]) -> str:
    if not result.get("valid"):
        return result.get("message", "Model could not be trained.")
    top = "not available"
    if isinstance(result.get("feature_importance"), pd.DataFrame) and not result["feature_importance"].empty:
        top = ", ".join(result["feature_importance"].head(3)["feature"].astype(str).tolist())
    if result["problem_type"] == "regression":
        return f"The {result['model_name']} model predicts {result['target']} with RMSE {result['rmse']:.3g}, MAE {result['mae']:.3g}, and R² {result['r2']:.3f}. The most influential available predictors are {top}. Compare residual spread before using this for decisions."
    return f"The {result['model_name']} model predicts {result['target']} with accuracy {result['accuracy']:.3f} and weighted F1 {result['f1']:.3f}. The most influential available predictors are {top}. Review class balance and confusion-matrix errors before acting on predictions."



CHAT_SUGGESTIONS = [
    "Summarize this dataset",
    "What variables are most important?",
    "What should I analyze first?",
    "What statistical tests do you recommend?",
    "What ML model would work best?",
    "What are the strongest correlations?",
    "What anomalies do you see?",
    "What visualizations should I generate?",
    "How clean is this dataset?",
    "Which variables are most predictive?",
    "What business insights stand out?",
]

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
IMPORT_LINE_RE = re.compile(r"^\s*(import\s+.+|from\s+.+\s+import\s+.+)\s*$", re.MULTILINE)
FILE_IO_RE = re.compile(r"^\s*(open\(|to_csv\(|to_excel\(|savefig\(|plt\.savefig\().*$", re.MULTILINE)

AI_DATA_CHAT_SYSTEM_PROMPT = """You are InsightForge's senior AI data scientist, statistician, and business analytics advisor.
You answer questions about an uploaded pandas DataFrame called df using only the supplied safe summaries, metadata, descriptive statistics, correlations, quality reports, outlier reports, chart/model summaries, and conversation memory. Never require or expose the full raw dataset.

Core behavior:
- Do not behave like a passive chatbot. Actively guide exploration like a senior exploratory analytics copilot.
- Explain what the dataset appears to represent, what matters first, why it matters, and what the user should do next.
- Identify patterns, suspicious distributions, outliers, missingness, imbalance, multicollinearity, low-variance fields, leakage risks, and redundant predictors.
- Recommend visualizations and explain what each chart may reveal.
- Recommend statistical tests based on variable types, sample sizes, assumptions, and missingness; explain what significant and non-significant results would mean.
- Recommend ML strategies, target variables, regression vs classification framing, baseline algorithms, validation strategy, feature importance interpretation, and overfitting/leakage risks.
- Use conversation history for follow-ups such as "why?", "which one?", "what next?", or "explain that".
- Sound like a professional analytics consultant: conversational, specific, connected, and action-oriented rather than generic.

Safety and performance rules:
- Use only provided safe context; do not claim direct access to unsupplied raw rows.
- Do not request or expose sensitive row-level data.
- If you include plotting code, use matplotlib only, no imports, no file I/O, and end with plt.tight_layout().
- Keep answers focused, but include enough reasoning for a user to learn what to do next.
"""

def extract_python_code(text: str) -> str | None:
    """Extract one optional matplotlib code block from an AI response."""
    match = CODE_BLOCK_RE.search(text or "")
    return match.group(1).strip() if match else None


def strip_python_code(text: str) -> str:
    """Remove code fences so the saved chat transcript stays readable."""
    return re.sub(CODE_BLOCK_RE, "", text or "").strip()


def sanitize_chat_plot_code(code: str) -> str:
    """Remove imports/file I/O from generated plotting code before execution."""
    cleaned = re.sub(IMPORT_LINE_RE, "", code or "")
    cleaned = re.sub(FILE_IO_RE, "", cleaned)
    cleaned = re.sub(r"^\s*plt\.show\(\)\s*$", "", cleaned, flags=re.MULTILINE)
    if "plt.tight_layout" not in cleaned:
        cleaned += "\n\nplt.tight_layout()"
    return cleaned.strip()


def safe_exec_chat_plot(code: str, df: pd.DataFrame) -> tuple[plt.Figure | None, str | None]:
    """Safely execute limited matplotlib chart code against the in-memory dataframe."""
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    safe_globals: dict[str, Any] = {
        "__builtins__": {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "sorted": sorted,
            "enumerate": enumerate,
            "print": print,
            "round": round,
        },
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
        "fig": fig,
        "ax": ax,
        "textwrap": textwrap,
    }
    try:
        exec(code, safe_globals, {})
        return plt.gcf(), None
    except Exception as exc:  # Keep the chat alive if generated chart code fails.
        return None, f"{type(exc).__name__}: {exc}"


def _safe_records(frame: pd.DataFrame, rows: int = 8) -> list[dict[str, Any]]:
    """Return a compact JSON-safe preview without shipping the full dataset."""
    if frame.empty:
        return []
    preview = frame.head(rows).replace({np.nan: None})
    return preview.to_dict(orient="records")


def _top_correlations(df: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        return []
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(numeric.columns):
        for right in numeric.columns[i + 1 :]:
            pair = numeric[[left, right]].dropna()
            if len(pair) >= 3 and pair[left].nunique() > 1 and pair[right].nunique() > 1:
                r, p = stats.pearsonr(pair[left], pair[right])
                rows.append(
                    {
                        "left": str(left),
                        "right": str(right),
                        "pearson_r": round(float(r), 4),
                        "p_value": float(p),
                        "n": int(len(pair)),
                        "significant_at_0_05": bool(p < ALPHA),
                    }
                )
    return sorted(rows, key=lambda row: abs(row["pearson_r"]), reverse=True)[:limit]


def _outlier_summary(df: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in df.select_dtypes(include=np.number).columns:
        count = iqr_outlier_count(df[col])
        if count:
            rows.append({"column": str(col), "iqr_outlier_count": int(count), "percent": round(count / len(df) * 100, 2) if len(df) else 0})
    return sorted(rows, key=lambda row: row["iqr_outlier_count"], reverse=True)[:limit]


def _categorical_frequencies(df: pd.DataFrame, limit_columns: int = 8, limit_values: int = 5) -> dict[str, list[dict[str, Any]]]:
    types = infer_column_types(df)
    output: dict[str, list[dict[str, Any]]] = {}
    for col in sorted(set(types.categorical + types.binary))[:limit_columns]:
        table = frequency_table(df, col).head(limit_values)
        output[str(col)] = [
            {"value": str(idx), "count": int(row["count"]), "percent": round(float(row["percent"]), 2)}
            for idx, row in table.iterrows()
        ]
    return output



def _format_percent(value: float) -> str:
    return f"{value:.1f}%"


def _candidate_target_columns(df: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    """Heuristically identify columns that are plausible analytics/ML targets."""
    types = infer_column_types(df)
    signals = {
        "target", "label", "outcome", "result", "score", "grade", "g3", "g2", "g1",
        "price", "cost", "sales", "revenue", "profit", "churn", "default", "risk", "class",
        "rating", "conversion", "success", "failure", "passed", "pass", "income", "amount",
    }
    candidates: list[dict[str, Any]] = []
    for col in df.columns:
        lower = str(col).lower()
        score = 0
        reasons: list[str] = []
        if any(token in lower for token in signals):
            score += 4
            reasons.append("name looks outcome-oriented")
        if lower in {"g3", "final", "final_grade", "final_score", "outcome", "target", "label"}:
            score += 2
            reasons.append("looks like a final outcome field")
        if col in types.binary:
            score += 3
            reasons.append("binary target candidate")
        if col in types.numeric:
            non_missing = pd.to_numeric(df[col], errors="coerce").dropna()
            if non_missing.nunique() >= 5:
                score += 2
                reasons.append("continuous numeric outcome candidate")
            if non_missing.nunique() <= 20:
                score += 1
                reasons.append("bounded/ordinal numeric outcome candidate")
        if col in types.categorical and df[col].nunique(dropna=True) <= 20:
            score += 2
            reasons.append("manageable categorical classes")
        if str(col).lower() in {"id", "uuid", "name", "email"} or lower.endswith("id"):
            score -= 5
            reasons.append("identifier-like leakage risk")
        if score > 0:
            if col in types.binary or (col in types.categorical and df[col].nunique(dropna=True) <= 20):
                problem = "classification"
            elif col in types.numeric and df[col].nunique(dropna=True) <= 20:
                problem = "regression or ordinal classification"
            else:
                problem = "regression"
            candidates.append({"column": str(col), "score": score, "problem_type": problem, "why": "; ".join(reasons)})
    return sorted(candidates, key=lambda row: row["score"], reverse=True)[:limit]


def _skew_kurtosis_flags(df: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for col in df.select_dtypes(include=np.number).columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(values) < 8 or values.nunique() <= 2:
            continue
        skew = float(values.skew())
        kurt = float(values.kurtosis())
        if abs(skew) >= 1 or abs(kurt) >= 3:
            flags.append({"column": str(col), "skewness": round(skew, 3), "kurtosis": round(kurt, 3)})
    return sorted(flags, key=lambda row: max(abs(row["skewness"]), abs(row["kurtosis"])), reverse=True)[:limit]


def _low_variance_features(df: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        non_missing = df[col].dropna()
        if non_missing.empty:
            rows.append({"column": str(col), "reason": "all values are missing"})
            continue
        top_share = float(non_missing.astype("object").value_counts(normalize=True).iloc[0] * 100)
        if non_missing.nunique(dropna=True) <= 1 or top_share >= 95:
            rows.append({"column": str(col), "reason": f"dominant value covers {top_share:.1f}% of non-missing rows"})
    return rows[:limit]


def _imbalance_flags(df: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    types = infer_column_types(df)
    rows: list[dict[str, Any]] = []
    for col in sorted(set(types.categorical + types.binary)):
        counts = df[col].astype("object").value_counts(dropna=True)
        if len(counts) < 2:
            continue
        top_share = float(counts.iloc[0] / counts.sum() * 100)
        minority_share = float(counts.iloc[-1] / counts.sum() * 100)
        if top_share >= 70 or minority_share <= 10:
            rows.append({"column": str(col), "top_value": str(counts.index[0]), "top_percent": round(top_share, 1), "minority_percent": round(minority_share, 1)})
    return sorted(rows, key=lambda row: row["top_percent"], reverse=True)[:limit]


def _business_theme(df: pd.DataFrame) -> str:
    cols = {str(c).lower() for c in df.columns}
    joined = " ".join(cols)
    if {"school", "sex", "age"}.issubset(cols) and ({"g1", "g2", "g3"} & cols):
        return "student performance / educational outcomes"
    if any(word in joined for word in ["sales", "revenue", "profit", "customer", "churn"]):
        return "commercial performance, customer behavior, or revenue analytics"
    if any(word in joined for word in ["patient", "diagnosis", "treatment", "medical", "health"]):
        return "health or clinical outcomes analytics"
    if any(word in joined for word in ["loan", "credit", "default", "risk", "income"]):
        return "financial risk or credit analytics"
    if any(word in joined for word in ["date", "time", "month", "year"]):
        return "time-based operational or trend analytics"
    return "general tabular business or research analytics"


def build_dataset_intelligence(df: pd.DataFrame) -> dict[str, Any]:
    """Compute the compact intelligence package behind the Ask Your Data copilot."""
    types = infer_column_types(df)
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    high_missing = [
        {"column": str(col), "percent": round(float(pct), 1), "count": int(df[col].isna().sum())}
        for col, pct in missing_pct.items()
        if pct >= 20
    ][:8]
    correlations = _top_correlations(df, limit=12)
    multicollinearity = [row for row in correlations if abs(row["pearson_r"]) >= 0.7]
    outliers = _outlier_summary(df, limit=10)
    skew_flags = _skew_kurtosis_flags(df)
    imbalance = _imbalance_flags(df)
    low_variance = _low_variance_features(df)
    target_candidates = _candidate_target_columns(df)
    duplicates = int(df.duplicated().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    missing_total = int(df.isna().sum().sum())
    quality_score = 100
    quality_score -= min(35, missing_total / total_cells * 100 if total_cells else 0)
    quality_score -= min(20, duplicates / len(df) * 100 if len(df) else 0)
    quality_score -= min(20, len(high_missing) * 4)
    quality_score -= min(15, len(outliers) * 2)
    quality_score = max(0, round(quality_score, 1))

    prediction_tasks: list[str] = []
    for candidate in target_candidates[:5]:
        prediction_tasks.append(f"Predict **{candidate['column']}** as a {candidate['problem_type']} task ({candidate['why']}).")
    if not prediction_tasks and types.numeric:
        prediction_tasks.append(f"Predict a high-value numeric field such as **{types.numeric[0]}** after confirming it is a real outcome, not an identifier.")

    clustering = []
    if len(types.numeric) >= 3:
        clustering.append("Segment rows with standardized numeric features using k-means or hierarchical clustering, then profile clusters with categorical fields.")
    if len(set(types.categorical + types.binary)) >= 2:
        clustering.append("Use categorical profiles to identify meaningful segments, but one-hot encode carefully to avoid high-cardinality noise.")

    return {
        "theme": _business_theme(df),
        "quality_score": quality_score,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "type_counts": {"numeric": len(types.numeric), "categorical_binary": len(set(types.categorical + types.binary)), "datetime": len(types.datetime)},
        "missing_total": missing_total,
        "missing_percent": round(float(missing_total / total_cells * 100), 2) if total_cells else 0,
        "high_missing": high_missing,
        "duplicates": duplicates,
        "strong_correlations": correlations[:8],
        "multicollinearity": multicollinearity[:8],
        "target_candidates": target_candidates,
        "prediction_tasks": prediction_tasks,
        "clustering_opportunities": clustering,
        "imbalance_warnings": imbalance,
        "outlier_heavy_columns": outliers,
        "suspicious_distributions": skew_flags,
        "low_variance_features": low_variance,
        "visualization_recommendations": recommend_visualizations(df),
        "statistical_recommendations": recommend_statistical_tests(df),
        "ml_recommendations": recommend_ml_strategy(df),
        "business_questions": likely_business_questions(df),
        "next_steps": recommended_next_steps(df),
    }


def recommend_visualizations(df: pd.DataFrame) -> list[dict[str, str]]:
    types = infer_column_types(df)
    recs: list[dict[str, str]] = []
    correlations = _top_correlations(df, limit=3)
    skewed = _skew_kurtosis_flags(df, limit=3)
    cat_cols = sorted(set(types.categorical + types.binary))
    if len(types.numeric) >= 2:
        recs.append({"chart": "Correlation heatmap", "why": "Quickly reveals redundant predictors, multicollinearity, and candidate relationships worth deeper testing."})
    for row in correlations[:2]:
        recs.append({"chart": f"Scatter plot with regression line: {row['left']} vs {row['right']}", "why": f"This is one of the strongest numeric relationships (r={row['pearson_r']:.3f}); the chart can reveal nonlinearity, clusters, and outliers."})
    for row in skewed[:2]:
        recs.append({"chart": f"Histogram + violin/boxplot for {row['column']}", "why": f"Skewness={row['skewness']:.2f} and kurtosis={row['kurtosis']:.2f}; distribution plots show whether transformations or robust tests are needed."})
    if cat_cols:
        recs.append({"chart": f"Grouped bar/count plot for {cat_cols[0]}", "why": "Shows category balance and possible sampling bias before comparing groups or training classifiers."})
    if types.numeric and cat_cols:
        recs.append({"chart": f"Box/violin plot of {types.numeric[0]} by {cat_cols[0]}", "why": "Compares distributions across groups and helps decide whether t-tests, ANOVA, or non-parametric tests are appropriate."})
    if len(types.numeric) >= 4:
        recs.append({"chart": "Pair plot / scatter-matrix of top numeric variables", "why": "Provides a compact view of multiple relationships before selecting model predictors."})
    if types.datetime and types.numeric:
        recs.append({"chart": f"Line chart of {types.numeric[0]} over {types.datetime[0]}", "why": "Checks trend, seasonality, and structural breaks over time."})
    return recs[:8]


def recommend_statistical_tests(df: pd.DataFrame) -> list[dict[str, str]]:
    types = infer_column_types(df)
    cat_cols = sorted(set(types.categorical + types.binary))
    recs: list[dict[str, str]] = []
    if len(types.numeric) >= 2:
        corr = _top_correlations(df, limit=1)
        if corr:
            recs.append({"test": f"Pearson/Spearman correlation for {corr[0]['left']} and {corr[0]['right']}", "why": "Pearson tests linear association; Spearman is safer when variables are skewed, ordinal, or affected by outliers."})
    if types.numeric and cat_cols:
        group_col = cat_cols[0]
        group_count = int(df[group_col].nunique(dropna=True))
        if group_count == 2:
            recs.append({"test": f"Welch t-test or Mann-Whitney U: {types.numeric[0]} by {group_col}", "why": "Use Welch t-test for two groups with unequal variances; use Mann-Whitney when distributions are non-normal or outlier-heavy."})
        elif group_count > 2:
            recs.append({"test": f"One-way ANOVA or Kruskal-Wallis: {types.numeric[0]} by {group_col}", "why": "ANOVA compares group means; Kruskal-Wallis is the robust alternative when normality or equal-variance assumptions are doubtful."})
    if len(cat_cols) >= 2:
        recs.append({"test": f"Chi-square test of independence: {cat_cols[0]} vs {cat_cols[1]}", "why": "Tests whether two categorical variables are associated; inspect expected cell counts and use grouped categories when cells are sparse."})
    target_candidates = _candidate_target_columns(df)
    if target_candidates:
        target = target_candidates[0]["column"]
        if target in types.numeric:
            recs.append({"test": f"Multiple regression for {target}", "why": "Quantifies adjusted relationships, flags important predictors, and reveals multicollinearity/residual issues."})
        else:
            recs.append({"test": f"Logistic/classification analysis for {target}", "why": "Estimates predictive signal while monitoring class imbalance, leakage, and misclassification costs."})
    return recs[:8]


def recommend_ml_strategy(df: pd.DataFrame) -> list[str]:
    types = infer_column_types(df)
    candidates = _candidate_target_columns(df)
    recs: list[str] = []
    if candidates:
        top = candidates[0]
        if top["problem_type"].startswith("classification"):
            recs.append(f"Start with interpretable logistic regression for **{top['column']}**, then compare a random forest for nonlinear interactions.")
            recs.append("Use stratified train/test splitting and report weighted F1, recall by class, and a confusion matrix because class imbalance may hide poor minority-class performance.")
        else:
            recs.append(f"Start with linear regression or regularized regression for **{top['column']}**, then compare random forest regression for nonlinear effects.")
            recs.append("Evaluate with MAE, RMSE, R², and residual plots; validate that influential predictors are available before the target is observed.")
    elif types.numeric:
        recs.append("Choose a domain-meaningful numeric target before modeling; avoid training on arbitrary columns just because they are numeric.")
    if len(types.numeric) >= 3:
        recs.append("Use clustering only after scaling numeric variables and removing identifiers; profile clusters after fitting rather than assuming they are meaningful.")
    recs.append("Check leakage risks: IDs, post-outcome fields, duplicate rows, and near-perfect correlations can inflate model performance.")
    recs.append("Use cross-validation for small datasets and keep preprocessing inside the sklearn Pipeline to avoid train/test contamination.")
    return recs[:7]


def likely_business_questions(df: pd.DataFrame) -> list[str]:
    types = infer_column_types(df)
    candidates = _candidate_target_columns(df)
    questions: list[str] = []
    theme = _business_theme(df)
    if candidates:
        target = candidates[0]["column"]
        questions.append(f"What factors most strongly explain or predict **{target}** in this {theme} dataset?")
        questions.append(f"Are there groups with systematically higher or lower **{target}**, and are those differences statistically meaningful?")
    if _top_correlations(df, limit=1):
        c = _top_correlations(df, limit=1)[0]
        questions.append(f"Is the relationship between **{c['left']}** and **{c['right']}** practically meaningful or driven by outliers?")
    if types.categorical or types.binary:
        questions.append("Which categories dominate the data, and could that imbalance bias conclusions or model performance?")
    if types.numeric:
        questions.append("Which numeric variables show unusual distributions that require transformation, winsorization, or robust statistics?")
    return questions[:6]


def recommended_next_steps(df: pd.DataFrame) -> list[str]:
    intelligence = {
        "missing": bool(df.isna().sum().max() > 0),
        "outliers": bool(_outlier_summary(df, limit=1)),
        "correlations": bool(_top_correlations(df, limit=1)),
        "targets": bool(_candidate_target_columns(df, limit=1)),
    }
    steps = ["Confirm the business/research objective and select a primary target variable before modeling."]
    if intelligence["missing"]:
        steps.append("Decide a missing-data strategy: impute, add missingness flags, or exclude unreliable columns depending on missingness mechanism.")
    if intelligence["outliers"]:
        steps.append("Inspect outlier-heavy columns with boxplots and determine whether unusual values are valid extremes, errors, or segment-specific behavior.")
    if intelligence["correlations"]:
        steps.append("Validate the strongest correlations with scatter plots and domain logic; correlation is not causation.")
    if intelligence["targets"]:
        steps.append("Run a baseline model with train/test validation, then inspect feature importance and leakage risks.")
    steps.append("Translate findings into a short decision memo: what changed, why it matters, and what action should be taken next.")
    return steps[:7]


def generate_expert_intelligence_markdown(df: pd.DataFrame) -> str:
    intel = build_dataset_intelligence(df)
    lines = [
        "## Automatic Dataset Intelligence Summary",
        f"I read this as a **{intel['theme']}** dataset with **{intel['shape']['rows']:,} rows** and **{intel['shape']['columns']:,} columns**. It contains **{intel['type_counts']['numeric']} numeric**, **{intel['type_counts']['categorical_binary']} categorical/binary**, and **{intel['type_counts']['datetime']} datetime** fields.",
        f"My initial data-quality score is **{intel['quality_score']}/100**. Missingness covers **{intel['missing_percent']:.2f}%** of cells and duplicate rows total **{intel['duplicates']:,}**.",
        "",
        "### What deserves attention first",
    ]
    if intel["target_candidates"]:
        lines.append("- Potential target variables: " + "; ".join(f"**{row['column']}** ({row['problem_type']}: {row['why']})" for row in intel["target_candidates"][:4]) + ".")
    else:
        lines.append("- I do not see an obvious target variable yet; choose one based on the decision you want to support.")
    if intel["strong_correlations"]:
        lines.append("- Strongest relationships: " + "; ".join(f"**{r['left']} ↔ {r['right']}** (r={r['pearson_r']:.3f})" for r in intel["strong_correlations"][:4]) + ".")
    if intel["high_missing"]:
        lines.append("- High missingness: " + "; ".join(f"**{r['column']}** ({r['percent']:.1f}%)" for r in intel["high_missing"][:4]) + ".")
    if intel["outlier_heavy_columns"]:
        lines.append("- Outlier-heavy columns: " + "; ".join(f"**{r['column']}** ({r['percent']:.1f}% flagged)" for r in intel["outlier_heavy_columns"][:4]) + ".")
    if intel["imbalance_warnings"]:
        lines.append("- Imbalance warnings: " + "; ".join(f"**{r['column']}** dominated by {r['top_value']} ({r['top_percent']:.1f}%)" for r in intel["imbalance_warnings"][:4]) + ".")
    if intel["suspicious_distributions"]:
        lines.append("- Suspicious distributions: " + "; ".join(f"**{r['column']}** (skew={r['skewness']:.2f}, kurtosis={r['kurtosis']:.2f})" for r in intel["suspicious_distributions"][:4]) + ".")

    sections = [
        ("Recommended Next Analyses", intel["next_steps"]),
        ("Visualization Intelligence", [f"{r['chart']} — {r['why']}" for r in intel["visualization_recommendations"]]),
        ("Statistical Reasoning", [f"{r['test']} — {r['why']}" for r in intel["statistical_recommendations"]]),
        ("ML Guidance", intel["ml_recommendations"]),
        ("Likely Business Questions", intel["business_questions"]),
    ]
    for heading, items in sections:
        lines.extend(["", f"### {heading}"])
        lines.extend(f"- {item}" for item in (items or ["No automated recommendation is available for this dataset shape yet."]))
    return "\n".join(lines)

def dataset_context(df: pd.DataFrame) -> dict[str, Any]:
    """Build the safe, dataset-aware context used by OpenAI and fallback chat."""
    types = infer_column_types(df)
    num_stats = numeric_descriptive_stats(df).round(4).head(30)
    cat_stats = categorical_summary(df).round(4).head(30)
    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    context: dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": df.columns.astype(str).tolist(),
        "column_types": {
            "numeric": types.numeric,
            "categorical": sorted(set(types.categorical + types.binary)),
            "datetime": types.datetime,
            "binary": types.binary,
        },
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {
            str(col): {"count": int(missing.loc[col]), "percent": round(float(missing_pct.loc[col]), 2)}
            for col in missing.index[:30]
            if int(missing.loc[col]) > 0
        },
        "numeric_stats": num_stats.to_dict(orient="index") if not num_stats.empty else {},
        "categorical_summary": cat_stats.to_dict(orient="index") if not cat_stats.empty else {},
        "categorical_frequencies": _categorical_frequencies(df),
        "top_correlations": _top_correlations(df),
        "outliers_iqr": _outlier_summary(df),
        "advanced_intelligence": build_dataset_intelligence(df),
        "visualization_recommendations": recommend_visualizations(df),
        "statistical_test_recommendations": recommend_statistical_tests(df),
        "ml_strategy_recommendations": recommend_ml_strategy(df),
        "key_findings": summarize_key_findings(df),
        "small_sample_rows": _safe_records(df, rows=5),
        "chart_history": st.session_state.get("chart_history", [])[-8:],
        "regression_results": st.session_state.get("regression_text", "No regression model has been run yet."),
        "ml_results": st.session_state.get("ml_results", {}).get("summary", "No ML model has been run yet."),
    }
    return context


def _format_missing_answer(df: pd.DataFrame) -> str:
    missing = df.isna().sum().sort_values(ascending=False)
    top = missing[missing > 0].head(8)
    if top.empty:
        return "I do not see missing-value problems in the cleaned dataset — every column is complete after the selected cleaning settings."
    lines = ["The main missing-value issues are:"]
    for col, count in top.items():
        lines.append(f"- **{col}**: {int(count):,} missing ({count / len(df) * 100:.1f}% of rows)")
    lines.append("Columns above roughly 20–30% missingness usually need a deliberate decision: impute, create a missingness flag, or exclude from modeling if the field is not reliable.")
    return "\n".join(lines)


def _format_correlation_answer(df: pd.DataFrame) -> str:
    correlations = _top_correlations(df)
    if not correlations:
        return "I need at least two numeric columns with enough non-missing variation to compute correlations, and this dataset does not currently meet that requirement."
    lines = ["Here are the strongest Pearson correlations I found:"]
    for row in correlations[:6]:
        direction = "positive" if row["pearson_r"] >= 0 else "negative"
        strength = "strong" if abs(row["pearson_r"]) >= 0.7 else "moderate" if abs(row["pearson_r"]) >= 0.4 else "weak"
        sig = "statistically significant" if row["significant_at_0_05"] else "not statistically significant"
        lines.append(f"- **{row['left']}** vs **{row['right']}**: r={row['pearson_r']:.3f} ({strength} {direction}), p={row['p_value']:.3g}, n={row['n']:,}; {sig} at α=0.05.")
    lines.append("Remember: correlation shows association, not causation. A scatter plot is the next best check for shape, clusters, and outliers.")
    return "\n".join(lines)


def _format_summary_answer(df: pd.DataFrame) -> str:
    types = infer_column_types(df)
    findings = summarize_key_findings(df)
    missing_total = int(df.isna().sum().sum())
    outliers = _outlier_summary(df, limit=3)
    lines = [
        f"This cleaned dataset has **{df.shape[0]:,} rows** and **{df.shape[1]:,} columns**.",
        f"I detected **{len(types.numeric)} numeric**, **{len(set(types.categorical + types.binary))} categorical/binary**, and **{len(types.datetime)} datetime** fields.",
        f"Missing values remaining: **{missing_total:,}**.",
        f"Strongest positive relationship: {findings['strongest_positive_correlation']}.",
        f"Strongest negative relationship: {findings['strongest_negative_correlation']}.",
        f"Most variable numeric field: {findings['most_variable_numeric_column']}.",
        f"Most imbalanced categorical field: {findings['most_imbalanced_categorical_column']}.",
    ]
    if outliers:
        lines.append("Potential outlier-heavy columns include " + ", ".join(f"**{row['column']}** ({row['iqr_outlier_count']:,})" for row in outliers) + ".")
    lines.append("A good next step is to inspect distributions for high-variation fields and scatter plots for the strongest correlations.")
    return "\n".join(lines)


def _format_visualization_answer(df: pd.DataFrame) -> str:
    recs = recommend_visualizations(df)
    if not recs:
        return "I need more typed columns before I can make strong visualization recommendations. Start by checking schema and converting date/numeric-looking fields."
    lines = ["Here are the visualizations I would generate first, in analyst priority order:"]
    lines.extend(f"- **{row['chart']}**: {row['why']}" for row in recs)
    lines.append("After creating each chart, ask: is the pattern strong, practically meaningful, potentially confounded, or driven by a small group of unusual rows?")
    return "\n".join(lines)


def _format_model_answer() -> str:
    regression_text = st.session_state.get("regression_text") or "No regression model has been run yet."
    ml_text = st.session_state.get("ml_results", {}).get("summary", "No ML model has been run yet.")
    guidance = "For regression, focus on coefficient direction, p-values below 0.05 for statistical significance, R²/adjusted R² for explained variance, and residual plots for model assumptions. For ML, compare test-set metrics and feature importance before trusting predictions."
    return f"**Regression tab:** {regression_text}\n\n**Machine Learning tab:** {ml_text}\n\n{guidance}"



def _format_statistical_recommendation_answer(df: pd.DataFrame) -> str:
    recs = recommend_statistical_tests(df)
    if not recs:
        return "I need more typed variables before I can recommend meaningful tests. Add or convert numeric/categorical fields, then start with descriptive distributions."
    lines = ["Here is the statistical testing plan I would use first:"]
    lines.extend(f"- **{row['test']}**: {row['why']}" for row in recs)
    lines.append("Before trusting p-values, check missingness, sample sizes per group, outliers, and whether observations are independent. Significant results mean the observed pattern is unlikely under the null hypothesis; they do not automatically imply causality or business importance.")
    return "\n".join(lines)


def _format_ml_strategy_answer(df: pd.DataFrame) -> str:
    candidates = _candidate_target_columns(df)
    lines = ["My ML guidance is to start simple, validate honestly, and watch for leakage."]
    if candidates:
        lines.append("Potential targets I would consider first:")
        lines.extend(f"- **{row['column']}** → {row['problem_type']} ({row['why']})" for row in candidates[:5])
    lines.append("Recommended strategy:")
    lines.extend(f"- {item}" for item in recommend_ml_strategy(df))
    return "\n".join(lines)


def _format_anomaly_answer(df: pd.DataFrame) -> str:
    intel = build_dataset_intelligence(df)
    lines = ["Here are the main anomaly and risk signals I see:"]
    if intel["outlier_heavy_columns"]:
        lines.append("- **Outliers:** " + "; ".join(f"{r['column']} has {r['iqr_outlier_count']:,} IQR flags ({r['percent']:.1f}%)" for r in intel["outlier_heavy_columns"][:6]) + ".")
    if intel["suspicious_distributions"]:
        lines.append("- **Skew/kurtosis:** " + "; ".join(f"{r['column']} skew={r['skewness']:.2f}, kurtosis={r['kurtosis']:.2f}" for r in intel["suspicious_distributions"][:6]) + ".")
    if intel["imbalance_warnings"]:
        lines.append("- **Imbalance:** " + "; ".join(f"{r['column']} is dominated by {r['top_value']} ({r['top_percent']:.1f}%)" for r in intel["imbalance_warnings"][:6]) + ".")
    if intel["multicollinearity"]:
        lines.append("- **Multicollinearity:** " + "; ".join(f"{r['left']} vs {r['right']} r={r['pearson_r']:.3f}" for r in intel["multicollinearity"][:6]) + ".")
    if intel["low_variance_features"]:
        lines.append("- **Low-variance fields:** " + "; ".join(f"{r['column']} ({r['reason']})" for r in intel["low_variance_features"][:6]) + ".")
    if len(lines) == 1:
        lines.append("- I do not see major automated anomaly flags, but I would still verify distributions visually and check domain-specific validity rules.")
    lines.append("Next action: visualize these fields before deleting anything. Some outliers are errors; others are exactly the high-value cases you want to understand.")
    return "\n".join(lines)


def _format_business_insights_answer(df: pd.DataFrame) -> str:
    questions = likely_business_questions(df)
    steps = recommended_next_steps(df)
    lines = [f"I would frame this as **{_business_theme(df)}** and turn the exploration into these decision questions:"]
    lines.extend(f"- {question}" for question in questions)
    lines.append("Suggested consulting-style next steps:")
    lines.extend(f"- {step}" for step in steps)
    return "\n".join(lines)


def _format_predictive_variables_answer(df: pd.DataFrame) -> str:
    candidates = _candidate_target_columns(df)
    correlations = _top_correlations(df, limit=8)
    lines = ["For predictive work, I would separate likely **targets** from likely **predictors** and then verify with models."]
    if candidates:
        lines.append("Likely target variables:")
        lines.extend(f"- **{row['column']}** ({row['problem_type']}): {row['why']}" for row in candidates[:5])
    if correlations:
        lines.append("Strong numeric predictor relationships to inspect first:")
        lines.extend(f"- **{row['left']} ↔ {row['right']}**: r={row['pearson_r']:.3f}, p={row['p_value']:.3g}" for row in correlations[:6])
    lines.append("Be careful: the most correlated variable may be leakage if it is measured after the target or is a near-duplicate of the outcome.")
    return "\n".join(lines)

def rule_based_answer(question: str, df: pd.DataFrame, history: list[dict[str, str]] | None = None) -> str:
    """Conversational fallback when OPENAI_API_KEY is unavailable or an API call fails."""
    q = question.lower()
    recent_context = " ".join(message.get("content", "").lower() for message in (history or [])[-4:])
    combined = f"{recent_context} {q}"
    if any(word in combined for word in ["missing", "null", "na", "quality", "clean", "cleanliness"]):
        return _format_missing_answer(df)
    if any(word in combined for word in ["correlation", "relationship", "related", "association", "strongest"]):
        return _format_correlation_answer(df)
    if any(word in combined for word in ["visual", "chart", "plot", "graph"]):
        return _format_visualization_answer(df)
    if any(word in combined for word in ["statistical test", "t-test", "anova", "chi-square", "chi square", "p-value", "p value", "significance", "hypothesis"]):
        return _format_statistical_recommendation_answer(df)
    if any(word in combined for word in ["ml", "machine learning", "algorithm", "classifier", "classification", "train/test", "overfit"]):
        return _format_ml_strategy_answer(df)
    if any(word in combined for word in ["predictive", "predictor", "important variable", "variables are most important", "feature importance", "target"]):
        return _format_predictive_variables_answer(df)
    if any(word in combined for word in ["model", "regression", "important", "importance", "predict"]):
        return _format_ml_strategy_answer(df)
    if any(word in combined for word in ["column", "field", "variable", "dtype", "type"]):
        types = infer_column_types(df)
        return (
            f"The dataset has **{df.shape[1]:,} columns**. Numeric: {', '.join(types.numeric[:20]) or 'none'}. "
            f"Categorical/binary: {', '.join(sorted(set(types.categorical + types.binary))[:20]) or 'none'}. "
            f"Datetime: {', '.join(types.datetime[:10]) or 'none'}."
        )
    if any(word in combined for word in ["outlier", "anomaly", "anomalies", "unusual", "risk", "suspicious", "skew", "kurtosis", "multicollinearity", "imbalance"]):
        return _format_anomaly_answer(df)
    if any(word in combined for word in ["business", "insight", "stand out", "decision", "advisor", "analyze first", "what should i analyze first"]):
        return _format_business_insights_answer(df)
    if any(word in combined for word in ["categor", "frequency", "frequencies", "count", "distribution"]):
        freqs = _categorical_frequencies(df, limit_columns=5, limit_values=3)
        if not freqs:
            return "I do not see categorical columns to summarize with frequencies."
        lines = ["Top categorical frequencies:"]
        for col, values in freqs.items():
            lines.append(f"- **{col}**: " + ", ".join(f"{v['value']} ({v['percent']:.1f}%)" for v in values))
        return "\n".join(lines)
    return _format_summary_answer(df)


def _openai_api_key() -> str | None:
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        key = None
    return (key or os.getenv("OPENAI_API_KEY") or "").strip() or None


def answer_with_openai(question: str, df: pd.DataFrame, history: list[dict[str, str]] | None = None) -> str:
    """Use OpenAI with conversation memory and compact dataset context; fallback if no key."""
    api_key = _openai_api_key()
    if not api_key:
        return rule_based_answer(question, df, history)

    client = OpenAI(api_key=api_key)
    ctx = dataset_context(df)
    remembered_messages = [
        {"role": message["role"], "content": message["content"]}
        for message in (history or [])[-12:]
        if message.get("role") in {"user", "assistant"} and message.get("content")
    ]
    messages = [
        {"role": "system", "content": AI_DATA_CHAT_SYSTEM_PROMPT},
        {"role": "system", "content": f"Safe dataset context (summaries only, not full raw data):\n{ctx}"},
        *remembered_messages,
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0.25,
        max_tokens=800,
    )
    return response.choices[0].message.content or "I could not generate an answer."

def markdown_to_html(report: str) -> str:
    body_lines: list[str] = []
    in_list = False
    for line in report.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            if not in_list:
                body_lines.append("<ul>")
                in_list = True
            body_lines.append(f"<li>{html.escape(stripped[2:])}</li>")
            continue
        if in_list:
            body_lines.append("</ul>")
            in_list = False
        if stripped.startswith("# "):
            body_lines.append(f"<h1>{html.escape(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            body_lines.append(f"<h2>{html.escape(stripped[3:])}</h2>")
        elif stripped.startswith("### "):
            body_lines.append(f"<h3>{html.escape(stripped[4:])}</h3>")
        elif not stripped:
            body_lines.append("<br>")
        else:
            body_lines.append(f"<p>{html.escape(stripped)}</p>")
    if in_list:
        body_lines.append("</ul>")
    return """<!doctype html><html><head><meta charset='utf-8'><title>InsightForge Business Report</title><style>body{font-family:Arial,sans-serif;max-width:980px;margin:40px auto;line-height:1.55;color:#172033}h1,h2,h3{color:#0f4c81}table{border-collapse:collapse;width:100%}code,pre{background:#f5f7fb}</style></head><body>""" + "\n".join(body_lines) + "</body></html>"


def build_business_report(df: pd.DataFrame, original_df: pd.DataFrame, cleaning_notes: list[str], test_results: list[str]) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    st.session_state.report_generated_at = generated
    findings = summarize_key_findings(df)
    num_stats = numeric_descriptive_stats(df)
    cat_stats = categorical_summary(df)
    charts = recommended_story_charts(df)
    ml_summary = st.session_state.get("ml_results", {}).get("summary", "No machine-learning model has been run during this session.")
    missing_pct = safe_pct(df.isna().sum().sum(), df.shape[0] * df.shape[1])
    lines = [
        "# InsightForge Business Analysis Report",
        f"Generated: {generated}",
        "",
        "## Dataset Overview",
        f"The analysis uses the uploaded dataset after selected cleaning steps: {df.shape[0]:,} rows and {df.shape[1]:,} columns. The original upload contained {original_df.shape[0]:,} rows and {original_df.shape[1]:,} columns.",
        "",
        "## Data Quality Summary",
        f"Missing values represent {missing_pct:.2f}% of cells. Duplicate rows total {int(df.duplicated().sum()):,}. Highest missing-value column: {findings['highest_missing_value_column']}.",
        "",
        "## Executive Summary",
        storytelling_text(df, original_df, cleaning_notes),
        "",
        "## Key Findings",
        f"- Strongest positive correlation: {findings['strongest_positive_correlation']}",
        f"- Strongest negative correlation: {findings['strongest_negative_correlation']}",
        f"- Most variable numeric column: {findings['most_variable_numeric_column']}",
        f"- Most imbalanced categorical column: {findings['most_imbalanced_categorical_column']}",
        "",
        "## Descriptive Statistics Summary",
        num_stats.head(12).to_markdown() if not num_stats.empty else "No numeric descriptive statistics are available.",
        "",
        "## Categorical Summary",
        cat_stats.head(12).to_markdown() if not cat_stats.empty else "No categorical summary is available.",
        "",
        "## Important Visual Insights",
    ]
    lines.extend([f"- {title}: {caption}" for title, _, caption in charts] or ["No recommended visual insights could be generated from the detected data types."])
    lines.extend([
        "",
        "## Statistical Test Results",
        *(test_results[-10:] if test_results else ["No statistical tests have been recorded during this session."]),
        "",
        "## ML Model Results",
        ml_summary,
        "",
        "## Recommended Next Steps",
        "- Validate the strongest relationships against domain knowledge before presenting conclusions.",
        "- Address missingness, duplicates, and imbalanced categories before high-stakes modeling.",
        "- Run targeted statistical tests or ML models tied to a specific business or academic hypothesis.",
        "- Re-run the report after selecting final cleaning settings and model configuration.",
        "",
        "## Limitations",
        "- Results are based only on the uploaded dataset and selected in-app cleaning choices.",
        "- Correlations and model importance do not prove causation.",
        "- AI chat responses use summaries and metadata rather than the full raw dataset.",
    ])
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Ask Your Data execution display helpers
# -----------------------------------------------------------------------------
def display_analysis_result(result: AnalysisResult, message_id: str | None = None, show_code: bool = False) -> None:
    """Render a safe Ask Your Data analysis result in a consistent expert format."""
    if result.warning:
        st.info(result.warning)

    if result.figures:
        for index, fig in enumerate(result.figures):
            st.plotly_chart(fig, use_container_width=True, key=f"{message_id}_fig_{index}" if message_id else None)

    st.markdown(f"### {result.title}")
    st.markdown("**What I did**")
    st.write(result.what_i_did)

    st.markdown("**Result**")
    st.write(result.result_summary)
    if result.metrics:
        metric_items = list(result.metrics.items())[:8]
        metric_cols = st.columns(min(4, max(1, len(metric_items))))
        for idx, (label, value) in enumerate(metric_items):
            if isinstance(value, float):
                display_value = f"{value:.4g}"
            else:
                display_value = str(value)
            metric_cols[idx % len(metric_cols)].metric(str(label), display_value)

    for table_index, (title, table) in enumerate(result.tables):
        if table is not None and not table.empty:
            with st.expander(title, expanded=table_index == 0):
                st.dataframe(table, use_container_width=True)

    st.markdown("**Interpretation**")
    st.write(result.interpretation)

    st.markdown("**What to explore next**")
    for item in result.next_steps:
        st.markdown(f"- {item}")

    if show_code and result.code:
        st.code(result.code, language="python")


def store_analysis_memory(question: str, detected: AnalysisIntent, result: AnalysisResult) -> None:
    """Persist compact analysis memory for follow-up questions and reports."""
    if "analysis_memory" not in st.session_state:
        st.session_state.analysis_memory = []
    st.session_state.analysis_memory.append(
        {
            "question": question,
            "detected_intent": detected.intent,
            "confidence": detected.confidence,
            "selected_columns": result.selected_columns,
            "result_summary": result.result_summary,
            "interpretation": result.interpretation,
            "table_titles": [title for title, _ in result.tables],
            "chart_count": len(result.figures),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    st.session_state.analysis_memory = st.session_state.analysis_memory[-30:]

# -----------------------------------------------------------------------------
# Visualization builder
# -----------------------------------------------------------------------------
def render_visualization(df: pd.DataFrame, chart_type: str, x_col: str | None, y_col: str | None, color_col: str | None, agg: str) -> go.Figure | None:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    color = None if color_col == "None" else color_col

    if chart_type == "Histogram" and x_col:
        return px.histogram(df, x=x_col, color=color, marginal="box", title=f"Histogram of {x_col}")
    if chart_type == "KDE/density plot" and x_col:
        groups = [df[x_col].dropna()] if not color else [g[x_col].dropna() for _, g in df.groupby(color)]
        labels = [x_col] if not color else [str(name) for name, _ in df.groupby(color)]
        if all(len(g) > 1 for g in groups):
            return ff.create_distplot(groups, labels, show_hist=False, show_rug=False)
    if chart_type == "Boxplot" and y_col:
        return px.box(df, x=x_col if x_col != "None" else None, y=y_col, color=color, title=f"Boxplot of {y_col}")
    if chart_type == "Violin plot" and y_col:
        return px.violin(df, x=x_col if x_col != "None" else None, y=y_col, color=color, box=True, title=f"Violin plot of {y_col}")
    if chart_type == "Scatter plot" and x_col and y_col:
        return px.scatter(df, x=x_col, y=y_col, color=color, hover_data=df.columns[:6], title=f"{y_col} vs {x_col}")
    if chart_type == "Regression scatter plot" and x_col and y_col:
        return px.scatter(df, x=x_col, y=y_col, color=color, trendline="ols", title=f"Regression scatter: {y_col} vs {x_col}")
    if chart_type == "Correlation heatmap" and len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        return px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Numeric Correlation Heatmap")
    if chart_type in {"Bar chart", "Grouped bar chart"} and x_col and y_col:
        grouped = df.groupby([x_col] + ([color] if color else []), dropna=False)[y_col].agg(agg.lower()).reset_index()
        return px.bar(grouped, x=x_col, y=y_col, color=color, barmode="group", title=f"{agg} of {y_col} by {x_col}")
    if chart_type == "Count plot" and x_col:
        counts = df.groupby([x_col] + ([color] if color else []), dropna=False).size().reset_index(name="count")
        return px.bar(counts, x=x_col, y="count", color=color, barmode="group", title=f"Count of records by {x_col}")
    if chart_type == "Line plot" and x_col and y_col:
        return px.line(df.sort_values(x_col), x=x_col, y=y_col, color=color, title=f"Line plot of {y_col} by {x_col}")
    if chart_type == "Pie chart" and x_col:
        counts = df[x_col].astype("object").value_counts(dropna=False).reset_index()
        counts.columns = [x_col, "count"]
        return px.pie(counts, names=x_col, values="count", title=f"Pie chart of {x_col}")
    if chart_type == "Stacked bar chart" and x_col and color:
        counts = df.groupby([x_col, color], dropna=False).size().reset_index(name="count")
        return px.bar(counts, x=x_col, y="count", color=color, barmode="stack", title=f"Stacked counts: {x_col} by {color}")
    if chart_type == "Area chart" and x_col and y_col:
        return px.area(df.sort_values(x_col), x=x_col, y=y_col, color=color, title=f"Area chart of {y_col} by {x_col}")
    if chart_type == "Pair plot style scatter matrix" and len(numeric_cols) >= 2:
        return px.scatter_matrix(df, dimensions=numeric_cols[:6], color=color, title="Scatter Matrix")
    if chart_type == "3D scatter plot" and len(numeric_cols) >= 3:
        return px.scatter_3d(df, x=numeric_cols[0], y=numeric_cols[1], z=numeric_cols[2], color=color, title="3D Scatter Plot")
    return None


# -----------------------------------------------------------------------------
# Streamlit interface
# -----------------------------------------------------------------------------
st.set_page_config(page_title="InsightForge Streamlit Analytics", page_icon="📊", layout="wide")
st.title("📊 InsightForge Streamlit Data Analysis")
st.caption("Upload CSV, Excel, or JSON files and generate real descriptive, inferential, modeling, visualization, ML, Q&A, and report outputs.")

if "test_results" not in st.session_state:
    st.session_state.test_results = []
if "regression_text" not in st.session_state:
    st.session_state.regression_text = ""
if "ml_results" not in st.session_state:
    st.session_state.ml_results = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_chat_prompt" not in st.session_state:
    st.session_state.pending_chat_prompt = None
if "chat_dataset_signature" not in st.session_state:
    st.session_state.chat_dataset_signature = None
if "chart_history" not in st.session_state:
    st.session_state.chart_history = []
if "analysis_memory" not in st.session_state:
    st.session_state.analysis_memory = []
if "report_generated_at" not in st.session_state:
    st.session_state.report_generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

with st.sidebar:
    st.caption("InsightForge Streamlit v2.0")
    if st.button("Reset workflow", help="Clear chat, model results, reports, and cached interactive findings."):
        st.session_state.test_results = []
        st.session_state.regression_text = ""
        st.session_state.ml_results = {}
        st.session_state.chat_history = []
        st.session_state.pending_chat_prompt = None
        st.session_state.chat_dataset_signature = None
        st.session_state.analysis_memory = []
        st.session_state.chart_history = []
        st.session_state.report_generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        st.rerun()

    st.header("1) Upload data")
    uploaded_file = st.file_uploader("Choose a CSV, XLSX, or JSON file", type=["csv", "xlsx", "json"])
    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")

    st.header("2) Dataset settings")
    sample_rows = st.slider("Sample rows to display", 5, 100, 20)
    drop_duplicates = st.checkbox("Drop duplicates", value=True)
    numeric_fill = st.selectbox("Fill numeric missing values", ["Do not fill", "Mean", "Median", "Zero"], index=0)
    categorical_fill = st.selectbox("Fill categorical missing values", ["Do not fill", "Mode", "Missing"], index=0)
    convert_datetimes = st.checkbox("Convert datetime columns when possible", value=True)
    remove_outliers = st.checkbox("Remove outliers using IQR", value=False)

    st.header("3) Analysis sections")
    selected_sections = st.multiselect(
        "Show sections",
        ["Dataset Overview", "Storytelling Dashboard", "Summary Analysis", "Descriptive Statistics", "Categorical Analysis", "Inferential Tests", "Regression Modeling", "Machine Learning", "Visualizations", "Ask Your Data", "Export Report", "Business Report"],
        default=["Dataset Overview", "Storytelling Dashboard", "Summary Analysis", "Descriptive Statistics", "Categorical Analysis", "Inferential Tests", "Regression Modeling", "Machine Learning", "Visualizations", "Ask Your Data", "Export Report", "Business Report"],
    )

    st.header("Theme/help notes")
    st.info("Use the cleaning controls before interpreting tests. Statistical results are computed from the cleaned uploaded dataset; no placeholder results are used.")

if uploaded_file is None:
    st.info("Upload a CSV, Excel, or JSON file in the sidebar to begin.")
    st.stop()

try:
    original_df = load_dataframe(uploaded_file.getvalue(), uploaded_file.name)
    df, cleaning_notes = clean_dataframe(original_df, drop_duplicates, numeric_fill, categorical_fill, convert_datetimes, remove_outliers)
except Exception as exc:
    st.error(f"Could not load or clean the dataset: {exc}")
    st.stop()

column_types = infer_column_types(df)
numeric_cols = column_types.numeric
categorical_cols = sorted(set(column_types.categorical + column_types.binary))
all_cols_with_none = ["None"] + df.columns.tolist()

summary_text = generate_summary_analysis(df, original_df, cleaning_notes, st.session_state.regression_text)

(
    overview_tab,
    story_tab,
    summary_tab,
    descriptive_tab,
    categorical_tab,
    inferential_tab,
    regression_tab,
    ml_tab,
    visualization_tab,
    ask_tab,
    export_tab,
    business_report_tab,
) = st.tabs(
    [
        "Dataset Overview",
        "Storytelling Dashboard",
        "Summary Analysis",
        "Descriptive Statistics",
        "Categorical Analysis",
        "Inferential Tests",
        "Regression Modeling",
        "Machine Learning",
        "Visualizations",
        "Ask Your Data",
        "Export Report",
        "Business Report",
    ]
)

with overview_tab:
    if "Dataset Overview" in selected_sections:
        st.subheader("Dataset Overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{df.shape[0]:,}")
        m2.metric("Columns", f"{df.shape[1]:,}")
        m3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
        m4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

        st.markdown("#### Detected column groups")
        st.write(
            {
                "numeric": numeric_cols,
                "categorical": column_types.categorical,
                "datetime": column_types.datetime,
                "binary": column_types.binary,
            }
        )

        st.markdown("#### Data types and missing values")
        schema = pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "missing_count": df.isna().sum(),
                "missing_percent": df.isna().mean() * 100,
                "unique_count": df.nunique(dropna=True),
            }
        )
        st.dataframe(schema, use_container_width=True)

        st.markdown("#### Cleaning summary")
        for note in cleaning_notes:
            st.write(f"- {note}")

        st.markdown("#### Sample rows")
        st.dataframe(df.head(sample_rows), use_container_width=True)
    else:
        st.info("Dataset Overview is hidden by the sidebar section selector.")


with story_tab:
    if "Storytelling Dashboard" in selected_sections:
        st.subheader("Storytelling Dashboard")
        st.markdown("### Executive summary")
        st.markdown(storytelling_text(df, original_df, cleaning_notes))

        st.markdown("### KPI cards")
        missing_pct = safe_pct(df.isna().sum().sum(), df.shape[0] * df.shape[1])
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Rows", f"{df.shape[0]:,}")
        k2.metric("Columns", f"{df.shape[1]:,}")
        k3.metric("Missing %", f"{missing_pct:.1f}%")
        k4.metric("Duplicates", f"{int(df.duplicated().sum()):,}")
        k5.metric("Numeric", f"{len(numeric_cols):,}")
        k6.metric("Categorical", f"{len(categorical_cols):,}")

        st.markdown("### Key findings")
        findings = summarize_key_findings(df)
        cards = [
            ("Strongest positive correlation", findings["strongest_positive_correlation"]),
            ("Strongest negative correlation", findings["strongest_negative_correlation"]),
            ("Most variable numeric column", findings["most_variable_numeric_column"]),
            ("Most imbalanced categorical column", findings["most_imbalanced_categorical_column"]),
            ("Highest missing-value column", findings["highest_missing_value_column"]),
        ]
        for row_start in range(0, len(cards), 3):
            cols = st.columns(min(3, len(cards) - row_start))
            for col_obj, (label, value) in zip(cols, cards[row_start: row_start + 3]):
                with col_obj:
                    st.info(f"**{label}**\n\n{value}")

        st.markdown("### Recommended charts")
        charts = recommended_story_charts(df)
        if charts:
            for title, fig, caption in charts:
                st.plotly_chart(fig, use_container_width=True)
                st.caption(caption)
        else:
            st.info("Upload data with numeric or categorical columns to generate recommended charts.")

        st.markdown("### Data Story")
        st.markdown(storytelling_text(df, original_df, cleaning_notes))
    else:
        st.info("Storytelling Dashboard is hidden by the sidebar section selector.")

with summary_tab:
    if "Summary Analysis" in selected_sections:
        st.markdown(summary_text)
    else:
        st.info("Summary Analysis is hidden by the sidebar section selector.")

with descriptive_tab:
    if "Descriptive Statistics" in selected_sections:
        st.subheader("Numeric descriptive statistics")
        num_stats = numeric_descriptive_stats(df)
        if num_stats.empty:
            st.info("No numeric columns detected.")
        else:
            st.dataframe(num_stats, use_container_width=True)
            st.download_button("Download numeric summary CSV", num_stats.to_csv().encode("utf-8"), "numeric_summary.csv", "text/csv")

        st.subheader("Categorical summaries")
        cat_stats = categorical_summary(df)
        if cat_stats.empty:
            st.info("No categorical columns detected.")
        else:
            st.dataframe(cat_stats, use_container_width=True)
            st.download_button("Download categorical summary CSV", cat_stats.to_csv().encode("utf-8"), "categorical_summary.csv", "text/csv")

            selected_cat = st.selectbox("Frequency table column", categorical_cols)
            freq = frequency_table(df, selected_cat)
            st.dataframe(freq, use_container_width=True)
            st.download_button(f"Download {selected_cat} frequency CSV", freq.to_csv().encode("utf-8"), f"{selected_cat}_frequency.csv", "text/csv")
    else:
        st.info("Descriptive Statistics is hidden by the sidebar section selector.")

with categorical_tab:
    if "Categorical Analysis" in selected_sections:
        st.subheader("Cross-tabulation and Chi-square test")
        if len(categorical_cols) < 2:
            st.info("At least two categorical or binary columns are required.")
        else:
            col_a = st.selectbox("First categorical column", categorical_cols, index=0)
            col_b = st.selectbox("Second categorical column", categorical_cols, index=min(1, len(categorical_cols) - 1))
            if col_a == col_b:
                st.warning("Choose two different categorical columns.")
            else:
                result = chi_square_analysis(df, col_a, col_b)
                if not result.get("valid"):
                    st.warning(result["message"])
                else:
                    st.markdown("#### Observed cross-tabulation")
                    st.dataframe(result["observed"], use_container_width=True)
                    st.markdown("#### Row-normalized cross-tabulation (%)")
                    st.dataframe(result["normalized"], use_container_width=True)
                    st.markdown("#### Chi-square result")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("χ² statistic", f"{result['chi2']:.4f}")
                    c2.metric("p-value", f"{result['p_value']:.4g}")
                    c3.metric("Degrees of freedom", result["dof"])
                    st.write(result["interpretation"])
                    st.markdown("#### Expected frequencies")
                    st.dataframe(result["expected"], use_container_width=True)
                    record_session_result(f"Chi-square {col_a} × {col_b}: χ²={result['chi2']:.4f}, p={result['p_value']:.4g}, dof={result['dof']}. {result['interpretation']}")
    else:
        st.info("Categorical Analysis is hidden by the sidebar section selector.")

with inferential_tab:
    if "Inferential Tests" in selected_sections:
        st.subheader("Inferential testing")
        test_name = st.selectbox("Select a test", ["Pearson correlation", "Spearman correlation", "Independent samples t-test", "One-way ANOVA", "Mann-Whitney U", "Kruskal-Wallis"])

        if test_name in {"Pearson correlation", "Spearman correlation"}:
            if len(numeric_cols) < 2:
                st.info("At least two numeric columns are required.")
            else:
                x = st.selectbox("First numeric variable", numeric_cols, index=0)
                y = st.selectbox("Second numeric variable", numeric_cols, index=min(1, len(numeric_cols) - 1))
                if x == y:
                    st.warning("Choose two different numeric variables.")
                else:
                    method = "Pearson" if test_name.startswith("Pearson") else "Spearman"
                    result = correlation_test(df, x, y, method)
                    if result.get("valid"):
                        st.metric(f"{method} statistic", f"{result['statistic']:.4f}")
                        st.metric("p-value", f"{result['p_value']:.4g}")
                        st.write(f"Sample size: {result['n']:,}. {result['interpretation']}")
                        record_session_result(f"{method} correlation {x} vs {y}: statistic={result['statistic']:.4f}, p={result['p_value']:.4g}, n={result['n']}. {result['interpretation']}")
                    else:
                        st.warning(result["message"])
        else:
            if not numeric_cols or not categorical_cols:
                st.info("These tests require one numeric variable and one grouping column.")
            else:
                numeric = st.selectbox("Numeric outcome", numeric_cols)
                group = st.selectbox("Grouping variable", categorical_cols)
                result = grouped_test(df, numeric, group, test_name)
                if result.get("valid"):
                    st.metric("Test statistic", f"{result['statistic']:.4f}")
                    st.metric("p-value", f"{result['p_value']:.4g}")
                    st.write(f"Sample size: {result['n']:,}. {result['interpretation']}")
                    st.write("Group sizes:", result["group_sizes"])
                    record_session_result(f"{test_name} for {numeric} by {group}: statistic={result['statistic']:.4f}, p={result['p_value']:.4g}, n={result['n']}. {result['interpretation']}")
                else:
                    st.warning(result["message"])
    else:
        st.info("Inferential Tests is hidden by the sidebar section selector.")

with regression_tab:
    if "Regression Modeling" in selected_sections:
        st.subheader("Regression modeling")
        if not df.columns.tolist():
            st.info("No columns available.")
        else:
            target = st.selectbox("Target variable", df.columns.tolist())
            possible_predictors = [col for col in df.columns if col != target]
            predictors = st.multiselect("Predictor variables", possible_predictors, default=possible_predictors[: min(3, len(possible_predictors))])
            target_is_numeric = target in numeric_cols
            target_is_binary = df[target].dropna().nunique() == 2
            model_type = st.radio("Model type", ["Linear regression", "Logistic regression"], horizontal=True, index=0 if target_is_numeric else 1)

            if st.button("Run regression model", type="primary"):
                if not predictors:
                    st.warning("Select at least one predictor.")
                elif model_type == "Linear regression":
                    if not target_is_numeric:
                        st.warning("Linear regression requires a numeric target.")
                    else:
                        result = linear_regression_model(df, target, predictors)
                        if result.get("valid"):
                            st.metric("R-squared", f"{result['r_squared']:.4f}")
                            st.metric("Adjusted R-squared", f"{result['adjusted_r_squared']:.4f}")
                            st.write(f"Sample size: {result['n']:,}; intercept: {result['intercept']:.4f}")
                            st.dataframe(result["coefficients"], use_container_width=True)
                            st.dataframe(result["residual_summary"], use_container_width=True)
                            fig = px.scatter(result["predictions"], x="predicted", y="residual", trendline="ols", title="Residual Plot")
                            fig.add_hline(y=0, line_dash="dash")
                            st.plotly_chart(fig, use_container_width=True)
                            important = result["coefficients"].drop(index="const", errors="ignore").sort_values("coefficient", key=np.abs, ascending=False).head(3)
                            st.session_state.regression_text = f"- Linear regression predicting **{target}** used {result['n']:,} complete rows and achieved R²={result['r_squared']:.3f} (adjusted R²={result['adjusted_r_squared']:.3f}). The largest absolute predictors were {', '.join(important.index.astype(str).tolist()) or 'not available'}."
                        else:
                            st.warning(result["message"])
                else:
                    if not target_is_binary:
                        st.warning("Logistic regression requires a binary target.")
                    else:
                        result = logistic_regression_model(df, target, predictors)
                        if result.get("valid"):
                            st.metric("Accuracy", f"{result['accuracy']:.4f}")
                            st.write(f"Train rows: {result['n_train']:,}; test rows: {result['n_test']:,}; classes: {result['classes']}")
                            st.dataframe(result["coefficients"], use_container_width=True)
                            st.dataframe(result["confusion_matrix"], use_container_width=True)
                            st.dataframe(result["classification_report"], use_container_width=True)
                            roc_fig = px.line(result["roc"], x="fpr", y="tpr", title="ROC Curve")
                            roc_fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
                            st.plotly_chart(roc_fig, use_container_width=True)
                            top_features = result["coefficients"].head(3)["feature"].tolist()
                            st.session_state.regression_text = f"- Logistic regression predicting **{target}** achieved accuracy={result['accuracy']:.3f} on {result['n_test']:,} test rows. The largest absolute coefficients were {', '.join(top_features)}."
                        else:
                            st.warning(result["message"])
    else:
        st.info("Regression Modeling is hidden by the sidebar section selector.")


with ml_tab:
    if "Machine Learning" in selected_sections:
        st.subheader("Machine Learning")
        if df.shape[1] < 2:
            st.info("Machine learning requires at least one target and one predictor column.")
        else:
            c1, c2 = st.columns([1, 2])
            with c1:
                ml_target = st.selectbox("Target column", df.columns.tolist(), key="ml_target")
                ml_predictor_options = [col for col in df.columns if col != ml_target]
                ml_predictors = st.multiselect("Predictor columns", ml_predictor_options, default=ml_predictor_options[: min(5, len(ml_predictor_options))], key="ml_predictors")
                problem_type = st.radio("Problem type", ["auto-detect", "regression", "classification"], horizontal=False)
                resolved_type = problem_type
                if resolved_type == "auto-detect":
                    resolved_type = "regression" if ml_target in numeric_cols and df[ml_target].nunique(dropna=True) > 10 else "classification"
                model_options = ["Linear Regression", "Random Forest Regressor", "Decision Tree Regressor"] if resolved_type == "regression" else ["Logistic Regression", "Random Forest Classifier", "Decision Tree Classifier", "KNN Classifier"]
                model_name = st.selectbox("Model", model_options)
                test_size = st.slider("Test split", 0.1, 0.5, 0.25, 0.05)
                random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
                run_ml = st.button("Train model", type="primary")
            with c2:
                st.write("**Modeling notes**")
                st.write("Categorical predictors are one-hot encoded, numeric predictors are imputed, and missing target rows are removed. Scaling is applied for Logistic Regression, KNN, and Linear Regression.")
                st.write(f"Detected problem type: **{resolved_type}**")

            if run_ml:
                result = train_ml_model(df, ml_target, ml_predictors, problem_type, model_name, float(test_size), int(random_state))
                if not result.get("valid"):
                    st.warning(result["message"])
                else:
                    st.session_state.ml_results = {"raw": result, "summary": model_interpretation(result)}
                    st.success("Model trained successfully.")

            result = st.session_state.get("ml_results", {}).get("raw")
            if result and result.get("valid"):
                st.markdown("### Model results")
                st.info(model_interpretation(result))
                if result["problem_type"] == "regression":
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Model", result["model_name"])
                    m2.metric("Train rows", f"{result['n_train']:,}")
                    m3.metric("MAE", f"{result['mae']:.4g}")
                    m4.metric("RMSE", f"{result['rmse']:.4g}")
                    m5.metric("R²", f"{result['r2']:.4f}")
                    pred_fig = px.scatter(result["predictions"], x="actual", y="predicted", title="Prediction vs Actual")
                    min_val = float(np.nanmin([result["predictions"]["actual"].min(), result["predictions"]["predicted"].min()]))
                    max_val = float(np.nanmax([result["predictions"]["actual"].max(), result["predictions"]["predicted"].max()]))
                    pred_fig.add_shape(type="line", x0=min_val, x1=max_val, y0=min_val, y1=max_val, line=dict(dash="dash"))
                    st.plotly_chart(pred_fig, use_container_width=True)
                    resid_fig = px.scatter(result["predictions"], x="predicted", y="residual", title="Residual Plot")
                    resid_fig.add_hline(y=0, line_dash="dash")
                    st.plotly_chart(resid_fig, use_container_width=True)
                else:
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Accuracy", f"{result['accuracy']:.4f}")
                    m2.metric("Precision", f"{result['precision']:.4f}")
                    m3.metric("Recall", f"{result['recall']:.4f}")
                    m4.metric("F1", f"{result['f1']:.4f}")
                    m5.metric("Test rows", f"{result['n_test']:,}")
                    st.dataframe(result["confusion_matrix"], use_container_width=True)
                    st.dataframe(result["classification_report"], use_container_width=True)
                    if "roc" in result:
                        roc_fig = px.line(result["roc"], x="fpr", y="tpr", title=f"ROC Curve (AUC={result['roc_auc']:.3f})")
                        roc_fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
                        st.plotly_chart(roc_fig, use_container_width=True)
                if isinstance(result.get("feature_importance"), pd.DataFrame) and not result["feature_importance"].empty:
                    st.markdown("### Feature importance")
                    st.dataframe(result["feature_importance"].head(30), use_container_width=True)
                    st.plotly_chart(px.bar(result["feature_importance"].head(15), x="importance", y="feature", orientation="h", title="Top feature importance"), use_container_width=True)
    else:
        st.info("Machine Learning is hidden by the sidebar section selector.")

with visualization_tab:
    if "Visualizations" in selected_sections:
        st.subheader("Interactive visualization builder")
        chart_type = st.selectbox(
            "Chart type",
            [
                "Histogram",
                "KDE/density plot",
                "Boxplot",
                "Violin plot",
                "Scatter plot",
                "Regression scatter plot",
                "Correlation heatmap",
                "Bar chart",
                "Count plot",
                "Grouped bar chart",
                "Line plot",
                "Pie chart",
                "Stacked bar chart",
                "Area chart",
                "Pair plot style scatter matrix",
                "3D scatter plot",
            ],
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            x_col = st.selectbox("X-axis", all_cols_with_none, index=1 if len(all_cols_with_none) > 1 else 0)
        with c2:
            y_options = ["None"] + numeric_cols
            y_col = st.selectbox("Y-axis", y_options, index=1 if len(y_options) > 1 else 0)
        with c3:
            color_col = st.selectbox("Color/group", all_cols_with_none)
        with c4:
            agg = st.selectbox("Aggregation", ["Mean", "Median", "Sum", "Count"], index=0)

        fig = render_visualization(
            df,
            chart_type,
            None if x_col == "None" else x_col,
            None if y_col == "None" else y_col,
            color_col,
            agg,
        )
        if fig is None:
            st.warning("This chart needs different variable selections or more numeric columns.")
        else:
            st.plotly_chart(fig, use_container_width=True)
            chart_record = {
                "chart_type": chart_type,
                "x": None if x_col == "None" else x_col,
                "y": None if y_col == "None" else y_col,
                "color": None if color_col == "None" else color_col,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            }
            if chart_record not in st.session_state.chart_history:
                st.session_state.chart_history.append(chart_record)
                st.session_state.chart_history = st.session_state.chart_history[-12:]
    else:
        st.info("Visualizations is hidden by the sidebar section selector.")


with ask_tab:
    if "Ask Your Data" in selected_sections:
        st.subheader("Ask Your Data — AI Data Scientist Copilot")
        st.caption(
            "Ask plain-English questions and I will run approved local analyses against the uploaded dataframe: "
            "charts, statistical tests, models, tables, and expert interpretation. Python runs internally and is not shown unless you explicitly ask for code."
        )

        dataset_signature = (tuple(df.columns.astype(str).tolist()), int(df.shape[0]), int(df.shape[1]))
        if st.session_state.chat_dataset_signature != dataset_signature:
            auto_summary = generate_expert_intelligence_markdown(df)
            st.session_state.chat_history = [
                {
                    "id": "msg_auto_dataset_intelligence",
                    "role": "assistant",
                    "content": auto_summary,
                    "analysis_result": None,
                }
            ]
            st.session_state.pending_chat_prompt = None
            st.session_state.pending_analysis_request = None
            st.session_state.chat_dataset_signature = dataset_signature
            st.session_state.analysis_memory = [{"type": "automatic_dataset_intelligence", "summary": auto_summary[:1500]}]

        intelligence = build_dataset_intelligence(df)
        panel1, panel2, panel3 = st.columns(3)
        with panel1:
            st.info(
                "**Expert Recommendations**\n\n"
                + "\n".join(f"- {step}" for step in intelligence["next_steps"][:3])
            )
        with panel2:
            risk_items: list[str] = []
            if intelligence["high_missing"]:
                risk_items.append(f"High missingness in {intelligence['high_missing'][0]['column']} ({intelligence['high_missing'][0]['percent']:.1f}%).")
            if intelligence["outlier_heavy_columns"]:
                risk_items.append(f"Outliers in {intelligence['outlier_heavy_columns'][0]['column']} ({intelligence['outlier_heavy_columns'][0]['percent']:.1f}% flagged).")
            if intelligence["imbalance_warnings"]:
                risk_items.append(f"Imbalance in {intelligence['imbalance_warnings'][0]['column']} ({intelligence['imbalance_warnings'][0]['top_percent']:.1f}% top class).")
            if intelligence["multicollinearity"]:
                risk_items.append(f"Multicollinearity: {intelligence['multicollinearity'][0]['left']} vs {intelligence['multicollinearity'][0]['right']}.")
            st.warning("**Potential Risks**\n\n" + "\n".join(f"- {item}" for item in (risk_items or ["No major automated risk flags; still validate with visuals."])))
        with panel3:
            st.success(
                "**What I can execute**\n\n"
                "- Plotly visualizations\n"
                "- Correlation, chi-square, t-test, ANOVA, Mann-Whitney, Kruskal-Wallis\n"
                "- Regression/classification models and feature importance"
            )

        with st.expander("Automatic Dataset Intelligence Summary", expanded=True):
            st.markdown(generate_expert_intelligence_markdown(df))

        with st.expander("What the assistant can use", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{df.shape[0]:,}")
            c2.metric("Columns", f"{df.shape[1]:,}")
            c3.metric("Quality score", f"{intelligence['quality_score']}/100")
            c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
            st.write(
                "The execution engine uses dataframe shape, schema, data types, missing-value reports, descriptive stats, "
                "categorical frequencies, correlations, IQR outlier counts, and previous Ask Your Data results. "
                "If OpenAI is configured, it may refine intent and narrative from metadata only; all calculations still run locally."
            )

        st.markdown("**Smart analysis buttons:**")
        smart_actions = suggest_smart_actions(df)
        for row_index in range(0, len(smart_actions), 4):
            row_actions = smart_actions[row_index : row_index + 4]
            cols = st.columns(len(row_actions))
            for col, suggestion in zip(cols, row_actions):
                if col.button(suggestion, key=f"ask_action_{row_index}_{suggestion}"):
                    st.session_state.pending_chat_prompt = suggestion
                    st.rerun()

        chat_container = st.container(height=620, border=True)
        with chat_container:
            if not st.session_state.chat_history:
                with st.chat_message("assistant"):
                    st.markdown(
                        "Hi — ask for an analysis such as **Show correlation heatmap**, **Run ANOVA for G3 by school**, "
                        "or **Build a regression model to predict G3 using G1, G2, studytime, failures, absences**."
                    )
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message.get("analysis_result"):
                        display_analysis_result(message["analysis_result"], message_id=message["id"], show_code=message.get("show_code", False))

        if "pending_analysis_request" not in st.session_state:
            st.session_state.pending_analysis_request = None

        if st.session_state.pending_analysis_request:
            pending = st.session_state.pending_analysis_request
            detected = pending["detected"]
            missing = pending["missing"]
            st.warning("I detected the analysis, but I need you to confirm one or more columns before I run it.")
            with st.form("ask_column_confirmation"):
                selected: dict[str, Any] = {}
                if "numeric_columns" in missing:
                    selected["columns"] = st.multiselect("Choose two numeric columns", missing["numeric_columns"], default=missing["numeric_columns"][:2], max_selections=2)
                if "categorical_columns" in missing:
                    selected["columns"] = st.multiselect("Choose two categorical columns", missing["categorical_columns"], default=missing["categorical_columns"][:2], max_selections=2)
                if "numeric_column" in missing:
                    value = st.selectbox("Choose numeric outcome/measure", missing["numeric_column"])
                    selected.setdefault("columns", []).append(value)
                if "group_column" in missing:
                    value = st.selectbox("Choose grouping/category column", missing["group_column"])
                    selected.setdefault("columns", []).append(value)
                if "categorical_column" in missing:
                    value = st.selectbox("Choose categorical column", missing["categorical_column"])
                    selected.setdefault("columns", []).append(value)
                if "target_column" in missing:
                    default_target = "G3" if "G3" in missing["target_column"] else missing["target_column"][-1]
                    selected["target"] = st.selectbox("Choose model target", missing["target_column"], index=missing["target_column"].index(default_target))
                if "predictor_columns" in missing:
                    predictor_options = [c for c in missing["predictor_columns"] if c != selected.get("target", detected.target)]
                    selected["predictors"] = st.multiselect("Choose predictor columns", predictor_options, default=predictor_options[: min(5, len(predictor_options))])
                run_now = st.form_submit_button("Run this analysis")
                cancel = st.form_submit_button("Cancel")
            if cancel:
                st.session_state.pending_analysis_request = None
                st.rerun()
            if run_now:
                question = pending["question"]
                result = run_requested_analysis(detected, df, selected, question)
                assistant_content = f"Detected intent: **{detected.intent}** (confidence {detected.confidence:.0%}). I ran the requested analysis locally."
                message_id = f"msg_{len(st.session_state.chat_history)}"
                st.session_state.chat_history.append({"id": message_id, "role": "assistant", "content": assistant_content, "analysis_result": result, "show_code": detected.needs_code})
                store_analysis_memory(question, detected, result)
                record_session_result(f"Ask Your Data — {question}: {result.result_summary}")
                st.session_state.pending_analysis_request = None
                st.rerun()

        prompt = st.session_state.pending_chat_prompt or st.chat_input("Ask for a chart, test, model, quality check, or recommendation…")
        st.session_state.pending_chat_prompt = None

        if prompt:
            detected = detect_analysis_intent(prompt, df, dataframe_metadata(df))
            user_message = {"id": f"msg_{len(st.session_state.chat_history)}", "role": "user", "content": prompt, "analysis_result": None}
            st.session_state.chat_history.append(user_message)
            missing = needs_column_selection(detected, df)
            if missing:
                st.session_state.pending_analysis_request = {"question": prompt, "detected": detected, "missing": missing}
            else:
                result = run_requested_analysis(detected, df, {}, prompt)
                assistant_content = f"Detected intent: **{detected.intent}** (confidence {detected.confidence:.0%}). I ran the requested analysis locally."
                if not result.valid:
                    assistant_content = "I could not confidently run that request yet. Here is a safe clarification and examples based on your columns."
                message_id = f"msg_{len(st.session_state.chat_history)}"
                st.session_state.chat_history.append({"id": message_id, "role": "assistant", "content": assistant_content, "analysis_result": result, "show_code": detected.needs_code})
                store_analysis_memory(prompt, detected, result)
                record_session_result(f"Ask Your Data — {prompt}: {result.result_summary}")
            st.rerun()

        with st.expander("Ask Your Data memory for this session"):
            st.json(st.session_state.analysis_memory[-10:])
    else:
        st.info("Ask Your Data is hidden by the sidebar section selector.")

with export_tab:
    if "Export Report" in selected_sections:
        st.subheader("Export report and tables")
        report = build_report(df, original_df, cleaning_notes, summary_text, st.session_state.test_results[-20:], st.session_state.regression_text)
        st.download_button("Download Markdown report", report.encode("utf-8"), "analysis_report.md", "text/markdown")
        st.download_button("Download cleaned dataset CSV", df.to_csv(index=False).encode("utf-8"), "cleaned_dataset.csv", "text/csv")
        num_stats = numeric_descriptive_stats(df)
        cat_stats = categorical_summary(df)
        if not num_stats.empty:
            st.download_button("Download numeric descriptive statistics", num_stats.to_csv().encode("utf-8"), "numeric_descriptive_statistics.csv", "text/csv")
        if not cat_stats.empty:
            st.download_button("Download categorical summary", cat_stats.to_csv().encode("utf-8"), "categorical_summary.csv", "text/csv")
        with st.expander("Preview report"):
            st.markdown(report)
    else:
        st.info("Export Report is hidden by the sidebar section selector.")


with business_report_tab:
    if "Business Report" in selected_sections:
        st.subheader("Business Report")
        report = build_business_report(df, original_df, cleaning_notes, st.session_state.test_results)
        html_report = markdown_to_html(report)
        timestamp = st.session_state.report_generated_at.replace(":", "").replace(" ", "_")
        st.caption(f"Timestamped report generation: {st.session_state.report_generated_at}")
        st.download_button("Download Markdown business report", report.encode("utf-8"), f"business_report_{timestamp}.md", "text/markdown")
        st.download_button("Download HTML business report", html_report.encode("utf-8"), f"business_report_{timestamp}.html", "text/html")
        summary_tables = []
        num_stats = numeric_descriptive_stats(df)
        cat_stats = categorical_summary(df)
        if not num_stats.empty:
            summary_tables.append(num_stats.reset_index().assign(table="numeric_descriptive_statistics"))
        if not cat_stats.empty:
            summary_tables.append(cat_stats.reset_index().assign(table="categorical_summary"))
        if summary_tables:
            csv_summary = pd.concat(summary_tables, ignore_index=True, sort=False).to_csv(index=False)
            st.download_button("Download CSV summary tables", csv_summary.encode("utf-8"), f"summary_tables_{timestamp}.csv", "text/csv")
        with st.expander("Preview Markdown report", expanded=True):
            st.markdown(report)
        with st.expander("Preview HTML source"):
            st.code(html_report[:8000], language="html")
    else:
        st.info("Business Report is hidden by the sidebar section selector.")
