"""Professional Streamlit data analysis app for InsightForge.

This app is intentionally self-contained and does not replace the existing Next.js
frontend. It reuses the same Python analytics stack (Pandas, SciPy, Statsmodels,
Scikit-learn, and Plotly) to compute real statistics from the uploaded dataset.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
st.caption("Upload CSV, Excel, or JSON files and generate real descriptive, inferential, modeling, visualization, and written analysis outputs.")

if "test_results" not in st.session_state:
    st.session_state.test_results = []
if "regression_text" not in st.session_state:
    st.session_state.regression_text = ""

with st.sidebar:
    st.header("1) Upload data")
    uploaded_file = st.file_uploader("Choose a CSV, XLSX, or JSON file", type=["csv", "xlsx", "json"])

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
        ["Dataset Overview", "Summary Analysis", "Descriptive Statistics", "Categorical Analysis", "Inferential Tests", "Regression Modeling", "Visualizations", "Export Report"],
        default=["Dataset Overview", "Summary Analysis", "Descriptive Statistics", "Categorical Analysis", "Inferential Tests", "Regression Modeling", "Visualizations", "Export Report"],
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
    summary_tab,
    descriptive_tab,
    categorical_tab,
    inferential_tab,
    regression_tab,
    visualization_tab,
    export_tab,
) = st.tabs(
    [
        "Dataset Overview",
        "Summary Analysis",
        "Descriptive Statistics",
        "Categorical Analysis",
        "Inferential Tests",
        "Regression Modeling",
        "Visualizations",
        "Export Report",
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
    else:
        st.info("Visualizations is hidden by the sidebar section selector.")

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
