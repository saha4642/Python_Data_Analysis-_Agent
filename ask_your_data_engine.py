"""Safe natural-language analysis engine for the Streamlit Ask Your Data tab.

The module intentionally maps natural language to a whitelist of local analysis
functions. It never evaluates user-supplied Python and never needs the full
DataFrame for optional OpenAI intent classification.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - requirements include openai, but keep imports resilient.
    OpenAI = None  # type: ignore[assignment]

ALPHA = 0.05
MAX_CHART_ROWS = 5000
MAX_CATEGORY_LEVELS = 30
SUPPORTED_INTENTS = {
    "summary",
    "descriptive_statistics",
    "correlation",
    "correlation_heatmap",
    "scatter_plot",
    "regression_plot",
    "histogram",
    "kde_plot",
    "boxplot",
    "violin_plot",
    "bar_chart",
    "count_plot",
    "grouped_bar_chart",
    "line_plot",
    "pie_chart",
    "stacked_bar_chart",
    "area_chart",
    "scatter_matrix",
    "scatter_3d",
    "cross_tabulation",
    "chi_square_test",
    "t_test",
    "anova",
    "mann_whitney",
    "kruskal_wallis",
    "linear_regression",
    "logistic_regression",
    "random_forest",
    "decision_tree",
    "feature_importance",
    "missing_values",
    "outlier_detection",
    "data_quality",
    "recommended_analysis",
    "unknown/general_question",
}


@dataclass
class AnalysisIntent:
    """Detected request intent and extracted column hints."""

    intent: str
    confidence: float
    columns: list[str] = field(default_factory=list)
    target: str | None = None
    predictors: list[str] = field(default_factory=list)
    method: str | None = None
    needs_code: bool = False
    clarification: str | None = None
    source: str = "rules"


@dataclass
class AnalysisResult:
    """Whitelisted execution output consumed by Streamlit."""

    valid: bool
    intent: str
    title: str
    what_i_did: str
    result_summary: str
    interpretation: str
    next_steps: list[str]
    figures: list[go.Figure] = field(default_factory=list)
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    selected_columns: dict[str, Any] = field(default_factory=dict)
    code: str | None = None
    warning: str | None = None


def dataframe_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """Return compact metadata safe for rules or optional OpenAI classification."""
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric and not pd.api.types.is_datetime64_any_dtype(df[c])]
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    binary = [c for c in df.columns if df[c].dropna().nunique() == 2]
    return {
        "rows": int(df.shape[0]),
        "columns": df.columns.astype(str).tolist(),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "datetime_columns": datetime_cols,
        "binary_columns": binary,
        "missing_columns": df.isna().sum().sort_values(ascending=False).head(10).astype(int).to_dict(),
    }


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def extract_columns_from_question(user_question: str, df: pd.DataFrame) -> list[str]:
    """Detect dataset columns mentioned in the user's natural-language question."""
    q = str(user_question or "")
    q_lower = q.lower()
    normalized_q = _normalize(q)
    matches: list[tuple[int, str]] = []
    for col in df.columns.astype(str):
        escaped = re.escape(col)
        boundary_match = re.search(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", q, flags=re.IGNORECASE)
        compact_match = _normalize(col) and _normalize(col) in normalized_q
        if boundary_match or compact_match:
            position = boundary_match.start() if boundary_match else q_lower.find(str(col).lower())
            matches.append((position if position >= 0 else 10_000, col))
    ordered = []
    for _, col in sorted(matches, key=lambda item: (item[0], len(item[1]))):
        if col not in ordered:
            ordered.append(col)
    return ordered


def _columns_after_keyword(question: str, df: pd.DataFrame, keyword_pattern: str) -> list[str]:
    match = re.search(keyword_pattern, question, flags=re.IGNORECASE)
    if not match:
        return []
    return extract_columns_from_question(question[match.end() :], df)


def infer_target_and_predictors(user_question: str, df: pd.DataFrame, mentioned: list[str]) -> tuple[str | None, list[str]]:
    """Infer target and predictors from phrases such as 'predict G3 using G1 and G2'."""
    target: str | None = None
    predictors: list[str] = []
    target_matches = _columns_after_keyword(user_question, df, r"\b(predict|target|for|model(?:\s+for)?|explain)\b")
    using_matches = _columns_after_keyword(user_question, df, r"\b(using|with|from|based on|predictors?)\b")
    if target_matches:
        target = target_matches[0]
    elif mentioned:
        # For modeling language, the first mentioned column is usually the target.
        target = mentioned[0]
    if using_matches:
        predictors = [c for c in using_matches if c != target]
    elif target:
        predictors = [c for c in mentioned if c != target]
    return target, predictors


def _keyword_intent(question: str) -> tuple[str, float, str | None]:
    q = question.lower()
    method = None
    if any(phrase in q for phrase in ["show me the code", "show code", "python code", "how did you code"]):
        return "unknown/general_question", 0.55, method
    if any(word in q for word in ["recommend", "what should", "explore next", "best visualization", "analyze first"]):
        return "recommended_analysis", 0.9, method
    if any(word in q for word in ["missing", "null", "na values"]):
        return "missing_values", 0.9, method
    if any(word in q for word in ["quality", "clean", "duplicates"]):
        return "data_quality", 0.85, method
    if any(word in q for word in ["outlier", "anomal"]):
        return "outlier_detection", 0.9, method
    if "chi" in q or "chisquare" in q or "chi-square" in q:
        return "chi_square_test", 0.95, method
    if "anova" in q or "analysis of variance" in q:
        return "anova", 0.95, method
    if "mann" in q or "whitney" in q:
        return "mann_whitney", 0.95, method
    if "kruskal" in q or "wallis" in q:
        return "kruskal_wallis", 0.95, method
    if "t-test" in q or "ttest" in q or " t test" in q:
        return "t_test", 0.95, method
    if "spearman" in q:
        return "correlation", 0.95, "Spearman"
    if "pearson" in q:
        return "correlation", 0.95, "Pearson"
    if "heatmap" in q and "corr" in q:
        return "correlation_heatmap", 0.96, method
    if "strongest correlation" in q or "top correlation" in q or ("correlation" in q and any(w in q for w in ["strongest", "find", "rank"])):
        return "correlation", 0.9, method
    if "correlation" in q or "correlat" in q or re.search(r"\bcorr\b", q):
        return "correlation", 0.85, method
    if "feature importance" in q or "most important" in q or "important for predicting" in q or "features predict" in q or "predictors" in q:
        return "feature_importance", 0.92, method
    if "random forest" in q:
        return "random_forest", 0.93, method
    if "decision tree" in q:
        return "decision_tree", 0.9, method
    if "logistic" in q:
        return "logistic_regression", 0.93, method
    if any(word in q for word in ["regression model", "linear regression", "multiple regression", "build regression", "predict "]):
        return "linear_regression", 0.9, method
    if "regression" in q and any(w in q for w in ["plot", "scatter", "line"]):
        return "regression_plot", 0.9, method
    if "scatter matrix" in q or "pair plot" in q or "pairplot" in q:
        return "scatter_matrix", 0.9, method
    if "3d" in q and "scatter" in q:
        return "scatter_3d", 0.9, method
    if "scatter" in q or " vs " in q or " versus " in q:
        return "scatter_plot", 0.87, method
    if "hist" in q or "distribution" in q:
        return "histogram", 0.88, method
    if "kde" in q or "density" in q:
        return "kde_plot", 0.88, method
    if "box" in q or "compare" in q:
        return "boxplot", 0.82, method
    if "violin" in q:
        return "violin_plot", 0.9, method
    if "cross tab" in q or "crosstab" in q or "contingency" in q:
        return "cross_tabulation", 0.9, method
    if "stack" in q and "bar" in q:
        return "stacked_bar_chart", 0.9, method
    if "group" in q and "bar" in q:
        return "grouped_bar_chart", 0.86, method
    if "count" in q:
        return "count_plot", 0.82, method
    if "bar" in q:
        return "bar_chart", 0.82, method
    if "pie" in q:
        return "pie_chart", 0.86, method
    if "line" in q or "trend" in q:
        return "line_plot", 0.78, method
    if "area" in q:
        return "area_chart", 0.8, method
    if any(word in q for word in ["describe", "statistics", "descriptive", "stats"]):
        return "descriptive_statistics", 0.86, method
    if any(word in q for word in ["summarize", "summary", "overview"]):
        return "summary", 0.8, method
    return "unknown/general_question", 0.2, method


def _openai_refine_intent(user_question: str, metadata: dict[str, Any], current: AnalysisIntent) -> AnalysisIntent:
    """Optionally refine low-confidence intent without sending the dataset."""
    if current.confidence >= 0.75 or OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return current
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            messages=[
                {"role": "system", "content": "Classify a data-analysis request. Return only JSON with intent, confidence, columns, target, predictors, method. Do not ask for data."},
                {"role": "user", "content": json.dumps({"question": user_question, "valid_intents": sorted(SUPPORTED_INTENTS), "metadata": metadata}, default=str)[:6000]},
            ],
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        intent = payload.get("intent", current.intent)
        if intent not in SUPPORTED_INTENTS:
            intent = current.intent
        return AnalysisIntent(
            intent=intent,
            confidence=float(payload.get("confidence", current.confidence)),
            columns=[c for c in payload.get("columns", current.columns) if c in metadata["columns"]],
            target=payload.get("target") if payload.get("target") in metadata["columns"] else current.target,
            predictors=[c for c in payload.get("predictors", current.predictors) if c in metadata["columns"]],
            method=payload.get("method") or current.method,
            needs_code=current.needs_code,
            source="openai",
        )
    except Exception:
        return current


def detect_analysis_intent(user_question: str, df: pd.DataFrame, metadata: dict[str, Any] | None = None) -> AnalysisIntent:
    """Classify the user's safe analysis request using rules plus optional OpenAI."""
    metadata = metadata or dataframe_metadata(df)
    intent, confidence, method = _keyword_intent(user_question)
    columns = extract_columns_from_question(user_question, df)
    target, predictors = infer_target_and_predictors(user_question, df, columns)
    needs_code = any(phrase in user_question.lower() for phrase in ["show me the code", "show code", "python code", "how did you code"])
    detected = AnalysisIntent(intent=intent, confidence=confidence, columns=columns, target=target, predictors=predictors, method=method, needs_code=needs_code)
    return _openai_refine_intent(user_question, metadata, detected)


def suggest_smart_actions(df: pd.DataFrame) -> list[str]:
    """Build dataset-aware one-click analysis prompts."""
    meta = dataframe_metadata(df)
    numeric = meta["numeric_columns"]
    categorical = meta["categorical_columns"] + [c for c in meta["binary_columns"] if c not in meta["categorical_columns"]]
    suggestions = []
    if len(numeric) >= 2:
        suggestions.append("Show correlation heatmap")
        pair = _strongest_corr_pair(df, numeric)
        if pair:
            suggestions.append(f"Make a scatter plot of {pair[0]} vs {pair[1]}")
    if numeric:
        target = "G3" if "G3" in numeric else numeric[-1]
        suggestions.append(f"Build a regression model to predict {target}")
        suggestions.append(f"Find strongest predictors for {target}")
        suggestions.append(f"Find outliers in {numeric[0]}")
    if numeric and categorical:
        y = "G3" if "G3" in numeric else numeric[0]
        x = "school" if "school" in categorical else categorical[0]
        suggestions.append(f"Show boxplot of {y} by {x}")
        suggestions.append(f"Run ANOVA for {y} by {x}")
    if len(categorical) >= 2:
        a = "Fjob" if "Fjob" in categorical else categorical[0]
        b = "Mjob" if "Mjob" in categorical else categorical[1]
        if a != b:
            suggestions.append(f"Run chi-square test between {a} and {b}")
    suggestions.extend(["Show missing value report", "Recommend best visualizations"])
    return list(dict.fromkeys(suggestions))[:10]


def needs_column_selection(intent: AnalysisIntent, df: pd.DataFrame) -> dict[str, Any]:
    """Return missing column roles that Streamlit should ask the user to choose."""
    meta = dataframe_metadata(df)
    numeric = meta["numeric_columns"]
    categorical = meta["categorical_columns"] + [c for c in meta["binary_columns"] if c not in meta["categorical_columns"]]
    cols = intent.columns
    missing: dict[str, Any] = {}
    two_numeric = {"correlation", "scatter_plot", "regression_plot"}
    numeric_by_group = {"boxplot", "violin_plot", "anova", "t_test", "mann_whitney", "kruskal_wallis"}
    two_categorical = {"chi_square_test", "cross_tabulation", "stacked_bar_chart"}
    one_numeric = {"histogram", "kde_plot", "outlier_detection"}
    one_categorical = {"count_plot", "pie_chart"}
    if intent.intent in two_numeric and len([c for c in cols if c in numeric]) < 2:
        missing["numeric_columns"] = numeric
    if intent.intent in numeric_by_group:
        if not any(c in numeric for c in cols):
            missing["numeric_column"] = numeric
        if not any(c in categorical for c in cols):
            missing["group_column"] = categorical or df.columns.astype(str).tolist()
    if intent.intent in two_categorical and len([c for c in cols if c in categorical]) < 2:
        missing["categorical_columns"] = categorical or df.columns.astype(str).tolist()
    if intent.intent in one_numeric and not any(c in numeric for c in cols):
        missing["numeric_column"] = numeric
    if intent.intent in one_categorical and not any(c in categorical for c in cols):
        missing["categorical_column"] = categorical or df.columns.astype(str).tolist()
    if intent.intent in {"linear_regression", "logistic_regression", "random_forest", "decision_tree", "feature_importance"}:
        if not intent.target:
            missing["target_column"] = df.columns.astype(str).tolist()
        if not intent.predictors:
            missing["predictor_columns"] = [c for c in df.columns.astype(str) if c != intent.target]
    return missing


def _sample_for_chart(df: pd.DataFrame, max_rows: int = MAX_CHART_ROWS) -> tuple[pd.DataFrame, str | None]:
    if len(df) <= max_rows:
        return df, None
    return df.sample(max_rows, random_state=42), f"Chart uses a reproducible sample of {max_rows:,} rows from {len(df):,} rows for performance."


def _numeric_cols(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def _categorical_cols(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]


def _first_numeric(df: pd.DataFrame, columns: list[str] | None = None) -> str | None:
    candidates = columns or df.columns.astype(str).tolist()
    return next((c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])), None)


def _first_categorical(df: pd.DataFrame, columns: list[str] | None = None) -> str | None:
    candidates = columns or df.columns.astype(str).tolist()
    return next((c for c in candidates if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])), None)


def _strongest_corr_pair(df: pd.DataFrame, numeric_cols: list[str] | None = None) -> tuple[str, str, float] | None:
    numeric_cols = numeric_cols or df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr(numeric_only=True)
    pairs = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value):
                pairs.append((left, right, float(value)))
    return max(pairs, key=lambda p: abs(p[2])) if pairs else None


def _format_p(p_value: float) -> str:
    return "nan" if pd.isna(p_value) else f"{p_value:.4g}"


def _sig(p_value: float) -> str:
    return "statistically significant" if pd.notna(p_value) and p_value < ALPHA else "not statistically significant"


def _invalid(intent: str, message: str, df: pd.DataFrame | None = None) -> AnalysisResult:
    next_steps = ["Try a more specific question that names the columns you want to analyze."]
    if df is not None:
        examples = suggest_smart_actions(df)[:4]
        if examples:
            next_steps.extend(examples)
    return AnalysisResult(False, intent, "I need a little more information", "I matched your request to the safest available analysis workflow, but required inputs were missing or invalid.", message, message, next_steps)


def _corr_table(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number)
    rows = []
    for i, left in enumerate(numeric.columns):
        for right in numeric.columns[i + 1 :]:
            pair = numeric[[left, right]].dropna()
            if len(pair) >= 3 and pair[left].nunique() > 1 and pair[right].nunique() > 1:
                r, p = stats.pearsonr(pair[left], pair[right])
                rows.append({"left": left, "right": right, "pearson_r": r, "p_value": p, "n": len(pair), "abs_r": abs(r)})
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False).drop(columns="abs_r") if rows else pd.DataFrame()


def _missing_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame({"missing_count": df.isna().sum(), "missing_percent": df.isna().mean() * 100, "dtype": df.dtypes.astype(str)})
    return report.sort_values(["missing_count", "missing_percent"], ascending=False)


def _outlier_report(df: pd.DataFrame, column: str | None = None) -> pd.DataFrame:
    cols = [column] if column else df.select_dtypes(include=np.number).columns.tolist()
    rows = []
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.nunique() <= 2:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((values < lower) | (values > upper)).sum())
        rows.append({"column": col, "lower_fence": lower, "upper_fence": upper, "outlier_count": count, "outlier_percent": count / len(values) * 100})
    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False) if rows else pd.DataFrame()


def _build_preprocessor(x: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    numeric_features = x.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [c for c in x.columns if c not in numeric_features]
    transformers = []
    if numeric_features:
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), numeric_features))
    if categorical_features:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def _auto_predictors(df: pd.DataFrame, target: str, predictors: list[str]) -> list[str]:
    if predictors:
        return [c for c in predictors if c in df.columns and c != target]
    candidates = [c for c in df.columns.astype(str) if c != target]
    # Keep modeling responsive and avoid high-cardinality text fields by default.
    selected = []
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique(dropna=True) <= MAX_CATEGORY_LEVELS:
            selected.append(c)
    return selected[:20]


def _train_model(df: pd.DataFrame, target: str, predictors: list[str], intent: str) -> AnalysisResult:
    predictors = _auto_predictors(df, target, predictors)
    if target not in df.columns or not predictors:
        return _invalid(intent, "Choose a target column and at least one predictor column.", df)
    model_df = df[[target] + predictors].dropna(subset=[target]).copy()
    if len(model_df) < 12:
        return _invalid(intent, "Modeling needs at least 12 rows with a non-missing target.", df)
    x = model_df[predictors]
    y_raw = model_df[target]
    is_regression = intent in {"linear_regression"} or (pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique(dropna=True) > 10 and intent not in {"logistic_regression"})
    if intent == "feature_importance":
        is_regression = pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique(dropna=True) > 10
    if is_regression:
        y = pd.to_numeric(y_raw, errors="coerce")
        valid = y.notna()
        x, y = x.loc[valid], y.loc[valid]
        if len(y) < 12 or y.nunique() < 2:
            return _invalid(intent, "Regression requires a numeric target with enough variation.", df)
        estimator_name = "Random Forest Regressor" if intent in {"random_forest", "feature_importance"} else "Decision Tree Regressor" if intent == "decision_tree" else "Linear Regression"
        estimator = RandomForestRegressor(n_estimators=250, random_state=42) if estimator_name == "Random Forest Regressor" else DecisionTreeRegressor(random_state=42) if estimator_name == "Decision Tree Regressor" else LinearRegression()
        pipeline = Pipeline([("preprocess", _build_preprocessor(x, scale_numeric=estimator_name == "Linear Regression")), ("model", estimator)])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
        pipeline.fit(x_train, y_train)
        pred = pipeline.predict(x_test)
        pred_df = pd.DataFrame({"actual": y_test, "predicted": pred, "residual": y_test - pred})
        rmse = mean_squared_error(y_test, pred) ** 0.5
        metrics = {"Model": estimator_name, "Target": target, "Train rows": len(x_train), "Test rows": len(x_test), "MAE": mean_absolute_error(y_test, pred), "RMSE": rmse, "R²": r2_score(y_test, pred)}
        figures = [px.scatter(pred_df, x="actual", y="predicted", trendline="ols", title="Predicted vs actual"), px.scatter(pred_df, x="predicted", y="residual", title="Residual plot")]
        figures[1].add_hline(y=0, line_dash="dash")
    else:
        y = y_raw.astype("object")
        if y.nunique(dropna=True) < 2:
            return _invalid(intent, "Classification requires a target with at least two classes.", df)
        estimator_name = "Random Forest Classifier" if intent in {"random_forest", "feature_importance"} else "Decision Tree Classifier" if intent == "decision_tree" else "Logistic Regression"
        estimator = RandomForestClassifier(n_estimators=250, random_state=42) if estimator_name == "Random Forest Classifier" else DecisionTreeClassifier(random_state=42) if estimator_name == "Decision Tree Classifier" else LogisticRegression(max_iter=2000)
        pipeline = Pipeline([("preprocess", _build_preprocessor(x, scale_numeric=estimator_name == "Logistic Regression")), ("model", estimator)])
        counts = y.value_counts()
        stratify = y if counts.min() >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=stratify)
        pipeline.fit(x_train, y_train)
        pred = pipeline.predict(x_test)
        labels = sorted(y.astype(str).unique())
        cm = pd.DataFrame(confusion_matrix(y_test.astype(str), pd.Series(pred).astype(str), labels=labels), index=[f"Actual {v}" for v in labels], columns=[f"Predicted {v}" for v in labels])
        report = pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).T
        metrics = {"Model": estimator_name, "Target": target, "Train rows": len(x_train), "Test rows": len(x_test), "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred, average="weighted", zero_division=0), "Recall": recall_score(y_test, pred, average="weighted", zero_division=0), "F1": f1_score(y_test, pred, average="weighted", zero_division=0)}
        figures = [px.imshow(cm, text_auto=True, title="Confusion matrix")]
        pred_df = pd.DataFrame()
        if len(labels) == 2 and hasattr(pipeline.named_steps["model"], "predict_proba"):
            proba = pipeline.predict_proba(x_test)[:, 1]
            y_bin = (y_test.astype(str) == labels[1]).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, proba)
            auc = roc_auc_score(y_bin, proba)
            metrics["ROC AUC"] = auc
            figures.append(px.area(pd.DataFrame({"fpr": fpr, "tpr": tpr}), x="fpr", y="tpr", title=f"ROC curve (AUC={auc:.3f})"))
    feature_names = [str(name) for name in pipeline.named_steps["preprocess"].get_feature_names_out()]
    fitted = pipeline.named_steps["model"]
    importance = pd.DataFrame()
    if hasattr(fitted, "feature_importances_"):
        importance = pd.DataFrame({"feature": feature_names, "importance": fitted.feature_importances_}).sort_values("importance", ascending=False)
    elif hasattr(fitted, "coef_"):
        coefs = np.ravel(fitted.coef_[0] if np.ndim(fitted.coef_) > 1 else fitted.coef_)
        importance = pd.DataFrame({"feature": feature_names[: len(coefs)], "importance": np.abs(coefs)}).sort_values("importance", ascending=False)
    top = ", ".join(importance.head(5)["feature"].astype(str)) if not importance.empty else "not available for this model"
    tables = [("Model metrics", pd.DataFrame([metrics]))]
    if not importance.empty:
        tables.append(("Feature importance / coefficient strength", importance.head(20)))
    if not is_regression:
        tables.extend([("Confusion matrix", cm), ("Classification report", report)])
    elif not pred_df.empty:
        tables.append(("Prediction sample", pred_df.head(25)))
    quality = f"R² is {metrics['R²']:.3f}; values closer to 1 indicate stronger predictive fit." if is_regression else f"Accuracy is {metrics['Accuracy']:.3f} and weighted F1 is {metrics['F1']:.3f}."
    return AnalysisResult(
        True,
        intent,
        f"{metrics['Model']} for {target}",
        f"I trained a local {metrics['Model']} using {len(predictors)} predictor(s) and a 75/25 train-test split. All calculations ran inside the app.",
        quality,
        f"The strongest model signals are {top}. Treat these as predictive associations, not proof of causation. Validate leakage, sample size, class balance, and residual/error patterns before acting.",
        ["Try a simpler model and compare metrics.", "Inspect top predictors for leakage or proxy variables.", "Use cross-validation before production decisions."],
        figures=figures,
        tables=tables,
        metrics=metrics,
        selected_columns={"target": target, "predictors": predictors},
    )


def run_requested_analysis(intent: AnalysisIntent | str, df: pd.DataFrame, selected_columns: dict[str, Any] | list[str] | None = None, user_question: str = "") -> AnalysisResult:
    """Execute one approved local analysis function and return display-ready outputs."""
    if isinstance(intent, str):
        analysis_intent = detect_analysis_intent(user_question, df)
        analysis_intent.intent = intent
    else:
        analysis_intent = intent
    selected_columns = selected_columns or {}
    columns = list(analysis_intent.columns)
    if isinstance(selected_columns, list):
        columns = selected_columns
    elif isinstance(selected_columns, dict):
        if selected_columns.get("columns"):
            columns = selected_columns["columns"]
        if selected_columns.get("target"):
            analysis_intent.target = selected_columns["target"]
        if selected_columns.get("predictors"):
            analysis_intent.predictors = selected_columns["predictors"]
    columns = [c for c in columns if c in df.columns]
    chart_df, sample_note = _sample_for_chart(df)
    intent_name = analysis_intent.intent

    if intent_name == "summary":
        missing = int(df.isna().sum().sum())
        table = pd.DataFrame({"metric": ["Rows", "Columns", "Missing cells", "Duplicate rows", "Numeric columns"], "value": [df.shape[0], df.shape[1], missing, int(df.duplicated().sum()), len(df.select_dtypes(include=np.number).columns)]})
        return AnalysisResult(True, intent_name, "Dataset summary", "I summarized the uploaded dataframe shape, data types, missingness, duplicates, and key fields.", f"The data has {df.shape[0]:,} rows, {df.shape[1]:,} columns, {missing:,} missing cells, and {int(df.duplicated().sum()):,} duplicate rows.", "Start with missing-value handling, strong relationships, and columns tied to your outcome of interest.", suggest_smart_actions(df)[:4], tables=[("Summary", table)], selected_columns={"columns": columns})

    if intent_name == "descriptive_statistics":
        num = df.select_dtypes(include=np.number).describe().T
        return AnalysisResult(True, intent_name, "Descriptive statistics", "I calculated count, mean, spread, quartiles, and range for numeric fields.", f"Generated descriptive statistics for {len(num)} numeric column(s).", "Look for large standard deviations, skewed quartiles, and min/max values that may indicate outliers or coding issues.", ["Plot distributions for skewed columns.", "Run outlier detection on high-variance fields."], tables=[("Numeric descriptive statistics", num)])

    if intent_name == "correlation_heatmap":
        numeric = df.select_dtypes(include=np.number)
        if numeric.shape[1] < 2:
            return _invalid(intent_name, "A correlation heatmap needs at least two numeric columns.", df)
        corr = numeric.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Numeric correlation heatmap")
        pair = _strongest_corr_pair(df, numeric.columns.tolist())
        summary = f"The strongest absolute correlation is {pair[0]} vs {pair[1]} (r={pair[2]:.3f})." if pair else "Correlation matrix created."
        return AnalysisResult(True, intent_name, "Correlation heatmap", "I computed Pearson correlations among all numeric columns and visualized them as a heatmap.", summary, "Darker red/blue cells indicate stronger positive/negative linear relationships; correlation does not prove causation.", ["Inspect the strongest pair with a scatter plot.", "Check whether outliers drive high correlations."], figures=[fig], tables=[("Correlation matrix", corr)], selected_columns={"columns": numeric.columns.tolist()})

    if intent_name == "correlation":
        nums = _numeric_cols(df, columns)
        method = analysis_intent.method or ("Spearman" if "spearman" in user_question.lower() else "Pearson")
        if len(nums) >= 2:
            pair = df[nums[:2]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 3:
                return _invalid(intent_name, "Correlation needs at least 3 paired non-missing observations.", df)
            stat, p = stats.spearmanr(pair[nums[0]], pair[nums[1]]) if method == "Spearman" else stats.pearsonr(pair[nums[0]], pair[nums[1]])
            table = pd.DataFrame([{"method": method, "x": nums[0], "y": nums[1], "statistic": stat, "p_value": p, "n": len(pair)}])
            fig = px.scatter(chart_df, x=nums[0], y=nums[1], trendline="ols", title=f"{nums[1]} vs {nums[0]}")
            return AnalysisResult(True, intent_name, f"{method} correlation", f"I ran a {method} correlation between {nums[0]} and {nums[1]}.", f"{method} statistic = {stat:.3f}, p = {_format_p(p)}, n = {len(pair):,}.", f"At α = 0.05, this relationship is {_sig(p)}. Positive values move together; negative values move in opposite directions.", ["Use a scatter plot to check shape and outliers.", "Use Spearman when the relationship is monotonic but not linear."], figures=[fig], tables=[("Correlation test", table)], selected_columns={"columns": nums[:2]}, warning=sample_note)
        table = _corr_table(df)
        if table.empty:
            return _invalid(intent_name, "No numeric column pairs have enough variation for correlation analysis.", df)
        return AnalysisResult(True, intent_name, "Strongest correlations", "I ranked all numeric column pairs by absolute Pearson correlation.", f"The strongest pair is {table.iloc[0]['left']} vs {table.iloc[0]['right']} (r={table.iloc[0]['pearson_r']:.3f}).", "High correlations highlight association and possible redundancy, but they do not establish causality.", ["Plot the strongest pair.", "Check multicollinearity before regression."], tables=[("Ranked correlations", table.head(20))])

    if intent_name in {"scatter_plot", "regression_plot"}:
        nums = _numeric_cols(df, columns)
        if len(nums) < 2:
            pair = _strongest_corr_pair(df)
            nums = [pair[0], pair[1]] if pair else nums
        if len(nums) < 2:
            return _invalid(intent_name, "Scatter plots need two numeric columns.", df)
        trend = "ols" if intent_name == "regression_plot" or "regression" in user_question.lower() else None
        fig = px.scatter(chart_df, x=nums[0], y=nums[1], trendline=trend, title=f"{nums[1]} vs {nums[0]}")
        return AnalysisResult(True, intent_name, "Scatter plot", f"I plotted {nums[1]} against {nums[0]} using the uploaded data.", f"The chart shows the point-level relationship between {nums[0]} and {nums[1]}.", "Look for direction, curvature, clusters, and influential outliers before relying on a correlation or regression line.", ["Run Pearson/Spearman correlation.", "Add a grouping color if a categorical field may explain clusters."], figures=[fig], selected_columns={"columns": nums[:2]}, warning=sample_note)

    if intent_name in {"histogram", "kde_plot"}:
        col = _first_numeric(df, columns) or _first_numeric(df)
        if not col:
            return _invalid(intent_name, "Distribution charts need a numeric column.", df)
        if intent_name == "kde_plot":
            values = chart_df[col].dropna()
            fig = ff.create_distplot([values], [col], show_hist=False, show_rug=False) if len(values) > 1 else px.histogram(chart_df, x=col)
        else:
            fig = px.histogram(chart_df, x=col, marginal="box", title=f"Distribution of {col}")
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        summary = f"{col}: mean={series.mean():.3g}, median={series.median():.3g}, std={series.std():.3g}, n={len(series):,}."
        return AnalysisResult(True, intent_name, f"Distribution of {col}", f"I visualized the distribution of {col}.", summary, "Compare the mean and median to judge skew, and inspect tails for outliers or coding limits.", ["Run outlier detection.", "Compare this distribution across a categorical group."], figures=[fig], tables=[("Distribution summary", series.describe().to_frame(col))], selected_columns={"columns": [col]}, warning=sample_note)

    if intent_name in {"boxplot", "violin_plot"}:
        y = _first_numeric(df, columns) or _first_numeric(df)
        x = _first_categorical(df, columns)
        if not y:
            return _invalid(intent_name, "Box/violin plots need a numeric measure.", df)
        fig = px.violin(chart_df, x=x, y=y, box=True, title=f"{y} by {x}" if x else f"Violin plot of {y}") if intent_name == "violin_plot" else px.box(chart_df, x=x, y=y, title=f"{y} by {x}" if x else f"Boxplot of {y}")
        table = df.groupby(x)[y].describe() if x else df[y].describe().to_frame(y)
        return AnalysisResult(True, intent_name, f"{y} by {x}" if x else f"{y} distribution", f"I compared {y}" + (f" across {x}." if x else "."), f"Displayed group medians, spread, and potential outliers for {y}.", "Groups with separated medians or very different spreads may warrant a formal t-test, ANOVA, or non-parametric test.", ["Run ANOVA/Kruskal-Wallis for group differences.", "Check group sample sizes."], figures=[fig], tables=[("Group summary", table)], selected_columns={"numeric": y, "group": x}, warning=sample_note)

    if intent_name in {"bar_chart", "count_plot", "grouped_bar_chart", "stacked_bar_chart", "pie_chart"}:
        cat_cols = _categorical_cols(df, columns)
        x = cat_cols[0] if cat_cols else _first_categorical(df) or (columns[0] if columns else df.columns[0])
        color = cat_cols[1] if len(cat_cols) > 1 else None
        y = _first_numeric(df, columns)
        if intent_name == "pie_chart":
            counts = df[x].astype("object").fillna("<Missing>").value_counts().head(MAX_CATEGORY_LEVELS).reset_index()
            counts.columns = [x, "count"]
            fig = px.pie(counts, names=x, values="count", title=f"Pie chart of {x}")
            table = counts
        elif intent_name in {"count_plot", "stacked_bar_chart"} or not y:
            table = df.groupby([x] + ([color] if color else []), dropna=False).size().reset_index(name="count").sort_values("count", ascending=False).head(MAX_CATEGORY_LEVELS * 2)
            fig = px.bar(table, x=x, y="count", color=color, barmode="stack" if intent_name == "stacked_bar_chart" else "group", title=f"Counts by {x}" + (f" and {color}" if color else ""))
        else:
            table = df.groupby([x] + ([color] if color else []), dropna=False)[y].mean().reset_index().sort_values(y, ascending=False).head(MAX_CATEGORY_LEVELS * 2)
            fig = px.bar(table, x=x, y=y, color=color, barmode="group", title=f"Mean {y} by {x}")
        return AnalysisResult(True, intent_name, "Categorical chart", f"I summarized records by {x}" + (f" and {color}." if color else "."), f"The table and chart show the largest categories for {x}.", "Dominant categories can shape model performance and statistical-test reliability.", ["Run a chi-square test for two categorical fields.", "Compare a numeric outcome by this category."], figures=[fig], tables=[("Chart data", table)], selected_columns={"x": x, "y": y, "color": color})

    if intent_name in {"line_plot", "area_chart"}:
        x = columns[0] if columns else df.columns[0]
        y = _first_numeric(df, columns[1:] if len(columns) > 1 else None) or _first_numeric(df)
        if not x or not y:
            return _invalid(intent_name, "Line/area charts need an x field and numeric y field.", df)
        plot_df = chart_df.sort_values(x)
        fig = px.area(plot_df, x=x, y=y, title=f"Area chart of {y} by {x}") if intent_name == "area_chart" else px.line(plot_df, x=x, y=y, title=f"Line plot of {y} by {x}")
        return AnalysisResult(True, intent_name, f"{y} over {x}", f"I plotted {y} across ordered values of {x}.", "The chart highlights trends, jumps, and possible seasonality/order effects.", "Use this mainly for time or ordered fields; unordered categories can create misleading lines.", ["Aggregate by time period before interpreting trends.", "Compare multiple groups with a color field."], figures=[fig], selected_columns={"x": x, "y": y}, warning=sample_note)

    if intent_name == "scatter_matrix":
        numeric = df.select_dtypes(include=np.number).columns.tolist()[:6]
        if len(numeric) < 2:
            return _invalid(intent_name, "Scatter matrices need at least two numeric columns.", df)
        fig = px.scatter_matrix(chart_df, dimensions=numeric, title="Scatter matrix")
        return AnalysisResult(True, intent_name, "Scatter matrix", "I created a pairwise scatter matrix for up to six numeric columns.", f"Displayed pairwise relationships for {len(numeric)} numeric fields.", "Use this to spot nonlinear patterns, clusters, and redundant variables before modeling.", ["Zoom into the strongest pair.", "Run correlation ranking."], figures=[fig], selected_columns={"columns": numeric}, warning=sample_note)

    if intent_name == "scatter_3d":
        numeric = _numeric_cols(df, columns) or df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric) < 3:
            return _invalid(intent_name, "3D scatter plots need at least three numeric columns.", df)
        fig = px.scatter_3d(chart_df, x=numeric[0], y=numeric[1], z=numeric[2], title=f"3D scatter: {numeric[0]}, {numeric[1]}, {numeric[2]}")
        return AnalysisResult(True, intent_name, "3D scatter plot", "I plotted three numeric dimensions in an interactive 3D scatter plot.", "The chart can reveal clusters or multivariate structure.", "3D charts are exploratory; validate patterns with simpler 2D plots or models.", ["Try coloring by a categorical group.", "Run feature importance for a target."], figures=[fig], selected_columns={"columns": numeric[:3]}, warning=sample_note)

    if intent_name in {"cross_tabulation", "chi_square_test"}:
        cats = _categorical_cols(df, columns)
        if len(cats) < 2:
            cats = [c for c in columns if c in df.columns][:2]
        if len(cats) < 2:
            return _invalid(intent_name, "This analysis needs two categorical columns.", df)
        observed = pd.crosstab(df[cats[0]], df[cats[1]])
        if intent_name == "cross_tabulation":
            return AnalysisResult(True, intent_name, "Cross-tabulation", f"I cross-tabulated {cats[0]} by {cats[1]}.", f"The table has {observed.shape[0]} row categories and {observed.shape[1]} column categories.", "Use row percentages to compare category mixes fairly when group sizes differ.", ["Run a chi-square test on this table.", "Visualize it as a stacked bar chart."], tables=[("Observed counts", observed), ("Row percentages", pd.crosstab(df[cats[0]], df[cats[1]], normalize="index") * 100)], selected_columns={"columns": cats[:2]})
        if observed.shape[0] < 2 or observed.shape[1] < 2:
            return _invalid(intent_name, "Chi-square requires at least two categories in each selected column.", df)
        chi2, p, dof, expected = stats.chi2_contingency(observed)
        result_table = pd.DataFrame([{"test": "Chi-square test of independence", "chi2": chi2, "p_value": p, "degrees_of_freedom": dof, "n": int(observed.values.sum())}])
        return AnalysisResult(True, intent_name, "Chi-square test", f"I tested whether {cats[0]} and {cats[1]} are independent using a chi-square test.", f"χ²={chi2:.3f}, dof={dof}, p={_format_p(p)}, n={int(observed.values.sum()):,}.", f"At α = 0.05, the association is {_sig(p)}. Assumptions: independent observations and sufficiently large expected cell counts; sparse tables should be interpreted carefully.", ["Inspect standardized residuals or row percentages.", "Collapse rare categories if expected counts are small."], tables=[("Test result", result_table), ("Observed counts", observed), ("Expected counts", pd.DataFrame(expected, index=observed.index, columns=observed.columns))], selected_columns={"columns": cats[:2]})

    if intent_name in {"anova", "t_test", "mann_whitney", "kruskal_wallis"}:
        y = _first_numeric(df, columns)
        group = _first_categorical(df, columns)
        if not y or not group:
            return _invalid(intent_name, "This test needs a numeric outcome and a grouping column.", df)
        subset = df[[y, group]].dropna()
        groups = [pd.to_numeric(g[y], errors="coerce").dropna() for _, g in subset.groupby(group)]
        names = [str(name) for name, _ in subset.groupby(group)]
        groups = [g for g in groups if len(g) > 0]
        if intent_name in {"t_test", "mann_whitney"} and len(groups) != 2:
            return _invalid(intent_name, "This two-group test requires exactly two non-empty groups.", df)
        if len(groups) < 2:
            return _invalid(intent_name, "The grouping column needs at least two non-empty groups.", df)
        if intent_name == "anova":
            stat, p, label = *stats.f_oneway(*groups), "One-way ANOVA"
        elif intent_name == "t_test":
            stat, p, label = *stats.ttest_ind(groups[0], groups[1], equal_var=False, nan_policy="omit"), "Welch independent t-test"
        elif intent_name == "mann_whitney":
            stat, p, label = *stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided"), "Mann-Whitney U test"
        else:
            stat, p, label = *stats.kruskal(*groups), "Kruskal-Wallis test"
        result_table = pd.DataFrame([{"test": label, "numeric_outcome": y, "group": group, "statistic": stat, "p_value": p, "n": int(sum(len(g) for g in groups)), "groups": len(groups)}])
        group_summary = subset.groupby(group)[y].describe()
        fig = px.box(chart_df, x=group, y=y, title=f"{y} by {group}")
        return AnalysisResult(True, intent_name, label, f"I compared {y} across groups in {group} using {label}.", f"Statistic={stat:.3f}, p={_format_p(p)}, n={int(sum(len(g) for g in groups)):,}, groups={len(groups)}.", f"At α = 0.05, the group difference is {_sig(p)}. Assumptions vary by test: ANOVA/t-test assume independent observations and approximately normal residuals; Welch t-test relaxes equal variances; rank tests compare distributions more robustly.", ["Review the boxplot and group sample sizes.", "If significant with more than two groups, run post-hoc pairwise comparisons."], figures=[fig], tables=[("Test result", result_table), ("Group summary", group_summary)], selected_columns={"numeric": y, "group": group}, warning=sample_note)

    if intent_name in {"linear_regression", "logistic_regression", "random_forest", "decision_tree", "feature_importance"}:
        return _train_model(df, analysis_intent.target or (_first_numeric(df, columns) or (columns[0] if columns else df.columns[-1])), analysis_intent.predictors, intent_name)

    if intent_name == "missing_values":
        report = _missing_report(df)
        missing_cols = report[report["missing_count"] > 0]
        summary = f"Found {int(df.isna().sum().sum()):,} missing cells across {len(missing_cols)} column(s)."
        fig = px.bar(missing_cols.head(30).reset_index().rename(columns={"index": "column"}), x="column", y="missing_percent", title="Missing values by column") if not missing_cols.empty else go.Figure()
        return AnalysisResult(True, intent_name, "Missing value report", "I counted missing values and percentages for every column.", summary, "Columns with high missingness can bias summaries and models; decide whether to impute, flag, or exclude based on domain meaning.", ["Inspect rows with missing target values.", "Compare results before and after imputation."], figures=[fig] if not missing_cols.empty else [], tables=[("Missing value report", report)], selected_columns={"columns": report.index.tolist()})

    if intent_name == "outlier_detection":
        col = _first_numeric(df, columns)
        report = _outlier_report(df, col)
        if report.empty:
            return _invalid(intent_name, "No IQR outliers were found or no suitable numeric column exists.", df)
        focus = report.iloc[0]["column"]
        fig = px.box(chart_df, y=focus, title=f"IQR outliers in {focus}")
        return AnalysisResult(True, intent_name, "Outlier detection", "I used the 1.5×IQR rule to flag unusually low or high numeric values.", f"The largest outlier count is in {focus}: {int(report.iloc[0]['outlier_count']):,} rows ({report.iloc[0]['outlier_percent']:.1f}%).", "Outliers may be valid extremes, entry errors, or influential cases; inspect context before removing them.", ["Compare model results with and without extreme values.", "Use robust statistics if outliers are valid."], figures=[fig], tables=[("IQR outlier report", report.head(30))], selected_columns={"columns": [col] if col else report["column"].tolist()}, warning=sample_note)

    if intent_name == "data_quality":
        missing = _missing_report(df)
        outliers = _outlier_report(df)
        dupes = int(df.duplicated().sum())
        summary_table = pd.DataFrame([{"rows": df.shape[0], "columns": df.shape[1], "missing_cells": int(df.isna().sum().sum()), "duplicate_rows": dupes, "columns_with_missing": int((df.isna().sum() > 0).sum())}])
        return AnalysisResult(True, intent_name, "Data quality review", "I reviewed missingness, duplicate rows, data types, and numeric outlier signals.", f"There are {dupes:,} duplicate rows and {int(df.isna().sum().sum()):,} missing cells.", "Resolve quality issues before final tests or models, especially missing target values, duplicated records, and impossible numeric ranges.", ["Open the missing value report.", "Run outlier detection on key numeric columns."], tables=[("Quality summary", summary_table), ("Missing values", missing), ("Outliers", outliers.head(20))])

    if intent_name == "recommended_analysis":
        actions = suggest_smart_actions(df)
        table = pd.DataFrame({"recommended_question": actions})
        return AnalysisResult(True, intent_name, "Recommended next analyses", "I inspected available column types and generated safe analyses that fit this dataset.", f"I found {len(actions)} strong next-step prompt(s) tailored to the uploaded columns.", "Start with broad structure (missingness/correlations), then move into hypothesis tests and prediction for important outcomes.", actions[:6], tables=[("Suggested prompts", table)])

    return _invalid(intent_name, "I could not confidently identify a supported analysis request from that wording.", df)
