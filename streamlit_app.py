from __future__ import annotations

import io
import os
import re
import textwrap
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

# SciPy for Q-Q plots
from scipy import stats

# sklearn tools for Random Forest / ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# capture print() output from executed code
from contextlib import redirect_stdout
import traceback

# ============================================================
# Load environment variables (OPENAI_API_KEY REQUIRED)
# ============================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY is not set.\n\n"
        "Add it to a `.env` file or your environment before running Streamlit."
    )
    st.stop()

# ============================================================
# Regex + Prompt
# ============================================================
CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
IMPORT_LINE_RE = re.compile(r"^\s*(import\s+.+|from\s+.+\s+import\s+.+)\s*$", re.MULTILINE)
FILE_IO_RE = re.compile(r"^\s*(open\(|to_csv\(|to_excel\(|savefig\(|plt\.savefig\().*$", re.MULTILINE)

SYSTEM_PROMPT = """You are a data analyst assistant.
You answer questions about a pandas DataFrame called df.

Rules:
- Be concise and correct.
- If a visualization helps, include matplotlib code.
- Choose the chart type that best fits the uploaded data and the user's question: time fields → line/trend, categories → sorted bar, numeric distributions → hist/box/Q-Q, numeric relationships → scatter/correlation heatmap.
- Optimize every chart for storytelling: clear title, readable labels, sorted/ranked categories where useful, and brief text explaining the key takeaway.
- DO NOT import anything.
- DO NOT read/write files.
- Use matplotlib ONLY. (NO seaborn / NO sns)
- Prefer bar/line/hist/box/scatter/Q-Q plots.
- For Q-Q plots, you may use stats.probplot(...) (stats is available).
- End plot code with plt.tight_layout().
- Put code in ONE ```python``` block only.
- If no plot is needed, do not include code.

Modeling rules (for ML like Random Forest):
- You may use these tools which are already available (do not import):
  train_test_split, RandomForestRegressor, RandomForestClassifier,
  Pipeline, ColumnTransformer, OneHotEncoder,
  r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix.
- Handle categorical columns with OneHotEncoder inside a Pipeline.
- Always print key metrics. If helpful, also plot feature importances and/or confusion matrix.
"""

# ============================================================
# Helpers
# ============================================================
def extract_python_code(text: str) -> Optional[str]:
    m = CODE_BLOCK_RE.search(text or "")
    return m.group(1).strip() if m else None


def sanitize_code(code: str) -> str:
    code = re.sub(IMPORT_LINE_RE, "", code)
    code = re.sub(FILE_IO_RE, "", code)
    code = re.sub(r"^\s*plt\.show\(\)\s*$", "", code, flags=re.MULTILINE)
    if "plt.tight_layout" not in code:
        code += "\n\nplt.tight_layout()"
    return code.strip()


def _fallback_corr_heatmap(df: pd.DataFrame) -> Tuple[Optional[plt.Figure], str, Optional[str]]:
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return None, "", "Need at least 2 numeric columns to plot a correlation heatmap."

    corr = num.corr(numeric_only=True)

    plt.close("all")
    fig, ax = plt.subplots()

    im = ax.imshow(corr.values, aspect="auto")
    ax.set_title("Correlation heatmap (numeric columns)")

    ax.set_xticks(range(corr.shape[1]))
    ax.set_yticks(range(corr.shape[0]))
    ax.set_xticklabels(corr.columns.tolist(), rotation=45, ha="right")
    ax.set_yticklabels(corr.index.tolist())

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig, "", None


def safe_exec_plot(code: str, df: pd.DataFrame) -> Tuple[Optional[plt.Figure], str, Optional[str]]:
    """
    Execute LLM-generated code safely.
    - Blocks seaborn usage (fallback heatmap).
    - Exposes scipy.stats + sklearn tools without imports.
    - Captures print() output so ML metrics show in the chat.
    """
    lowered = (code or "").lower()

    if "sns." in lowered or "seaborn" in lowered:
        return _fallback_corr_heatmap(df)

    plt.close("all")

    # Create a default figure/axes; the code may use plt directly.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    safe_globals: Dict[str, Any] = {
        "__builtins__": {
            # core python builtins needed by numpy/sklearn
            "abs": abs, "min": min, "max": max, "sum": sum,
            "len": len, "range": range, "sorted": sorted, "enumerate": enumerate,
            "print": print,

            "str": str, "int": int, "float": float, "bool": bool,
            "list": list, "dict": dict, "tuple": tuple, "set": set,

            "isinstance": isinstance, "type": type,
            "zip": zip, "map": map, "any": any, "all": all,
            "round": round,

            # ✅ needed by numpy/sklearn printing/formatting
            "repr": repr,
            "format": format,
            "getattr": getattr,
            "setattr": setattr,
            "hasattr": hasattr,
            "iter": iter,
            "next": next,
            "slice": slice,
        },

        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
        "fig": fig,
        "ax": ax,
        "textwrap": textwrap,

        # SciPy stats for Q-Q plots
        "stats": stats,

        # sklearn tools for ML
        "train_test_split": train_test_split,
        "r2_score": r2_score,
        "mean_absolute_error": mean_absolute_error,
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "RandomForestRegressor": RandomForestRegressor,
        "RandomForestClassifier": RandomForestClassifier,
        "OneHotEncoder": OneHotEncoder,
        "ColumnTransformer": ColumnTransformer,
        "Pipeline": Pipeline,
    }

    out_buf = io.StringIO()
    try:
        with redirect_stdout(out_buf):
            exec(code, safe_globals, {})
        text_out = out_buf.getvalue().strip()

        # If the executed code created no usable plot, avoid returning an empty figure
        if len(plt.get_fignums()) == 0:
            return None, text_out, None

        return plt.gcf(), text_out, None

    except Exception as e:
        text_out = out_buf.getvalue().strip()
        msg = f"{type(e).__name__}: {e}"
        if text_out:
            msg = msg + "\n\n(Partial output captured)\n" + text_out
        return None, "", msg


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )


def dataframe_context(df: pd.DataFrame, rows: int = 8) -> str:
    return f"""
Shape: {df.shape}

Columns:
{', '.join(df.columns[:60])}

Preview:
{df.head(rows).to_markdown(index=False)}
""".strip()


# -----------------------------
# Smart idea generator
# -----------------------------
def _pick_columns(df: pd.DataFrame):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

    if not dt_cols:
        for c in df.columns[:80]:
            if df[c].dtype == object:
                s = df[c].dropna()
                if len(s) >= 10:
                    parsed = pd.to_datetime(s.head(50), errors="coerce")
                    if parsed.notna().mean() >= 0.7:
                        dt_cols.append(c)

    cat_cols = []
    for c in df.columns:
        if c in dt_cols:
            continue
        if str(df[c].dtype) in {"object", "category", "bool"}:
            cat_cols.append(c)
        else:
            if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique(dropna=True) <= 20:
                cat_cols.append(c)

    def uniq(xs):
        out, seen = [], set()
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return uniq(num_cols), uniq(cat_cols), uniq(dt_cols)


def _find_classification_target(df: pd.DataFrame) -> Optional[str]:
    candidates = []
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            continue
        nun = s.nunique()
        if 2 <= nun <= 20:
            if str(df[c].dtype) in {"object", "category", "bool"}:
                candidates.append((0, nun, c))
            else:
                candidates.append((1, nun, c))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _find_regression_target(df: pd.DataFrame, num_cols: List[str]) -> Optional[str]:
    if not num_cols:
        return None
    scored = []
    for c in num_cols:
        s = df[c]
        miss = float(s.isna().mean())
        if miss > 0.5:
            continue
        var = float(np.nanvar(s.values))
        scored.append((miss, -var, c))
    if not scored:
        return num_cols[0]
    scored.sort()
    return scored[0][2]


def generate_analysis_ideas(df: pd.DataFrame, max_ideas: int = 12) -> List[str]:
    num_cols, cat_cols, dt_cols = _pick_columns(df)

    ideas: List[str] = []
    ideas.append("Give me a quick summary of the dataset: missing values, duplicates, and basic stats.")
    ideas.append("Which columns have the most missing values? Show a table of missing counts and percentages.")

    if len(num_cols) >= 1:
        c1 = num_cols[0]
        ideas.append(f"Plot the distribution (histogram) of '{c1}' and summarize outliers.")
        ideas.append(f"Show the top 10 highest and lowest rows by '{c1}'.")
        ideas.append(f"Make a Q-Q plot for '{c1}' to check if it is normally distributed.")

    if len(num_cols) >= 2:
        x, y = num_cols[0], num_cols[1]
        ideas.append(f"Make a scatter plot of '{x}' vs '{y}' and describe the relationship/correlation.")
        ideas.append("Show a correlation table/heatmap for numeric columns and highlight the strongest pairs.")

        target_reg = _find_regression_target(df, num_cols)
        if target_reg:
            feature_hint = [c for c in num_cols if c != target_reg][:6]
            ideas.append(
                f"Train a Random Forest regressor to predict '{target_reg}' using features {feature_hint}. "
                f"Report R2/MAE and plot feature importances."
            )

    if len(cat_cols) >= 1:
        ccat = cat_cols[0]
        ideas.append(f"Show the value counts for '{ccat}' (top categories) and plot a bar chart.")

    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        ccat, cnum = cat_cols[0], num_cols[0]
        ideas.append(f"Compare average '{cnum}' by '{ccat}' (bar chart) and identify the top group.")
        ideas.append(f"Compare '{cnum}' across '{ccat}' and make a Q-Q plot for '{cnum}' to assess normality.")

    cls_target = _find_classification_target(df)
    if cls_target:
        feats = [c for c in df.columns if c != cls_target][:6]
        ideas.append(
            f"Train a Random Forest classifier to predict '{cls_target}' using features {feats}. "
            f"Show accuracy, classification report, and confusion matrix."
        )

    if len(dt_cols) >= 1:
        t = dt_cols[0]
        ideas.append(f"Convert '{t}' to datetime if needed, then show record counts over time (line chart).")

    if len(dt_cols) >= 1 and len(num_cols) >= 1:
        t, cnum = dt_cols[0], num_cols[0]
        ideas.append(f"Over time using '{t}', plot '{cnum}' (or its average) as a line chart and describe trends.")

    if not num_cols and not cat_cols and not dt_cols:
        ideas.append("Show the dataset schema (dtypes) and recommend 3 useful analyses based on the columns.")

    seen = set()
    uniq_ideas = []
    for it in ideas:
        if it not in seen:
            uniq_ideas.append(it)
            seen.add(it)

    return uniq_ideas[:max_ideas]


# ============================================================
# Streamlit UI (UNCHANGED)
# ============================================================
st.set_page_config(page_title="Chat with CSV", layout="wide")
st.title("📊 Chat with Your Data")
st.caption("Upload a CSV → ask questions → see answers and plots")

if "df" not in st.session_state:
    st.session_state.df = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# track uploaded file signature so we only reload/reset on NEW upload
if "_upload_sig" not in st.session_state:
    st.session_state._upload_sig = None

with st.sidebar:
    st.header("Upload CSV")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    show_df = st.checkbox("Show dataframe preview", True)
    show_stats = st.checkbox("Show numeric stats", True)
    show_code = st.checkbox("Show generated code (debug)", False)

# only reload/reset when the file changes (prevents wiping user input every rerun)
if uploaded is not None:
    raw = uploaded.getvalue()
    sig = (uploaded.name, len(raw))
    if st.session_state._upload_sig != sig:
        try:
            df = pd.read_csv(io.BytesIO(raw))
            df.columns = [c.strip() for c in df.columns]
            st.session_state.df = df
            st.sidebar.success(f"Loaded {uploaded.name} ({df.shape[0]}×{df.shape[1]})")

            st.session_state._upload_sig = sig

            # reset chat + inputs on NEW upload only
            st.session_state.messages = []
            if "idea_selectbox" in st.session_state:
                del st.session_state["idea_selectbox"]
            if "question_text_input" in st.session_state:
                del st.session_state["question_text_input"]

        except Exception as e:
            st.sidebar.error(str(e))

df = st.session_state.df
left, right = st.columns([1.1, 1.2], gap="large")

with left:
    st.subheader("Data")
    if df is None:
        st.info("Upload a CSV to begin.")
    else:
        if show_df:
            st.dataframe(df.head(50), width="stretch")
        if show_stats:
            num = df.select_dtypes(include="number")
            if not num.empty:
                st.dataframe(num.describe().T, width="stretch")

with right:
    st.subheader("Chat")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if df is not None:
        st.markdown("### 💡 Quick ideas (based on your data)")
        ideas = generate_analysis_ideas(df)
        idea_options = ["— Select an idea —"] + ideas

        if "question_text_input" not in st.session_state:
            st.session_state.question_text_input = ""

        def on_idea_change():
            selected = st.session_state.get("idea_selectbox", "— Select an idea —")
            if selected != "— Select an idea —":
                st.session_state.question_text_input = selected

        st.selectbox(
            "Pick an idea to auto-fill your question",
            idea_options,
            key="idea_selectbox",
            on_change=on_idea_change,
        )

        q_text = st.text_input(
            "Your question (edit or type your own)",
            key="question_text_input",
            placeholder="e.g., Plot sales over time and highlight the top 5 spikes",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            run_btn = st.button("Run analysis", type="primary", use_container_width=True)
        with c2:
            clear_btn = st.button("Clear chat", use_container_width=True)

        if clear_btn:
            st.session_state.messages = []
            st.session_state.question_text_input = ""
            if "idea_selectbox" in st.session_state:
                del st.session_state["idea_selectbox"]
            st.rerun()

        if run_btn:
            q = (st.session_state.question_text_input or "").strip()
            if not q:
                st.warning("Please type a question or select an idea.")
            else:
                st.session_state.messages.append({"role": "user", "content": q})
                with st.chat_message("user"):
                    st.markdown(q)

                llm = get_llm()
                prompt = SYSTEM_PROMPT + "\n\n" + dataframe_context(df)

                resp = llm.invoke([
                    {"role": "system", "content": prompt},
                    *st.session_state.messages[-10:],
                    {"role": "user", "content": q},
                ])

                answer = getattr(resp, "content", str(resp))
                code = extract_python_code(answer)

                clean_answer = re.sub(CODE_BLOCK_RE, "", answer).strip()
                st.session_state.messages.append({"role": "assistant", "content": clean_answer})

                with st.chat_message("assistant"):
                    st.markdown(clean_answer if clean_answer else "_(No text output)_")

                    if code:
                        code = sanitize_code(code)
                        fig, stdout_text, err = safe_exec_plot(code, df)

                        # show printed output (metrics/results) even if no plot
                        if stdout_text:
                            st.text(stdout_text)

                        if err:
                            st.error(err)
                            if show_code:
                                st.code(code, "python")
                        else:
                            if fig is not None:
                                st.pyplot(fig, clear_figure=True)
                            if show_code:
                                st.code(code, "python")
