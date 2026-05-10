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
- Choose the chart type that best fits the uploaded data and the user's question: time fields → line/trend, categories → sorted bar, numeric distributions → hist/box, numeric relationships → scatter/correlation heatmap.
- Optimize every chart for storytelling: clear title, readable labels, sorted/ranked categories where useful, and brief text explaining the key takeaway.
- DO NOT import anything.
- DO NOT read/write files.
- Use matplotlib ONLY. (NO seaborn / NO sns)
- Prefer bar/line/hist/box/scatter plots.
- End plot code with plt.tight_layout().
- Put code in ONE ```python``` block only.
- If no plot is needed, do not include code.
"""

# ============================================================
# Helpers
# ============================================================
def extract_python_code(text: str) -> Optional[str]:
    m = CODE_BLOCK_RE.search(text or "")
    return m.group(1).strip() if m else None


def sanitize_code(code: str) -> str:
    # remove imports
    code = re.sub(IMPORT_LINE_RE, "", code)
    # remove file I/O
    code = re.sub(FILE_IO_RE, "", code)
    # remove plt.show
    code = re.sub(r"^\s*plt\.show\(\)\s*$", "", code, flags=re.MULTILINE)

    if "plt.tight_layout" not in code:
        code += "\n\nplt.tight_layout()"
    return code.strip()


# --------- NEW: fallback heatmap (matplotlib only) ----------
def _fallback_corr_heatmap(df: pd.DataFrame) -> Tuple[Optional[plt.Figure], Optional[str]]:
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return None, "Need at least 2 numeric columns to plot a correlation heatmap."

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
    return fig, None


def safe_exec_plot(code: str, df: pd.DataFrame) -> Tuple[Optional[plt.Figure], Optional[str]]:
    """
    Execute LLM-generated code safely.
    If the model outputs seaborn code (sns/seaborn), fall back to a matplotlib plot
    instead of crashing.
    """
    lowered = (code or "").lower()

    # If seaborn appears, avoid executing and use fallback
    if "sns." in lowered or "seaborn" in lowered:
        return _fallback_corr_heatmap(df)

    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    safe_globals: Dict[str, Any] = {
        "__builtins__": {
            "abs": abs, "min": min, "max": max, "sum": sum,
            "len": len, "range": range, "sorted": sorted,
            "enumerate": enumerate, "print": print
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
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"

        # common recovery if the model used sns anyway
        if "nameerror" in msg.lower() and "sns" in msg.lower():
            fig2, err2 = _fallback_corr_heatmap(df)
            if err2 is None:
                return fig2, None

        return None, msg


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
# Smart idea generator (unchanged)
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


def generate_analysis_ideas(df: pd.DataFrame, max_ideas: int = 12) -> List[str]:
    num_cols, cat_cols, dt_cols = _pick_columns(df)

    ideas: List[str] = []
    ideas.append("Give me a quick summary of the dataset: missing values, duplicates, and basic stats.")
    ideas.append("Which columns have the most missing values? Show a table of missing counts and percentages.")

    if len(num_cols) >= 1:
        c1 = num_cols[0]
        ideas.append(f"Plot the distribution (histogram) of '{c1}' and summarize outliers.")
        ideas.append(f"Show the top 10 highest and lowest rows by '{c1}'.")

    if len(num_cols) >= 2:
        x, y = num_cols[0], num_cols[1]
        ideas.append(f"Make a scatter plot of '{x}' vs '{y}' and describe the relationship/correlation.")
        ideas.append(f"Show a correlation table/heatmap for numeric columns and highlight the strongest pairs.")

    if len(cat_cols) >= 1:
        ccat = cat_cols[0]
        ideas.append(f"Show the value counts for '{ccat}' (top categories) and plot a bar chart.")

    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        ccat, cnum = cat_cols[0], num_cols[0]
        ideas.append(f"Compare average '{cnum}' by '{ccat}' (bar chart) and identify the top group.")

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
# Streamlit UI (unchanged)
# ============================================================
st.set_page_config(page_title="Chat with CSV (OpenAI)", layout="wide")
st.title("📊 Chat with Your Data (OpenAI)")
st.caption("Upload a CSV → ask questions → see answers and plots")

if "df" not in st.session_state:
    st.session_state.df = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "idea_selected" not in st.session_state:
    st.session_state.idea_selected = "— Select an idea —"
if "draft_question" not in st.session_state:
    st.session_state.draft_question = ""

with st.sidebar:
    st.header("Upload CSV")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    show_df = st.checkbox("Show dataframe preview", True)
    show_stats = st.checkbox("Show numeric stats", True)
    show_code = st.checkbox("Show generated code (debug)", False)

if uploaded:
    try:
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        df.columns = [c.strip() for c in df.columns]
        st.session_state.df = df
        st.sidebar.success(f"Loaded {uploaded.name} ({df.shape[0]}×{df.shape[1]})")
        st.session_state.idea_selected = "— Select an idea —"
        st.session_state.draft_question = ""
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
            selected = st.session_state.idea_selectbox
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
            st.session_state.idea_selectbox = "— Select an idea —"
            st.session_state.question_text_input = ""
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
                        fig, err = safe_exec_plot(code, df)
                        if err:
                            st.error(err)
                            if show_code:
                                st.code(code, "python")
                        else:
                            st.pyplot(fig, clear_figure=True)
                            if show_code:
                                st.code(code, "python")
