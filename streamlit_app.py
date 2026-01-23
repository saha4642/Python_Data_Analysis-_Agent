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
- DO NOT import anything.
- DO NOT read/write files.
- Use matplotlib only.
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
    code = re.sub(IMPORT_LINE_RE, "", code)
    code = re.sub(FILE_IO_RE, "", code)
    code = re.sub(r"^\s*plt\.show\(\)\s*$", "", code, flags=re.MULTILINE)
    if "plt.tight_layout" not in code:
        code += "\n\nplt.tight_layout()"
    return code.strip()


def safe_exec_plot(code: str, df: pd.DataFrame) -> Tuple[Optional[plt.Figure], Optional[str]]:
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
        return None, f"{type(e).__name__}: {e}"


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


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Chat with CSV (OpenAI)", layout="wide")
st.title("📊 Chat with Your Data (OpenAI)")
st.caption("Upload a CSV → ask questions → see answers and plots")

if "df" not in st.session_state:
    st.session_state.df = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Upload CSV")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    show_df = st.checkbox("Show dataframe preview", True)
    show_stats = st.checkbox("Show numeric stats", True)
    show_code = st.checkbox("Show generated code (debug)", False)

# Load CSV
if uploaded:
    try:
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        df.columns = [c.strip() for c in df.columns]
        st.session_state.df = df
        st.sidebar.success(f"Loaded {uploaded.name} ({df.shape[0]}×{df.shape[1]})")
    except Exception as e:
        st.sidebar.error(str(e))

df = st.session_state.df

left, right = st.columns([1.1, 1.2], gap="large")

# ---------------- Data ----------------
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

# ---------------- Chat ----------------
with right:
    st.subheader("Chat")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if df is not None:
        q = st.chat_input("Ask a question about your data…")
        if q:
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

            st.session_state.messages.append({"role": "assistant", "content": re.sub(CODE_BLOCK_RE, "", answer)})

            with st.chat_message("assistant"):
                st.markdown(re.sub(CODE_BLOCK_RE, "", answer))

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
