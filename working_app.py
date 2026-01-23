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

# -----------------------------
# Load .env early (IMPORTANT)
# -----------------------------
load_dotenv()

# -----------------------------
# Regex / Prompts
# -----------------------------
CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
IMPORT_LINE_RE = re.compile(r"^\s*(import\s+.+|from\s+.+\s+import\s+.+)\s*$", re.MULTILINE)
FILE_IO_RE = re.compile(r"^\s*(open\(|to_csv\(|to_excel\(|savefig\(|plt\.savefig\().*$", re.MULTILINE)
STYLE_RE = re.compile(r"^\s*plt\.style\.use\(.+\).*$", re.MULTILINE)

SYSTEM_PROMPT = """You are a data analyst assistant.
You will answer questions about a pandas DataFrame called df.

You must follow these rules exactly:
- Always provide a clear, concise answer.
- If a visualization is helpful or explicitly requested, include Python code for the plot.
- The plot code MUST NOT import anything. (df, pd, np, plt are already available.)
- The plot code MUST NOT read/write files.
- Use matplotlib only. Do not use seaborn/plotly.
- Prefer simple matplotlib charts: bar/line/hist/box/scatter.
- Avoid specifying colors unless user asks.
- If you need multiple plots, you MAY use plt.subplot(...) or plt.subplots(...).
- Always end plot code with plt.tight_layout().
- Put plot code in ONE python fenced block: ```python ... ```
- If no plot is needed, do not output any code block.
"""

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# -----------------------------
# Code extraction / cleanup
# -----------------------------
def extract_python_code(text: str) -> Optional[str]:
    m = CODE_BLOCK_RE.search(text or "")
    if not m:
        return None
    return m.group(1).strip() or None


def strip_unsupported_lines(code: str) -> str:
    code = re.sub(IMPORT_LINE_RE, "", code)
    code = re.sub(FILE_IO_RE, "", code)
    code = re.sub(STYLE_RE, "", code)
    return code.strip()


def ensure_tight_layout(code: str) -> str:
    if "plt.tight_layout" not in code:
        code = code.rstrip() + "\n\nplt.tight_layout()\n"
    return code


def normalize_plot_code(code: str) -> str:
    code = strip_unsupported_lines(code)
    code = re.sub(r"^\s*plt\.show\(\)\s*$", "", code, flags=re.MULTILINE).strip()
    code = ensure_tight_layout(code)
    return code


def remove_code_block_from_markdown(text: str) -> str:
    """Hide the code block from the chat display (graph only)."""
    return CODE_BLOCK_RE.sub("", text or "").strip()


# -----------------------------
# Safe plotting sandbox
# -----------------------------
def safe_exec_plot(code: str, df: pd.DataFrame) -> Tuple[Optional[plt.Figure], Optional[str]]:
    """
    Execute plotting code in a restricted namespace.
    Returns (figure, error_message).
    """
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    safe_globals: Dict[str, Any] = {
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
            "zip": zip,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
        },
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
        "fig": fig,
        "ax": ax,
        "textwrap": textwrap,
    }
    safe_locals: Dict[str, Any] = {}

    try:
        exec(code, safe_globals, safe_locals)
        return plt.gcf(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# -----------------------------
# LLM + context
# -----------------------------
def get_llm(model_name: str, api_key: str) -> ChatOpenAI:
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in .env, as an environment variable, "
            "or paste it in the sidebar."
        )
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
    )


def dataframe_context_snippet(df: pd.DataFrame, max_rows: int = 8) -> str:
    cols = ", ".join([str(c) for c in df.columns[:60]])
    head = df.head(max_rows).to_markdown(index=False)
    dtypes = df.dtypes.astype(str).to_dict()
    dtype_lines = "\n".join([f"- {k}: {v}" for k, v in list(dtypes.items())[:60]])

    return f"""
DataFrame shape: {df.shape}
Columns (first up to 60): {cols}

Dtypes (first up to 60):
{dtype_lines}

Preview (first {max_rows} rows):
{head}
""".strip()


def llm_answer_and_optional_plot(
    llm: ChatOpenAI,
    df: pd.DataFrame,
    chat_history: List[dict],
    user_question: str,
) -> str:
    df_ctx = dataframe_context_snippet(df)

    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + df_ctx}]
    for m in chat_history[-10:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_question})

    resp = llm.invoke(messages)
    return getattr(resp, "content", str(resp))


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Chat with your DataFrame (OpenAI)", layout="wide")
st.title("📊 Chat with your DataFrame (OpenAI)")
st.caption("Upload a CSV → ask questions → get answers + visualizations (Matplotlib).")

# Session state
if "df" not in st.session_state:
    st.session_state.df = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("OpenAI model", value=DEFAULT_MODEL)

    st.divider()
    st.subheader("OpenAI API Key")
    api_key_input = st.text_input(
        "OPENAI_API_KEY",
        type="password",
        value=st.session_state.openai_api_key,
        help="You can set this in .env too. This field overrides env for this session.",
    )
    st.session_state.openai_api_key = api_key_input.strip()

    st.divider()
    st.subheader("Upload CSV")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    show_df = st.checkbox("Show dataframe preview", value=True)
    show_profile = st.checkbox("Show basic stats", value=True)

    st.divider()
    st.subheader("Optional")
    hide_code_in_chat = st.checkbox("Hide code in chat", value=True)
    show_code = st.checkbox("Show generated code", value=False)

# Load CSV
if uploaded is not None:
    try:
        bytes_data = uploaded.getvalue()
        df = pd.read_csv(io.BytesIO(bytes_data))
        df.columns = [c.strip() for c in df.columns]
        st.session_state.df = df
        st.sidebar.success(f"Loaded: {uploaded.name}  |  shape={df.shape}")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {type(e).__name__}: {e}")

df: Optional[pd.DataFrame] = st.session_state.df

col1, col2 = st.columns([1.1, 1.2], gap="large")

with col1:
    st.subheader("Data")
    if df is None:
        st.info("Upload a CSV from the sidebar to begin.")
    else:
        st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")

        if show_df:
            st.dataframe(df.head(50), width="stretch")

        if show_profile:
            st.markdown("**Quick stats (numeric columns)**")
            num_df = df.select_dtypes(include="number")
            if num_df.shape[1] == 0:
                st.write("No numeric columns detected.")
            else:
                st.dataframe(num_df.describe().T, width="stretch")

with col2:
    st.subheader("Chat")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if df is None:
        st.info("Upload a CSV to enable chat.")
    else:
        if not st.session_state.openai_api_key:
            st.warning("Add your OpenAI API key in the sidebar (or set it in .env) to start chatting.")

        user_q = st.chat_input("Ask a question about your data…")
        if user_q:
            st.session_state.messages.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            try:
                llm = get_llm(model_name=model_name, api_key=st.session_state.openai_api_key)
                assistant_text = llm_answer_and_optional_plot(
                    llm=llm,
                    df=df,
                    chat_history=st.session_state.messages,
                    user_question=user_q,
                )

                # Extract plot code (if any)
                code = extract_python_code(assistant_text)
                plot_fig = None
                plot_err = None

                if code:
                    code = normalize_plot_code(code)
                    plot_fig, plot_err = safe_exec_plot(code, df)

                # Display assistant message (optionally hide code)
                display_text = remove_code_block_from_markdown(assistant_text) if hide_code_in_chat else assistant_text
                st.session_state.messages.append({"role": "assistant", "content": display_text})

                with st.chat_message("assistant"):
                    st.markdown(display_text)

                    if code:
                        st.markdown("**Visualization:**")
                        if plot_err:
                            st.error(f"Plot error: {plot_err}")
                            if show_code:
                                st.code(code, language="python")
                        else:
                            st.pyplot(plot_fig, clear_figure=True)
                            if show_code:
                                st.code(code, language="python")

            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"{type(e).__name__}: {e}")
