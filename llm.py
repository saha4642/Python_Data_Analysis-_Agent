from __future__ import annotations

from langchain_openai import ChatOpenAI
from config import get_settings


def get_llm() -> ChatOpenAI:
    s = get_settings()
    # ChatOpenAI reads OPENAI_API_KEY from env automatically, but we validate in config.py
    return ChatOpenAI(
        model=s.openai_model,
        temperature=0,
    )
