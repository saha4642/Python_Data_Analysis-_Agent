from __future__ import annotations

import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

from llm import get_llm


def build_df_agent(df: pd.DataFrame):
    llm = get_llm()

    # AgentType.OPENAI_FUNCTIONS works well with ChatOpenAI-compatible models.
    # verbose=True prints the reasoning steps/tools used (helpful for debugging).
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        allow_dangerous_code=True,  # enables code execution inside agent when needed
    )
    return agent
