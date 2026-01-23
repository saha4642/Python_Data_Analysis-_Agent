from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    csv_path: str


def get_settings() -> Settings:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in .env or set it in your environment."
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    csv_path = os.getenv("CSV_PATH", "./data.csv").strip()

    return Settings(
        openai_api_key=api_key,
        openai_model=model,
        csv_path=csv_path,
    )
