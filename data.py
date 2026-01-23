from __future__ import annotations

import pandas as pd
from config import get_settings


def load_dataframe() -> pd.DataFrame:
    s = get_settings()
    df = pd.read_csv(s.csv_path)

    # Small cleanup: avoid weird column names
    df.columns = [c.strip() for c in df.columns]

    return df
