from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

def _read_edges_csv_strict(path: Path) -> pd.DataFrame:
    """Read a CSV and guarantee the exact 8 columns exist (fill missing with NaN)."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[edges] WARN: failed to read {path}: {e}")
        return pd.DataFrame(columns=REQUIRED_EDGES_COLUMNS)
    for col in REQUIRED_EDGES_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[REQUIRED_EDGES_COLUMNS].copy()