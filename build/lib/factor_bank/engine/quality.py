"""Data-quality diagnostics returned with every evaluation (spec §6)."""
from __future__ import annotations

import pandas as pd


def quality_report(df: pd.DataFrame, value_col: str = "_factor") -> dict:
    valid = df[value_col].notna()
    coverage_by_year = (
        valid.groupby(df["date"].dt.year).mean().round(4).to_dict()
    )

    def _stale_frac(s: pd.Series) -> float | None:
        s = s.dropna()
        if len(s) < 2:
            return None
        return float((s.diff().iloc[1:] == 0).mean())

    stale = df.groupby("ticker")[value_col].apply(_stale_frac).dropna()

    return {
        "coverage": round(float(valid.mean()), 4) if len(df) else 0.0,
        "coverage_by_year": {int(k): float(v) for k, v in coverage_by_year.items()},
        "duplicates": int(df.duplicated(["ticker", "date"]).sum()),
        "staleness_mean": round(float(stale.mean()), 4) if len(stale) else None,
        "n_rows": int(len(df)),
    }
