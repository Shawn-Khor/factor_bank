"""Factor compute engine — passthroughs, IBD ordinals, yields, cap structure,
rolling time-series transforms, sector-relative, and generated `{base}__{transform}`
names via the TRANSFORMS registry.

Precondition: `df` must be sorted by (ticker, date) for all per-ticker rolling
and shift-based factors/transforms to be correct.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from factor_bank.data.enriched import COLUMN_GROUPS

# IBD ordinal encodings (higher = more bullish/safer)
IBD_TREND_MAP = {"BULL": 2, "MIXED_UP": 1, "MIXED_DOWN": -1, "BEAR": -2}
IBD_STATE_MAP = {
    "RECOVERY": 2, "AVERAGE": 1, "DETERIORATION": -1, "COLLAPSE": -2,
}
IBD_RISK_MAP = {"LOW": 1, "MED": 0, "HIGH": -1}

# Set of factors that are simple passthroughs from the enriched parquet.
_PASSTHROUGH_COLS: set[str] = set(
    COLUMN_GROUPS["daily"] + COLUMN_GROUPS["sa"] + COLUMN_GROUPS["zacks"]
    + COLUMN_GROUPS["ibd_numeric"] + COLUMN_GROUPS["technical"]
    + COLUMN_GROUPS["returns"]
)


def compute_factor(df: pd.DataFrame, name: str) -> pd.Series:
    """Compute one factor on the enriched DataFrame, returning a Series indexed
    to the same rows as df.

    Engineered factors (yields, z-scores, percentiles, sector-relative) are
    derived from the raw columns already in df. No DB roundtrip.

    Precondition: `df` sorted by (ticker, date).
    """
    # Generated names — "{base}__{transform}" — recurse then apply TRANSFORMS.
    if "__" in name:
        base, _, transform = name.partition("__")
        if transform not in TRANSFORMS:
            raise ValueError(f"Unknown transform: {transform}")
        return TRANSFORMS[transform](df, compute_factor(df, base))

    # Passthroughs — raw daily / SA / Zacks / IBD-numeric / technicals / returns
    if name in _PASSTHROUGH_COLS:
        if name not in df.columns:
            raise ValueError(f"Column '{name}' not present in enriched parquet")
        return pd.to_numeric(df[name], errors="coerce").astype(float)

    # IBD string columns — apply ordinal encoder
    if name == "ibd_trend_ord":
        return df["ibd_trend"].map(IBD_TREND_MAP).astype(float)
    if name == "ibd_state_ord":
        return df["ibd_state"].map(IBD_STATE_MAP).astype(float)
    if name == "ibd_risk_ord":
        return df["ibd_risk"].map(IBD_RISK_MAP).astype(float)

    # Wave 2 — yield transforms
    if name == "earnings_yield":
        return 1.0 / df["pe"].replace(0, np.nan)
    if name == "book_yield":
        return 1.0 / df["pb"].replace(0, np.nan)
    if name == "sales_yield":
        return 1.0 / df["ps"].replace(0, np.nan)
    if name == "ebitda_yield":
        return 1.0 / df["evebitda"].replace(0, np.nan)

    # Wave 3 — capital structure
    if name == "ev_to_mcap":
        return df["ev"] / df["marketcap"].replace(0, np.nan)
    if name == "log_marketcap":
        mc = df["marketcap"].astype(float)
        return np.log(mc.where(mc > 0))

    # Wave 4 — time-series transforms (per-ticker rolling)
    if name == "pe_zscore_252d":
        return _rolling_zscore(df, "pe", 252)
    if name == "ebitda_yield_zscore_252d":
        ey = 1.0 / df["evebitda"].replace(0, np.nan)
        return _rolling_zscore_series(df["ticker"], ey, 252)
    if name == "pe_change_30d":
        return _per_ticker_shift_diff(df, "pe", 30)
    if name == "pb_percentile_252d":
        return _rolling_percentile(df, "pb", 252)

    # Wave 5 — cross-sectional
    if name == "evebitda_sector_relative":
        ee = df["evebitda"]
        sec_median = df.assign(_ee=ee).groupby(["date", "sector"])["_ee"].transform("median")
        return ee - sec_median

    raise ValueError(f"Unknown factor: {name}")


def _rolling_zscore(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return _rolling_zscore_series(df["ticker"], df[col], window)


def _rolling_zscore_series(ticker: pd.Series, values: pd.Series, window: int) -> pd.Series:
    """Per-ticker rolling z-score. df must be sorted by (ticker, date)."""
    grouped = values.groupby(ticker)
    mean = grouped.transform(lambda x: x.rolling(window, min_periods=window).mean())
    std = grouped.transform(lambda x: x.rolling(window, min_periods=window).std(ddof=0))
    z = (values - mean) / std.replace(0, np.nan)
    return z


def _per_ticker_shift_diff(df: pd.DataFrame, col: str, lag: int) -> pd.Series:
    """Per-ticker (col - col.shift(lag)). Sort assumed (ticker, date)."""
    return df.groupby("ticker")[col].transform(lambda x: x - x.shift(lag))


def _rolling_percentile(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    """Per-ticker rolling rank percentile in 1yr window."""
    def _rp(x: pd.Series) -> pd.Series:
        # rolling.rank with pct=True returns the rank within each window
        return x.rolling(window, min_periods=window).rank(pct=True)
    return df.groupby("ticker")[col].transform(_rp)


# ─── Transform registry — shared with Plan C's Factor Lab grid ─────────────


def _chg(df: pd.DataFrame, s: pd.Series, lag: int) -> pd.Series:
    return s.groupby(df["ticker"]).transform(lambda x: x - x.shift(lag))


def _pctile(df: pd.DataFrame, s: pd.Series, window: int) -> pd.Series:
    return s.groupby(df["ticker"]).transform(
        lambda x: x.rolling(window, min_periods=window).rank(pct=True)
    )


TRANSFORMS: dict[str, Callable[[pd.DataFrame, pd.Series], pd.Series]] = {
    "zscore_63d":  lambda df, s: _rolling_zscore_series(df["ticker"], s, 63),
    "zscore_126d": lambda df, s: _rolling_zscore_series(df["ticker"], s, 126),
    "zscore_252d": lambda df, s: _rolling_zscore_series(df["ticker"], s, 252),
    "chg_5d":      lambda df, s: _chg(df, s, 5),
    "chg_21d":     lambda df, s: _chg(df, s, 21),
    "chg_63d":     lambda df, s: _chg(df, s, 63),
    "pctile_252d": lambda df, s: _pctile(df, s, 252),
    "sector_rel":  lambda df, s: s - s.groupby([df["date"], df["sector"]]).transform("median"),
}
