"""Alphalens-style quantile spread — cross-sectional bucketing and long/short
cumulative return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_bank.config import ALLOWED_QUANTILES


def quantile_spread(
    factor_matrix: pd.DataFrame,
    fwd_matrix: pd.DataFrame,
    n_quantiles: int,
) -> dict:
    """Alphalens-style quantile spread.

    For each date, rank the cross-section into N equal-population buckets.
    Then take the mean forward-return per bucket across all dates.
    Long-short cumulative = compound daily (Q_top - Q_bottom).
    """
    if n_quantiles not in ALLOWED_QUANTILES:
        raise ValueError(f"n_quantiles must be one of {ALLOWED_QUANTILES}")

    F = factor_matrix
    R = fwd_matrix
    valid = F.notna() & R.notna()

    # Vectorised quantile assignment per date:
    # 1. Rank each row to percentile in [0, 1]   (NaN where invalid)
    # 2. Convert to integer bin 1..N via ceil(pct * N), clipped to N
    pct = F.rank(axis=1, pct=True, method="first")  # 0 < pct ≤ 1 for non-NaN
    Q = np.ceil(pct * n_quantiles).clip(upper=n_quantiles)
    # Mask rows with too few valid observations (need ≥ 2 per bucket)
    too_sparse = valid.sum(axis=1) < n_quantiles * 2
    if too_sparse.any():
        Q.loc[too_sparse] = np.nan

    # Per-date, per-quantile mean return — vectorised
    per_date_means: dict[int, pd.Series] = {}
    for q in range(1, n_quantiles + 1):
        mask = (Q == q) & valid
        masked_R = R.where(mask)
        per_date_means[q] = masked_R.mean(axis=1)

    # Quantile means (across all dates) — for the bar chart
    quantile_means = [
        {"quantile": q,
         "mean_return": float(per_date_means[q].mean()) if per_date_means[q].notna().any() else None,
         "n_obs": int(per_date_means[q].notna().sum())}
        for q in range(1, n_quantiles + 1)
    ]

    # Long/short daily series = Q_top - Q_bottom per date
    ls_daily = (per_date_means[n_quantiles] - per_date_means[1]).dropna()
    if len(ls_daily) == 0:
        return {
            "quantile_means": quantile_means,
            "longshort_cumulative": [],
            "longshort_stats": None,
        }

    cum = (1.0 + ls_daily).cumprod() - 1.0
    longshort_cumulative = [
        {"date": d.strftime("%Y-%m-%d"), "value": float(v)}
        for d, v in cum.items()
    ]

    # Annualized stats
    n = len(ls_daily)
    mean_d = float(ls_daily.mean())
    std_d = float(ls_daily.std(ddof=1))
    sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 0 else None
    total_return = float(cum.iloc[-1])
    longshort_stats = {
        "total_return": round(total_return, 6),
        "annualized_sharpe": round(sharpe, 4) if sharpe is not None else None,
        "n_days": n,
    }

    return {
        "quantile_means": quantile_means,
        "longshort_cumulative": longshort_cumulative,
        "longshort_stats": longshort_stats,
    }
