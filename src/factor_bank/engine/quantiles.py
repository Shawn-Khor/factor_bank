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
    horizon: int = 1,
) -> dict:
    """Alphalens-style quantile spread.

    For each date, rank the cross-section into N equal-population buckets.
    Then take the mean forward-return per bucket across all dates.

    `fwd_matrix` holds `horizon`-trading-session forward returns sampled
    every session, so the per-date long/short spread series (`ls_daily`) is
    made of *overlapping* horizon-session returns whenever horizon > 1 (each
    value shares horizon-1 sessions with its neighbors). Two derived stats
    correct for that overlap:

    - Cumulative return: each overlapping horizon-session spread is first
      converted to its per-session-equivalent daily rate,
      `(1 + ls) ** (1/horizon) - 1`, before compounding — otherwise a single
      21-session return gets compounded as if it were earned in one day, 21
      consecutive times, inflating the reported total ~horizon-fold.
      Pathological spreads where `1 + ls <= 0` (an economically meaningless
      < -100% horizon return) can't take a fractional power; those dates are
      set to NaN and dropped rather than compounded.
    - Annualized Sharpe: computed on the non-overlapping subsample
      `ls_daily.iloc[::horizon]` (every horizon-th date only), annualized by
      `sqrt(252 / horizon)` instead of `sqrt(252)`.

    At horizon=1 (the default) both reduce exactly to plain daily compounding
    — no overlap exists at horizon 1.
    """
    if n_quantiles not in ALLOWED_QUANTILES:
        raise ValueError(f"n_quantiles must be one of {ALLOWED_QUANTILES}")

    F = factor_matrix
    R = fwd_matrix
    valid = F.notna() & R.notna()

    # Vectorised quantile assignment per date:
    # 1. Rank each row to percentile in [0, 1]   (NaN where invalid)
    # 2. Convert to integer bin 1..N via ceil(pct * N), clipped to N
    # Rank on jointly-valid (F, R) pairs only (F.where(valid)) so tickers
    # whose forward return is missing don't consume a bucket slot —
    # consistent with cross_sectional_metrics' masking.
    pct = F.where(valid).rank(axis=1, pct=True, method="first")  # 0 < pct ≤ 1 for non-NaN
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

    # Long/short daily series = Q_top - Q_bottom per date. This is a series
    # of *horizon*-session forward returns sampled every session, so it is
    # overlapping (autocorrelated) for horizon > 1 — see docstring.
    ls_daily = (per_date_means[n_quantiles] - per_date_means[1]).dropna()
    if len(ls_daily) == 0:
        return {
            "quantile_means": quantile_means,
            "longshort_cumulative": [],
            "longshort_stats": None,
        }

    # ─── Cumulative: de-overlap by converting each horizon-session return to
    # its per-session-equivalent daily rate before compounding.
    base = 1.0 + ls_daily
    r_d = base.pow(1.0 / horizon) - 1.0
    r_d = r_d.where(base > 0)  # guard pathological spreads (1+ls <= 0)
    r_d = r_d.dropna()
    cum = (1.0 + r_d).cumprod() - 1.0
    longshort_cumulative = [
        {"date": d.strftime("%Y-%m-%d"), "value": float(v)}
        for d, v in cum.items()
    ]
    total_return = float(cum.iloc[-1]) if len(cum) else None

    # ─── Annualized Sharpe: non-overlapping subsample only, so consecutive
    # samples don't share horizon-1 sessions of return history.
    ls_nol = ls_daily.iloc[::horizon]
    n = len(ls_daily)
    n_independent = len(ls_nol)
    mean_d = float(ls_nol.mean())
    std_d = float(ls_nol.std(ddof=1))
    sharpe = (mean_d / std_d * np.sqrt(252.0 / horizon)) if std_d > 0 else None
    longshort_stats = {
        "total_return": round(total_return, 6) if total_return is not None else None,
        "annualized_sharpe": round(sharpe, 4) if sharpe is not None else None,
        "n_days": n,
        "n_independent": n_independent,
    }

    return {
        "quantile_means": quantile_means,
        "longshort_cumulative": longshort_cumulative,
        "longshort_stats": longshort_stats,
    }
