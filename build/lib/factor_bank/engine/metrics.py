"""Cross-sectional IC battery — Pearson/Rank IC, IC IR, t-stat, p-value,
winsorization, and a plain-English verdict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_bank.config import ALLOWED_HORIZONS

# ─── Forward returns — trading-session index shift ──────────────────────────


def trading_session_forward_returns(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """Forward return after N trading sessions — pure integer index shift.

    No `np.searchsorted` on calendar dates needed. Clean, fast, no weekend edge
    cases. NaN where the future row doesn't exist (end of date range).
    """
    if n not in ALLOWED_HORIZONS:
        raise ValueError(f"Horizon {n} not in {ALLOWED_HORIZONS}")
    shifted = prices.shift(-n)
    return shifted / prices - 1


# ─── Winsorization ───────────────────────────────────────────────────────────


def winsorize_xs(F: pd.DataFrame, p: float = 0.01) -> pd.DataFrame:
    """Per-date (row-wise) clip at the p / 1-p cross-sectional quantiles."""
    lo = F.quantile(p, axis=1)
    hi = F.quantile(1 - p, axis=1)
    return F.clip(lo, hi, axis=0)


# ─── Cross-sectional IC ──────────────────────────────────────────────────────


def _pearson_rowwise(F: pd.DataFrame, R: pd.DataFrame) -> pd.Series:
    Fm = F.sub(F.mean(axis=1), axis=0)
    Rm = R.sub(R.mean(axis=1), axis=0)
    numer = (Fm * Rm).sum(axis=1)
    denom = np.sqrt((Fm ** 2).sum(axis=1)) * np.sqrt((Rm ** 2).sum(axis=1))
    return (numer / denom.replace(0, np.nan)).dropna()


def cross_sectional_metrics(
    factor_matrix: pd.DataFrame,
    fwd_matrix: pd.DataFrame,
    winsorize: float | None = 0.01,
) -> dict:
    """Vectorised cross-sectional IC, Rank IC, IC IR, t-stat, p-value.

    factor_matrix / fwd_matrix are both (date × ticker) wide-format.
    Both must share the same index/columns ordering.

    Pearson IC is computed on the winsorized factor matrix (per-date clip at
    the `winsorize` / `1-winsorize` quantiles) when `winsorize` is set;
    `ic_raw` always keeps the unclipped Pearson value for comparison. Rank IC
    is unaffected by winsorization since ranks are clip-invariant.
    """
    from scipy import stats

    F = factor_matrix
    R = fwd_matrix

    valid = F.notna() & R.notna()
    good_dates = valid.sum(axis=1)
    good_dates = good_dates[good_dates >= 5].index
    if len(good_dates) < 3:
        return _empty_metrics(len(good_dates))

    F = F.loc[good_dates].where(valid.loc[good_dates])
    R = R.loc[good_dates].where(valid.loc[good_dates])

    # Pearson IC — vectorised row-wise, on winsorized values; raw kept for comparison
    Fw = winsorize_xs(F, winsorize) if winsorize else F
    ic_s = _pearson_rowwise(Fw, R)
    ic_raw_s = _pearson_rowwise(F, R)

    # Spearman IC — rank rows, then Pearson on ranks
    Fr = F.rank(axis=1)
    Rr = R.rank(axis=1)
    Frm = Fr.sub(Fr.mean(axis=1), axis=0)
    Rrm = Rr.sub(Rr.mean(axis=1), axis=0)
    numer_r = (Frm * Rrm).sum(axis=1)
    denom_r = np.sqrt((Frm ** 2).sum(axis=1)) * np.sqrt((Rrm ** 2).sum(axis=1))
    ric_s = (numer_r / denom_r.replace(0, np.nan)).dropna()

    idx = ic_s.index.intersection(ric_s.index).intersection(ic_raw_s.index)
    ic_arr = ic_s.loc[idx].to_numpy()
    ric_arr = ric_s.loc[idx].to_numpy()
    ic_raw_arr = ic_raw_s.loc[idx].to_numpy()
    n = len(ric_arr)
    if n < 3:
        return _empty_metrics(n)

    mean_ic = float(np.mean(ic_arr))
    mean_ic_raw = float(np.mean(ic_raw_arr))
    mean_ric = float(np.mean(ric_arr))
    std_ric = float(np.std(ric_arr, ddof=1))
    pct_positive = float(np.mean(ric_arr > 0))
    ic_ir = mean_ric / std_ric if std_ric > 0 else None
    t_stat = mean_ric / (std_ric / np.sqrt(n)) if std_ric > 0 else None
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1)) if t_stat is not None else None

    def _r(v, nd=6):
        return round(v, nd) if v is not None and np.isfinite(v) else None

    return {
        "ic": _r(mean_ic),
        "ic_raw": _r(mean_ic_raw),
        "rank_ic": _r(mean_ric),
        "std_ic": _r(std_ric),
        "ic_ir": _r(ic_ir),
        "pct_positive": _r(pct_positive),
        "t_stat": _r(t_stat),
        "p_value": _r(p_value),
        "n_obs": n,
    }


def _empty_metrics(n: int) -> dict:
    return {
        "ic": None, "ic_raw": None, "rank_ic": None, "std_ic": None, "ic_ir": None,
        "pct_positive": None, "t_stat": None, "p_value": None, "n_obs": n,
    }


# ─── Verdict ──────────────────────────────────────────────────────────────


def verdict(m: dict) -> str:
    t = abs(m.get("t_stat") or 0.0)
    ric = abs(m.get("rank_ic") or 0.0)
    if t >= 3 and ric >= 0.02:
        return "STRONG"
    if t >= 2 and ric >= 0.01:
        return "MODERATE"
    if t >= 2:
        return "WEAK"
    return "INSIGNIFICANT"
