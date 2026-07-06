"""End-to-end single-factor evaluation (spec §1.1 scope rules)."""
from __future__ import annotations

import pandas as pd

from factor_bank.config import ALLOWED_HORIZONS, ALLOWED_QUANTILES, get_settings
from factor_bank.data.enriched import load_enriched
from factor_bank.data.universe import filter_to_sp500, get_spells
from factor_bank.engine.catalog import all_factor_names
from factor_bank.engine.factors import TRANSFORMS, compute_factor
from factor_bank.engine.metrics import (
    cross_sectional_metrics,
    trading_session_forward_returns,
    verdict,
)
from factor_bank.engine.quality import quality_report
from factor_bank.engine.quantiles import quantile_spread


def _known_factor(name: str) -> bool:
    if "__" in name:
        base, _, tr = name.partition("__")
        return base in all_factor_names() and tr in TRANSFORMS
    return name in all_factor_names()


def evaluate(
    factor_name: str,
    from_date: str,
    to_date: str,
    horizon: int,
    n_quantiles: int = 5,
    *,
    enriched: pd.DataFrame | None = None,
    spells: pd.DataFrame | None = None,
    winsorize: float | None = 0.01,
) -> dict:
    if not _known_factor(factor_name):
        raise ValueError(f"Unknown factor: {factor_name}")
    if horizon not in ALLOWED_HORIZONS:
        raise ValueError(f"Horizon must be one of {ALLOWED_HORIZONS}")
    if n_quantiles not in ALLOWED_QUANTILES:
        raise ValueError(f"n_quantiles must be one of {ALLOWED_QUANTILES}")

    floor = pd.Timestamp(get_settings().date_floor)
    from_ts, to_ts = pd.Timestamp(from_date), pd.Timestamp(to_date)
    if from_ts < floor:
        raise ValueError(f"from_date {from_date} below floor {floor.date()}")

    df_all = enriched if enriched is not None else load_enriched()
    sp = spells if spells is not None else get_spells()

    buf_start = from_ts - pd.Timedelta(days=400)
    buf_end = to_ts + pd.Timedelta(days=int(horizon * 1.6) + 7)
    df = df_all[(df_all["date"] >= buf_start) & (df_all["date"] <= buf_end)].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df = filter_to_sp500(df, sp)
    if df.empty:
        return {"error": "No data after S&P 500 filtering"}

    df["_factor"] = compute_factor(df, factor_name)
    in_window = (df["date"] >= from_ts) & (df["date"] <= to_ts)
    df_eval = df[in_window].reset_index(drop=True)
    if df_eval.empty:
        return {"error": "No data in requested window"}

    quality = quality_report(df_eval)

    factor_matrix = df_eval.pivot_table(index="date", columns="ticker", values="_factor")
    price_pivot = df.pivot_table(
        index="date", columns="ticker", values="prev_close_price"
    ).sort_index()
    fwd_full = trading_session_forward_returns(price_pivot, horizon)
    common_dates = factor_matrix.index.intersection(fwd_full.index)
    common_tickers = factor_matrix.columns.intersection(fwd_full.columns)
    F = factor_matrix.loc[common_dates, common_tickers]
    R = fwd_full.loc[common_dates, common_tickers]

    metrics = cross_sectional_metrics(F, R, winsorize=winsorize)
    qs = quantile_spread(F, R, n_quantiles)

    return {
        "factor": factor_name,
        "horizon": horizon,
        "n_quantiles": n_quantiles,
        "from_date": from_date,
        "to_date": to_date,
        "metrics": metrics,
        "verdict": verdict(metrics),
        "quality": quality,
        "quantile_means": qs["quantile_means"],
        "longshort_cumulative": qs["longshort_cumulative"],
        "longshort_stats": qs["longshort_stats"],
        "meta": {
            "n_dates": int(len(common_dates)),
            "n_tickers_universe": int(len(common_tickers)),
        },
    }
