"""Adapter: factor-bank matrices → alpha_eval.ml_eval → JSON-safe dict (spec §7)."""
from __future__ import annotations

import json
import time
from typing import Callable

import pandas as pd

from factor_bank.engine.factors import compute_factor
from factor_bank.engine.metrics import trading_session_forward_returns
from factor_bank.engine.panel import get_window

ML_HORIZONS = (1, 5, 21, 63)
MODE_PERMUTATIONS = {"quick": 0, "standard": 50, "thorough": 200}
MAX_FACTORS = 20


def _records(df) -> list[dict]:
    """DataFrame → JSON-safe records (to_json handles numpy scalars + NaN)."""
    if df is None or len(df) == 0:
        return []
    return json.loads(df.to_json(orient="records"))


def run_ml_eval(
    factors: list[str],
    horizons: list[int],
    from_date: str,
    to_date: str,
    quantiles: int = 5,
    mode: str = "standard",
    tier2: bool = False,
    progress: Callable[[str], None] | None = None,
    *,
    enriched: pd.DataFrame | None = None,
    spells: pd.DataFrame | None = None,
) -> dict:
    from alpha_eval.ml_eval import ml_eval, prepare_ml_eval_data

    if not (2 <= len(set(factors)) <= MAX_FACTORS):
        raise ValueError(f"factors must contain 2..{MAX_FACTORS} distinct names")
    if not horizons or any(h not in ML_HORIZONS for h in horizons):
        raise ValueError(f"horizons must be a non-empty subset of {ML_HORIZONS}")
    if mode not in MODE_PERMUTATIONS:
        raise ValueError(f"mode must be one of {sorted(MODE_PERMUTATIONS)}")

    tick = progress or (lambda msg: None)
    t0 = time.time()
    horizons = sorted(set(int(h) for h in horizons))

    tick("loading data window")
    df = get_window(from_date, to_date, max(horizons), enriched=enriched, spells=spells)
    if df.empty:
        raise ValueError("No data after S&P 500 filtering")

    from_ts, to_ts = pd.Timestamp(from_date), pd.Timestamp(to_date)
    in_window = (df["date"] >= from_ts) & (df["date"] <= to_ts)

    features: dict[str, pd.DataFrame] = {}
    for i, name in enumerate(dict.fromkeys(factors), 1):
        tick(f"factor matrices {i}/{len(set(factors))}: {name}")
        # Unlike lab/screen.py (which skips a bad candidate out of hundreds
        # and keeps going), ml-eval users pick a small, explicit factor list
        # — a column that vanished from the enriched frame (schema drift)
        # should fail the whole job loudly, naming the factor, rather than
        # silently dropping it from the battery.
        try:
            vals = compute_factor(df, name)
        except ValueError as e:
            raise ValueError(f"Factor '{name}' failed: {e}") from e
        features[name] = (
            df[in_window].assign(_f=vals[in_window])
            .pivot_table(index="date", columns="ticker", values="_f")
        )

    tick("forward returns")
    price_pivot = df.pivot_table(
        index="date", columns="ticker", values="prev_close_price"
    ).sort_index()
    eval_dates = df.loc[in_window, "date"].unique()
    target = {
        h: trading_session_forward_returns(price_pivot, h).loc[
            lambda m: m.index.isin(eval_dates)
        ]
        for h in horizons
    }

    tick("preparing data (align + trim)")
    data = prepare_ml_eval_data(
        features=features,
        prices=price_pivot,
        target=target,
        min_common_tickers=5,
        min_observations=60,
        winsorize=0.01,
    )

    methods = ["mutual_info", "distance_corr", "quantile_spread", "monotonicity", "redundancy"]
    if tier2:
        methods += ["mdi", "quick_mda"]
    n_perm = MODE_PERMUTATIONS[mode]

    tick(f"running ml_eval ({mode}, {n_perm} permutations — the long stage)")
    result = ml_eval(
        data, quantiles=quantiles, methods=methods,
        model_type="lightgbm", n_jobs=1, n_permutations=n_perm,
    )

    tick("serializing")
    red_df = result.redundancy_matrix()
    redundancy = None
    if red_df is not None and not red_df.empty:
        redundancy = {
            "features": [str(c) for c in red_df.columns],
            "values": json.loads(red_df.to_json(orient="values")),
        }
    mono = result.monotonicity()
    mono_summary = mono.get("summary") if isinstance(mono, dict) else None

    return {
        "screening": _records(result.screening_summary("best")),
        "mutual_info": _records(result.mutual_info()),
        "distance_corr": _records(result.distance_corr()),
        "redundancy": redundancy,
        "monotonicity": _records(mono_summary),
        "mdi": _records(result.mdi_importance()) if tier2 else None,
        "mda": _records(result.quick_mda()) if tier2 else None,
        "meta": {
            "factors": sorted(features),
            "horizons": horizons,
            "from_date": from_date,
            "to_date": to_date,
            "quantiles": quantiles,
            "mode": mode,
            "n_permutations": n_perm,
            "n_tickers": data.n_tickers,
            "n_dates": data.n_dates,
            "warnings": list(data.warnings),
            "elapsed_s": round(time.time() - t0, 1),
        },
    }
