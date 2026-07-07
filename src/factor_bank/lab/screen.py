"""Factor Lab two-stage screen: train-only stage-1 Rank IC battery with
BH-FDR across the candidate grid, then a stage-2 deep pass (holdout Rank IC,
sign-flip, mutual information, distance correlation) on the top-K survivors.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from sklearn.feature_selection import mutual_info_regression

from factor_bank.engine.factors import compute_factor
from factor_bank.engine.metrics import cross_sectional_metrics, trading_session_forward_returns
from factor_bank.engine.panel import get_window
from factor_bank.lab.grid import candidate_grid

MIN_TRAIN_DATES = 60
MIN_HOLDOUT_DATES = 20
MI_MAX_N = 50_000
DCOR_MAX_N = 3_000
SEED = 42


def _distance_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    """Bias-corrected distance correlation — O(n^2), use n up to a few thousand.

    Pure-numpy port of the legacy `_distance_corr` in
    alpha-discovery/dashboard/api.py (lines 810-824).
    """
    n = len(x)
    if n < 4:
        return None
    A = np.abs(x[:, None] - x[None, :]).astype(np.float64)
    B = np.abs(y[:, None] - y[None, :]).astype(np.float64)

    def _dc(M: np.ndarray) -> np.ndarray:
        return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()

    Adc, Bdc = _dc(A), _dc(B)
    dcov2_xy = (Adc * Bdc).mean()
    denom = np.sqrt(abs((Adc * Adc).mean()) * abs((Bdc * Bdc).mean()))
    if denom < 1e-15:
        return None
    return float(np.sqrt(max(0.0, dcov2_xy / denom)))


def _round(v, nd: int = 6):
    return round(float(v), nd) if v is not None and np.isfinite(v) else None


def _pivot_factor(df: pd.DataFrame, in_window: pd.Series, columns, name: str) -> pd.DataFrame:
    """compute_factor(df, name) restricted to the eval window, pivoted wide,
    columns aligned to the shared ticker universe (`columns`, from the price
    pivot) so it lines up exactly with the forward-return matrix.

    Raises ValueError (propagated from compute_factor) if the candidate's
    base column is missing from the window.
    """
    vals = compute_factor(df, name)
    F = (
        df[in_window].assign(_f=vals[in_window])
        .pivot_table(index="date", columns="ticker", values="_f")
    )
    return F.reindex(columns=columns)


def _flatten_valid(F: pd.DataFrame, R: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Flatten two aligned wide matrices, keeping only finite (factor, fwd
    return) pairs."""
    f = F.to_numpy().reshape(-1)
    r = R.reindex(index=F.index, columns=F.columns).to_numpy().reshape(-1)
    mask = np.isfinite(f) & np.isfinite(r)
    return f[mask], r[mask]


def _subsample(f: np.ndarray, r: np.ndarray, max_n: int) -> tuple[np.ndarray, np.ndarray]:
    if len(f) <= max_n:
        return f, r
    idx = np.random.default_rng(SEED).choice(len(f), max_n, replace=False)
    return f[idx], r[idx]


def screen(
    horizon: int,
    from_date: str,
    to_date: str,
    top_k: int = 30,
    progress: Callable[[str], None] | None = None,
    *,
    enriched: pd.DataFrame | None = None,
    spells: pd.DataFrame | None = None,
    candidates: list[str] | None = None,
) -> dict:
    """Two-stage screen over a candidate grid.

    Stage 1 (train-only): Rank IC battery per candidate, BH-FDR across the
    non-null p-values.

    Stage 2 (top-K by |ic_ir|): holdout Rank IC, sign-flip flag, mutual
    information, distance correlation — all computed on data untouched by
    stage 1 or the top-K ranking.
    """
    tick = progress or (lambda msg: None)
    t0 = time.time()

    df = get_window(from_date, to_date, horizon, enriched=enriched, spells=spells)

    from_ts, to_ts = pd.Timestamp(from_date), pd.Timestamp(to_date)
    in_window = (df["date"] >= from_ts) & (df["date"] <= to_ts)

    dates = np.sort(df.loc[in_window, "date"].unique())
    split = int(len(dates) * 0.7)
    train_dates_raw = dates[:split]
    holdout_dates = dates[split:]

    # Embargo: a train date's forward return at `horizon` sessions ahead can
    # land inside the holdout price path, which would let stage-1 selection
    # see holdout information. Drop the last `horizon` train dates so every
    # train-stage forward return is realized strictly before the holdout
    # segment starts.
    train_dates = train_dates_raw[:-horizon] if horizon > 0 else train_dates_raw
    if len(train_dates) < MIN_TRAIN_DATES:
        raise ValueError(
            f"Only {len(train_dates)} train dates in window (need >= {MIN_TRAIN_DATES})"
        )
    if len(holdout_dates) < MIN_HOLDOUT_DATES:
        raise ValueError(
            f"Only {len(holdout_dates)} holdout dates in window (need >= {MIN_HOLDOUT_DATES})"
        )

    price_pivot = df.pivot_table(index="date", columns="ticker", values="prev_close_price").sort_index()
    R_full = trading_session_forward_returns(price_pivot, horizon)
    R_train = R_full.reindex(index=train_dates)
    R_holdout = R_full.reindex(index=holdout_dates)

    cand_list = list(candidates) if candidates is not None else candidate_grid()
    n = len(cand_list)

    # ─── Stage 1: train-only Rank IC battery ────────────────────────────────
    processed: list[dict] = []
    skipped: list[str] = []

    for i, cand in enumerate(cand_list, 1):
        try:
            F = _pivot_factor(df, in_window, price_pivot.columns, cand)
            m = cross_sectional_metrics(F.reindex(index=train_dates), R_train, winsorize=None)
            processed.append({"candidate": cand, "metrics": m})
        except ValueError:
            skipped.append(cand)
        if i % 25 == 0:
            tick(f"stage 1: {i}/{n} candidates")

    # BH-FDR across the non-null stage-1 p-values; null p -> null q.
    idx_with_p = [j for j, r in enumerate(processed) if r["metrics"]["p_value"] is not None]
    if idx_with_p:
        pvals = np.array([processed[j]["metrics"]["p_value"] for j in idx_with_p], dtype=float)
        qvals = false_discovery_control(pvals)
        for j, q in zip(idx_with_p, qvals):
            processed[j]["q_value"] = _round(q)
    for r in processed:
        r.setdefault("q_value", None)

    def _rank_key(r: dict) -> tuple:
        icir = r["metrics"]["ic_ir"]
        if icir is None:
            return (1, 0.0)
        return (0, -abs(icir))

    processed.sort(key=_rank_key)
    finalists = processed[:top_k]

    # ─── Stage 2: deep pass on the top-K survivors ──────────────────────────
    leaderboard: list[dict] = []
    for j, r in enumerate(finalists, 1):
        cand = r["candidate"]
        m = r["metrics"]
        try:
            F = _pivot_factor(df, in_window, price_pivot.columns, cand)
            F_train = F.reindex(index=train_dates)
            F_holdout = F.reindex(index=holdout_dates)
            hm = cross_sectional_metrics(F_holdout, R_holdout, winsorize=None)
            holdout_ric = hm["rank_ic"]

            sign_flip = None
            if m["rank_ic"] is not None and holdout_ric is not None:
                sign_flip = bool(np.sign(m["rank_ic"]) != np.sign(holdout_ric))

            f_tr, r_tr = _flatten_valid(F_train, R_train)
            mi = None
            dcor = None
            if len(f_tr) >= 10:
                f_mi, r_mi = _subsample(f_tr, r_tr, MI_MAX_N)
                mi = float(mutual_info_regression(
                    f_mi.reshape(-1, 1), r_mi, random_state=SEED)[0])
                f_dc, r_dc = _subsample(f_tr, r_tr, DCOR_MAX_N)
                dcor = _distance_corr(f_dc, r_dc)

            leaderboard.append({
                "candidate": cand,
                "train_rank_ic": m["rank_ic"],
                "ic_ir": m["ic_ir"],
                "t_stat": m["t_stat"],
                "p_value": m["p_value"],
                "q_value": r["q_value"],
                "n_obs": m["n_obs"],
                "holdout_rank_ic": holdout_ric,
                "sign_flip": sign_flip,
                "mi": _round(mi),
                "dcor": _round(dcor),
            })
        except ValueError:
            skipped.append(cand)
        tick(f"stage 2: {j}/{top_k}")

    leaderboard.sort(key=lambda r: (r["ic_ir"] is None, -abs(r["ic_ir"] or 0.0)))

    return {
        "leaderboard": leaderboard,
        "n_candidates": n,
        "n_skipped": len(skipped),
        "skipped": skipped,
        "split": {
            "train_dates": int(len(train_dates)),
            "holdout_dates": int(len(holdout_dates)),
            "embargo_dates": int(horizon),
        },
        "meta": {
            "horizon": horizon,
            "from_date": from_date,
            "to_date": to_date,
            "top_k": top_k,
            "elapsed_s": round(time.time() - t0, 1),
        },
    }
