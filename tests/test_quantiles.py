import numpy as np
import pandas as pd

from factor_bank.engine.quantiles import quantile_spread


def test_monotone_signal_spread():
    """Factor = fwd return level per ticker → Q5 mean > Q1 mean, positive L/S."""
    n_dates, n_tickers = 80, 25
    dates = pd.bdate_range("2019-01-01", periods=n_dates)
    cols = [f"T{i}" for i in range(n_tickers)]
    levels = np.linspace(-0.01, 0.01, n_tickers)
    R = pd.DataFrame(np.tile(levels, (n_dates, 1)), index=dates, columns=cols)
    F = R.copy()
    out = quantile_spread(F, R, 5, horizon=1)
    means = [q["mean_return"] for q in out["quantile_means"]]
    assert means == sorted(means)          # perfectly monotone
    assert out["longshort_stats"]["total_return"] > 0
    assert len(out["longshort_cumulative"]) == n_dates
    assert out["longshort_stats"]["n_days"] == n_dates
    assert out["longshort_stats"]["n_independent"] == n_dates  # horizon=1 -> no overlap


def test_sparse_dates_are_masked():
    dates = pd.bdate_range("2019-01-01", periods=3)
    F = pd.DataFrame(np.random.default_rng(0).normal(size=(3, 4)),
                     index=dates, columns=list("ABCD"))
    R = F.copy()
    out = quantile_spread(F, R, 5, horizon=1)  # 4 tickers < 5*2 → every date too sparse
    assert out["longshort_cumulative"] == []
    assert out["longshort_stats"] is None


def test_buckets_partition_jointly_valid_names_only():
    """F4: ranking must happen on F.where(valid), not F alone — otherwise
    tickers with a missing forward return still consume a bucket slot and
    the top bucket(s) can end up with zero real observations whenever return
    availability correlates with factor level (e.g. index deletions)."""
    n_dates = 5
    dates = pd.bdate_range("2019-01-01", periods=n_dates)
    valid_cols = [f"V{i}" for i in range(10)]
    invalid_cols = [f"X{i}" for i in range(10)]  # highest factor values, R always NaN
    cols = valid_cols + invalid_cols
    factor_vals = np.arange(1, 21, dtype=float)  # 1..10 valid range, 11..20 invalid range
    F = pd.DataFrame(np.tile(factor_vals, (n_dates, 1)), index=dates, columns=cols)
    R = F.copy() / 100.0
    R[invalid_cols] = np.nan

    out = quantile_spread(F, R, 5, horizon=1)
    top_bucket = out["quantile_means"][-1]
    # Under the old "rank on F alone" behavior, the top-factor-value bucket
    # is entirely inside the invalid (R == NaN) group -> n_obs == 0.
    assert top_bucket["n_obs"] > 0
    assert top_bucket["mean_return"] is not None


def test_horizon_deoverlap_matches_analytic_and_beats_naive_compounding():
    """F1 (HIGH): overlapping horizon-session L/S returns must be converted
    to a per-session-equivalent daily rate before compounding, not compounded
    as if each horizon-session return were earned in a single day."""
    n_dates, n_tickers, horizon = 300, 30, 21
    dates = pd.bdate_range("2019-01-01", periods=n_dates)
    cols = [f"T{i}" for i in range(n_tickers)]
    levels = np.linspace(-0.008, 0.008, n_tickers)
    # Deterministic, constant-across-dates spread (same construction as
    # test_monotone_signal_spread) so the analytic answer is exact.
    R = pd.DataFrame(np.tile(levels, (n_dates, 1)), index=dates, columns=cols)
    F = R.copy()
    out = quantile_spread(F, R, 5, horizon=horizon)

    n_per_bucket = n_tickers // 5
    rho = levels[-n_per_bucket:].mean() - levels[:n_per_bucket].mean()  # Q5 - Q1, per date

    r_d = (1.0 + rho) ** (1.0 / horizon) - 1.0
    analytic_total = (1.0 + r_d) ** n_dates - 1.0
    naive_total = (1.0 + rho) ** n_dates - 1.0

    reported = out["longshort_stats"]["total_return"]
    assert abs(reported - analytic_total) < 1e-6
    assert reported < naive_total * 0.05          # far below naive daily compounding
    assert out["longshort_stats"]["n_days"] == n_dates
    assert out["longshort_stats"]["n_independent"] == len(range(0, n_dates, horizon))


def test_horizon_one_reduces_to_plain_daily_compounding():
    """horizon=1 (the default) must reproduce the pre-de-overlap numbers
    exactly — no overlap exists at horizon 1."""
    n_dates, n_tickers = 80, 25
    dates = pd.bdate_range("2019-01-01", periods=n_dates)
    cols = [f"T{i}" for i in range(n_tickers)]
    levels = np.linspace(-0.01, 0.01, n_tickers)
    R = pd.DataFrame(np.tile(levels, (n_dates, 1)), index=dates, columns=cols)
    F = R.copy()

    out_h1 = quantile_spread(F, R, 5, horizon=1)
    out_default = quantile_spread(F, R, 5)  # default horizon=1
    assert out_h1["longshort_stats"] == out_default["longshort_stats"]
    assert out_h1["longshort_cumulative"] == out_default["longshort_cumulative"]
