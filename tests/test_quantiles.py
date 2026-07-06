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
    out = quantile_spread(F, R, 5)
    means = [q["mean_return"] for q in out["quantile_means"]]
    assert means == sorted(means)          # perfectly monotone
    assert out["longshort_stats"]["total_return"] > 0
    assert len(out["longshort_cumulative"]) == n_dates
    assert out["longshort_stats"]["n_days"] == n_dates


def test_sparse_dates_are_masked():
    dates = pd.bdate_range("2019-01-01", periods=3)
    F = pd.DataFrame(np.random.default_rng(0).normal(size=(3, 4)),
                     index=dates, columns=list("ABCD"))
    R = F.copy()
    out = quantile_spread(F, R, 5)  # 4 tickers < 5*2 → every date too sparse
    assert out["longshort_cumulative"] == []
    assert out["longshort_stats"] is None
