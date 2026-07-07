import numpy as np
import pandas as pd

from factor_bank.engine.metrics import (
    cross_sectional_metrics,
    trading_session_forward_returns,
    verdict,
    winsorize_xs,
)


def _perfect_signal(n_dates=60, n_tickers=20, seed=0):
    """F = R + tiny noise → near-perfect IC. (Exact equality would make the
    daily IC series constant at 1.0, std=0, t_stat=None — degenerate.)"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_dates)
    cols = [f"T{i}" for i in range(n_tickers)]
    R = pd.DataFrame(rng.normal(0, 0.01, (n_dates, n_tickers)), index=dates, columns=cols)
    F = R + rng.normal(0, 1e-4, R.shape)
    return F, R


def test_perfect_signal_ic_near_one():
    F, R = _perfect_signal()
    m = cross_sectional_metrics(F, R)
    assert m["rank_ic"] > 0.99
    assert m["ic"] > 0.99
    assert m["pct_positive"] == 1.0
    assert m["p_value"] < 1e-10
    assert m["n_obs"] == 60


def test_noise_signal_ic_near_zero():
    rng = np.random.default_rng(1)
    F, R = _perfect_signal(seed=2)
    F.loc[:, :] = rng.normal(0, 1, F.shape)  # decorrelate
    m = cross_sectional_metrics(F, R)
    assert abs(m["rank_ic"]) < 0.05
    assert m["p_value"] > 0.01


def test_winsorize_tames_outlier_pearson():
    F, R = _perfect_signal(n_tickers=50)
    F.iloc[:, 0] = 1e9  # one absurd column wrecks raw Pearson
    m = cross_sectional_metrics(F, R, winsorize=0.05)
    assert m["ic"] > 0.9          # clipped Pearson recovers the signal
    assert abs(m["ic_raw"]) < 0.5  # raw Pearson is wrecked by the outlier
    assert m["ic"] > m["ic_raw"]


def test_winsorize_out_of_range_rejected():
    """F2: winsorize >= 0.5 crosses the lo/hi clip quantiles and silently
    corrupts Pearson IC via DataFrame.clip's snap-to-crossed-bounds behavior
    — must be rejected outright, not accepted and produce garbage."""
    F, R = _perfect_signal()
    for bad in (0.5, 0.9, 1.0, -0.01, -1.0):
        try:
            cross_sectional_metrics(F, R, winsorize=bad)
            raise AssertionError(f"winsorize={bad} should have raised ValueError")
        except ValueError as e:
            assert "winsorize" in str(e)
    # Boundary and disabled cases must NOT raise.
    cross_sectional_metrics(F, R, winsorize=0.0)
    cross_sectional_metrics(F, R, winsorize=0.499)
    cross_sectional_metrics(F, R, winsorize=None)


def test_winsorize_xs_row_wise():
    F = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 100.0]])
    W = winsorize_xs(F, p=0.2)
    assert W.iloc[0, -1] < 100.0 and W.iloc[0, 0] > 1.0 - 1e9


def test_verdict_thresholds():
    assert verdict({"t_stat": 3.5, "rank_ic": 0.03}) == "STRONG"
    assert verdict({"t_stat": 2.5, "rank_ic": 0.015}) == "MODERATE"
    assert verdict({"t_stat": 2.5, "rank_ic": 0.005}) == "WEAK"
    assert verdict({"t_stat": 0.5, "rank_ic": 0.001}) == "INSIGNIFICANT"
    assert verdict({"t_stat": None, "rank_ic": None}) == "INSIGNIFICANT"


def test_forward_returns_integer_shift():
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0]},
        index=pd.bdate_range("2019-01-01", periods=3),
    )
    fwd = trading_session_forward_returns(prices, 1)
    assert abs(fwd.iloc[0, 0] - 0.10) < 1e-12
    assert pd.isna(fwd.iloc[2, 0])  # no future row at the end
