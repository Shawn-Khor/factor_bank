import numpy as np
import pandas as pd

from factor_bank.engine.quality import quality_report


def test_coverage_and_years():
    df = pd.DataFrame({
        "ticker": ["A"] * 4 + ["B"] * 4,
        "date": pd.to_datetime(["2019-01-02", "2019-01-03", "2020-01-02", "2020-01-03"] * 2),
        "_factor": [1.0, 2.0, np.nan, np.nan, 3.0, 4.0, 5.0, 6.0],
    })
    q = quality_report(df)
    assert q["coverage"] == 0.75
    assert q["coverage_by_year"] == {2019: 1.0, 2020: 0.5}
    assert q["duplicates"] == 0
    assert q["n_rows"] == 8


def test_staleness_flags_frozen_series():
    dates = pd.bdate_range("2019-01-01", periods=10)
    df = pd.DataFrame({
        "ticker": ["FROZEN"] * 10 + ["LIVE"] * 10,
        "date": list(dates) * 2,
        "_factor": [5.0] * 10 + list(np.arange(10.0)),
    })
    q = quality_report(df)
    # FROZEN: 9/9 consecutive diffs are zero; LIVE: 0/9 → mean 0.5
    assert abs(q["staleness_mean"] - 0.5) < 1e-9


def test_all_nan_staleness_none():
    df = pd.DataFrame({
        "ticker": ["A", "A"],
        "date": pd.to_datetime(["2019-01-02", "2019-01-03"]),
        "_factor": [np.nan, np.nan],
    })
    assert quality_report(df)["staleness_mean"] is None
