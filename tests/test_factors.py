import numpy as np
import pandas as pd
import pytest

from factor_bank.engine.catalog import FACTOR_CATALOG, all_factor_names, numeric_base_factors
from factor_bank.engine.factors import TRANSFORMS, compute_factor


@pytest.fixture
def df():
    n = 300
    dates = pd.bdate_range("2019-01-01", periods=n)
    frames = []
    rng = np.random.default_rng(7)
    for i, t in enumerate(["AAA", "BBB"]):
        frames.append(pd.DataFrame({
            "ticker": t,
            "date": dates,
            "sector": "Tech" if i == 0 else "Energy",
            "pe": rng.normal(20 + i * 10, 2, n),
            "evebitda": rng.normal(12, 1, n),
            "marketcap": np.full(n, 1e9 * (i + 1)),
            "ev": np.full(n, 1.5e9 * (i + 1)),
            "ibd_trend": "BULL",
            "prev_close_price": 100.0 + np.arange(n) * (0.1 + i * 0.1),
        }))
    return pd.concat(frames, ignore_index=True).sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


def test_catalog_integrity():
    names = all_factor_names()
    assert len(names) == len(set(names)), "duplicate factor names"
    assert "pe" in names and "evebitda_sector_relative" in names
    assert set(FACTOR_CATALOG) >= {"Value", "Technicals", "Seeking Alpha", "Value × TS"}
    numeric = numeric_base_factors()
    assert "ibd_trend_ord" not in numeric and "pe" in numeric


def test_passthrough_and_yield(df):
    pe = compute_factor(df, "pe")
    pd.testing.assert_series_equal(pe, df["pe"].astype(float), check_names=False)
    ey = compute_factor(df, "earnings_yield")
    pd.testing.assert_series_equal(ey, 1.0 / df["pe"], check_names=False)


def test_ibd_ordinal(df):
    assert (compute_factor(df, "ibd_trend_ord") == 2.0).all()


def test_zscore_transform_matches_pandas_reference(df):
    got = compute_factor(df, "pe__zscore_63d")
    one = df[df["ticker"] == "AAA"]["pe"]
    mean = one.rolling(63, min_periods=63).mean()
    std = one.rolling(63, min_periods=63).std(ddof=0)
    expected = (one - mean) / std
    pd.testing.assert_series_equal(
        got[df["ticker"] == "AAA"], expected, check_names=False
    )
    assert got[df["ticker"] == "AAA"].head(62).isna().all()  # min_periods guard


def test_chg_transform(df):
    got = compute_factor(df, "pe__chg_5d")
    one = df[df["ticker"] == "AAA"]["pe"]
    pd.testing.assert_series_equal(
        got[df["ticker"] == "AAA"], one - one.shift(5), check_names=False
    )


def test_sector_rel_transform(df):
    got = compute_factor(df, "pe__sector_rel")
    # one ticker per sector here → factor minus its own sector median = 0
    assert np.allclose(got.dropna(), 0.0)


@pytest.mark.parametrize("name,missing_col", [
    ("earnings_yield", "pe"),
    ("book_yield", "pb"),
    ("sales_yield", "ps"),
    ("ebitda_yield", "evebitda"),
    ("ev_to_mcap", "ev"),
    ("log_marketcap", "marketcap"),
    ("pe_zscore_252d", "pe"),
    ("ebitda_yield_zscore_252d", "evebitda"),
    ("pe_change_30d", "pe"),
    ("pb_percentile_252d", "pb"),
    ("evebitda_sector_relative", "evebitda"),
])
def test_derived_factor_missing_column_raises_valueerror_not_keyerror(df, name, missing_col):
    """F5/schema-drift hardening: a derived factor whose underlying column
    has disappeared from the enriched frame (upstream drift, or
    disk_cache._select silently dropping it) must raise a clean ValueError —
    the same exception type/handling every other factor branch already uses
    — not a raw KeyError that callers up the stack don't expect."""
    d = df.drop(columns=[missing_col], errors="ignore")  # already absent from the fixture for some cases
    with pytest.raises(ValueError, match=f"'{missing_col}'"):
        compute_factor(d, name)


def test_generated_name_errors():
    d = pd.DataFrame({"ticker": ["A"], "date": [pd.Timestamp("2019-01-02")],
                      "sector": ["T"], "pe": [1.0]})
    with pytest.raises(ValueError, match="Unknown transform"):
        compute_factor(d, "pe__bogus")
    with pytest.raises(ValueError, match="Unknown factor"):
        compute_factor(d, "nope")


def test_transform_registry_exact_keys():
    assert set(TRANSFORMS) == {
        "zscore_63d", "zscore_126d", "zscore_252d",
        "chg_5d", "chg_21d", "chg_63d", "pctile_252d", "sector_rel",
    }
