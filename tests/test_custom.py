import pandas as pd
import pytest

from factor_bank.data import custom
from factor_bank.engine.factors import compute_factor


@pytest.fixture(autouse=True)
def _tmp_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("FB_CACHE_DIR", str(tmp_path))
    custom.clear_memo()
    yield
    custom.clear_memo()


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB", "BBB"],
        "date": ["2019-01-02", "2019-01-03", "2019-01-02", "2019-01-03"],
        "value": [1.5, 2.5, -0.5, 0.25],
    })


# ─── Valid roundtrip ─────────────────────────────────────────────────────────

def test_valid_roundtrip():
    result = custom.validate_and_store("my_sig", _csv_bytes(_valid_df()))
    assert result["name"] == "my_sig"
    assert result["n_rows"] == 4
    assert result["n_tickers"] == 2
    assert result["date_min"] == "2019-01-02"
    assert result["date_max"] == "2019-01-03"

    assert "my_sig" in custom.custom_names()

    loaded = custom.load_custom("my_sig")
    assert len(loaded) == 4
    row = loaded[(loaded["ticker"] == "AAA") & (loaded["date"] == pd.Timestamp("2019-01-03"))]
    assert row["value"].iloc[0] == 2.5

    # Second load hits the memo (same object).
    assert custom.load_custom("my_sig") is loaded


def test_delete_roundtrip():
    custom.validate_and_store("my_sig", _csv_bytes(_valid_df()))
    assert custom.delete_custom_factor("my_sig") is True
    assert "my_sig" not in custom.custom_names()
    assert custom.delete_custom_factor("my_sig") is False


def test_load_unknown_raises():
    with pytest.raises(ValueError, match="Unknown custom factor"):
        custom.load_custom("nope")


# ─── Validation failures ────────────────────────────────────────────────────

def test_bad_name_rejected():
    with pytest.raises(ValueError, match="name must match"):
        custom.validate_and_store("Bad-Name", _csv_bytes(_valid_df()))
    with pytest.raises(ValueError, match="name must match"):
        custom.validate_and_store("1starts_with_digit", _csv_bytes(_valid_df()))
    with pytest.raises(ValueError, match="name must match"):
        custom.validate_and_store("a" * 41, _csv_bytes(_valid_df()))


def test_double_underscore_name_rejected():
    with pytest.raises(ValueError, match="reserved for transforms"):
        custom.validate_and_store("ab__cd", _csv_bytes(_valid_df()))


def test_catalog_collision_rejected():
    with pytest.raises(ValueError, match="collides with catalog factor"):
        custom.validate_and_store("pe", _csv_bytes(_valid_df()))


def test_too_large_rejected(monkeypatch):
    monkeypatch.setattr(custom, "MAX_BYTES", 10)
    with pytest.raises(ValueError, match="20 MB limit"):
        custom.validate_and_store("my_sig", _csv_bytes(_valid_df()))


def test_not_csv_rejected():
    # Ragged row widths trip the C parser's tokenizer -> pandas.errors.ParserError.
    bad_bytes = b"ticker,date,value\nAAA,2019-01-02\nBBB,2019-01-03,1,2,3\n"
    with pytest.raises(ValueError, match="not parseable as CSV"):
        custom.validate_and_store("my_sig", bad_bytes)


def test_wrong_columns_rejected():
    bad = pd.DataFrame({"symbol": ["AAA"], "date": ["2019-01-02"], "value": [1.0]})
    with pytest.raises(ValueError, match="columns must be exactly"):
        custom.validate_and_store("my_sig", _csv_bytes(bad))


def test_empty_csv_rejected():
    bad = pd.DataFrame({"ticker": [], "date": [], "value": []})
    with pytest.raises(ValueError, match="no rows"):
        custom.validate_and_store("my_sig", _csv_bytes(bad))


def test_bad_dates_rejected():
    bad = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "date": ["2019-01-02", "not-a-date-at-all-oops"],
        "value": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="unparseable dates"):
        custom.validate_and_store("my_sig", _csv_bytes(bad))


def test_non_numeric_values_rejected():
    bad = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "date": ["2019-01-02", "2019-01-03"],
        "value": [1.0, "not-a-number"],
    })
    with pytest.raises(ValueError, match="non-numeric values"):
        custom.validate_and_store("my_sig", _csv_bytes(bad))


def test_duplicate_rows_rejected():
    bad = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "date": ["2019-01-02", "2019-01-02"],
        "value": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="duplicate"):
        custom.validate_and_store("my_sig", _csv_bytes(bad))


# ─── compute_factor merge correctness ───────────────────────────────────────

def test_compute_factor_merges_custom_values(synthetic_market):
    enriched, _ = synthetic_market
    df = enriched.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Pick 2 tickers x 2 known dates present in the fixture.
    sub = df[df["ticker"].isin(["T00", "T01"])].iloc[[0, 1]]
    up = pd.DataFrame({
        "ticker": sub["ticker"].tolist(),
        "date": sub["date"].dt.strftime("%Y-%m-%d").tolist(),
        "value": [42.0, -7.0],
    })
    custom.validate_and_store("custom_probe", _csv_bytes(up))

    result = compute_factor(df, "custom_probe")
    assert len(result) == len(df)
    assert result.index.equals(df.index)

    t0_row = sub.iloc[0]
    t1_row = sub.iloc[1]
    idx0 = df[(df["ticker"] == t0_row["ticker"]) & (df["date"] == t0_row["date"])].index[0]
    idx1 = df[(df["ticker"] == t1_row["ticker"]) & (df["date"] == t1_row["date"])].index[0]
    assert result.loc[idx0] == 42.0
    assert result.loc[idx1] == -7.0

    # Everywhere else -> NaN (only 2 rows out of the whole fixture have values).
    assert result.isna().sum() == len(df) - 2


def test_generated_name_over_custom_base(synthetic_market):
    enriched, _ = synthetic_market
    df = enriched.sort_values(["ticker", "date"]).reset_index(drop=True)
    up = pd.DataFrame({
        "ticker": df["ticker"],
        "date": df["date"].dt.strftime("%Y-%m-%d"),
        "value": df["pe"],
    })
    custom.validate_and_store("custom_pe2", _csv_bytes(up))

    base = compute_factor(df, "custom_pe2")
    transformed = compute_factor(df, "custom_pe2__chg_21d")
    # chg_21d(base) recomputed directly should match the generated-name path.
    expected = base.groupby(df["ticker"]).transform(lambda x: x - x.shift(21))
    pd.testing.assert_series_equal(transformed, expected, check_names=False)


# ─── End-to-end evaluate() ───────────────────────────────────────────────────

def test_evaluate_custom_factor_reproduces_pe_verdict(synthetic_market):
    from factor_bank.engine.evaluate import evaluate

    enriched, spells = synthetic_market
    up = pd.DataFrame({
        "ticker": enriched["ticker"],
        "date": enriched["date"].dt.strftime("%Y-%m-%d"),
        "value": enriched["pe"],
    })
    custom.validate_and_store("custom_pe", _csv_bytes(up))

    out = evaluate(
        "custom_pe", "2019-01-01", "2020-01-01", horizon=21, n_quantiles=5,
        enriched=enriched, spells=spells,
    )
    assert out["verdict"] == "STRONG"
    assert out["metrics"]["rank_ic"] > 0.9
    assert out["quality"]["coverage"] == 1.0
