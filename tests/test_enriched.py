import io

import pandas as pd
import pytest

import factor_bank.data.enriched as enriched_mod
from factor_bank.data.enriched import COLUMN_GROUPS, load_enriched


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setenv("FB_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FB_S3_ENRICHED_PATHS", "r1:bucket/enriched.parquet")
    enriched_mod.clear_memo()
    monkeypatch.setattr(
        "factor_bank.data.enriched.get_permaticker_map", lambda fs=None: {"AAA": 1, "BBB": 2}
    )


def _frame(rows):
    return pd.DataFrame(rows, columns=["ticker", "date", "sector", "prev_close_price", "pe"])


class OneFrameFS:
    def __init__(self, df):
        self._df = df

    def open(self, path):
        buf = io.BytesIO()
        self._df.to_parquet(buf)
        buf.seek(0)
        return buf

    def info(self, path):
        return {"ETag": "e1"}


def test_load_applies_floor_perma_and_column_defense():
    df = _frame([
        ["AAA", "2017-06-01", "Tech", 10.0, 5.0],   # below floor → dropped
        ["AAA", "2019-06-03", "Tech", 11.0, 6.0],
        ["BBB", "2019-06-03", "Energy", 20.0, 7.0],
    ])
    out = load_enriched(fs_factory=lambda region: OneFrameFS(df))
    assert out["date"].min() >= pd.Timestamp("2018-01-01")
    assert out["permaticker"].tolist() == [1, 2]
    # requested columns absent from the parquet (e.g. zack_vgm) are simply not present
    assert "zack_vgm" not in out.columns


def test_duplicate_ticker_date_hard_fails():
    df = _frame([
        ["AAA", "2019-06-03", "Tech", 11.0, 6.0],
        ["AAA", "2019-06-03", "Tech", 11.0, 6.0],
    ])
    with pytest.raises(ValueError, match="duplicate"):
        load_enriched(fs_factory=lambda region: OneFrameFS(df))


def test_fallback_to_second_path(monkeypatch):
    monkeypatch.setenv(
        "FB_S3_ENRICHED_PATHS", "bad:bucket/x.parquet,r2:bucket/y.parquet"
    )
    good = _frame([["AAA", "2019-06-03", "Tech", 11.0, 6.0]])

    class FailFS:
        def open(self, path):
            raise OSError("nope")

        def info(self, path):
            raise OSError("nope")

    def factory(region):
        return FailFS() if region == "bad" else OneFrameFS(good)

    out = load_enriched(fs_factory=factory)
    assert len(out) == 1


def test_column_groups_shape():
    assert COLUMN_GROUPS["daily"] == ["pe", "pb", "ps", "evebit", "evebitda", "ev", "marketcap"]
    assert "ibd_string" in COLUMN_GROUPS and "returns" in COLUMN_GROUPS
