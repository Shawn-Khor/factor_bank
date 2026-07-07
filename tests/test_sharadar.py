import numpy as np
import pandas as pd

from factor_bank.data.sharadar import _snapshot_parquet_path, ticker_to_permaticker


class _FakeFS:
    """Minimal fs stand-in for `_snapshot_parquet_path`: `ls` lists files
    under a prefix, `info` optionally returns `LastModified`."""

    def __init__(self, files, last_modified=None, info_raises=False):
        self.files = files
        self.last_modified = last_modified or {}
        self.info_raises = info_raises
        self.info_calls = 0

    def ls(self, prefix):
        return list(self.files)

    def info(self, path):
        self.info_calls += 1
        if self.info_raises:
            raise OSError("no LastModified support on this backend")
        return {"LastModified": self.last_modified[path]}


def test_snapshot_prefers_newest_last_modified_over_lexicographic_order():
    # Lexicographically "a_old" sorts before "b_new", but b_new was written
    # more recently -> LastModified must win, not filename order.
    fs = _FakeFS(
        ["p/a_old.parquet", "p/b_new.parquet"],
        last_modified={"p/a_old.parquet": 100, "p/b_new.parquet": 200},
    )
    assert _snapshot_parquet_path(fs, "p/") == "p/b_new.parquet"


def test_snapshot_falls_back_to_sorted_last_when_info_unavailable():
    fs = _FakeFS(["p/b.parquet", "p/a.parquet"], info_raises=True)
    # Previous behavior (`sorted(files)[0]`) would have picked "p/a.parquet"
    # (the oldest by lexicographic order) — the fixed fallback picks the
    # lexicographically *last* path instead.
    assert _snapshot_parquet_path(fs, "p/") == "p/b.parquet"


def test_snapshot_single_candidate_skips_info_call():
    fs = _FakeFS(["p/only.parquet"], info_raises=True)  # would raise if called
    assert _snapshot_parquet_path(fs, "p/") == "p/only.parquet"
    assert fs.info_calls == 0


def test_snapshot_no_files_raises():
    import pytest

    fs = _FakeFS([])
    with pytest.raises(FileNotFoundError):
        _snapshot_parquet_path(fs, "p/")


def test_permaticker_basic_and_related():
    df = pd.DataFrame({
        "table": ["SEP", "SEP", "SEP"],
        "ticker": ["META", "AAPL", "BADROW"],
        "permaticker": [111, 222, np.nan],
        "relatedtickers": ["FB", None, "X"],
    })
    m = ticker_to_permaticker(df)
    assert m["META"] == 111
    assert m["FB"] == 111        # historical alias maps to same id
    assert m["AAPL"] == 222
    assert "BADROW" not in m     # NaN permaticker dropped
    assert "X" not in m          # alias of dropped row not mapped


def test_permaticker_alias_does_not_clobber_primary():
    df = pd.DataFrame({
        "table": ["SEP", "SEP"],
        "ticker": ["A", "B"],
        "permaticker": [1, 2],
        "relatedtickers": [None, "A"],  # B lists A as related — A's own id wins
    })
    m = ticker_to_permaticker(df)
    assert m["A"] == 1
    assert m["B"] == 2


def test_permaticker_filters_non_sep_when_table_present():
    df = pd.DataFrame({
        "table": ["SFP", "SEP"],
        "ticker": ["FUND", "STK"],
        "permaticker": [9, 10],
        "relatedtickers": [None, None],
    })
    m = ticker_to_permaticker(df)
    assert "FUND" not in m and m["STK"] == 10
