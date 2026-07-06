import bisect

import pandas as pd
import pytest

from factor_bank.data.universe import build_spells, filter_to_sp500


@pytest.fixture
def events():
    # A: added, removed, re-added (tests multi-spell)
    # B: added and never removed (open spell)
    # C: removed without add (pre-history noise — must be ignored)
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2018-01-02", "2018-01-02", "2019-06-01", "2020-03-01", "2018-05-01",
        ]),
        "action": ["added", "added", "removed", "added", "removed"],
        "ticker": ["A", "B", "A", "A", "C"],
    })


def naive_members_on(events: pd.DataFrame, d: pd.Timestamp) -> frozenset:
    """Reference semantics: walk the event log (port of the Phase 2 implementation)."""
    ev = events.sort_values("date")
    current: set = set()
    out: dict = {}
    for _, row in ev.iterrows():
        a = str(row["action"]).lower()
        if a == "added":
            current.add(row["ticker"])
        elif a == "removed":
            current.discard(row["ticker"])
        out[row["date"]] = frozenset(current)
    keys = sorted(out)
    i = bisect.bisect_right(keys, d) - 1
    return out[keys[i]] if i >= 0 else frozenset()


def test_spells_match_naive_reference(events):
    spells = build_spells(events)
    dates = pd.date_range("2017-12-01", "2021-01-01", freq="7D")
    tickers = ["A", "B", "C"]
    for d in dates:
        expected = naive_members_on(events, d)
        got = frozenset(
            spells[(spells["start"] <= d) & (d < spells["end"])]["ticker"]
        )
        assert got == expected & frozenset(tickers), f"mismatch on {d}"


def test_filter_to_sp500_interval_merge(events):
    spells = build_spells(events)
    df = pd.DataFrame({
        "ticker": ["A", "A", "A", "B", "C"],
        "date": pd.to_datetime(
            ["2018-06-01", "2019-06-01", "2020-06-01", "2019-01-01", "2019-01-01"]
        ),
        "val": [1, 2, 3, 4, 5],
    })
    out = filter_to_sp500(df, spells)
    # A on 2019-06-01 is its removal date → excluded (end-exclusive)
    assert out["val"].tolist() == [1, 3, 4]
    assert list(out.columns) == ["ticker", "date", "val"]


def test_filter_empty_when_no_members(events):
    spells = build_spells(events)
    df = pd.DataFrame({
        "ticker": ["Z"], "date": [pd.Timestamp("2019-01-01")], "val": [1],
    })
    assert filter_to_sp500(df, spells).empty
