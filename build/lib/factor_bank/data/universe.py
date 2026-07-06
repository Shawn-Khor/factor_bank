"""Historical S&P 500 membership as interval spells + vectorized filtering."""
from __future__ import annotations

import logging

import pandas as pd

from factor_bank.data.sharadar import load_sp500_events

logger = logging.getLogger(__name__)

_memo: dict = {}


def clear_memo() -> None:
    _memo.clear()


def build_spells(events: pd.DataFrame) -> pd.DataFrame:
    """Walk the add/remove event log once into (ticker, start, end) spells.

    Member from start (inclusive) to end (exclusive) — matches the Phase 2
    bisect semantics where a removal takes effect on its own date.
    """
    ev = events.sort_values("date")
    open_spells: dict[str, pd.Timestamp] = {}
    rows: list[tuple] = []
    for date, action, ticker in ev[["date", "action", "ticker"]].itertuples(index=False):
        a = str(action).lower()
        if a == "added":
            open_spells.setdefault(ticker, date)
        elif a == "removed" and ticker in open_spells:
            rows.append((ticker, open_spells.pop(ticker), date))
    rows.extend((t, s, pd.Timestamp.max) for t, s in open_spells.items())
    spells = pd.DataFrame(rows, columns=["ticker", "start", "end"])
    logger.info("Built %d membership spells for %d tickers",
                len(spells), spells["ticker"].nunique())
    return spells


def filter_to_sp500(df: pd.DataFrame, spells: pd.DataFrame) -> pd.DataFrame:
    """One interval merge instead of a per-date Python loop."""
    merged = df.merge(spells, on="ticker", how="inner")
    mask = (merged["start"] <= merged["date"]) & (merged["date"] < merged["end"])
    return (
        merged[mask]
        .drop(columns=["start", "end"])
        .reset_index(drop=True)
    )


def get_spells(fs=None) -> pd.DataFrame:
    if "spells" not in _memo:
        _memo["spells"] = build_spells(load_sp500_events(fs))
    return _memo["spells"]
