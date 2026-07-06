"""Memoized buffered+filtered data window shared by slow multi-factor runs.

Memoized frames are stored privately; get_window always hands callers a copy, so in-place mutation cannot poison the cache."""
from __future__ import annotations

import time

import pandas as pd

from factor_bank.data.enriched import load_enriched
from factor_bank.data.universe import filter_to_sp500, get_spells

_TTL_SECONDS = 3600.0
_memo: dict[tuple, tuple[float, pd.DataFrame]] = {}


def clear_memo() -> None:
    _memo.clear()


def get_window(
    from_date: str,
    to_date: str,
    max_horizon: int,
    *,
    enriched: pd.DataFrame | None = None,
    spells: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from_ts, to_ts = pd.Timestamp(from_date), pd.Timestamp(to_date)
    use_memo = enriched is None and spells is None
    key = (from_date, to_date, int(max_horizon))
    if use_memo:
        hit = _memo.get(key)
        if hit and time.time() - hit[0] < _TTL_SECONDS:
            return hit[1].copy()

    df_all = enriched if enriched is not None else load_enriched()
    sp = spells if spells is not None else get_spells()

    buf_start = from_ts - pd.Timedelta(days=400)
    buf_end = to_ts + pd.Timedelta(days=int(max_horizon * 1.6) + 7)
    df = df_all[(df_all["date"] >= buf_start) & (df_all["date"] <= buf_end)].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df = filter_to_sp500(df, sp)

    if use_memo:
        _memo[key] = (time.time(), df)
        return df.copy()
    return df
