"""Nexus enriched_stocks.parquet loader — multi-region fallback, floor, permaticker."""
from __future__ import annotations

import logging

import pandas as pd

from factor_bank.config import get_settings
from factor_bank.data.disk_cache import cached_parquet
from factor_bank.data.sharadar import get_permaticker_map, make_fs

logger = logging.getLogger(__name__)

_memo: dict = {}


def clear_memo() -> None:
    _memo.clear()


COLUMN_GROUPS: dict[str, list[str]] = {
    "daily": ["pe", "pb", "ps", "evebit", "evebitda", "ev", "marketcap"],
    "sa": [
        "sa_quant_rating", "sa_wall_street_rating", "sa_analyst_rating",
        "sa_momentum", "sa_revisions", "sa_value", "sa_growth", "sa_profitability",
    ],
    "zacks": ["zack_rating", "zack_value", "zack_growth", "zack_momentum", "zack_vgm"],
    "ibd_numeric": ["ibd_rs_rank"],
    "ibd_string": ["ibd_trend", "ibd_state", "ibd_risk"],
    "technical": [
        "beta", "kalman_beta", "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_pct", "bb_width", "atr_pct", "stoch_k", "stoch_d", "adx_14",
        "volume_ratio", "dist_sma_50", "dist_sma_200",
        "pct_from_52w_high", "pct_from_52w_low",
    ],
    "returns": [
        "ret_5d", "ret_10d", "ret_20d", "ret_60d", "ret_120d", "ret_252d",
        "volatility_20d", "volatility_60d",
    ],
}

BASE_COLS = ["ticker", "date", "sector", "prev_close_price"]


def _all_wanted_cols() -> list[str]:
    return BASE_COLS + [c for group in COLUMN_GROUPS.values() for c in group]


def load_enriched(fs_factory=None) -> pd.DataFrame:
    if "enriched" in _memo:
        return _memo["enriched"]
    s = get_settings()
    factory = fs_factory or (lambda region: make_fs(region))

    df = None
    for entry in s.enriched_paths:
        region, _, path = entry.partition(":")
        try:
            df = cached_parquet(factory(region), path, s.cache_dir, columns=_all_wanted_cols())
            logger.info("Loaded enriched from %s: %d rows", entry, len(df))
            break
        except Exception as e:
            logger.warning("enriched load failed from %s: %s", entry, e)
    if df is None:
        raise RuntimeError(f"Could not load enriched_stocks from any of {s.enriched_paths}")

    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.Timestamp(s.date_floor)].reset_index(drop=True)

    n_dup = int(df.duplicated(["ticker", "date"]).sum())
    if n_dup:
        raise ValueError(f"enriched_stocks has {n_dup} duplicate (ticker, date) rows")

    df["permaticker"] = df["ticker"].map(get_permaticker_map())
    _memo["enriched"] = df
    return df
