"""Custom factor values: CSV validation, parquet storage, lookup."""
from __future__ import annotations

import io
import re

import pandas as pd

from factor_bank.config import get_settings
from factor_bank.data import store

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
MAX_BYTES = 20 * 1024 * 1024

_memo: dict[str, pd.DataFrame] = {}


def clear_memo() -> None:
    _memo.clear()


def validate_and_store(name: str, raw: bytes) -> dict:
    """Validate CSV bytes; on success write parquet + registry row.

    Raises ValueError with a user-facing message on any problem.
    """
    from factor_bank.engine.catalog import all_factor_names

    if not NAME_RE.match(name):
        raise ValueError("name must match ^[a-z][a-z0-9_]{0,39}$")
    if "__" in name:
        raise ValueError("name must not contain '__' (reserved for transforms)")
    if name in all_factor_names():
        raise ValueError(f"name collides with catalog factor '{name}'")
    if store.get_custom(name) is not None:
        raise ValueError(f"custom factor '{name}' already exists — delete it first")
    if len(raw) > MAX_BYTES:
        raise ValueError("file exceeds 20 MB limit")

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(f"not parseable as CSV: {e}") from e
    if list(df.columns) != ["ticker", "date", "value"]:
        raise ValueError(f"columns must be exactly ticker,date,value — got {list(df.columns)}")
    if df.empty:
        raise ValueError("CSV has no rows")
    try:
        # .dt.normalize() drops any time-of-day component so intraday
        # timestamps (e.g. "2019-06-03 15:30:00") align on exact date equality
        # against midnight-normalized enriched dates at merge time (M-1) —
        # otherwise validation passes but every merged value comes back NaN.
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    except Exception as e:
        raise ValueError(f"unparseable dates: {e}") from e
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    n_bad = int(df["value"].isna().sum())
    if n_bad:
        raise ValueError(f"{n_bad} non-numeric values")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    n_dup = int(df.duplicated(["ticker", "date"]).sum())
    if n_dup:
        raise ValueError(f"{n_dup} duplicate (ticker, date) rows")

    out_dir = get_settings().cache_dir / "custom_factors"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    df.to_parquet(path)
    store.register_custom(
        name, str(path), len(df), int(df["ticker"].nunique()),
        str(df["date"].min().date()), str(df["date"].max().date()),
    )
    _memo.pop(name, None)
    return {
        "name": name, "n_rows": len(df), "n_tickers": int(df["ticker"].nunique()),
        "date_min": str(df["date"].min().date()), "date_max": str(df["date"].max().date()),
    }


def custom_names() -> list[str]:
    return [r["name"] for r in store.list_custom()]


def load_custom(name: str) -> pd.DataFrame:
    if name in _memo:
        return _memo[name]
    rec = store.get_custom(name)
    if rec is None:
        raise ValueError(f"Unknown custom factor: {name}")
    df = pd.read_parquet(rec["path"])
    df["date"] = pd.to_datetime(df["date"])
    _memo[name] = df
    return df


def delete_custom_factor(name: str) -> bool:
    rec = store.get_custom(name)
    if rec is None:
        return False
    from pathlib import Path
    Path(rec["path"]).unlink(missing_ok=True)
    _memo.pop(name, None)
    return store.delete_custom(name)
