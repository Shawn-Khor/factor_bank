"""Sharadar S3 loaders: SP500 membership events, TICKERS metadata, permaticker map."""
from __future__ import annotations

import logging

import pandas as pd

from factor_bank.config import get_settings
from factor_bank.data.disk_cache import cached_parquet

logger = logging.getLogger(__name__)

_memo: dict = {}


def clear_memo() -> None:
    _memo.clear()


def make_fs(region: str | None = None):
    import s3fs

    s = get_settings()
    return s3fs.S3FileSystem(
        key=s.aws_access_key_id,
        secret=s.aws_secret_access_key,
        client_kwargs={"region_name": region or s.sharadar_region},
    )


def _snapshot_parquet_path(fs, prefix: str) -> str:
    """Resolve the current file under a `latest_snapshot/` prefix.

    Today this prefix holds exactly one file, so any tie-break rule is a
    no-op in practice. But nothing upstream guarantees that stays true, so:
    when there's more than one candidate, prefer whichever has the newest
    `fs.info(f)["LastModified"]` (the file actually most recently written) —
    guarded by try/except since not every S3-compatible backend returns
    `LastModified`, and a network hiccup shouldn't be fatal here. If there's
    only one candidate, or the LastModified lookup fails for any reason,
    fall back to `sorted(files)[-1]` — the lexicographically *last* (newest,
    for date-stamped filenames) path, which is deterministic and doesn't
    require any extra calls. (The previous fallback, `sorted(files)[0]`, was
    a bug: it picked the lexicographically *first* — i.e. oldest — file.)
    """
    files = [x for x in fs.ls(prefix) if x.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet under {prefix}")
    if len(files) > 1:
        try:
            dated = [(fs.info(f)["LastModified"], f) for f in files]
            return max(dated, key=lambda t: t[0])[1]
        except Exception as e:
            logger.warning(
                "_snapshot_parquet_path: LastModified lookup failed (%s); "
                "falling back to sorted(files)[-1] for %s", e, prefix,
            )
    return sorted(files)[-1]


def load_sp500_events(fs=None) -> pd.DataFrame:
    if "sp500" in _memo:
        return _memo["sp500"]
    s = get_settings()
    fs = fs or make_fs()
    path = _snapshot_parquet_path(fs, f"{s.sharadar_bucket}/sharadar/SP500/latest_snapshot/")
    df = cached_parquet(fs, path, s.cache_dir)
    df["date"] = pd.to_datetime(df["date"])
    _memo["sp500"] = df
    logger.info("Loaded SP500 events: %d rows", len(df))
    return df


def load_tickers(fs=None) -> pd.DataFrame:
    if "tickers" in _memo:
        return _memo["tickers"]
    s = get_settings()
    fs = fs or make_fs()
    path = _snapshot_parquet_path(fs, f"{s.sharadar_bucket}/sharadar/TICKERS/latest_snapshot/")
    df = cached_parquet(fs, path, s.cache_dir)
    _memo["tickers"] = df
    logger.info("Loaded TICKERS: %d rows", len(df))
    return df


def ticker_to_permaticker(tickers_df: pd.DataFrame) -> dict[str, int]:
    """Vectorized symbol → permaticker map, including historical aliases."""
    df = tickers_df
    if "table" in df.columns:
        df = df[df["table"] == "SEP"]
    df = df.dropna(subset=["permaticker"])
    df = df[df["ticker"].astype(str).str.strip() != ""]

    out: dict[str, int] = dict(
        zip(df["ticker"].astype(str).str.strip(), df["permaticker"].astype(int))
    )
    rel = df[["permaticker", "relatedtickers"]].dropna(subset=["relatedtickers"])
    rel = rel[rel["relatedtickers"].astype(str).str.strip() != ""]
    if not rel.empty:
        exploded = rel.assign(
            alias=rel["relatedtickers"].astype(str).str.split()
        ).explode("alias")
        for alias, pt in zip(exploded["alias"], exploded["permaticker"].astype(int)):
            out.setdefault(alias.strip(), pt)
    return out


def get_permaticker_map(fs=None) -> dict[str, int]:
    if "perma" not in _memo:
        _memo["perma"] = ticker_to_permaticker(load_tickers(fs))
    return _memo["perma"]
