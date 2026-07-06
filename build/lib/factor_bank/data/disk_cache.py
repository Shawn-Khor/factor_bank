"""ETag + TTL disk cache for S3 parquet objects. All loaders go through this."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _paths(cache_dir: Path, s3_path: str) -> tuple[Path, Path]:
    key = hashlib.sha1(s3_path.encode()).hexdigest()[:16]
    obj_dir = cache_dir / "objects"
    obj_dir.mkdir(parents=True, exist_ok=True)
    return obj_dir / f"{key}.parquet", obj_dir / f"{key}.json"


def _select(df: pd.DataFrame, columns: list[str] | None) -> pd.DataFrame:
    if columns is None:
        return df
    present = [c for c in columns if c in df.columns]
    missing = set(columns) - set(present)
    if missing:
        logger.warning("cached_parquet: missing columns %s", sorted(missing))
    return df[present]


def cached_parquet(
    fs,
    s3_path: str,
    cache_dir: Path,
    ttl_hours: float = 24.0,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    pq_path, meta_path = _paths(cache_dir, s3_path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    local_ok = pq_path.exists() and meta is not None

    if local_ok and time.time() - meta["fetched_at"] < ttl_hours * 3600:
        return _select(pd.read_parquet(pq_path), columns)

    if local_ok:  # stale — revalidate by ETag
        try:
            remote_etag = fs.info(s3_path).get("ETag")
        except Exception as e:
            logger.warning("S3 unreachable (%s); serving stale cache for %s", e, s3_path)
            return _select(pd.read_parquet(pq_path), columns)
        if remote_etag == meta.get("etag"):
            meta["fetched_at"] = time.time()
            meta_path.write_text(json.dumps(meta))
            return _select(pd.read_parquet(pq_path), columns)

    with fs.open(s3_path) as f:
        df = pd.read_parquet(f)
    df.to_parquet(pq_path)
    try:
        etag = fs.info(s3_path).get("ETag")
    except Exception:
        etag = None
    meta_path.write_text(json.dumps({"s3_path": s3_path, "etag": etag, "fetched_at": time.time()}))
    logger.info("cached_parquet: fetched %s (%d rows)", s3_path, len(df))
    return _select(df, columns)
