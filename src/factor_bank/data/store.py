"""SQLite persistence: saved scans + custom-factor registry."""
from __future__ import annotations

import json
import secrets
import sqlite3
import string
import time

from factor_bank.config import get_settings

_ALPHABET = string.ascii_lowercase + string.digits


def _conn() -> sqlite3.Connection:
    path = get_settings().cache_dir / "factor_bank.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    c.execute(
        "CREATE TABLE IF NOT EXISTS scans ("
        "id TEXT PRIMARY KEY, name TEXT NOT NULL, tab TEXT NOT NULL, "
        "config_json TEXT NOT NULL, created_at REAL NOT NULL)"
    )
    c.execute(
        "CREATE TABLE IF NOT EXISTS custom_factors ("
        "name TEXT PRIMARY KEY, path TEXT NOT NULL, n_rows INTEGER NOT NULL, "
        "n_tickers INTEGER NOT NULL, date_min TEXT, date_max TEXT, created_at REAL NOT NULL)"
    )
    return c


_CREATE_SCAN_RETRIES = 5


def create_scan(name: str, tab: str, config: dict) -> str:
    # 6-char base36 id space is huge, but not infinite — retry on collision
    # (sqlite3.IntegrityError on the PRIMARY KEY) rather than 500ing the user.
    last_err: sqlite3.IntegrityError | None = None
    for _ in range(_CREATE_SCAN_RETRIES):
        scan_id = "".join(secrets.choice(_ALPHABET) for _ in range(6))
        try:
            with _conn() as c:
                c.execute(
                    "INSERT INTO scans VALUES (?, ?, ?, ?, ?)",
                    (scan_id, name, tab, json.dumps(config), time.time()),
                )
            return scan_id
        except sqlite3.IntegrityError as e:
            last_err = e
    raise last_err


def list_scans() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM scans ORDER BY created_at DESC").fetchall()
    return [dict(r) | {"config": json.loads(r["config_json"])} for r in rows]


def get_scan(scan_id: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    return dict(r) | {"config": json.loads(r["config_json"])} if r else None


def delete_scan(scan_id: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
    return cur.rowcount > 0


# ── custom-factor registry (rows only; parquet I/O lives in data/custom.py) ──

def register_custom(name: str, path: str, n_rows: int, n_tickers: int,
                    date_min: str, date_max: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO custom_factors VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, path, n_rows, n_tickers, date_min, date_max, time.time()),
        )


def list_custom() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM custom_factors ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def get_custom(name: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM custom_factors WHERE name = ?", (name,)).fetchone()
    return dict(r) if r else None


def delete_custom(name: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM custom_factors WHERE name = ?", (name,))
    return cur.rowcount > 0
