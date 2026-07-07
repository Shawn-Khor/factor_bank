# Factor Bank Plan C — Workflow Features (Scans, Custom Factors, Factor Lab) + Handover

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Phase 4 — saved/shareable scans, custom-factor CSV upload, and the Factor Lab bulk transform-grid screener (two-stage funnel with BH-FDR and a 70/30 holdout) — plus final polish and the handover checklist.

**Architecture:** SQLite persistence (`FB_CACHE_DIR/factor_bank.db`) via a new `data/store.py` (scans + custom-factor registry); custom factor values as parquet under `FB_CACHE_DIR/custom_factors/`; `lab/` package for grid + screening reusing `get_window`/`compute_factor`/`cross_sectional_metrics`; screening runs through the existing `JOBS` store; two new frontend tabs (Factor Lab, Scans) + save-scan buttons + `?scan=` restore.

**Spec:** `~/alpha-discovery/docs/superpowers/specs/2026-07-06-factor-bank-package-design.md` §8 (scans + CSV), §9 (Factor Lab), §11 leftovers (export, tooltips), §13 (handover).

## Global Constraints

- Repo `/home/shawnkhor/factor_bank`, branch `main`, venv `.venv`. Suite at Plan-C start: 63 green.
- Persistence in `get_settings().cache_dir / "factor_bank.db"` and `cache_dir / "custom_factors/"` — NEVER any other location; NO absolute paths under `src/`.
- Scan ids: 6-char lowercase base36 slugs. Custom factor names: regex `^[a-z][a-z0-9_]{0,39}$`, must not collide with catalog names or transform-generated names (no `__`).
- Factor Lab (spec §9, verbatim rules): candidates = numeric base factors × the 8 `TRANSFORMS`; stage-1 metrics computed on the FIRST 70% of eval dates only; BH-FDR (`scipy.stats.false_discovery_control`) across the full grid's p-values; stage 2 = top-K (default 30, ranked by |IC IR|) get MI (sklearn `mutual_info_regression`, ≤50k subsample, seed 42) + dCor (pure-numpy, ≤3k subsample, seed 42 — port `_distance_corr` from `~/alpha-discovery/dashboard/api.py:810-824`) + holdout Rank IC on the untouched last 30%.
- Upload cap 20 MB; CSV columns exactly `ticker,date,value`.
- Do not break existing contracts: `/api/evaluate`, `/api/ml-eval`, `/api/jobs`, `/api/factors` response shapes (additive changes only — `/api/factors` gains a `"Custom"` group).
- Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task C1: SQLite store + scans API + Scans tab + ?scan= restore

**Files:**
- Create: `src/factor_bank/data/store.py`, `src/factor_bank/server/static/js/scans.js`
- Modify: `src/factor_bank/server/api.py`, `static/index.html` (enable Scans tab + markup + save-scan buttons), `static/js/common.js` (saveScan + restore dispatch), `static/js/evaluate.js` + `js/mleval.js` (register restore handlers + current-config getters)
- Test: `tests/test_store.py`, `tests/test_api.py` (extend)

**Interfaces:**
- Produces `data/store.py`:

```python
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


def create_scan(name: str, tab: str, config: dict) -> str:
    scan_id = "".join(secrets.choice(_ALPHABET) for _ in range(6))
    with _conn() as c:
        c.execute(
            "INSERT INTO scans VALUES (?, ?, ?, ?, ?)",
            (scan_id, name, tab, json.dumps(config), time.time()),
        )
    return scan_id


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
```

- API routes (in `api.py`): `POST /api/scans` body `{name, tab, config}` → 201 `{"id"}` (400 if name empty or tab not in evaluate/mleval/lab); `GET /api/scans` → `{"scans": [...]}`; `GET /api/scans/{id}` → record or 404; `DELETE /api/scans/{id}` → `{"deleted": bool}`.
- Frontend contract: `common.js` gains `window.fbRestore = {}` registry + `saveScan(tab, getConfig)` helper (prompts for a name via `prompt()`, POSTs, shows the share URL `${location.origin}/?scan=${id}` via the tab's status line and copies it to clipboard when available) + on-load: if `?scan=` param present, `GET /api/scans/{id}`, `switchTab(rec.tab)`, call `window.fbRestore[rec.tab](rec.config)`. `evaluate.js` and `mleval.js` each register `window.fbRestore.evaluate/.mleval` (set controls from config, then trigger their run function) and expose `getEvaluateConfig()` / `getMlConfig()` used by the save buttons. `scans.js` renders the Scans tab: table of name/tab/created/open/delete; open = navigate to `/?scan=<id>`.

**Tests** (`tests/test_store.py` — run against `FB_CACHE_DIR=tmp_path` via monkeypatch):

```python
import pytest


@pytest.fixture(autouse=True)
def _tmp_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("FB_CACHE_DIR", str(tmp_path))


def test_scan_crud_roundtrip():
    from factor_bank.data import store
    sid = store.create_scan("my scan", "evaluate", {"factor": "pe", "horizon": 21})
    assert len(sid) == 6
    rec = store.get_scan(sid)
    assert rec["name"] == "my scan" and rec["config"]["factor"] == "pe"
    assert any(s["id"] == sid for s in store.list_scans())
    assert store.delete_scan(sid) is True
    assert store.get_scan(sid) is None
    assert store.delete_scan(sid) is False


def test_custom_registry_roundtrip():
    from factor_bank.data import store
    store.register_custom("my_sig", "/x/my_sig.parquet", 100, 10, "2019-01-02", "2020-01-02")
    rec = store.get_custom("my_sig")
    assert rec["n_rows"] == 100 and rec["n_tickers"] == 10
    assert store.delete_custom("my_sig") is True
    assert store.get_custom("my_sig") is None
```

Plus in `tests/test_api.py`: scans CRUD through the endpoints (create → get → list → delete → 404), invalid tab → 400.

**Steps:** TDD as usual; commit `feat: saved scans — SQLite store, API, Scans tab, ?scan= restore`.

---

### Task C2: Custom-factor CSV upload

**Files:**
- Create: `src/factor_bank/data/custom.py`
- Modify: `src/factor_bank/server/api.py` (upload/delete routes; `/api/factors` gains Custom group), `src/factor_bank/engine/factors.py` (custom lookup branch), `src/factor_bank/engine/evaluate.py` (`_known_factor` accepts custom), `static/index.html` + a small upload widget on the Evaluate tab (Custom group header gets an "＋ upload" affordance), `static/js/evaluate.js`
- Test: `tests/test_custom.py`, `tests/test_api.py` (extend)

**Interfaces:**
- `data/custom.py`:

```python
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
        df["date"] = pd.to_datetime(df["date"])
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
```

- `engine/factors.py` — in `compute_factor`, AFTER the transform branch and BEFORE the final `raise ValueError(f"Unknown factor: {name}")`, add:

```python
    # Custom uploaded factors — merged on (ticker, date), NaN where absent.
    from factor_bank.data.custom import custom_names, load_custom
    if name in custom_names():
        cdf = load_custom(name).rename(columns={"value": "_custom_val"})
        merged = df[["ticker", "date"]].merge(cdf, on=["ticker", "date"], how="left")
        return merged["_custom_val"].astype(float).set_axis(df.index)
```

- `engine/evaluate.py` — `_known_factor` also returns True for `name in custom_names()` (and for `base__transform` where base is custom).
- API: `POST /api/custom-factors` (multipart: `name` form field + `file`) → 201 with `validate_and_store`'s dict, 400 `{"error"}` on ValueError; `DELETE /api/custom-factors/{name}` → `{"deleted": bool}`; `GET /api/factors` response `groups` gains a `"Custom"` group `{name: "custom upload (N rows, date_min→date_max)"}` built from `store.list_custom()` (empty group omitted); `/api/warmup` also calls `custom.clear_memo()`.
- Frontend: on the Evaluate tab factor card, a small upload control (name input + file input + button) that POSTs multipart, then re-fetches the catalog so the Custom pill appears; uploaded factors evaluate through the normal flow (coverage shows in the quality panel — that is the §8 coverage requirement, already built in Plan A).

**Tests** (`tests/test_custom.py`, `FB_CACHE_DIR=tmp_path` fixture): valid roundtrip (store → custom_names → load_custom → values match); each validation failure (bad name, `__` in name, catalog collision, wrong columns, bad dates, non-numeric, dupes, >20MB via a size monkeypatch on MAX_BYTES); `compute_factor` merge correctness on the `synthetic_market` frame (upload values for 2 tickers × known dates → series aligned, NaN elsewhere); end-to-end `evaluate()` on a custom factor built to equal the fixture's `pe` (should reproduce STRONG verdict). API tests: multipart upload through TestClient, 400 paths, catalog group appears, delete works.

**Steps:** TDD; commit `feat: custom-factor CSV upload — validation, parquet store, catalog integration`.

---

### Task C3: Factor Lab engine (grid + two-stage screen)

**Files:**
- Create: `src/factor_bank/lab/__init__.py`, `src/factor_bank/lab/grid.py`, `src/factor_bank/lab/screen.py`
- Test: `tests/test_lab.py`

**Interfaces:**
- `lab/grid.py`:

```python
"""Candidate grid: numeric base factors × the 8 transform registry keys."""
from __future__ import annotations

from factor_bank.engine.catalog import numeric_base_factors
from factor_bank.engine.factors import TRANSFORMS


def candidate_grid(include_custom: bool = True) -> list[str]:
    bases = list(numeric_base_factors())
    if include_custom:
        from factor_bank.data.custom import custom_names
        bases += custom_names()
    return [f"{base}__{t}" for base in bases for t in TRANSFORMS]
```

- `lab/screen.py` — `screen(horizon, from_date, to_date, top_k=30, progress=None, *, enriched=None, spells=None, candidates=None) -> dict`. Flow (all rules from Global Constraints):
  1. `df = get_window(from_date, to_date, horizon, enriched=..., spells=...)`; eval-window dates sorted; `split = int(len(dates) * 0.7)`; train/holdout date sets. Raise ValueError if fewer than 60 train dates or 20 holdout dates.
  2. Price pivot + `trading_session_forward_returns(price_pivot, horizon)` computed ONCE.
  3. Stage 1 — for each candidate (default `candidate_grid()`): `compute_factor(df, cand)` (skip candidates whose base column is missing from the window — wrap in try/except ValueError → record skipped), pivot to eval dates, `cross_sectional_metrics(F.loc[train], R.loc[train], winsorize=None)`; keep `rank_ic, ic_ir, t_stat, p_value, n_obs`. `progress(f"stage 1: {i}/{n} candidates")` every 25.
  4. BH-FDR: `scipy.stats.false_discovery_control` over the non-null p-values → `q_value` per candidate (null p → null q).
  5. Rank stage-1 survivors by `abs(ic_ir)` (nulls last) → top-K.
  6. Stage 2 — per finalist: holdout `rank_ic` (same metrics call on holdout dates); `sign_flip = sign(train_ric) != sign(holdout_ric)`; MI on ≤50k flattened valid train pairs (`mutual_info_regression`, `random_state=42`); dCor on ≤3k (ported `_distance_corr`). `progress(f"stage 2: {j}/{top_k}")`.
  7. Return `{"leaderboard": [...], "n_candidates": n, "n_skipped": k, "skipped": [names...], "split": {"train_dates": ..., "holdout_dates": ...}, "meta": {horizon, from_date, to_date, top_k, elapsed_s}}` — leaderboard records: `candidate, train_rank_ic, ic_ir, t_stat, p_value, q_value, n_obs, holdout_rank_ic, sign_flip, mi, dcor`, sorted by `abs(ic_ir)` desc. JSON-safe (round floats to 6dp, None for non-finite).

**Tests** (`tests/test_lab.py`, using `synthetic_market` + `ps` overwritten with noise as in `tests/test_bridge.py`):

```python
def test_grid_shape():
    # bases exclude the 3 ibd ordinals; 8 transforms each; names well-formed
    from factor_bank.lab.grid import candidate_grid
    g = candidate_grid(include_custom=False)
    from factor_bank.engine.catalog import numeric_base_factors
    from factor_bank.engine.factors import TRANSFORMS
    assert len(g) == len(numeric_base_factors()) * len(TRANSFORMS)
    assert all("__" in c for c in g)


def test_screen_finds_signal_and_reports_holdout(market_with_noise):
    # candidates restricted to a small hand-picked set for test speed:
    # pe__chg_5d (zero-dispersion → skipped or null), pe__sector_rel (signal
    # survives sector demeaning in the fixture), ps__zscore_63d (noise)
    from factor_bank.lab.screen import screen
    enriched, spells = market_with_noise
    out = screen(21, "2019-06-01", "2020-06-01", top_k=3,
                 enriched=enriched, spells=spells,
                 candidates=["pe__sector_rel", "ps__zscore_63d", "pe__chg_5d"])
    lb = {r["candidate"]: r for r in out["leaderboard"]}
    assert "pe__sector_rel" in lb
    top = out["leaderboard"][0]
    assert top["candidate"] == "pe__sector_rel"          # signal wins
    assert top["holdout_rank_ic"] is not None and not top["sign_flip"]
    assert lb["ps__zscore_63d"]["q_value"] >= lb["pe__sector_rel"]["q_value"] or lb["pe__sector_rel"]["q_value"] is None
    assert out["split"]["train_dates"] > out["split"]["holdout_dates"] > 0
    import json; json.dumps(out)


def test_screen_validates_window(market_with_noise):
    from factor_bank.lab.screen import screen
    enriched, spells = market_with_noise
    import pytest
    with pytest.raises(ValueError):
        screen(21, "2019-06-01", "2019-07-01", enriched=enriched, spells=spells)  # too few dates
```

(Move the `market_with_noise` fixture from `tests/test_bridge.py` into `tests/conftest.py` so both files share it — adjust test_bridge.py imports accordingly.)

**Steps:** TDD; verify BH-FDR q-values are monotone in p empirically inside the test if trivial to assert; commit `feat: factor lab — candidate grid + two-stage screen with BH-FDR and holdout`.

---

### Task C4: Factor Lab API + tab UI

**Files:**
- Modify: `src/factor_bank/server/api.py`, `static/index.html` (enable Factor Lab tab + markup), `static/fb.css`
- Create: `static/js/lab.js`
- Test: `tests/test_api.py` (extend)

**Interfaces:**
- `POST /api/lab/screen` body `{horizon, from_date, to_date, top_k=30}` → 202 `{"job_id"}` after fast validation (horizon ∈ ALLOWED_HORIZONS, 1 ≤ top_k ≤ 100); runs `lab.screen.screen(...)` through `JOBS` (no pinned frames — same memo-friendly pattern as `/api/ml-eval`); `GET /api/lab/candidates` → `{"n_candidates": len(candidate_grid()), "transforms": list(TRANSFORMS)}` (for the UI header).
- `lab.js`: controls (horizon single-select buttons reusing the evaluate-tab pattern, date range, top_k number input), permanent banner (verbatim from spec §9): "Grid-mined results are in-sample until validated — trust the holdout column."; run → job progress (reuse `pollJob`) → leaderboard table: rank, candidate, train Rank IC, IC IR, FDR q-value with `pStar`, MI, dCor, holdout Rank IC (red `.badge.WEAK`-style badge when `sign_flip`), and an **Open in Evaluate** action per row → calls `window.fbOpenInEvaluate(candidate, from_date, to_date, horizon)` (new export from `evaluate.js`: sets the factor/dates/horizon controls, switches tab, triggers the run). Save-as-scan button (tab "lab", config = controls). Renders `n_skipped` + names collapsed under a details element when non-zero.
- API test: monkeypatch `api.lab_screen` (the imported screen fn) with a fake returning a small leaderboard; POST → 202 → poll → done; validation 400s.

**Steps:** TDD for the API; static-trace discipline for the JS (every id cross-referenced); commit `feat: factor lab tab — screen endpoint, leaderboard UI, open-in-evaluate`.

---

### Task C5: Polish — CSV export, tooltips, escaping fix, badge styling

**Files:**
- Modify: `static/js/common.js`, `static/js/evaluate.js`, `static/js/mleval.js`, `static/js/lab.js`, `static/index.html`, `static/fb.css`
- Test: none new beyond suite staying green (frontend-only; verified in C6)

**Contract:**
1. `common.js`: `tableToCsv(tableEl) -> string` + `downloadCsv(filename, csv)`; every results table (evaluate metrics grid → skip; quantile means, ML screening, MI records, redundancy, lab leaderboard) gets a small "⬇ CSV" button in its card title that exports the rendered table.
2. `common.js`: `GLOSSARY` map (metric key → one-sentence plain-English definition; source text from `~/plan/factor-bank.md` §5: IC, Rank IC, Std IC, IC IR, % Positive, t-stat, p-value, MI, dCor, q-value, holdout, composite rank, consistent, nonlinear_gain, verdict) + `applyTooltips(scopeEl)` that sets `title=` on `th`/`.metric-label` elements by matching label text. Call it after every render in all three tabs.
3. Fix the B6 minor: `renderScreening`'s warnings block must use `textContent` (build elements, no unescaped innerHTML interpolation).
4. ML screening `consistent`/`nonlinear_gain` get real `.badge` styling (small green/amber badges instead of plain ✓/—).
5. Bump all static `?v=` params to `?v=2`.

Commit: `feat: polish — CSV export, metric tooltips, warnings escaping, badges`.

---

### Task C6: Handover — fresh-venv install test, README, final review, cutover decision

**Files:**
- Modify: `README.md`
- No other source changes unless the final review demands them

**Steps:**
1. **Fresh-venv install test** (spec §13, the headline check): in a temp dir, `python3 -m venv /tmp/fb-handover-test && /tmp/fb-handover-test/bin/pip install "factor-bank @ git+https://github.com/Shawn-Khor/factor_bank.git"` (base install, NO ml extra — must succeed without access to alpha_eval), then `factor-bank --help` works, then with the repo's `.env` copied to a scratch working dir, `FB_PORT=8202 factor-bank serve` starts, `/api/health` ok, `/api/factors` returns the catalog, `/api/evaluate` works (cache dir default is shared ~/.cache/factor_bank so it's warm). ML extra check: `pip install -e /home/shawnkhor/alpha_eval` into the same venv, restart, `/api/ml-eval` accepts a job. Kill server, remove venv.
2. **README final pass**: add Scans / Custom factors / Factor Lab sections (one paragraph + endpoint each); update the tab list; document the in-sample banner semantics and the holdout column; ensure the env-var table and glossary cover q-value/holdout.
3. **Full-suite + suite count recorded in ledger.**
4. **Final whole-plan-C review**: review package over the Plan C commit range, most capable model, findings triaged and fixed (one fixer, re-verify).
5. **Push.**
6. **Cutover**: do NOT touch port 8200 — the legacy alpha-discovery dashboard serves other tabs (Findings/Telescope etc.), so the old uvicorn stays. Instead surface the question to the user at the end: keep new dashboard on 8201 permanently (recommended — update memory + README accordingly), or also add a redirect from the old factor-bank.html to 8201.

---

## Plan C Self-Review Notes

- Spec §8 coverage: scans CRUD + slug ids + share URL + auto-run restore (C1); CSV upload validation/parquet/registry/Custom group/coverage-via-quality-panel/delete (C2). Result snapshots explicitly out of scope (spec: configs only).
- Spec §9 coverage: grid = numeric × 8 transforms incl. custom bases (C3 grid); 70/30 split, BH-FDR across full grid, top-K=30 by |IC IR|, MI/dCor on finalists, holdout column + sign-flip flag, skipped-candidates transparency (C3/C4); banner verbatim (C4); `base__transform` addressability in Evaluate came free in Plan A (Task 7 design).
- Spec §11 leftovers: CSV export + tooltips (C5); verdict badge + quality panel shipped in Plan A; progress bars shipped in Plan B.
- Spec §13: fresh-venv install, README completeness, colleague access notes (C6).
- Type consistency: `store.py` consumed by C1 routes + C2 (`register_custom/list_custom/get_custom/delete_custom`); `custom_names/load_custom` consumed by `factors.py` branch (C2) and `grid.py` (C3); `screen()` signature consumed by C4 endpoint + fake in its test; `window.fbRestore`/`fbOpenInEvaluate`/`pollJob` cross-module JS contracts named identically in C1/C4/C5.
- Deliberate deviation: engine→data.custom import in `compute_factor` is lazy (function-level) to avoid a hard layering cycle; acceptable for one lookup, recorded here.
