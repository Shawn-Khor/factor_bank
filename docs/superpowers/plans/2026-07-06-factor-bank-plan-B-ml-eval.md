# Factor Bank Plan B — ML Eval Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the ML Eval tab — non-linear multi-factor evaluation (MI with permutation significance, dCor, monotonicity, redundancy matrix, optional LightGBM MDI/MDA, composite screening) wrapping `alpha_eval.ml_eval`, run as background jobs with progress polling.

**Architecture:** A `JobStore` (single-worker executor, in-process, bounded) runs slow evaluations; `engine/panel.py` memoizes the expensive buffered+filtered data window; `ml/bridge.py` adapts factor-bank matrices into `alpha_eval.prepare_ml_eval_data(...)`/`ml_eval(...)` and serializes `MLEvalResult` to JSON; two new API routes (`POST /api/ml-eval`, `GET /api/jobs/{id}`); a new frontend tab module `js/mleval.js`.

**Tech Stack:** Existing package (Plan A complete, 47 tests green) + `alpha_eval` (weekiat's package, local checkout at `/home/shawnkhor/alpha_eval`, declares lightgbm/joblib/dcor deps), FastAPI, vanilla JS + Chart.js.

**Spec:** `~/alpha-discovery/docs/superpowers/specs/2026-07-06-factor-bank-package-design.md` §7 (ML Eval), §10 (jobs), §5.3 (matrix memoization — implemented here at window level, see Task 4 note).

## Global Constraints

- Repo: `/home/shawnkhor/factor_bank`, branch `main`, venv `.venv` (run tests with `.venv/bin/python -m pytest`).
- NO absolute paths under `src/` — the alpha_eval dependency is imported normally (installed into the venv), never via `sys.path`.
- ML Eval request bounds (spec §7): `2 ≤ len(factors) ≤ 20`; `horizons` non-empty ⊆ `{1, 5, 21, 63}`; modes `quick|standard|thorough` → `n_permutations` `0|50|200`; `tier2` adds LightGBM `mdi` + `quick_mda`.
- Single-worker job executor (spec §10) — deliberate, RAM headroom over parallelism. Job store bounded at 20 records; only finished jobs are evicted.
- Existing interfaces (Plan A, do not break): `evaluate()` signature and response shape; `/api/factors`, `/api/evaluate`, `/api/warmup` contracts; `_sanitize`; `synthetic_market` conftest fixture.
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task B1: Install and pin the alpha_eval dependency

**Files:**
- Modify: `pyproject.toml` (comment only), `README.md` (dev-install note)
- Test: `tests/test_ml_import.py`

**Interfaces:**
- Produces: importable `alpha_eval.ml_eval` (`prepare_ml_eval_data`, `ml_eval`) inside `.venv`; later tasks assume it.

- [ ] **Step 1: Install alpha_eval into the venv from the local checkout**

```bash
cd ~/factor_bank && .venv/bin/pip install -q -e /home/shawnkhor/alpha_eval
.venv/bin/python -c "from alpha_eval.ml_eval import prepare_ml_eval_data, ml_eval; import lightgbm; print('ok')"
```

Expected: `ok`. (The `[ml]` extra in pyproject already declares the git URL for colleagues; the local editable install is the dev equivalent. Do NOT add a path dependency to pyproject.)

- [ ] **Step 2: Write the import smoke test** — `tests/test_ml_import.py`

```python
def test_ml_eval_importable():
    from alpha_eval.ml_eval import ml_eval, prepare_ml_eval_data  # noqa: F401
    from alpha_eval.ml_eval.prepare import PreparedData  # noqa: F401
```

- [ ] **Step 3: Run it** — `.venv/bin/python -m pytest tests/test_ml_import.py -v` — Expected: 1 PASS.

- [ ] **Step 4: README dev note** — in README's Install section add: dev installs run `pip install -e /path/to/alpha_eval` (or the `[ml]` extra's git URL, which needs read access to softdevintegrations/alpha_eval).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "chore: wire alpha_eval dependency for ML Eval (dev editable install + smoke test)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task B2: Job store + GET /api/jobs/{id}

**Files:**
- Create: `src/factor_bank/server/jobs.py`
- Modify: `src/factor_bank/server/api.py` (add jobs route)
- Test: `tests/test_jobs.py`

**Interfaces:**
- Produces: `JobStore` class and module singleton `JOBS`. `JOBS.submit(fn) -> str` where `fn(progress: Callable[[str], None]) -> dict`; `JOBS.get(job_id) -> dict | None` returning `{id, status: "queued"|"running"|"done"|"error", progress: str, started_at, finished_at, result, error, detail}`. Route `GET /api/jobs/{job_id}` → the record (404 `{"error": "unknown job"}` if absent). Task B5 submits bridge runs through this.

- [ ] **Step 1: Write the failing test** — `tests/test_jobs.py`

```python
import time

from fastapi.testclient import TestClient

from factor_bank.server.app import create_app
from factor_bank.server.jobs import JobStore


def _wait(store, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = store.get(job_id)
        if rec["status"] in ("done", "error"):
            return rec
        time.sleep(0.02)
    raise TimeoutError


def test_success_lifecycle_and_progress():
    store = JobStore()
    def work(progress):
        progress("stage 1")
        progress("stage 2")
        return {"answer": 42}
    jid = store.submit(work)
    rec = _wait(store, jid)
    assert rec["status"] == "done"
    assert rec["result"] == {"answer": 42}
    assert rec["progress"] == "stage 2"
    assert rec["started_at"] is not None and rec["finished_at"] is not None


def test_error_captures_traceback():
    store = JobStore()
    def boom(progress):
        raise RuntimeError("kapow")
    rec = _wait(store, store.submit(boom))
    assert rec["status"] == "error"
    assert "kapow" in rec["error"]
    assert "RuntimeError" in rec["detail"]  # full traceback
    assert rec["result"] is None


def test_serial_execution_order():
    store = JobStore()
    order = []
    j1 = store.submit(lambda p: order.append(1) or {})
    j2 = store.submit(lambda p: order.append(2) or {})
    _wait(store, j1); _wait(store, j2)
    assert order == [1, 2]  # single worker → strictly serial


def test_eviction_keeps_active_jobs():
    store = JobStore(max_jobs=3)
    done = [_wait(store, store.submit(lambda p: {})) for _ in range(3)]
    j_new = store.submit(lambda p: {})
    _wait(store, j_new)
    assert store.get(j_new) is not None
    assert store.get(done[0]["id"]) is None  # oldest finished evicted


def test_jobs_endpoint_unknown_404():
    client = TestClient(create_app())
    r = client.get("/api/jobs/nope1234")
    assert r.status_code == 404 and "error" in r.json()
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_jobs.py -v` — Expected: FAIL (module missing).

- [ ] **Step 3: Write `src/factor_bank/server/jobs.py`**

```python
"""In-process background jobs: submit → poll → result (spec §10).

Single worker by design: the enriched frame dominates RAM, so heavy ML runs
are serialized rather than parallelized.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable


class JobStore:
    def __init__(self, max_jobs: int = 20):
        self._max = max_jobs
        self._ex = ThreadPoolExecutor(max_workers=1)
        self._jobs: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def submit(self, fn: Callable[[Callable[[str], None]], dict]) -> str:
        job_id = uuid.uuid4().hex[:8]
        rec = {
            "id": job_id, "status": "queued", "progress": "",
            "started_at": None, "finished_at": None,
            "result": None, "error": None, "detail": None,
        }
        with self._lock:
            self._jobs[job_id] = rec
            self._evict()

        def _run():
            rec["status"] = "running"
            rec["started_at"] = time.time()
            try:
                rec["result"] = fn(lambda msg: rec.__setitem__("progress", str(msg)))
                rec["status"] = "done"
            except Exception as e:  # noqa: BLE001 — job boundary, full capture
                rec["error"] = str(e)
                rec["detail"] = traceback.format_exc()
                rec["status"] = "error"
            finally:
                rec["finished_at"] = time.time()

        self._ex.submit(_run)
        return job_id

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _evict(self) -> None:
        # Only finished jobs are evicted, oldest first (lock held by caller).
        while len(self._jobs) > self._max:
            victim = next(
                (k for k, r in self._jobs.items() if r["status"] in ("done", "error")),
                None,
            )
            if victim is None:
                return
            self._jobs.pop(victim)


JOBS = JobStore()
```

- [ ] **Step 4: Add the route to `api.py`**

```python
from factor_bank.server.jobs import JOBS

@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    rec = JOBS.get(job_id)
    if rec is None:
        return JSONResponse(status_code=404, content={"error": "unknown job"})
    return _sanitize(rec)
```

- [ ] **Step 5: Run tests** — `tests/test_jobs.py` 5 PASS; full suite green.

- [ ] **Step 6: Commit** — `feat: in-process job store + GET /api/jobs/{id}` (+ trailer).

---

### Task B3: Memoized data window (`engine/panel.py`)

**Files:**
- Create: `src/factor_bank/engine/panel.py`
- Test: `tests/test_panel.py`

**Interfaces:**
- Consumes: `load_enriched`, `get_spells`, `filter_to_sp500` (Plan A)
- Produces: `get_window(from_date: str, to_date: str, max_horizon: int, *, enriched=None, spells=None) -> pd.DataFrame` — the buffered (400d lookback, `max_horizon*1.6+7` forward), S&P-filtered, `(ticker, date)`-sorted frame `evaluate()` builds internally, extracted so ML Eval reuses ONE window across N factors and repeat requests. TTL-memoized (1h) ONLY when `enriched is None` (real loaders); injected test frames bypass the memo. `clear_memo()` exported; Task B5 wires it into `/api/warmup`.
- Note: spec §5.3 asked for factor-matrix + fwd-matrix memoization too; window-level memoization captures the dominant cost (the S&P filter over ~8M rows) — per-matrix memoization deferred until profiling demands it (YAGNI, recorded here deliberately).

- [ ] **Step 1: Write the failing test** — `tests/test_panel.py`

```python
import pandas as pd

import factor_bank.engine.panel as panel_mod
from factor_bank.engine.panel import clear_memo, get_window


def test_window_buffers_filters_sorts(synthetic_market):
    enriched, spells = synthetic_market
    w = get_window("2019-06-02", "2019-12-31", 21, enriched=enriched, spells=spells)
    # 400-day lookback buffer reaches the fixture start
    assert w["date"].min() < pd.Timestamp("2019-06-02")
    # forward buffer extends past to_date
    assert w["date"].max() > pd.Timestamp("2019-12-31")
    # sorted by (ticker, date)
    assert w.equals(w.sort_values(["ticker", "date"]).reset_index(drop=True))
    assert w["ticker"].nunique() == 12


def test_membership_filter_applied(synthetic_market):
    enriched, spells = synthetic_market
    w = get_window("2019-06-02", "2019-12-31", 21,
                   enriched=enriched, spells=spells[spells["ticker"] != "T00"])
    assert "T00" not in set(w["ticker"])


def test_memo_only_for_real_loaders(synthetic_market, monkeypatch):
    enriched, spells = synthetic_market
    clear_memo()
    calls = {"n": 0}

    def fake_load():
        calls["n"] += 1
        return enriched

    monkeypatch.setattr(panel_mod, "load_enriched", fake_load)
    monkeypatch.setattr(panel_mod, "get_spells", lambda: spells)
    get_window("2019-06-02", "2019-12-31", 21)
    get_window("2019-06-02", "2019-12-31", 21)   # same key → memo hit
    assert calls["n"] == 1
    assert len(panel_mod._memo) == 1
    # injected frames must NOT populate the memo
    clear_memo()
    get_window("2019-06-02", "2019-12-31", 21, enriched=enriched, spells=spells)
    assert len(panel_mod._memo) == 0
```

- [ ] **Step 2: Run to verify failure** — Expected: FAIL (module missing).

- [ ] **Step 3: Write `src/factor_bank/engine/panel.py`**

```python
"""Memoized buffered+filtered data window shared by slow multi-factor runs."""
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
            return hit[1]

    df_all = enriched if enriched is not None else load_enriched()
    sp = spells if spells is not None else get_spells()

    buf_start = from_ts - pd.Timedelta(days=400)
    buf_end = to_ts + pd.Timedelta(days=int(max_horizon * 1.6) + 7)
    df = df_all[(df_all["date"] >= buf_start) & (df_all["date"] <= buf_end)].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df = filter_to_sp500(df, sp)

    if use_memo:
        _memo[key] = (time.time(), df)
    return df
```

- [ ] **Step 4: Run tests** — 3 PASS; full suite green.
- [ ] **Step 5: Commit** — `feat: memoized data window for multi-factor runs` (+ trailer).

---

### Task B4: ML bridge (`ml/bridge.py`)

**Files:**
- Create: `src/factor_bank/ml/__init__.py`, `src/factor_bank/ml/bridge.py`
- Test: `tests/test_bridge.py`

**Interfaces:**
- Consumes: `get_window` (B3), `compute_factor`, `trading_session_forward_returns`, `alpha_eval.ml_eval.{prepare_ml_eval_data, ml_eval}`
- Produces:

```python
run_ml_eval(
    factors: list[str], horizons: list[int], from_date: str, to_date: str,
    quantiles: int = 5, mode: str = "standard", tier2: bool = False,
    progress: Callable[[str], None] | None = None,
    *, enriched=None, spells=None,
) -> dict
```

Returns JSON-safe dict: `screening` (records from `screening_summary("best")`), `mutual_info` / `distance_corr` (records with feature/horizon), `redundancy` (`{"features": [...], "values": [[...]]}` or None), `monotonicity` (records from the summary frame), `mdi` / `mda` (records or None; only when tier2), `meta` (n_tickers, n_dates, horizons, mode, n_permutations, elapsed_s, warnings). Raises ValueError on bounds violations (Global Constraints). `MODE_PERMUTATIONS = {"quick": 0, "standard": 50, "thorough": 200}` exported (B5 validates against its keys).

- [ ] **Step 1: Write the failing test** — `tests/test_bridge.py`

```python
import json

import numpy as np
import pytest

from factor_bank.ml.bridge import MODE_PERMUTATIONS, run_ml_eval


@pytest.fixture
def market_with_noise(synthetic_market):
    """pe and evebitda are (redundant) true signals in the fixture; overwrite
    ps with pure noise so screening has something to rank last."""
    enriched, spells = synthetic_market
    enriched = enriched.copy()
    rng = np.random.default_rng(5)
    enriched["ps"] = rng.normal(0, 1, len(enriched))
    return enriched, spells


def test_bounds_validation(market_with_noise):
    enriched, spells = market_with_noise
    with pytest.raises(ValueError, match="factors"):
        run_ml_eval(["pe"], [21], "2019-01-01", "2020-01-01",
                    enriched=enriched, spells=spells)          # < 2 factors
    with pytest.raises(ValueError, match="[Hh]orizon"):
        run_ml_eval(["pe", "ps"], [10], "2019-01-01", "2020-01-01",
                    enriched=enriched, spells=spells)          # 10 not in ML set
    with pytest.raises(ValueError, match="mode"):
        run_ml_eval(["pe", "ps"], [21], "2019-01-01", "2020-01-01",
                    mode="warp", enriched=enriched, spells=spells)


def test_end_to_end_screening_and_redundancy(market_with_noise):
    enriched, spells = market_with_noise
    msgs = []
    out = run_ml_eval(
        ["pe", "evebitda", "ps"], [5, 21], "2019-01-01", "2020-01-01",
        mode="quick", progress=msgs.append,
        enriched=enriched, spells=spells,
    )
    json.dumps(out)  # fully JSON-serializable

    feats = {r["feature"] for r in out["screening"]}
    assert feats == {"pe", "evebitda", "ps"}
    by_feat = {r["feature"]: r for r in out["screening"]}
    # noise ranks worse (higher composite_rank) than both true signals
    assert by_feat["ps"]["composite_rank"] > by_feat["pe"]["composite_rank"]
    assert by_feat["ps"]["composite_rank"] > by_feat["evebitda"]["composite_rank"]

    # redundancy: pe & evebitda are both monotone in the same ticker ladder →
    # their pairwise MI exceeds pe↔noise
    red = out["redundancy"]
    i, j, k = (red["features"].index(f) for f in ("pe", "evebitda", "ps"))
    assert red["values"][i][j] > red["values"][i][k]

    assert out["mdi"] is None            # tier2 off
    assert out["meta"]["n_permutations"] == 0
    assert msgs, "progress callback never called"


def test_mode_map():
    assert MODE_PERMUTATIONS == {"quick": 0, "standard": 50, "thorough": 200}
```

- [ ] **Step 2: Run to verify failure** — Expected: FAIL (module missing).

- [ ] **Step 3: Write `src/factor_bank/ml/bridge.py`** (and empty `ml/__init__.py`)

```python
"""Adapter: factor-bank matrices → alpha_eval.ml_eval → JSON-safe dict (spec §7)."""
from __future__ import annotations

import json
import time
from typing import Callable

import pandas as pd

from factor_bank.engine.factors import compute_factor
from factor_bank.engine.metrics import trading_session_forward_returns
from factor_bank.engine.panel import get_window

ML_HORIZONS = (1, 5, 21, 63)
MODE_PERMUTATIONS = {"quick": 0, "standard": 50, "thorough": 200}
MAX_FACTORS = 20


def _records(df) -> list[dict]:
    """DataFrame → JSON-safe records (to_json handles numpy scalars + NaN)."""
    if df is None or len(df) == 0:
        return []
    return json.loads(df.to_json(orient="records"))


def run_ml_eval(
    factors: list[str],
    horizons: list[int],
    from_date: str,
    to_date: str,
    quantiles: int = 5,
    mode: str = "standard",
    tier2: bool = False,
    progress: Callable[[str], None] | None = None,
    *,
    enriched: pd.DataFrame | None = None,
    spells: pd.DataFrame | None = None,
) -> dict:
    from alpha_eval.ml_eval import ml_eval, prepare_ml_eval_data

    if not (2 <= len(set(factors)) <= MAX_FACTORS):
        raise ValueError(f"factors must contain 2..{MAX_FACTORS} distinct names")
    if not horizons or any(h not in ML_HORIZONS for h in horizons):
        raise ValueError(f"horizons must be a non-empty subset of {ML_HORIZONS}")
    if mode not in MODE_PERMUTATIONS:
        raise ValueError(f"mode must be one of {sorted(MODE_PERMUTATIONS)}")

    tick = progress or (lambda msg: None)
    t0 = time.time()
    horizons = sorted(set(int(h) for h in horizons))

    tick("loading data window")
    df = get_window(from_date, to_date, max(horizons), enriched=enriched, spells=spells)
    if df.empty:
        raise ValueError("No data after S&P 500 filtering")

    from_ts, to_ts = pd.Timestamp(from_date), pd.Timestamp(to_date)
    in_window = (df["date"] >= from_ts) & (df["date"] <= to_ts)

    features: dict[str, pd.DataFrame] = {}
    for i, name in enumerate(dict.fromkeys(factors), 1):
        tick(f"factor matrices {i}/{len(set(factors))}: {name}")
        vals = compute_factor(df, name)
        features[name] = (
            df[in_window].assign(_f=vals[in_window])
            .pivot_table(index="date", columns="ticker", values="_f")
        )

    tick("forward returns")
    price_pivot = df.pivot_table(
        index="date", columns="ticker", values="prev_close_price"
    ).sort_index()
    eval_dates = df.loc[in_window, "date"].unique()
    target = {
        h: trading_session_forward_returns(price_pivot, h).loc[
            lambda m: m.index.isin(eval_dates)
        ]
        for h in horizons
    }

    tick("preparing data (align + trim)")
    data = prepare_ml_eval_data(
        features=features,
        prices=price_pivot,
        target=target,
        min_common_tickers=5,
        min_observations=60,
        winsorize=0.01,
    )

    methods = ["mutual_info", "distance_corr", "quantile_spread", "monotonicity", "redundancy"]
    if tier2:
        methods += ["mdi", "quick_mda"]
    n_perm = MODE_PERMUTATIONS[mode]

    tick(f"running ml_eval ({mode}, {n_perm} permutations — the long stage)")
    result = ml_eval(
        data, quantiles=quantiles, methods=methods,
        model_type="lightgbm", n_jobs=1, n_permutations=n_perm,
    )

    tick("serializing")
    red_df = result.redundancy_matrix()
    redundancy = None
    if red_df is not None and not red_df.empty:
        redundancy = {
            "features": [str(c) for c in red_df.columns],
            "values": json.loads(red_df.to_json(orient="values")),
        }
    mono = result.monotonicity()
    mono_summary = mono.get("summary") if isinstance(mono, dict) else None

    return {
        "screening": _records(result.screening_summary("best")),
        "mutual_info": _records(result.mutual_info()),
        "distance_corr": _records(result.distance_corr()),
        "redundancy": redundancy,
        "monotonicity": _records(mono_summary),
        "mdi": _records(result.mdi_importance()) if tier2 else None,
        "mda": _records(result.quick_mda()) if tier2 else None,
        "meta": {
            "factors": sorted(features),
            "horizons": horizons,
            "from_date": from_date,
            "to_date": to_date,
            "quantiles": quantiles,
            "mode": mode,
            "n_permutations": n_perm,
            "n_tickers": data.n_tickers,
            "n_dates": data.n_dates,
            "warnings": list(data.warnings),
            "elapsed_s": round(time.time() - t0, 1),
        },
    }
```

Implementation note for the engineer: `result.monotonicity()` returns a dict of DataFrames — inspect its actual keys on first run (`python -c` with the synthetic fixture, or read `~/alpha_eval/alpha_eval/ml_eval/ml_result.py:216`) and take the per-feature SUMMARY frame; if the key is not literally `"summary"`, adapt this one lookup and record the actual key in your report. Same for `mdi_importance()`/`quick_mda()` column names — do not guess in the frontend (Task B6 renders whatever columns the records carry).

- [ ] **Step 4: Run tests** — `tests/test_bridge.py` 3 PASS (the end-to-end one takes ~10–60 s; that is expected). Full suite green.
- [ ] **Step 5: Commit** — `feat: ml bridge — prepare_ml_eval_data + ml_eval + JSON serialization` (+ trailer).

---

### Task B5: POST /api/ml-eval + warmup clears panel memo

**Files:**
- Modify: `src/factor_bank/server/api.py`
- Test: `tests/test_api.py` (extend)

**Interfaces:**
- Produces: `POST /api/ml-eval` body `{factors, horizons, from_date, to_date, quantiles=5, mode="standard", tier2=false}` → `{"job_id": "..."}` (202). Bounds violations fail FAST with 400 (validated inline via a dry `ValueError` check BEFORE submitting, so users don't wait on a queued job to learn their request was malformed). Poll via `GET /api/jobs/{id}` (B2). `/api/warmup` additionally calls `panel.clear_memo()`.

- [ ] **Step 1: Extend `tests/test_api.py`** (append; reuse the existing `client` fixture with `synthetic_market` injection)

```python
def test_ml_eval_flow(client, monkeypatch):
    import factor_bank.server.api as api

    def fake_run(factors, horizons, from_date, to_date, quantiles=5,
                 mode="standard", tier2=False, progress=None, **kw):
        progress("halfway")
        return {"screening": [{"feature": f} for f in factors], "meta": {"mode": mode}}

    monkeypatch.setattr(api, "run_ml_eval", fake_run)
    r = client.post("/api/ml-eval", json={
        "factors": ["pe", "pb"], "horizons": [21],
        "from_date": "2019-01-01", "to_date": "2020-01-01",
    })
    assert r.status_code == 202
    jid = r.json()["job_id"]
    import time
    for _ in range(100):
        rec = client.get(f"/api/jobs/{jid}").json()
        if rec["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    assert rec["status"] == "done"
    assert rec["result"]["meta"]["mode"] == "standard"


def test_ml_eval_validates_before_submit(client):
    r = client.post("/api/ml-eval", json={
        "factors": ["pe"], "horizons": [21],          # only 1 factor
        "from_date": "2019-01-01", "to_date": "2020-01-01",
    })
    assert r.status_code == 400 and "factors" in r.json()["error"]
    r = client.post("/api/ml-eval", json={
        "factors": ["pe", "pb"], "horizons": [10],    # 10 not an ML horizon
        "from_date": "2019-01-01", "to_date": "2020-01-01",
    })
    assert r.status_code == 400


def test_warmup_clears_panel_memo(client, monkeypatch):
    import factor_bank.engine.panel as panel
    panel._memo[("x", "y", 1)] = (0.0, None)
    client.post("/api/warmup")
    assert panel._memo == {}
```

- [ ] **Step 2: Run to verify failure**, then **Step 3: implement in `api.py`**

```python
from factor_bank.engine import panel as panel_mod
from factor_bank.ml.bridge import ML_HORIZONS, MAX_FACTORS, MODE_PERMUTATIONS, run_ml_eval


class MLEvalRequest(BaseModel):
    factors: list[str]
    horizons: list[int] = [21]
    from_date: str
    to_date: str
    quantiles: int = 5
    mode: str = "standard"
    tier2: bool = False


@router.post("/ml-eval", status_code=202)
def submit_ml_eval(req: MLEvalRequest):
    # Fail fast on bounds BEFORE queueing (same rules the bridge enforces).
    if not (2 <= len(set(req.factors)) <= MAX_FACTORS):
        return JSONResponse(status_code=400, content={"error": f"factors must contain 2..{MAX_FACTORS} distinct names"})
    if not req.horizons or any(h not in ML_HORIZONS for h in req.horizons):
        return JSONResponse(status_code=400, content={"error": f"horizons must be a non-empty subset of {list(ML_HORIZONS)}"})
    if req.mode not in MODE_PERMUTATIONS:
        return JSONResponse(status_code=400, content={"error": f"mode must be one of {sorted(MODE_PERMUTATIONS)}"})
    if req.quantiles not in ALLOWED_QUANTILES:
        return JSONResponse(status_code=400, content={"error": f"quantiles must be one of {ALLOWED_QUANTILES}"})

    enriched, spells = _get_enriched(), _get_spells()
    job_id = JOBS.submit(lambda progress: _sanitize(run_ml_eval(
        req.factors, req.horizons, req.from_date, req.to_date,
        quantiles=req.quantiles, mode=req.mode, tier2=req.tier2,
        progress=progress, enriched=enriched, spells=spells,
    )))
    return {"job_id": job_id}
```

And in the warmup route add `panel_mod.clear_memo()` beside the existing three `clear_memo()` calls.

NOTE on the lambda: it closes over `enriched`/`spells` resolved at submit time — this keeps the injection seam working under monkeypatched tests AND pins the job to a consistent data snapshot even if warmup swaps the memo mid-run. Preserve that property.

- [ ] **Step 4: Run tests** — 3 new PASS; full suite green.
- [ ] **Step 5: Commit** — `feat: POST /api/ml-eval background endpoint; warmup clears panel memo` (+ trailer).

---

### Task B6: ML Eval tab frontend

**Files:**
- Modify: `src/factor_bank/server/static/index.html` (enable tab + markup), `static/fb.css` (heatmap + progress styles), `static/js/common.js` (poller + heatmap helpers)
- Create: `src/factor_bank/server/static/js/mleval.js`
- Test: `tests/test_api.py` (one static-asset assertion), manual browser pass in Task B7

**Interfaces:**
- Consumes: `GET /api/factors` (catalog + groups), `POST /api/ml-eval`, `GET /api/jobs/{id}`
- Produces: working ML Eval tab. Contract with the backend response: `screening` records (columns as returned — render `feature, chosen_horizon, mean_ic, mi_score, mi_pvalue_adj, dcor, composite_rank, consistent, nonlinear_gain` when present), `mutual_info` records (feature/horizon/mi_score) → heatmap, `redundancy` `{features, values}` → matrix, `mdi` records → bar chart.

- [ ] **Step 1: index.html** — remove `disabled` from the ML Eval tab button; fill `<main id="tab-mleval" class="tab-content">` with: a Settings card (date range + quantiles inputs with ids `ml-from`/`ml-to`/`ml-quantiles`; horizon checkboxes `1D/5D/21D/63D` ids `mlh-1/mlh-5/mlh-21/mlh-63`, 21D checked; mode `<select id="ml-mode">` quick/standard/thorough default standard; `<input type="checkbox" id="ml-tier2">` labelled "Tier 2 (LightGBM importance)"); a Factors card with `<div id="ml-factor-pills">` (multi-select — clicking toggles; reuse `.factor-pill` styles) and a selected-count span `#ml-count`; a Run button `#ml-run`; a progress area `#ml-progress` (hidden by default) with a status line `#ml-progress-text` and elapsed timer `#ml-elapsed`; results cards (hidden until done): `#ml-screening` (table container), `#ml-heatmap` (MI feature×horizon), `#ml-redundancy`, `#ml-mdi-wrap` with `<canvas id="ml-mdi-chart">`. Load `<script src="/js/mleval.js?v=1"></script>` after evaluate.js.

- [ ] **Step 2: common.js additions** (exact code)

```javascript
async function pollJob(jobId, onProgress, intervalMs = 2000) {
  for (;;) {
    const rec = await fetch(`/api/jobs/${jobId}`).then(r => r.json());
    if (rec.status === "done") return rec.result;
    if (rec.status === "error") throw new Error(rec.error + "\n" + (rec.detail || ""));
    onProgress(rec);
    await new Promise(res => setTimeout(res, intervalMs));
  }
}

function heatColor(v, vmin, vmax) {
  if (v == null || !isFinite(v)) return "";
  const t = vmax > vmin ? (v - vmin) / (vmax - vmin) : 0;
  const alpha = (0.08 + 0.72 * t).toFixed(3);
  return `background: rgba(37, 99, 235, ${alpha}); color: ${t > 0.6 ? "#fff" : "inherit"}`;
}

function renderMatrixTable(el, rowLabels, colLabels, values) {
  const flat = values.flat().filter(v => v != null && isFinite(v));
  const vmin = Math.min(...flat), vmax = Math.max(...flat);
  el.innerHTML = `<div style="overflow-x:auto"><table class="matrix-table"><thead><tr><th></th>${
    colLabels.map(c => `<th>${c}</th>`).join("")
  }</tr></thead><tbody>${
    rowLabels.map((r, i) => `<tr><th>${r}</th>${
      values[i].map(v => `<td style="${heatColor(v, vmin, vmax)}">${fmt(v, 3)}</td>`).join("")
    }</tr>`).join("")
  }</tbody></table></div>`;
}
```

- [ ] **Step 3: mleval.js** — structure (follow evaluate.js idioms: same fetch/error/spinner patterns, same pill CSS classes):
  - `loadMlCatalog()`: reuse the `/api/factors` response already fetched by evaluate.js if cached in a shared `window.fbCatalog`, else fetch; render group-labelled multi-select pills into `#ml-factor-pills`; maintain `state.selected: Set`; update `#ml-count`; cap selection at 20 with a brief inline warning.
  - `runMlEval()`: gather controls → POST `/api/ml-eval` → on 400 show the error inline; on 202 show `#ml-progress`, start an elapsed timer, `pollJob(jobId, rec => progressText.textContent = rec.progress)` → render on resolve; always clear timer + spinner in `finally`.
  - `renderScreening(records)`: table sorted by `composite_rank` ascending; significance stars on `mi_pvalue_adj` (reuse `pStar`); render `consistent`/`nonlinear_gain` as ✓/— badges; skip columns absent from the records.
  - `renderMiHeatmap(records)`: pivot records (feature × horizon, value mi_score) in JS → `renderMatrixTable`.
  - `renderRedundancy(red)`: `renderMatrixTable(el, red.features, red.features, red.values)`.
  - `renderMdi(records)`: horizontal Chart.js bar of top-15 by importance; destroy previous chart instance first (same pattern evaluate.js uses for its charts); hide the card when `records` is null/empty.
- [ ] **Step 4: fb.css** — add `.matrix-table` (compact, monospace numerics, sticky first column) and `#ml-progress` bar styles (reuse the existing spinner class).
- [ ] **Step 5: static smoke assertion** — extend `tests/test_api.py::test_static_index_served` to also assert `"mleval.js"` is referenced in `/` HTML and `client.get("/js/mleval.js").status_code == 200`.
- [ ] **Step 6: Run full suite** — green. **Step 7: Commit** — `feat: ML Eval tab — multi-select, job progress, screening + heatmaps` (+ trailer).

---

### Task B7: Real-data verification + push

**Files:** none new (verification + push)

- [ ] **Step 1:** Restart the detached server on 8201 (`kill` old pid; `nohup .venv/bin/factor-bank serve --port 8201 ... & disown`). `curl /api/health` → ok.
- [ ] **Step 2:** Submit a real run: `POST /api/ml-eval` with `{"factors": ["pe", "ebitda_yield", "rsi_14", "ret_20d", "sa_quant_rating"], "horizons": [5, 21], "from_date": "2019-01-01", "to_date": "2025-01-01", "mode": "standard"}` → poll `/api/jobs/{id}` until done; record elapsed. Expected: completes (standard mode, 5 factors × 2 horizons; minutes-scale), `screening` has 5 rows with finite mi_score, redundancy is 5×5. Sanity: `rsi_14` and `ret_20d` (fast technicals) should show different chosen_horizon/IC character than the slow value factors — eyeball, record in report.
- [ ] **Step 3:** Submit a `quick`-mode 2-factor run and confirm it returns in well under a minute (warm window memo — second request reuses the window; note both elapsed times to demonstrate the B3 memo working).
- [ ] **Step 4:** Error path: submit with `"factors": ["pe", "nope_factor"]` → job goes to `status: error` with a traceback in `detail` (bounds pass; compute fails) — confirms the job error surface end-to-end.
- [ ] **Step 5:** `git push origin main`. Update the progress ledger.

---

## Plan B Self-Review Notes

- Spec §7 coverage: multi-select 2–20 ✓ (B4/B5/B6), horizons subset ✓, modes ✓, tier2 MDI/MDA ✓, screening/MI-heatmap/redundancy/monotonicity ✓ (monotonicity serialized in B4; rendered only if records non-empty — plan text of §7 lists it as table data, no dedicated chart), per-feature drill-down link deferred to Plan C polish (recorded).
- Spec §10 coverage: submit/poll/progress/bounded store/tracebacks ✓ (B2), 2s polling ✓ (B6).
- Spec §5.3: window-level memoization only — deliberate scope cut, documented in B3.
- Type consistency: `run_ml_eval` signature identical in B4 (def), B5 (call + fake), B6 (request body fields match `MLEvalRequest`). `MODE_PERMUTATIONS`/`ML_HORIZONS`/`MAX_FACTORS` defined B4, imported B5. `pollJob`/`renderMatrixTable`/`heatColor` defined B6-Step2, used B6-Step3.
- Known risk: `prepare_ml_eval_data` global NaN trimming may drop heavily-NaN factors (e.g. 252d z-scores early in a range) — `data.warnings` is surfaced into `meta.warnings` and the UI should show them; B6 renders `meta.warnings` under the screening table if non-empty (add one line in `renderScreening`).
