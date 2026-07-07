"""HTTP API. Plans B/C add routes to this same router."""
from __future__ import annotations

import json
import math
import time
import traceback

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from factor_bank.config import ALLOWED_HORIZONS, ALLOWED_QUANTILES, get_settings
from factor_bank.data import custom as custom_mod
from factor_bank.data import enriched as enriched_mod
from factor_bank.data import sharadar as sharadar_mod
from factor_bank.data import store as store_mod
from factor_bank.data import universe as universe_mod
from factor_bank.data.enriched import load_enriched
from factor_bank.data.sharadar import load_sp500_events, load_tickers
from factor_bank.engine import panel as panel_mod
from factor_bank.engine.catalog import FACTOR_CATALOG
from factor_bank.engine.evaluate import evaluate
from factor_bank.engine.factors import TRANSFORMS
from factor_bank.lab.grid import candidate_grid
from factor_bank.lab.screen import screen as lab_screen
from factor_bank.ml.bridge import ML_HORIZONS, MAX_FACTORS, MODE_PERMUTATIONS, run_ml_eval
from factor_bank.server.jobs import JOBS

router = APIRouter()

ALLOWED_SCAN_TABS = {"evaluate", "mleval", "lab"}


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _data_cache_age_hours() -> float | None:
    """Hours since the most recently fetched disk-cache object (the newest
    `fetched_at` across every `objects/*.json` meta file). None if the cache
    dir doesn't exist yet or holds no objects. No S3 calls — reads local meta
    JSON only, so this stays fast enough to call on every health check."""
    objects_dir = get_settings().cache_dir / "objects"
    newest = None
    for meta_path in objects_dir.glob("*.json"):
        try:
            fetched_at = json.loads(meta_path.read_text()).get("fetched_at")
        except (OSError, ValueError):
            continue
        if fetched_at is not None and (newest is None or fetched_at > newest):
            newest = fetched_at
    if newest is None:
        return None
    return (time.time() - newest) / 3600.0


@router.get("/health")
def health():
    age_hours = _data_cache_age_hours()
    counts = JOBS.counts()
    return {
        "ok": True,
        "data_cache_age_hours": round(age_hours, 2) if age_hours is not None else None,
        "jobs_queued": counts["queued"],
        "jobs_running": counts["running"],
    }


@router.get("/factors")
def list_factors():
    groups = dict(FACTOR_CATALOG)
    custom_group = {
        rec["name"]: f"custom upload ({rec['n_rows']} rows, {rec['date_min']}→{rec['date_max']})"
        for rec in store_mod.list_custom()
    }
    if custom_group:
        groups["Custom"] = custom_group
    return {
        "groups": groups,
        "horizons": list(ALLOWED_HORIZONS),
        "quantile_options": list(ALLOWED_QUANTILES),
        "date_floor": get_settings().date_floor,
    }


class EvaluateRequest(BaseModel):
    factor: str
    from_date: str
    to_date: str
    horizon: int
    n_quantiles: int = 5
    winsorize: float | None = 0.01


class MLEvalRequest(BaseModel):
    factors: list[str]
    horizons: list[int] = [21]
    from_date: str
    to_date: str
    quantiles: int = 5
    mode: str = "standard"
    tier2: bool = False


@router.post("/evaluate")
def run_evaluate(req: EvaluateRequest):
    try:
        # No pinned enriched/spells here (same pattern /api/ml-eval and
        # /api/lab/screen already use) — evaluate() routes through
        # engine.panel.get_window, whose TTL memo only engages when the
        # caller passes neither frame, so repeat requests over the same
        # (from_date, to_date, horizon) hit the memo instead of re-slicing
        # the full enriched frame every time.
        result = evaluate(
            req.factor, req.from_date, req.to_date, req.horizon, req.n_quantiles,
            winsorize=req.winsorize,
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": traceback.format_exc()},
        )
    return _sanitize(result)


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

    # Don't resolve/pass enriched/spells here: get_window only memoizes its
    # TTL cache when both are None, so passing no frames lets the window memo
    # engage across jobs (repeat date ranges hit cache instead of re-filtering
    # the full frame). /api/warmup clears the memo for a refresh.
    job_id = JOBS.submit(lambda progress: _sanitize(run_ml_eval(
        req.factors, req.horizons, req.from_date, req.to_date,
        quantiles=req.quantiles, mode=req.mode, tier2=req.tier2,
        progress=progress,
    )))
    return {"job_id": job_id}


class LabScreenRequest(BaseModel):
    horizon: int
    from_date: str
    to_date: str
    top_k: int = 30


@router.post("/lab/screen", status_code=202)
def submit_lab_screen(req: LabScreenRequest):
    # Fail fast on bounds BEFORE queueing.
    if req.horizon not in ALLOWED_HORIZONS:
        return JSONResponse(
            status_code=400,
            content={"error": f"horizon must be one of {list(ALLOWED_HORIZONS)}"},
        )
    if not (1 <= req.top_k <= 100):
        return JSONResponse(status_code=400, content={"error": "top_k must be between 1 and 100"})

    # Same memo-friendly pattern as /api/ml-eval: no pinned frames, so
    # get_window's TTL memo can engage across jobs.
    job_id = JOBS.submit(lambda progress: _sanitize(lab_screen(
        req.horizon, req.from_date, req.to_date, top_k=req.top_k, progress=progress,
    )))
    return {"job_id": job_id}


@router.get("/lab/candidates")
def get_lab_candidates():
    return {"n_candidates": len(candidate_grid()), "transforms": list(TRANSFORMS)}


@router.post("/warmup")
def warmup():
    """Drop in-memory state, then reload through the disk-cache layer (which
    revalidates by ETag) — clearing the module-level memos first is what makes
    this endpoint an actual refresh instead of a silent no-op against
    process-lifetime caches."""
    enriched_mod.clear_memo()
    sharadar_mod.clear_memo()
    universe_mod.clear_memo()
    panel_mod.clear_memo()
    custom_mod.clear_memo()
    try:
        tickers = load_tickers()
        events = load_sp500_events()
        enriched = load_enriched()
        return {
            "loaded": True,
            "n_tickers_meta": len(tickers),
            "n_sp500_events": len(events),
            "n_enriched_rows": len(enriched),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": traceback.format_exc()},
        )


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    rec = JOBS.get(job_id)
    if rec is None:
        return JSONResponse(status_code=404, content={"error": "unknown job"})
    return _sanitize(rec)


class ScanRequest(BaseModel):
    name: str
    tab: str
    config: dict


@router.post("/scans", status_code=201)
def create_scan(req: ScanRequest):
    if not req.name.strip():
        return JSONResponse(status_code=400, content={"error": "name must not be blank"})
    if req.tab not in ALLOWED_SCAN_TABS:
        return JSONResponse(
            status_code=400,
            content={"error": f"tab must be one of {sorted(ALLOWED_SCAN_TABS)}"},
        )
    scan_id = store_mod.create_scan(req.name, req.tab, req.config)
    return {"id": scan_id}


@router.get("/scans")
def list_scans():
    return {"scans": store_mod.list_scans()}


@router.get("/scans/{scan_id}")
def get_scan(scan_id: str):
    rec = store_mod.get_scan(scan_id)
    if rec is None:
        return JSONResponse(status_code=404, content={"error": "unknown scan"})
    return rec


@router.delete("/scans/{scan_id}")
def delete_scan(scan_id: str):
    return {"deleted": store_mod.delete_scan(scan_id)}


@router.post("/custom-factors", status_code=201)
async def upload_custom_factor(name: str = Form(...), file: UploadFile = File(...)):
    raw = await file.read()
    try:
        result = custom_mod.validate_and_store(name, raw)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    return result


@router.delete("/custom-factors/{name}")
def delete_custom_factor(name: str):
    return {"deleted": custom_mod.delete_custom_factor(name)}
