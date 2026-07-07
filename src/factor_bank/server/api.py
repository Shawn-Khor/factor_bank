"""HTTP API. Plans B/C add routes to this same router."""
from __future__ import annotations

import math
import traceback

import pandas as pd
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from factor_bank.config import ALLOWED_HORIZONS, ALLOWED_QUANTILES, get_settings
from factor_bank.data import enriched as enriched_mod
from factor_bank.data import sharadar as sharadar_mod
from factor_bank.data import store as store_mod
from factor_bank.data import universe as universe_mod
from factor_bank.data.enriched import load_enriched
from factor_bank.data.sharadar import load_sp500_events, load_tickers
from factor_bank.data.universe import get_spells
from factor_bank.engine import panel as panel_mod
from factor_bank.engine.catalog import FACTOR_CATALOG
from factor_bank.engine.evaluate import evaluate
from factor_bank.ml.bridge import ML_HORIZONS, MAX_FACTORS, MODE_PERMUTATIONS, run_ml_eval
from factor_bank.server.jobs import JOBS

router = APIRouter()

ALLOWED_SCAN_TABS = {"evaluate", "mleval", "lab"}


# Indirection points so tests (and Plan B jobs) can inject data:
def _get_enriched() -> pd.DataFrame:
    return load_enriched()


def _get_spells() -> pd.DataFrame:
    return get_spells()


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


@router.get("/health")
def health():
    return {"ok": True}


@router.get("/factors")
def list_factors():
    return {
        "groups": FACTOR_CATALOG,
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
        result = evaluate(
            req.factor, req.from_date, req.to_date, req.horizon, req.n_quantiles,
            enriched=_get_enriched(), spells=_get_spells(), winsorize=req.winsorize,
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
