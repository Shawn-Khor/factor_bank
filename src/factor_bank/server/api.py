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
from factor_bank.data import universe as universe_mod
from factor_bank.data.enriched import load_enriched
from factor_bank.data.sharadar import load_sp500_events, load_tickers
from factor_bank.data.universe import get_spells
from factor_bank.engine.catalog import FACTOR_CATALOG
from factor_bank.engine.evaluate import evaluate

router = APIRouter()


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


@router.post("/warmup")
def warmup():
    """Drop in-memory state, then reload through the disk-cache layer (which
    revalidates by ETag) — clearing the module-level memos first is what makes
    this endpoint an actual refresh instead of a silent no-op against
    process-lifetime caches."""
    enriched_mod.clear_memo()
    sharadar_mod.clear_memo()
    universe_mod.clear_memo()
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
