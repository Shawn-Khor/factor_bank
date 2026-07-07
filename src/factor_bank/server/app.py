from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from factor_bank.server.api import router


def create_app() -> FastAPI:
    app = FastAPI(title="Factor Bank")
    # /api/evaluate's response (quantile means + longshort_cumulative series)
    # compresses ~72% (measured); 2048B minimum avoids wasting CPU gzip'ing
    # tiny responses like /api/health.
    app.add_middleware(GZipMiddleware, minimum_size=2048)
    app.include_router(router, prefix="/api")
    static_dir = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app
