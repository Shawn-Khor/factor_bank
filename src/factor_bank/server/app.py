from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from factor_bank.server.api import router


def create_app() -> FastAPI:
    app = FastAPI(title="Factor Bank")
    app.include_router(router, prefix="/api")
    static_dir = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app
