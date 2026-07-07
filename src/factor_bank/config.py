"""All runtime configuration. Env vars only — no hardcoded paths anywhere else."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import find_dotenv, load_dotenv
    # find_dotenv()'s default (usecwd=False) walks up from the *calling
    # module's* file location, not the process cwd. For a pip-installed
    # (non-editable) package that means it walks up from site-packages and
    # never finds a `.env` copied into the working directory as the README
    # instructs — usecwd=True makes it search from cwd upward instead, which
    # is the documented behavior. (An editable dev install masked this,
    # since site-packages then resolves to the repo checkout, which does
    # have a `.env`.)
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:  # dotenv is a convenience, not a requirement at runtime
    pass

DEFAULT_ENRICHED_PATHS = (
    "us-east-1:tsgs-market-data-prod-us-east-1/nexus-data-prod/cache/enriched_stocks.parquet",
    "ap-southeast-1:tsgs-market-data-prod-ap-southeast-1/nexus-data-prod/cache/enriched_stocks.parquet",
)

ALLOWED_HORIZONS = (1, 5, 10, 21, 42, 63)
ALLOWED_QUANTILES = (3, 5, 10)


@dataclass(frozen=True)
class Settings:
    aws_access_key_id: str | None
    aws_secret_access_key: str | None
    cache_dir: Path
    enriched_paths: tuple[str, ...]  # entries are "region:bucket/key"
    date_floor: str
    port: int
    sharadar_bucket: str = "tsgs-market-data-prod-ap-southeast-1"
    sharadar_region: str = "ap-southeast-1"


def get_settings() -> Settings:
    raw_paths = os.environ.get("FB_S3_ENRICHED_PATHS", "")
    paths = tuple(p.strip() for p in raw_paths.split(",") if p.strip()) or DEFAULT_ENRICHED_PATHS
    return Settings(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        cache_dir=Path(os.environ.get("FB_CACHE_DIR", "~/.cache/factor_bank")).expanduser(),
        enriched_paths=paths,
        date_floor=os.environ.get("FB_DATE_FLOOR", "2018-01-01"),
        port=int(os.environ.get("FB_PORT", "8200")),
    )
