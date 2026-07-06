"""Console entry point: factor-bank serve | warmup."""
from __future__ import annotations

import argparse
import logging


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="factor-bank")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="Run the dashboard server")
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--host", default="0.0.0.0")
    sub.add_parser("warmup", help="Pre-populate the disk cache from S3")
    args = p.parse_args()

    from factor_bank.config import get_settings

    if args.cmd == "serve":
        import uvicorn

        from factor_bank.server.app import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port or get_settings().port)
    elif args.cmd == "warmup":
        from factor_bank.data.enriched import load_enriched
        from factor_bank.data.sharadar import load_sp500_events, load_tickers
        from factor_bank.data.universe import get_spells

        load_tickers()
        load_sp500_events()
        get_spells()
        df = load_enriched()
        print(f"Warmup complete: {len(df):,} enriched rows cached.")
