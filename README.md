# factor-bank

A standalone, pip-installable dashboard for cross-sectional evaluation of stock
factors against the historical S&P 500. Point it at S3-hosted Sharadar and Nexus
data, run one command, and get a web UI that scores a factor's predictive power
(IC, Rank IC, IC IR, t-stat/p-value, quantile spreads) with data-quality
diagnostics attached to every result — no Postgres, no alpha-discovery
checkout, just AWS credentials and a browser.

## Install

With the ML extras (LightGBM-backed non-linear evaluation, used by the ML Eval
tab once it ships):

```bash
pip install "factor-bank[ml] @ git+https://github.com/Shawn-Khor/factor_bank.git"
```

Plain (Evaluate tab only, no LightGBM/alpha_eval dependency):

```bash
pip install "factor-bank @ git+https://github.com/Shawn-Khor/factor_bank.git"
```

Requires Python >= 3.10. The `[ml]` extra pulls
`alpha_eval @ git+https://github.com/softdevintegrations/alpha_eval.git` — you
need read access to that repo for the extras to install.

## Configure

All configuration is env vars — no hardcoded paths. Copy `.env.example` to
`.env` in your working directory (auto-loaded via `python-dotenv`) or export
these directly:

| Var | Default | Purpose |
|-----|---------|---------|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | — (required) | S3 reads (standard names; also honors `~/.aws` via boto chain) |
| `FB_CACHE_DIR` | `~/.cache/factor_bank` | disk cache + custom factors + scans.db |
| `FB_S3_ENRICHED_PATHS` | current two-region defaults | comma-separated fallbacks |
| `FB_DATE_FLOOR` | `2018-01-01` | scope rule, overridable |
| `FB_PORT` | `8200` | `factor-bank serve` default |

## Run

```bash
factor-bank warmup   # pre-populate the disk cache from S3 (minutes, cold; seconds thereafter)
factor-bank serve    # http://localhost:8200 by default
```

`factor-bank serve --port 8201 --host 0.0.0.0` overrides the port/host without
touching env vars. Open the printed URL — the Evaluate tab lets you pick a
factor, date range, forward-return horizon, and quantile count, then renders a
verdict badge, the quality panel, and both charts. The ML Eval, Factor Lab,
and Scans tabs are visible but disabled — they land in follow-on work.

## Metric glossary

- **IC** — mean cross-sectional Pearson correlation between the factor and
  the forward return, computed on the winsorized factor (1%/99% per-date clip
  by default) to limit outlier skew. Winsorization is an API parameter —
  `POST /api/evaluate` accepts `winsorize` (default `0.01`; pass `null` to
  disable and compute `ic` on the raw factor); a UI toggle for it is planned
  but not yet wired up. `ic_raw` is always the unclipped calculation, kept for
  comparison against tools that don't winsorize.
- **Rank IC** — mean cross-sectional Spearman correlation between factor rank
  and forward-return rank; unaffected by winsorization since ranks are
  clip-invariant, and the primary metric the verdict is based on.
- **IC IR** — Rank IC information ratio: mean daily Rank IC divided by its
  standard deviation, i.e. how consistently the signal points the same
  direction rather than how big it is on average.
- **t-stat** — t-statistic testing whether the mean daily Rank IC is
  significantly different from zero, given its day-to-day standard deviation
  and the number of independent days observed.
- **p-value** — two-tailed significance level for that t-stat; conventionally
  read as "the odds this Rank IC arose from a factor with true zero signal."
- **Verdict** — a plain-English traffic light derived from `|Rank IC|` and
  `t-stat`: **STRONG** (t ≥ 3 and |Rank IC| ≥ 0.02), **MODERATE** (t ≥ 2 and
  |Rank IC| ≥ 0.01), **WEAK** (t ≥ 2 but below the Rank IC bar), otherwise
  **INSIGNIFICANT**.

## Data caveats

- **T-1 `prev_close_price` convention.** The enriched Nexus parquet stores
  each row's price as the prior session's close, not same-day close. Both the
  factor snapshot and the forward-return calculation use this same T-1
  convention, so returns stay internally consistent and — critically — no
  lookahead bias is introduced. The tradeoff is conservative rather than
  optimistic: a same-day-close convention would inflate measured IC by
  smuggling in same-day information a live signal wouldn't have had yet.
- **Vendor start dates.** Not every vendor column is populated back to the
  `2018-01-01` date floor. Seeking Alpha (`sa_*`) factors only start ~2020;
  requesting an earlier `from_date` for those factors will show reduced
  coverage in the quality panel's per-year breakdown rather than silently
  shrinking N with no explanation.
- **252-day-window factors need a run-up.** Any factor built on a trailing
  252-trading-day window (e.g. `pctile_252d`-style transforms, 52-week
  high/low distance) is only first valid roughly 10 months after the start of
  the requested date range, since the window itself needs to fill before it
  can produce a value — this is expected, not a data gap.
- **Charts need internet access.** Both result charts are rendered with
  Chart.js loaded from the jsdelivr CDN (`index.html`); on a machine without
  outbound internet from the browser, metrics still render but the charts
  silently fail (`Chart is not defined`).

## Architecture

```
factor_bank/
├─ pyproject.toml           # name=factor-bank, console script, pinned deps
├─ README.md
├─ .env.example
├─ src/factor_bank/
│  ├─ config.py             # all env-var config; no hardcoded paths anywhere
│  ├─ data/
│  │  ├─ sharadar.py        # SP500 membership, TICKERS/permaticker map
│  │  ├─ enriched.py        # nexus enriched_stocks loader (multi-region fallback)
│  │  ├─ universe.py        # vectorized S&P 500 membership filter (interval spells)
│  │  └─ disk_cache.py      # local parquet cache layer, ETag+TTL keyed
│  ├─ engine/
│  │  ├─ catalog.py         # FACTOR_CATALOG + custom-factor registry hook
│  │  ├─ factors.py         # compute_factor + transform primitives
│  │  ├─ metrics.py         # vectorized IC battery, winsorization, verdict
│  │  ├─ quality.py         # coverage / staleness / duplicate diagnostics
│  │  ├─ quantiles.py       # quantile spread + long/short stats
│  │  └─ evaluate.py        # end-to-end orchestration (load → compute → metrics)
│  ├─ server/
│  │  ├─ app.py             # FastAPI factory, static mount
│  │  ├─ api.py             # all /api/* routes
│  │  └─ static/            # index.html + fb.css + js/{common,evaluate}.js
│  └─ cli.py                # `factor-bank serve|warmup`
└─ tests/
```

`ml/` (ML Eval tab), `lab/` (Factor Lab), and `server/jobs.py` + `server/scans.py`
(saved scans, async jobs) are designed in the spec but not yet implemented —
the Evaluate tab above is the complete, shipped v1 surface.
