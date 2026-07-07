# factor-bank

A standalone, pip-installable dashboard for cross-sectional evaluation of stock
factors against the historical S&P 500. Point it at S3-hosted Sharadar and Nexus
data, run one command, and get a web UI that scores a factor's predictive power
(IC, Rank IC, IC IR, t-stat/p-value, quantile spreads) with data-quality
diagnostics attached to every result — no Postgres, no alpha-discovery
checkout, just AWS credentials and a browser.

## Install

With the ML extras (LightGBM-backed non-linear evaluation, used by the ML Eval
tab):

```bash
pip install "factor-bank[ml] @ git+https://github.com/Shawn-Khor/factor_bank.git"
```

Plain (Evaluate, Factor Lab, Scans, and custom factors — no LightGBM/alpha_eval
dependency):

```bash
pip install "factor-bank @ git+https://github.com/Shawn-Khor/factor_bank.git"
```

Requires Python >= 3.10. The `[ml]` extra pulls
`alpha_eval @ git+https://github.com/softdevintegrations/alpha_eval.git` — you
need read access to that repo for the extras to install.

The plain install is deliberately fully functional on its own: every tab
renders and every endpoint except `/api/ml-eval` works. Submitting an ML Eval
job without the `[ml]` extra installed doesn't crash the server — the job
fails fast with a status you can poll (`GET /api/jobs/{id}` →
`status: "error"`, `error` mentioning `alpha_eval`), because the
`alpha_eval` import happens lazily inside the job, not at server startup.
Install `[ml]` and restart to pick it up.

> **Dev note:** the `[ml]` extra's `alpha_eval` git dependency needs read access to `softdevintegrations/alpha_eval`. On a dev box with a local checkout, `pip install -e /path/to/alpha_eval` into the same venv is equivalent.

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
touching env vars. On this host, port 8200 is already occupied by the legacy
alpha-discovery dashboard, so factor-bank runs on 8201 here — set `FB_PORT=8201`
or pass `--port 8201`. Open the printed URL — four tabs:

- **Evaluate** — pick a factor, date range, forward-return horizon, and
  quantile count, then renders a verdict badge, the quality panel, and both
  charts. `POST /api/evaluate`.
- **ML Eval** — pick 2–20 factors and one or more horizons for a
  LightGBM-backed non-linear battery (mutual information, distance
  correlation, quantile spread, monotonicity, redundancy, and — with Tier 2 —
  MDI/MDA importance) run as a background job; requires the `[ml]` extra (see
  Install). `POST /api/ml-eval` (202 + `job_id`), polled via
  `GET /api/jobs/{job_id}`.
- **Factor Lab** — mines the ~430-candidate factor grid for you; see
  [Factor Lab](#factor-lab) below. `POST /api/lab/screen` (202 + `job_id`).
- **Scans** — every tab's "Save scan" button persists its current config for
  one-click (or one-URL) recall; see [Saved scans](#saved-scans) below.

The Evaluate tab also has an **＋ Upload custom factor** control for bringing
your own signal into the catalog, from which it's available on every other
tab too; see [Custom factors](#custom-factors) below.

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
- **q-value** — Factor Lab only. The stage-1 p-value for a candidate after
  Benjamini-Hochberg false-discovery-rate correction across every candidate
  screened in the same run (`scipy.stats.false_discovery_control`). Screening
  hundreds of candidates at once inflates the odds that some clear the raw
  p-value bar by chance alone; q-value is the p-value you'd need to beat once
  that multiple-comparisons cost is priced in, so **rank leaderboard rows by
  q-value, not p-value**.
- **Holdout Rank IC** — Factor Lab only. Rank IC recomputed on the trailing
  30% of the requested date range (the "holdout" dates), which stage 1's
  screening and top-K ranking never see. The stage-1 (train) Rank IC that
  drives ranking and q-values is in-sample by construction — it's exactly
  what a 432-candidate grid search is optimizing over — so the holdout column
  is the only number in that row not subject to that selection bias. A
  candidate whose sign flips between train and holdout is flagged with a
  `flip` badge.

## Saved scans

Every tab has a **Save scan** button that persists its current settings
(factor(s), dates, horizon(s), mode — whatever that tab's config is) as a
named row in `<cache_dir>/factor_bank.db` (SQLite). Saving returns a share
URL of the form `http://host:port/?scan=<id>` — opening that URL loads the
app and rehydrates the target tab's controls from the saved config
automatically, so a scan is a shareable link, not just a local bookmark.
The **Scans** tab lists every saved scan (name, tab, created-at) and lets you
delete old ones.

Endpoints: `POST /api/scans` (`{name, tab, config}`, `tab` ∈
`evaluate`/`mleval`/`lab`) → `{id}`; `GET /api/scans` → list; `GET
/api/scans/{id}` → one record (what `?scan=` fetches); `DELETE
/api/scans/{id}`.

## Custom factors

Don't have your signal in the catalog? Upload it. From the Evaluate tab's
**＋ Upload custom factor** control, pick a name and a CSV with exactly three
columns — `ticker,date,value` — and it's validated, written to
`<cache_dir>/custom_factors/<name>.parquet`, and registered under a
**Custom** group in the factor catalog, available from every tab (Evaluate,
ML Eval, Factor Lab) exactly like a built-in factor.

Validation, all client-facing (a failure returns a 400 with the specific
reason, nothing is written on failure):

- **Name** must match `^[a-z][a-z0-9_]{0,39}$`, must not contain `__`
  (reserved for the catalog's generated-transform names, e.g.
  `pe__zscore_63d`), and must not collide with an existing catalog factor
  name.
- **File** must be ≤ 20 MB and parse as CSV with columns exactly
  `ticker,date,value` (no extras, no reordering). `date` must be parseable;
  `value` must be numeric (non-numeric cells fail the whole upload, listing
  the bad-row count); `(ticker, date)` pairs must be unique (duplicates fail
  the whole upload too, same way).

Endpoints: `POST /api/custom-factors` (multipart form: `name`, `file`) →
`{name, n_rows, n_tickers, date_min, date_max}`; `DELETE
/api/custom-factors/{name}`.

## Factor Lab

Answers "which of the ~430 catalog transforms (plus any custom factors)
actually holds up?" without you hand-picking candidates first — but a grid
search over that many candidates will always turn up some that look good by
chance, so the Lab is built around a two-stage funnel specifically to
surface that risk rather than hide it:

1. **Stage 1 (train, in-sample).** Split the requested date range 70/30 into
   train and holdout. Run the Rank IC battery on every candidate against the
   train dates only, then apply Benjamini-Hochberg FDR correction across all
   of that stage's p-values to get each candidate's **q-value**. Rank by
   `|IC IR|` and keep the top `top_k` (default 30) as finalists.
2. **Stage 2 (holdout, deep pass).** For each finalist, compute Rank IC on
   the holdout dates the finalist never touched in stage 1 or the top-K cut
   — plus a train/holdout sign-flip flag, mutual information, and distance
   correlation. The leaderboard is sorted by `|IC IR|` and shows both the
   train Rank IC and the **Holdout Rank IC** side by side.

The Lab tab shows a permanent banner — *"Grid-mined results are in-sample
until validated — trust the holdout column."* — because the train column and
the q-value ranking derived from it are exactly what the search optimized
over, so they're expected to look better than a candidate's true forward
signal; only the holdout column wasn't part of that optimization. See the
[metric glossary](#metric-glossary) for precise q-value/holdout-Rank-IC
definitions.

Endpoints: `POST /api/lab/screen` (`{horizon, from_date, to_date, top_k}`) →
202 + `job_id`, polled via `GET /api/jobs/{job_id}`; `GET
/api/lab/candidates` → `{n_candidates, transforms}` (grid size and the
transform vocabulary it's built from).

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

## Operations

- **Process supervision.** Run under the systemd unit shipped at
  `deploy/factor-bank.service` rather than a bare `nohup`/shell job — it
  gives you restart-on-crash, logs in `journalctl` instead of a temp file
  that vanishes on reboot, and (with `loginctl enable-linger <user>`, once)
  survival across reboots for a per-user unit. `systemctl --user status
  factor-bank` / `journalctl --user -u factor-bank -f` are the first two
  commands to reach for when something looks wrong; `journalctl --user -u
  factor-bank -p warning` surfaces the stale-cache/S3 warnings below without
  scrolling past the per-request access log.
- **Stale cache on S3 failure.** `data/disk_cache.py` treats *any* exception
  from `fs.info()` (expired/rotated AWS creds, `AccessDenied`, network
  unreachable — all indistinguishable at that call site) as "S3 is
  unreachable" and serves the last good local parquet, logging a WARNING
  rather than failing the request. This is deliberate (uptime over
  freshness) — it means credential rot doesn't hard-fail anything for as
  long as the local TTL cache stays populated, so `/api/health`'s
  `data_cache_age_hours` (see below) plus the journald warning log are the
  only proactive signals that the data is actually stale; nothing pages
  anyone. If the local cache is empty (fresh box, or a cleared
  `FB_CACHE_DIR`) and creds are bad, `load_enriched()` raises and
  `/api/evaluate`/`/api/warmup` surface a 500 with a traceback instead.
- **Single-worker job queue + stuck-job recovery.** `/api/ml-eval` and
  `/api/lab/screen` run on one background worker by design (the enriched
  frame dominates RAM, so heavy runs are serialized rather than
  parallelized — see `server/jobs.py`). There is no per-job timeout or
  cancel API today, so one wedged job blocks every job submitted after it
  (`/api/jobs/{id}`'s `n_ahead` keeps growing for everyone else, while
  `/api/evaluate` — which doesn't go through the job queue — keeps working
  normally, which can make a stuck queue easy to miss). `/api/health`'s
  `jobs_queued`/`jobs_running` fields are the fastest way to notice this
  from a script or a glance. The only recovery today is restarting the
  process: `systemctl --user restart factor-bank`.
- **Health endpoint.** `GET /api/health` returns `{ok, data_cache_age_hours,
  jobs_queued, jobs_running}` — `data_cache_age_hours` is the age of the
  newest successfully fetched S3 object under `<cache_dir>/objects` (`null`
  if the cache is empty), computed from local meta JSON only (no S3 calls,
  so it stays cheap to poll).
- **T+h survivorship in long/short economics.** Both the quantile-bucket
  return calculation and the IC battery require a ticker to have prices at
  *both* T and T+horizon; the last `horizon` sessions before an S&P 500
  deletion never contribute a forward return, since that final leg would
  need a post-deletion price the pivot doesn't have. This is a mild,
  horizon-scaling survivorship tilt (not a code bug) — worth remembering
  when comparing long-horizon (42D/63D) results against a venue that handles
  deletions differently.
- **De-overlapped long/short economics (this version).** `longshort_stats`
  and `longshort_cumulative` from `/api/evaluate` no longer compound
  overlapping horizon-session returns as if each were earned in a single
  day. At `horizon=1` nothing changes. At `horizon > 1`, `total_return` and
  `annualized_sharpe` are now computed on a properly de-overlapped basis
  (each horizon-session spread converted to its per-session-equivalent daily
  rate before compounding; Sharpe computed on the non-overlapping
  `::horizon` subsample) — **these numbers will differ substantially from
  the legacy alpha-discovery dashboard**, which does not de-overlap and
  therefore overstates both total return and Sharpe roughly `horizon`-fold
  at horizon > 1. Do not compare the two tools' long/short numbers directly
  above horizon 1.

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
│  │  ├─ disk_cache.py      # local parquet cache layer, ETag+TTL keyed
│  │  ├─ custom.py          # custom-factor CSV validation + parquet store
│  │  └─ store.py           # SQLite: saved scans + custom-factor registry
│  ├─ engine/
│  │  ├─ catalog.py         # FACTOR_CATALOG + custom-factor registry hook
│  │  ├─ factors.py         # compute_factor + transform primitives
│  │  ├─ metrics.py         # vectorized IC battery, winsorization, verdict
│  │  ├─ quality.py         # coverage / staleness / duplicate diagnostics
│  │  ├─ quantiles.py       # quantile spread + long/short stats
│  │  ├─ panel.py           # memoized multi-factor data window (ML Eval + Lab)
│  │  └─ evaluate.py        # end-to-end orchestration (load → compute → metrics)
│  ├─ ml/
│  │  └─ bridge.py          # factor-bank matrices → alpha_eval.ml_eval adapter
│  ├─ lab/
│  │  ├─ grid.py            # ~430-candidate factor grid
│  │  └─ screen.py          # two-stage screen: train Rank IC + BH-FDR, holdout deep pass
│  ├─ server/
│  │  ├─ app.py             # FastAPI factory, static mount
│  │  ├─ api.py             # all /api/* routes
│  │  ├─ jobs.py            # in-process background job store (ML Eval + Lab)
│  │  └─ static/            # index.html + fb.css + js/{common,evaluate,mleval,lab,scans}.js
│  └─ cli.py                # `factor-bank serve|warmup`
└─ tests/
```

The Evaluate tab is the base-install v1 surface; ML Eval, Factor Lab, Scans,
and custom factors shipped in follow-on work and are all covered above. Only
ML Eval needs the `[ml]` extra — everything else in this tree runs off the
base install.
