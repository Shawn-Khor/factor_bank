import json

import pandas as pd
import pytest

from factor_bank.ml.bridge import MODE_PERMUTATIONS, run_ml_eval


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


def test_bad_factor_column_fails_job_naming_the_factor(market_with_noise):
    """Ops-scan #5: a derived factor whose underlying column vanished from
    the enriched frame (schema drift) must abort the whole ml-eval job with
    a message naming the specific factor, not a bare KeyError."""
    enriched, spells = market_with_noise
    enriched = enriched.drop(columns=["pb"])
    with pytest.raises(ValueError, match="book_yield"):
        run_ml_eval(
            ["pe", "book_yield"], [21], "2019-01-01", "2020-01-01",
            mode="quick", enriched=enriched, spells=spells,
        )


def test_mode_map():
    assert MODE_PERMUTATIONS == {"quick": 0, "standard": 50, "thorough": 200}


def test_custom_factor_accepted_through_run_ml_eval(monkeypatch, tmp_path, synthetic_market):
    """Seam test (M-8): a custom factor registered via data/custom.py resolves
    end-to-end through run_ml_eval exactly like a built-in factor — this seam
    crosses data/custom.py, engine/factors.py (compute_factor), and
    ml/bridge.py, and was previously verified by code reading only, not by a
    test."""
    monkeypatch.setenv("FB_CACHE_DIR", str(tmp_path))
    from factor_bank.data import custom

    custom.clear_memo()
    enriched, spells = synthetic_market
    df = enriched.sort_values(["ticker", "date"]).reset_index(drop=True)
    upload = pd.DataFrame({
        "ticker": df["ticker"],
        "date": df["date"].dt.strftime("%Y-%m-%d"),
        "value": df["pe"],
    })
    custom.validate_and_store("custom_pe_seam", upload.to_csv(index=False).encode())
    try:
        out = run_ml_eval(
            ["pe", "custom_pe_seam"], [21], "2019-01-01", "2020-01-01",
            mode="quick", enriched=enriched, spells=spells,
        )
        json.dumps(out)
        feats = {r["feature"] for r in out["screening"]}
        assert "custom_pe_seam" in feats
    finally:
        custom.clear_memo()
