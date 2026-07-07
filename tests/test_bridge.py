import json

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


def test_mode_map():
    assert MODE_PERMUTATIONS == {"quick": 0, "standard": 50, "thorough": 200}
