import json

import pytest


def test_grid_shape():
    # bases exclude the 3 ibd ordinals; 8 transforms each; names well-formed
    from factor_bank.lab.grid import candidate_grid
    g = candidate_grid(include_custom=False)
    from factor_bank.engine.catalog import numeric_base_factors
    from factor_bank.engine.factors import TRANSFORMS
    assert len(g) == len(numeric_base_factors()) * len(TRANSFORMS)
    assert all("__" in c for c in g)


def test_screen_finds_signal_and_reports_holdout(market_with_noise):
    # candidates restricted to a small hand-picked set for test speed:
    # pe__chg_5d (zero-dispersion -> skipped or null), pe__sector_rel (signal
    # survives sector demeaning in the fixture), ps__zscore_63d (noise)
    from factor_bank.lab.screen import screen
    enriched, spells = market_with_noise
    out = screen(21, "2019-06-01", "2020-06-01", top_k=3,
                 enriched=enriched, spells=spells,
                 candidates=["pe__sector_rel", "ps__zscore_63d", "pe__chg_5d"])
    lb = {r["candidate"]: r for r in out["leaderboard"]}
    assert "pe__sector_rel" in lb
    top = out["leaderboard"][0]
    assert top["candidate"] == "pe__sector_rel"          # signal wins
    assert top["holdout_rank_ic"] is not None and not top["sign_flip"]
    assert lb["ps__zscore_63d"]["q_value"] >= lb["pe__sector_rel"]["q_value"] or lb["pe__sector_rel"]["q_value"] is None
    assert out["split"]["train_dates"] > out["split"]["holdout_dates"] > 0
    json.dumps(out)


def test_screen_validates_window(market_with_noise):
    from factor_bank.lab.screen import screen
    enriched, spells = market_with_noise
    with pytest.raises(ValueError):
        screen(21, "2019-06-01", "2019-07-01", enriched=enriched, spells=spells)  # too few dates


def test_bh_fdr_monotone_in_p(market_with_noise):
    """BH-FDR q-values must be monotone (non-decreasing) with rising p-value."""
    from factor_bank.lab.screen import screen
    enriched, spells = market_with_noise
    out = screen(21, "2019-06-01", "2020-06-01", top_k=3,
                 enriched=enriched, spells=spells,
                 candidates=["pe__sector_rel", "ps__zscore_63d", "pe__chg_5d"])
    rows = [r for r in out["leaderboard"] if r["p_value"] is not None and r["q_value"] is not None]
    rows.sort(key=lambda r: r["p_value"])
    qs = [r["q_value"] for r in rows]
    assert qs == sorted(qs)
