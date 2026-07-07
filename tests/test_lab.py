import json

import pandas as pd
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
    from factor_bank.engine.panel import get_window
    from factor_bank.lab.screen import screen
    enriched, spells = market_with_noise
    horizon = 21
    out = screen(horizon, "2019-06-01", "2020-06-01", top_k=3,
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

    # I-1 embargo: the last `horizon` train dates are held back from stage-1
    # metrics (their forward-return targets would otherwise reach into the
    # holdout price path), and every date in the window is accounted for
    # across train + embargo + holdout — no date silently vanishes or is
    # double-counted.
    assert out["split"]["embargo_dates"] == horizon
    df = get_window("2019-06-01", "2020-06-01", horizon, enriched=enriched, spells=spells)
    from_ts, to_ts = pd.Timestamp("2019-06-01"), pd.Timestamp("2020-06-01")
    in_window = (df["date"] >= from_ts) & (df["date"] <= to_ts)
    total_dates = int(df.loc[in_window, "date"].nunique())
    assert (
        out["split"]["train_dates"] + out["split"]["embargo_dates"] + out["split"]["holdout_dates"]
        == total_dates
    )


def test_screen_validates_window(market_with_noise):
    from factor_bank.lab.screen import screen
    enriched, spells = market_with_noise
    with pytest.raises(ValueError):
        screen(21, "2019-06-01", "2019-07-01", enriched=enriched, spells=spells)  # too few dates


def test_stage2_failure_recorded_separately_not_double_counted(market_with_noise, monkeypatch):
    """F8/item7: a candidate that fails during stage 2 (deep pass) is already
    counted in `processed` (it survived stage 1) — recording it into
    `skipped` too double-counts it. It must land in `stage2_failures`
    instead, and the stage-2 progress tick denominator must reflect the
    actual finalist count, not a hardcoded top_k."""
    import factor_bank.lab.screen as screen_mod

    enriched, spells = market_with_noise
    orig_pivot = screen_mod._pivot_factor
    call_counts: dict[str, int] = {}

    def fake_pivot(df, in_window, columns, name):
        call_counts[name] = call_counts.get(name, 0) + 1
        if name == "pe__sector_rel" and call_counts[name] == 2:
            raise ValueError("synthetic stage-2 failure")
        return orig_pivot(df, in_window, columns, name)

    monkeypatch.setattr(screen_mod, "_pivot_factor", fake_pivot)

    progress_msgs = []
    out = screen_mod.screen(
        21, "2019-06-01", "2020-06-01", top_k=3,
        enriched=enriched, spells=spells,
        candidates=["pe__sector_rel", "ps__zscore_63d", "pe__chg_5d"],
        progress=progress_msgs.append,
    )

    assert out["stage2_failures"] == ["pe__sector_rel"]
    assert "pe__sector_rel" not in out["skipped"]      # not double-counted
    assert out["n_skipped"] == len(out["skipped"])

    stage2_msgs = [m for m in progress_msgs if m.startswith("stage 2:")]
    assert stage2_msgs, "stage 2 progress never ticked"
    # All 3 candidates became finalists (top_k=3, only 3 candidates supplied)
    # -> denominator must be 3, not the requested top_k if it ever diverges.
    assert all(m.endswith("/3") for m in stage2_msgs)


def test_bh_fdr_monotone_in_p(market_with_noise):
    """BH-FDR q-values must be monotone (non-decreasing) with rising p-value."""
    from factor_bank.lab.screen import screen
    enriched, spells = market_with_noise
    out = screen(21, "2019-06-01", "2020-06-01", top_k=3,
                 enriched=enriched, spells=spells,
                 candidates=["pe__sector_rel", "ps__zscore_63d", "pe__chg_5d"])
    rows = [r for r in out["leaderboard"] if r["p_value"] is not None and r["q_value"] is not None]
    assert len(rows) >= 2, "monotonicity assertion below is vacuous with <2 rows"
    rows.sort(key=lambda r: r["p_value"])
    qs = [r["q_value"] for r in rows]
    assert qs == sorted(qs)
