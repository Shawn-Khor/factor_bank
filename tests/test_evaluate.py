import pytest

from factor_bank.engine.evaluate import evaluate


def test_perfect_factor_end_to_end(synthetic_market):
    enriched, spells = synthetic_market
    out = evaluate("pe", "2019-01-01", "2020-01-01", horizon=21, n_quantiles=5,
                   enriched=enriched, spells=spells)
    m = out["metrics"]
    assert m["rank_ic"] > 0.9           # drift dominates noise → near-perfect rank IC
    assert out["verdict"] == "STRONG"
    q = out["quantile_means"]
    assert q[-1]["mean_return"] > q[0]["mean_return"]
    assert out["longshort_stats"]["total_return"] > 0
    assert out["quality"]["coverage"] == 1.0
    assert out["meta"]["n_tickers_universe"] == 12


def test_membership_filter_applied(synthetic_market):
    enriched, spells = synthetic_market
    spells = spells[spells["ticker"] != "T00"]  # T00 never a member
    out = evaluate("pe", "2019-01-01", "2020-01-01", horizon=21,
                   enriched=enriched, spells=spells)
    assert out["meta"]["n_tickers_universe"] == 11


def test_validation_errors(synthetic_market):
    enriched, spells = synthetic_market
    with pytest.raises(ValueError, match="floor"):
        evaluate("pe", "2015-01-01", "2020-01-01", 21, enriched=enriched, spells=spells)
    with pytest.raises(ValueError, match="Horizon"):
        evaluate("pe", "2019-01-01", "2020-01-01", 2, enriched=enriched, spells=spells)
    with pytest.raises(ValueError, match="Unknown factor"):
        evaluate("nope", "2019-01-01", "2020-01-01", 21, enriched=enriched, spells=spells)


def test_generated_factor_name_works(synthetic_market):
    enriched, spells = synthetic_market
    out = evaluate("pe__chg_21d", "2019-06-01", "2020-01-01", horizon=21,
                   enriched=enriched, spells=spells)
    # pe is constant per ticker → its 21d change is 0 everywhere → no cross-
    # sectional dispersion → metrics must be null-safe, not crash
    assert out["metrics"]["n_obs"] >= 0
