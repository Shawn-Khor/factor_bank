import pandas as pd

import factor_bank.engine.panel as panel_mod
from factor_bank.engine.panel import clear_memo, get_window


def test_window_buffers_filters_sorts(synthetic_market):
    enriched, spells = synthetic_market
    w = get_window("2019-06-02", "2019-12-31", 21, enriched=enriched, spells=spells)
    # 400-day lookback buffer reaches the fixture start
    assert w["date"].min() < pd.Timestamp("2019-06-02")
    # forward buffer extends past to_date
    assert w["date"].max() > pd.Timestamp("2019-12-31")
    # sorted by (ticker, date)
    assert w.equals(w.sort_values(["ticker", "date"]).reset_index(drop=True))
    assert w["ticker"].nunique() == 12


def test_membership_filter_applied(synthetic_market):
    enriched, spells = synthetic_market
    w = get_window("2019-06-02", "2019-12-31", 21,
                   enriched=enriched, spells=spells[spells["ticker"] != "T00"])
    assert "T00" not in set(w["ticker"])


def test_memo_only_for_real_loaders(synthetic_market, monkeypatch):
    enriched, spells = synthetic_market
    clear_memo()
    calls = {"n": 0}

    def fake_load():
        calls["n"] += 1
        return enriched

    monkeypatch.setattr(panel_mod, "load_enriched", fake_load)
    monkeypatch.setattr(panel_mod, "get_spells", lambda: spells)
    get_window("2019-06-02", "2019-12-31", 21)
    get_window("2019-06-02", "2019-12-31", 21)   # same key → memo hit
    assert calls["n"] == 1
    assert len(panel_mod._memo) == 1
    # injected frames must NOT populate the memo
    clear_memo()
    get_window("2019-06-02", "2019-12-31", 21, enriched=enriched, spells=spells)
    assert len(panel_mod._memo) == 0
