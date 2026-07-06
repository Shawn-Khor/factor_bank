import pandas as pd
import pytest
from fastapi.testclient import TestClient

import factor_bank.server.api as api_mod
from factor_bank.data import enriched as enriched_mod
from factor_bank.data import sharadar as sharadar_mod
from factor_bank.data import universe as universe_mod
from factor_bank.server.app import create_app


@pytest.fixture
def client(synthetic_market, monkeypatch):
    enriched, spells = synthetic_market
    monkeypatch.setattr(api_mod, "_get_enriched", lambda: enriched)
    monkeypatch.setattr(api_mod, "_get_spells", lambda: spells)
    return TestClient(create_app())


def test_health(client):
    assert client.get("/api/health").json() == {"ok": True}


def test_factors_catalog(client):
    body = client.get("/api/factors").json()
    assert "Value" in body["groups"]
    assert body["horizons"] == [1, 5, 10, 21, 42, 63]
    assert body["quantile_options"] == [3, 5, 10]


def test_evaluate_roundtrip(client):
    r = client.post("/api/evaluate", json={
        "factor": "pe", "from_date": "2019-01-01", "to_date": "2020-01-01",
        "horizon": 21, "n_quantiles": 5,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "STRONG"
    assert body["quality"]["coverage"] == 1.0


def test_evaluate_validation_400(client):
    r = client.post("/api/evaluate", json={
        "factor": "pe", "from_date": "2015-01-01", "to_date": "2020-01-01",
        "horizon": 21, "n_quantiles": 5,
    })
    assert r.status_code == 400
    assert "floor" in r.json()["error"]


def test_evaluate_winsorize_null_matches_raw(client):
    r = client.post("/api/evaluate", json={
        "factor": "pe", "from_date": "2019-01-01", "to_date": "2020-01-01",
        "horizon": 21, "n_quantiles": 5, "winsorize": None,
    })
    assert r.status_code == 200
    metrics = r.json()["metrics"]
    assert metrics["ic"] == metrics["ic_raw"]


def test_static_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Factor Bank" in r.text


def test_warmup_clears_memos_and_reloads(client, monkeypatch):
    # Seed each module's process-lifetime memo with a sentinel so we can prove
    # warmup actually drops it rather than short-circuiting on the old value.
    enriched_mod._memo["enriched"] = "STALE_SENTINEL"
    sharadar_mod._memo["tickers"] = "STALE_SENTINEL"
    sharadar_mod._memo["sp500"] = "STALE_SENTINEL"
    universe_mod._memo["spells"] = "STALE_SENTINEL"

    fake_tickers = pd.DataFrame({"ticker": ["AAA"]})
    fake_events = pd.DataFrame({"ticker": ["AAA"], "action": ["added"], "date": [pd.Timestamp("2019-01-01")]})
    fake_enriched = pd.DataFrame({"ticker": ["AAA"], "date": [pd.Timestamp("2019-01-01")]})

    monkeypatch.setattr(api_mod, "load_tickers", lambda: fake_tickers)
    monkeypatch.setattr(api_mod, "load_sp500_events", lambda: fake_events)
    monkeypatch.setattr(api_mod, "load_enriched", lambda: fake_enriched)

    r = client.post("/api/warmup")
    assert r.status_code == 200
    body = r.json()
    assert body["loaded"] is True
    assert body["n_tickers_meta"] == 1
    assert body["n_sp500_events"] == 1
    assert body["n_enriched_rows"] == 1

    # The sentinels must be gone — clear_memo() ran before the reload calls.
    assert "STALE_SENTINEL" not in enriched_mod._memo.values()
    assert "STALE_SENTINEL" not in sharadar_mod._memo.values()
    assert "STALE_SENTINEL" not in universe_mod._memo.values()
