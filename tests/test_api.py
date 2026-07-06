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
    assert "mleval.js" in r.text
    assert client.get("/js/mleval.js").status_code == 200


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


def test_ml_eval_flow(client, monkeypatch):
    import factor_bank.server.api as api

    def fake_run(factors, horizons, from_date, to_date, quantiles=5,
                 mode="standard", tier2=False, progress=None, **kw):
        progress("halfway")
        return {"screening": [{"feature": f} for f in factors], "meta": {"mode": mode}}

    monkeypatch.setattr(api, "run_ml_eval", fake_run)
    r = client.post("/api/ml-eval", json={
        "factors": ["pe", "pb"], "horizons": [21],
        "from_date": "2019-01-01", "to_date": "2020-01-01",
    })
    assert r.status_code == 202
    jid = r.json()["job_id"]
    import time
    for _ in range(100):
        rec = client.get(f"/api/jobs/{jid}").json()
        if rec["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    assert rec["status"] == "done"
    assert rec["result"]["meta"]["mode"] == "standard"


def test_ml_eval_validates_before_submit(client):
    r = client.post("/api/ml-eval", json={
        "factors": ["pe"], "horizons": [21],          # only 1 factor
        "from_date": "2019-01-01", "to_date": "2020-01-01",
    })
    assert r.status_code == 400 and "factors" in r.json()["error"]
    r = client.post("/api/ml-eval", json={
        "factors": ["pe", "pb"], "horizons": [10],    # 10 not an ML horizon
        "from_date": "2019-01-01", "to_date": "2020-01-01",
    })
    assert r.status_code == 400


def test_warmup_clears_panel_memo(client, monkeypatch):
    import factor_bank.engine.panel as panel
    panel._memo[("x", "y", 1)] = (0.0, None)
    client.post("/api/warmup")
    assert panel._memo == {}
