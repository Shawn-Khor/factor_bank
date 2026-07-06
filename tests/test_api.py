import pytest
from fastapi.testclient import TestClient

import factor_bank.server.api as api_mod
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


def test_static_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Factor Bank" in r.text
