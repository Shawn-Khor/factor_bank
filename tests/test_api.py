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


@pytest.fixture
def scans_client(synthetic_market, monkeypatch, tmp_path):
    # Dedicated fixture (rather than folding this into `client`) so the saved-scans
    # DB lands in a tmp dir without forcing every other client-based test onto a
    # cold FB_CACHE_DIR — several of them (e.g. test_warmup_clears_panel_memo) hit
    # the real disk-cache/S3 path unmocked and are fast only because that cache is
    # normally warm.
    enriched, spells = synthetic_market
    monkeypatch.setattr(api_mod, "_get_enriched", lambda: enriched)
    monkeypatch.setattr(api_mod, "_get_spells", lambda: spells)
    monkeypatch.setenv("FB_CACHE_DIR", str(tmp_path))
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
        assert kw.get("enriched") is None and kw.get("spells") is None
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


def test_scans_crud_roundtrip(scans_client):
    r = scans_client.post("/api/scans", json={
        "name": "s1", "tab": "evaluate", "config": {"factor": "pe", "horizon": 21},
    })
    assert r.status_code == 201
    sid = r.json()["id"]

    r = scans_client.get(f"/api/scans/{sid}")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "s1" and body["tab"] == "evaluate"
    assert body["config"]["factor"] == "pe"

    r = scans_client.get("/api/scans")
    assert any(s["id"] == sid for s in r.json()["scans"])

    r = scans_client.delete(f"/api/scans/{sid}")
    assert r.json() == {"deleted": True}

    r = scans_client.get(f"/api/scans/{sid}")
    assert r.status_code == 404
    assert "error" in r.json()


def test_scans_blank_name_400(scans_client):
    r = scans_client.post("/api/scans", json={"name": "   ", "tab": "evaluate", "config": {}})
    assert r.status_code == 400


def test_scans_bad_tab_400(scans_client):
    r = scans_client.post("/api/scans", json={"name": "s1", "tab": "nope", "config": {}})
    assert r.status_code == 400


def test_scans_delete_unknown_returns_false(scans_client):
    r = scans_client.delete("/api/scans/zzzzzz")
    assert r.json() == {"deleted": False}


def _custom_csv_bytes() -> bytes:
    return (
        pd.DataFrame({
            "ticker": ["T00", "T00", "T01", "T01"],
            "date": ["2019-01-02", "2019-01-03", "2019-01-02", "2019-01-03"],
            "value": [1.0, 2.0, -1.0, -2.0],
        })
        .to_csv(index=False)
        .encode()
    )


def test_custom_factor_upload_roundtrip_and_catalog_group(scans_client):
    r = scans_client.post(
        "/api/custom-factors",
        data={"name": "my_custom"},
        files={"file": ("my_custom.csv", _custom_csv_bytes(), "text/csv")},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "my_custom"
    assert body["n_rows"] == 4
    assert body["n_tickers"] == 2
    assert body["date_min"] == "2019-01-02"
    assert body["date_max"] == "2019-01-03"

    catalog = scans_client.get("/api/factors").json()
    assert "Custom" in catalog["groups"]
    assert "my_custom" in catalog["groups"]["Custom"]
    assert "4 rows" in catalog["groups"]["Custom"]["my_custom"]

    r = scans_client.delete("/api/custom-factors/my_custom")
    assert r.json() == {"deleted": True}

    catalog2 = scans_client.get("/api/factors").json()
    assert "Custom" not in catalog2["groups"]


def test_custom_factor_upload_bad_name_400(scans_client):
    r = scans_client.post(
        "/api/custom-factors",
        data={"name": "Bad-Name"},
        files={"file": ("x.csv", _custom_csv_bytes(), "text/csv")},
    )
    assert r.status_code == 400
    assert "name must match" in r.json()["error"]


def test_custom_factor_upload_catalog_collision_400(scans_client):
    r = scans_client.post(
        "/api/custom-factors",
        data={"name": "pe"},
        files={"file": ("x.csv", _custom_csv_bytes(), "text/csv")},
    )
    assert r.status_code == 400
    assert "collides with catalog factor" in r.json()["error"]


def test_custom_factor_upload_bad_columns_400(scans_client):
    bad = pd.DataFrame({"a": [1], "b": [2], "c": [3]}).to_csv(index=False).encode()
    r = scans_client.post(
        "/api/custom-factors",
        data={"name": "my_custom2"},
        files={"file": ("x.csv", bad, "text/csv")},
    )
    assert r.status_code == 400
    assert "columns must be exactly" in r.json()["error"]


def test_custom_factor_delete_unknown_returns_false(scans_client):
    r = scans_client.delete("/api/custom-factors/nope")
    assert r.json() == {"deleted": False}


def test_custom_factor_evaluate_through_normal_flow(scans_client):
    scans_client.post(
        "/api/custom-factors",
        data={"name": "my_custom3"},
        files={"file": ("x.csv", _custom_csv_bytes(), "text/csv")},
    )
    r = scans_client.post("/api/evaluate", json={
        "factor": "my_custom3", "from_date": "2019-01-01", "to_date": "2020-01-01",
        "horizon": 21, "n_quantiles": 5,
    })
    assert r.status_code == 200
    assert "coverage" in r.json()["quality"]


def test_warmup_clears_custom_memo(scans_client, monkeypatch):
    import factor_bank.data.custom as custom_mod

    # scans_client points FB_CACHE_DIR at a cold tmp dir, so an unmocked
    # /api/warmup would fall through disk-cache to a real S3 download —
    # stub the same three loaders test_warmup_clears_memos_and_reloads does.
    monkeypatch.setattr(api_mod, "load_tickers", lambda: pd.DataFrame({"ticker": ["AAA"]}))
    monkeypatch.setattr(api_mod, "load_sp500_events", lambda: pd.DataFrame(
        {"ticker": ["AAA"], "action": ["added"], "date": [pd.Timestamp("2019-01-01")]}
    ))
    monkeypatch.setattr(api_mod, "load_enriched", lambda: pd.DataFrame(
        {"ticker": ["AAA"], "date": [pd.Timestamp("2019-01-01")]}
    ))

    scans_client.post(
        "/api/custom-factors",
        data={"name": "my_custom4"},
        files={"file": ("x.csv", _custom_csv_bytes(), "text/csv")},
    )
    # Populate the memo by evaluating (compute_factor -> load_custom).
    scans_client.post("/api/evaluate", json={
        "factor": "my_custom4", "from_date": "2019-01-01", "to_date": "2020-01-01",
        "horizon": 21, "n_quantiles": 5,
    })
    assert "my_custom4" in custom_mod._memo
    scans_client.post("/api/warmup")
    assert custom_mod._memo == {}
