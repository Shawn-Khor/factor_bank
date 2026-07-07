import pytest


@pytest.fixture(autouse=True)
def _tmp_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("FB_CACHE_DIR", str(tmp_path))


def test_scan_crud_roundtrip():
    from factor_bank.data import store
    sid = store.create_scan("my scan", "evaluate", {"factor": "pe", "horizon": 21})
    assert len(sid) == 6
    rec = store.get_scan(sid)
    assert rec["name"] == "my scan" and rec["config"]["factor"] == "pe"
    assert any(s["id"] == sid for s in store.list_scans())
    assert store.delete_scan(sid) is True
    assert store.get_scan(sid) is None
    assert store.delete_scan(sid) is False


def test_custom_registry_roundtrip():
    from factor_bank.data import store
    store.register_custom("my_sig", "/x/my_sig.parquet", 100, 10, "2019-01-02", "2020-01-02")
    rec = store.get_custom("my_sig")
    assert rec["n_rows"] == 100 and rec["n_tickers"] == 10
    assert store.delete_custom("my_sig") is True
    assert store.get_custom("my_sig") is None
