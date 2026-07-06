from pathlib import Path


def test_defaults(monkeypatch):
    for var in ["FB_CACHE_DIR", "FB_DATE_FLOOR", "FB_PORT", "FB_S3_ENRICHED_PATHS"]:
        monkeypatch.delenv(var, raising=False)
    from factor_bank.config import get_settings
    s = get_settings()
    assert s.date_floor == "2018-01-01"
    assert s.port == 8200
    assert s.cache_dir == Path("~/.cache/factor_bank").expanduser()
    assert len(s.enriched_paths) == 2
    assert all(":" in p for p in s.enriched_paths)
    assert s.sharadar_bucket == "tsgs-market-data-prod-ap-southeast-1"


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("FB_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FB_DATE_FLOOR", "2020-01-01")
    monkeypatch.setenv("FB_PORT", "9999")
    monkeypatch.setenv("FB_S3_ENRICHED_PATHS", "r1:b/k1.parquet")
    from factor_bank.config import get_settings
    s = get_settings()
    assert s.cache_dir == tmp_path
    assert s.date_floor == "2020-01-01"
    assert s.port == 9999
    assert s.enriched_paths == ("r1:b/k1.parquet",)
