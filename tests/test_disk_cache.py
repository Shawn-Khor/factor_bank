import io
import json
import time

import pandas as pd
import pytest

from factor_bank.data.disk_cache import cached_parquet


class FakeFS:
    """Minimal s3fs stand-in with call counting."""

    def __init__(self, frames: dict[str, pd.DataFrame], etags: dict[str, str]):
        self.frames, self.etags = frames, etags
        self.open_calls = 0
        self.info_calls = 0
        self.fail = False

    def open(self, path):
        if self.fail:
            raise OSError("s3 down")
        self.open_calls += 1
        buf = io.BytesIO()
        self.frames[path].to_parquet(buf)
        buf.seek(0)
        return buf

    def info(self, path):
        if self.fail:
            raise OSError("s3 down")
        self.info_calls += 1
        return {"ETag": self.etags[path]}


@pytest.fixture
def fs():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    return FakeFS({"bucket/k.parquet": df}, {"bucket/k.parquet": "v1"})


def test_first_fetch_then_fresh_hit(fs, tmp_path):
    df1 = cached_parquet(fs, "bucket/k.parquet", tmp_path)
    assert fs.open_calls == 1
    df2 = cached_parquet(fs, "bucket/k.parquet", tmp_path)
    assert fs.open_calls == 1  # served from disk, no second fetch
    pd.testing.assert_frame_equal(df1, df2)


def test_stale_same_etag_reads_local(fs, tmp_path):
    cached_parquet(fs, "bucket/k.parquet", tmp_path)
    meta = next(tmp_path.glob("objects/*.json"))
    m = json.loads(meta.read_text())
    m["fetched_at"] = time.time() - 999_999
    meta.write_text(json.dumps(m))
    cached_parquet(fs, "bucket/k.parquet", tmp_path)
    assert fs.open_calls == 1 and fs.info_calls == 2  # etag checked, not refetched


def test_stale_new_etag_refetches(fs, tmp_path):
    cached_parquet(fs, "bucket/k.parquet", tmp_path)
    meta = next(tmp_path.glob("objects/*.json"))
    m = json.loads(meta.read_text())
    m["fetched_at"] = time.time() - 999_999
    meta.write_text(json.dumps(m))
    fs.etags["bucket/k.parquet"] = "v2"
    fs.frames["bucket/k.parquet"] = pd.DataFrame({"a": [9], "b": ["q"]})
    df = cached_parquet(fs, "bucket/k.parquet", tmp_path)
    assert fs.open_calls == 2
    assert df["a"].tolist() == [9]


def test_offline_falls_back_to_local(fs, tmp_path):
    cached_parquet(fs, "bucket/k.parquet", tmp_path)
    meta = next(tmp_path.glob("objects/*.json"))
    m = json.loads(meta.read_text())
    m["fetched_at"] = time.time() - 999_999
    meta.write_text(json.dumps(m))
    fs.fail = True
    df = cached_parquet(fs, "bucket/k.parquet", tmp_path)
    assert df["a"].tolist() == [1, 2, 3]


def test_columns_selected_after_read(fs, tmp_path):
    df = cached_parquet(fs, "bucket/k.parquet", tmp_path, columns=["a"])
    assert list(df.columns) == ["a"]
    # full frame still cached on disk
    full = pd.read_parquet(next(tmp_path.glob("objects/*.parquet")))
    assert list(full.columns) == ["a", "b"]
