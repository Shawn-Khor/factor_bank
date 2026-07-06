import time

from fastapi.testclient import TestClient

from factor_bank.server.app import create_app
from factor_bank.server.jobs import JobStore


def _wait(store, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = store.get(job_id)
        if rec["status"] in ("done", "error"):
            return rec
        time.sleep(0.02)
    raise TimeoutError


def test_success_lifecycle_and_progress():
    store = JobStore()
    def work(progress):
        progress("stage 1")
        progress("stage 2")
        return {"answer": 42}
    jid = store.submit(work)
    rec = _wait(store, jid)
    assert rec["status"] == "done"
    assert rec["result"] == {"answer": 42}
    assert rec["progress"] == "stage 2"
    assert rec["started_at"] is not None and rec["finished_at"] is not None


def test_error_captures_traceback():
    store = JobStore()
    def boom(progress):
        raise RuntimeError("kapow")
    rec = _wait(store, store.submit(boom))
    assert rec["status"] == "error"
    assert "kapow" in rec["error"]
    assert "RuntimeError" in rec["detail"]  # full traceback
    assert rec["result"] is None


def test_serial_execution_order():
    store = JobStore()
    order = []
    j1 = store.submit(lambda p: order.append(1) or {})
    j2 = store.submit(lambda p: order.append(2) or {})
    _wait(store, j1); _wait(store, j2)
    assert order == [1, 2]  # single worker → strictly serial


def test_eviction_keeps_active_jobs():
    store = JobStore(max_jobs=3)
    done = [_wait(store, store.submit(lambda p: {})) for _ in range(3)]
    j_new = store.submit(lambda p: {})
    _wait(store, j_new)
    assert store.get(j_new) is not None
    assert store.get(done[0]["id"]) is None  # oldest finished evicted


def test_jobs_endpoint_unknown_404():
    client = TestClient(create_app())
    r = client.get("/api/jobs/nope1234")
    assert r.status_code == 404 and "error" in r.json()
