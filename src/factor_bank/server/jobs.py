"""In-process background jobs: submit → poll → result (spec §10).

Single worker by design: the enriched frame dominates RAM, so heavy ML runs
are serialized rather than parallelized.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable


class JobStore:
    def __init__(self, max_jobs: int = 20):
        self._max = max_jobs
        self._ex = ThreadPoolExecutor(max_workers=1)
        self._jobs: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def submit(self, fn: Callable[[Callable[[str], None]], dict]) -> str:
        job_id = uuid.uuid4().hex[:8]
        rec = {
            "id": job_id, "status": "queued", "progress": "",
            "started_at": None, "finished_at": None,
            "result": None, "error": None, "detail": None,
        }
        with self._lock:
            self._jobs[job_id] = rec
            self._evict()

        def _run():
            rec["status"] = "running"
            rec["started_at"] = time.time()
            try:
                rec["result"] = fn(lambda msg: rec.__setitem__("progress", str(msg)))
                rec["status"] = "done"
            except Exception as e:  # noqa: BLE001 — job boundary, full capture
                rec["error"] = str(e)
                rec["detail"] = traceback.format_exc()
                rec["status"] = "error"
            finally:
                rec["finished_at"] = time.time()

        self._ex.submit(_run)
        return job_id

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _evict(self) -> None:
        # Only finished jobs are evicted, oldest first (lock held by caller).
        while len(self._jobs) > self._max:
            victim = next(
                (k for k, r in self._jobs.items() if r["status"] in ("done", "error")),
                None,
            )
            if victim is None:
                return
            self._jobs.pop(victim)


JOBS = JobStore()
