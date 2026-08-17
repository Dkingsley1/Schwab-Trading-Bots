#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PAPER_PROFITABILITY_LOCK_ENV = "PAPER_PROFITABILITY_GENERATION_LOCK_HELD"
PAPER_PROFITABILITY_LOCK_NAME = "paper_profitability_generation.lock"


def paper_profitability_lock_path(project_root: Path) -> Path:
    override = os.getenv("PAPER_PROFITABILITY_GENERATION_LOCK_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return project_root / "governance" / "locks" / PAPER_PROFITABILITY_LOCK_NAME


@contextmanager
def paper_profitability_generation_lock(
    project_root: Path,
    *,
    timeout_seconds: float = 120.0,
) -> Iterator[Any | None]:
    if os.getenv(PAPER_PROFITABILITY_LOCK_ENV, "").strip() == "1":
        yield None
        return

    lock_path = paper_profitability_lock_path(project_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    deadline = time.monotonic() + max(float(timeout_seconds), 0.0)
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                handle.close()
                raise TimeoutError(f"paper profitability generation lock timed out: {lock_path}")
            time.sleep(0.05)

    handle.seek(0)
    handle.truncate(0)
    handle.write(
        f"pid={os.getpid()} acquired_utc={datetime.now(timezone.utc).isoformat()}"
    )
    handle.flush()
    try:
        yield handle
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
