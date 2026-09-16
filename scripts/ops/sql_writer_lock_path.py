from __future__ import annotations

import os
from pathlib import Path


DEFAULT_SQL_WRITER_LOCK = Path("governance") / "locks" / "jsonl_sql_writer.lock"


def configured_sql_writer_lock_path(project_root: Path) -> Path:
    raw = str(os.getenv("SQL_LINK_SERVICE_LOCK_PATH", "") or "").strip()
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else project_root / path
    return project_root / DEFAULT_SQL_WRITER_LOCK
