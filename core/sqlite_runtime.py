from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _truthy(raw: Any, default: bool = False) -> bool:
    text = str(raw or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _load_resource_guard(project_root: Path) -> dict[str, Any]:
    path = Path(project_root) / "governance" / "health" / "resource_guard_latest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _env_value(primary: str, fallback: str = "", default: str = "") -> str:
    import os

    for name in (primary, fallback):
        if not name:
            continue
        value = str(os.getenv(name, "") or "").strip()
        if value:
            return value
    return str(default or "")


def normalize_temp_store_mode(raw: Any, default: str = "MEMORY") -> str:
    mode = str(raw or default).strip().upper()
    if mode in {"DEFAULT", "FILE", "MEMORY"}:
        return mode
    return str(default or "MEMORY").strip().upper()


def resolve_sqlite_runtime_settings(project_root: Path) -> dict[str, Any]:
    resource_guard = _load_resource_guard(project_root)
    memory_state = str(resource_guard.get("memory_pressure_state") or "").strip().lower()
    memory_kind = str(resource_guard.get("memory_pressure_kind") or "").strip().lower()
    swap_used_gb = _safe_float(resource_guard.get("swap_used_gb"), 0.0)
    memory_free_pct = _safe_float(resource_guard.get("memory_free_pct"), 0.0)

    pressure_level = "green"
    if memory_state == "red" or memory_kind in {"red", "throttled"} or swap_used_gb >= 20.0 or memory_free_pct <= 10.0:
        pressure_level = "red"
    elif (
        memory_state == "yellow"
        or memory_kind.startswith("swap_only")
        or swap_used_gb >= 10.0
        or (0.0 < memory_free_pct <= 18.0)
    ):
        pressure_level = "yellow"

    defaults = {
        "green": {"temp_store_mode": "MEMORY", "cache_size_kb": 8192, "mmap_size_mb": 64},
        "yellow": {"temp_store_mode": "FILE", "cache_size_kb": 4096, "mmap_size_mb": 24},
        "red": {"temp_store_mode": "FILE", "cache_size_kb": 2048, "mmap_size_mb": 8},
    }[pressure_level]
    temp_store_mode = normalize_temp_store_mode(
        _env_value("BOT_OPS_SQLITE_TEMP_STORE_MODE", "SQLITE_TEMP_STORE_MODE", defaults["temp_store_mode"]),
        default=defaults["temp_store_mode"],
    )
    cache_size_kb = max(
        _safe_int(
            _env_value("BOT_OPS_SQLITE_CACHE_SIZE_KB", "SQLITE_CACHE_SIZE_KB", str(defaults["cache_size_kb"])),
            defaults["cache_size_kb"],
        ),
        1024,
    )
    mmap_size_mb = max(
        _safe_int(
            _env_value("BOT_OPS_SQLITE_MMAP_SIZE_MB", "SQLITE_MMAP_SIZE_MB", str(defaults["mmap_size_mb"])),
            defaults["mmap_size_mb"],
        ),
        0,
    )
    busy_timeout_ms = max(_safe_int(_env_value("BOT_OPS_SQLITE_BUSY_TIMEOUT_MS", "", "30000"), 30000), 1000)
    cache_spill = _truthy(_env_value("BOT_OPS_SQLITE_CACHE_SPILL", "SQLITE_CACHE_SPILL", "1"), True)
    wal_autocheckpoint_pages = max(
        _safe_int(_env_value("BOT_OPS_SQLITE_WAL_AUTOCHECKPOINT_PAGES", "SQLITE_WAL_AUTOCHECKPOINT_PAGES", "1000"), 1000),
        0,
    )
    return {
        "pressure_level": pressure_level,
        "memory_pressure_state": memory_state,
        "memory_pressure_kind": memory_kind,
        "swap_used_gb": round(swap_used_gb, 3),
        "memory_free_pct": round(memory_free_pct, 3),
        "temp_store_mode": temp_store_mode,
        "cache_size_kb": cache_size_kb,
        "cache_size_pragma": -cache_size_kb,
        "mmap_size_mb": mmap_size_mb,
        "mmap_size_bytes": int(mmap_size_mb * 1024 * 1024),
        "busy_timeout_ms": busy_timeout_ms,
        "cache_spill": cache_spill,
        "wal_autocheckpoint_pages": wal_autocheckpoint_pages,
    }


def apply_sqlite_runtime_settings(
    conn: sqlite3.Connection,
    settings: Mapping[str, Any],
    *,
    query_only: bool = False,
    readonly: bool = False,
) -> None:
    if not readonly:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA busy_timeout={int(settings.get('busy_timeout_ms') or 30000)}")
    conn.execute(f"PRAGMA temp_store={normalize_temp_store_mode(settings.get('temp_store_mode'), 'MEMORY')}")
    conn.execute(f"PRAGMA cache_size={int(settings.get('cache_size_pragma') or -8192)}")
    conn.execute(f"PRAGMA mmap_size={int(settings.get('mmap_size_bytes') or 0)}")
    conn.execute(f"PRAGMA cache_spill={1 if bool(settings.get('cache_spill', True)) else 0}")
    wal_autocheckpoint_pages = max(_safe_int(settings.get("wal_autocheckpoint_pages"), 1000), 0)
    if (not readonly) and wal_autocheckpoint_pages > 0:
        conn.execute(f"PRAGMA wal_autocheckpoint={wal_autocheckpoint_pages}")
    if query_only:
        conn.execute("PRAGMA query_only=ON")


def connect_sqlite(
    path: Path | str,
    *,
    project_root: Path,
    timeout_seconds: float = 30.0,
    query_only: bool = False,
    readonly: bool = False,
    extra_pragmas: Sequence[str] | None = None,
) -> sqlite3.Connection:
    db_path = Path(path).expanduser()
    if not readonly:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    if readonly:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=max(float(timeout_seconds), 1.0))
    else:
        conn = sqlite3.connect(str(db_path), timeout=max(float(timeout_seconds), 1.0))
    apply_sqlite_runtime_settings(conn, resolve_sqlite_runtime_settings(project_root), query_only=query_only, readonly=readonly)
    for pragma in list(extra_pragmas or ()):
        conn.execute(str(pragma))
    return conn


def sqlite_integrity_summary(
    path: Path | str,
    *,
    project_root: Path,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    db_path = Path(path).expanduser()
    wal_path = Path(f"{db_path}-wal")
    shm_path = Path(f"{db_path}-shm")
    if not db_path.exists():
        return {
            "db_path": str(db_path),
            "present": False,
            "ok": False,
            "quick_check": "missing",
            "db_size_bytes": 0,
            "wal_size_bytes": 0,
            "shm_size_bytes": 0,
        }
    try:
        conn = connect_sqlite(
            db_path,
            project_root=project_root,
            timeout_seconds=timeout_seconds,
            query_only=True,
            readonly=True,
        )
        try:
            quick_check = str(conn.execute("PRAGMA quick_check").fetchone()[0] or "").strip() or "unknown"
            page_count = int(conn.execute("PRAGMA page_count").fetchone()[0] or 0)
            freelist_count = int(conn.execute("PRAGMA freelist_count").fetchone()[0] or 0)
        finally:
            conn.close()
    except Exception as exc:
        return {
            "db_path": str(db_path),
            "present": True,
            "ok": False,
            "quick_check": f"error:{type(exc).__name__}:{exc}",
            "db_size_bytes": int(db_path.stat().st_size),
            "wal_size_bytes": int(wal_path.stat().st_size) if wal_path.exists() else 0,
            "shm_size_bytes": int(shm_path.stat().st_size) if shm_path.exists() else 0,
        }
    return {
        "db_path": str(db_path),
        "present": True,
        "ok": quick_check == "ok",
        "quick_check": quick_check,
        "page_count": page_count,
        "freelist_count": freelist_count,
        "db_size_bytes": int(db_path.stat().st_size),
        "wal_size_bytes": int(wal_path.stat().st_size) if wal_path.exists() else 0,
        "shm_size_bytes": int(shm_path.stat().st_size) if shm_path.exists() else 0,
    }
