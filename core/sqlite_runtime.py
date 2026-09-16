from __future__ import annotations

import json
import math
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.storage_router import inspect_storage_path


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
        return value if math.isfinite(value) else float(default)
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


def _fresh_observation(
    payload: Mapping[str, Any], *, producer_only: bool = False
) -> bool:
    now = datetime.now(timezone.utc)
    keys = ["timestamp_utc"]
    if not producer_only:
        if (
            payload.get("input_evidence_ready") is False
            or payload.get("measurement_refreshed") is False
        ):
            return False
        if "source_timestamp_utc" in payload:
            keys.append("source_timestamp_utc")
    for key in keys:
        try:
            stamp = datetime.fromisoformat(str(payload.get(key)).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return False
        if stamp.tzinfo is None or not 0 <= (now - stamp).total_seconds() <= 120:
            return False
    return True


def sqlite_pressure_snapshot(project_root: Path) -> dict[str, Any]:
    resource_guard = _load_resource_guard(project_root)
    evidence_ready = _fresh_observation(resource_guard)
    memory_state = (
        str(resource_guard.get("memory_pressure_state") or "").strip().lower()
    )
    memory_kind = str(resource_guard.get("memory_pressure_kind") or "").strip().lower()
    swap_used_gb = _safe_float(resource_guard.get("swap_used_gb"), 0.0)
    memory_free_pct = _safe_float(
        resource_guard.get(
            "memory_free_pct", resource_guard.get("memory_available_pct")
        ),
        0.0,
    )
    for raw in (
        resource_guard.get("swap_used_gb"),
        resource_guard.get(
            "memory_free_pct", resource_guard.get("memory_available_pct")
        ),
    ):
        if isinstance(raw, bool) or _safe_float(raw, -1.0) < 0:
            evidence_ready = False

    pressure_level = "green"
    if (
        not evidence_ready
        or memory_state == "red"
        or memory_kind in {"red", "throttled", "observation_unavailable"}
        or swap_used_gb >= 20.0
        or memory_free_pct <= 10.0
        or memory_free_pct > 100
    ):
        pressure_level = "red"
    elif (
        memory_state == "yellow"
        or memory_kind.startswith("swap_only")
        or swap_used_gb >= 10.0
        or (0.0 < memory_free_pct <= 18.0)
    ):
        pressure_level = "yellow"

    try:
        runtime = json.loads(
            (
                Path(project_root)
                / "governance/health/runtime_throttle_control_latest.json"
            ).read_text()
        )
    except (OSError, ValueError):
        runtime = {}
    safety = runtime.get("adaptive_safety_limits") if isinstance(runtime, dict) else {}
    safety = safety if isinstance(safety, dict) else {}
    adaptive_hold = bool(
        isinstance(runtime, dict)
        and _fresh_observation(runtime, producer_only=True)
        and safety.get("active") is True
    )
    if adaptive_hold:
        pressure_level = (
            "red"
            if safety.get("minimum_memory_pressure_level") == "high"
            else "yellow" if pressure_level == "green" else pressure_level
        )
    return {
        "pressure_level": pressure_level,
        "resource_evidence_ready": evidence_ready,
        "resource_source_timestamp_utc": resource_guard.get(
            "source_timestamp_utc", resource_guard.get("timestamp_utc")
        ),
        "adaptive_safety_hold": adaptive_hold,
        "memory_pressure_state": memory_state,
        "memory_pressure_kind": memory_kind,
        "swap_used_gb": round(swap_used_gb, 3),
        "memory_free_pct": round(memory_free_pct, 3),
    }


def resolve_sqlite_runtime_settings(project_root: Path) -> dict[str, Any]:
    pressure = sqlite_pressure_snapshot(project_root)
    pressure_level = pressure["pressure_level"]

    defaults = {
        "green": {
            "temp_store_mode": "MEMORY",
            "cache_size_kb": 8192,
            "mmap_size_mb": 0,
        },
        "yellow": {"temp_store_mode": "FILE", "cache_size_kb": 4096, "mmap_size_mb": 0},
        "red": {"temp_store_mode": "FILE", "cache_size_kb": 2048, "mmap_size_mb": 0},
    }[pressure_level]
    temp_store_mode = normalize_temp_store_mode(
        _env_value(
            "BOT_OPS_SQLITE_TEMP_STORE_MODE",
            "SQLITE_TEMP_STORE_MODE",
            defaults["temp_store_mode"],
        ),
        default=defaults["temp_store_mode"],
    )
    cache_size_kb = max(
        _safe_int(
            _env_value(
                "BOT_OPS_SQLITE_CACHE_SIZE_KB",
                "SQLITE_CACHE_SIZE_KB",
                str(defaults["cache_size_kb"]),
            ),
            defaults["cache_size_kb"],
        ),
        1024,
    )
    requested_mmap_size_mb = max(
        _safe_int(
            _env_value(
                "BOT_OPS_SQLITE_MMAP_SIZE_MB",
                "SQLITE_MMAP_SIZE_MB",
                str(defaults["mmap_size_mb"]),
            ),
            defaults["mmap_size_mb"],
        ),
        0,
    )
    mmap_explicitly_allowed = _truthy(
        _env_value("BOT_OPS_SQLITE_ALLOW_MMAP", "SQLITE_ALLOW_MMAP", "0"), False
    )
    mmap_size_mb = requested_mmap_size_mb if mmap_explicitly_allowed else 0
    if pressure_level != "green":
        temp_store_mode = "FILE"
        cache_size_kb = min(cache_size_kb, defaults["cache_size_kb"])
        mmap_size_mb = 0
    busy_timeout_ms = max(
        _safe_int(_env_value("BOT_OPS_SQLITE_BUSY_TIMEOUT_MS", "", "30000"), 30000), 0
    )
    cache_spill = _truthy(
        _env_value("BOT_OPS_SQLITE_CACHE_SPILL", "SQLITE_CACHE_SPILL", "1"), True
    )
    wal_autocheckpoint_pages = max(
        _safe_int(
            _env_value(
                "BOT_OPS_SQLITE_WAL_AUTOCHECKPOINT_PAGES",
                "SQLITE_WAL_AUTOCHECKPOINT_PAGES",
                "1000",
            ),
            1000,
        ),
        0,
    )
    return {
        **pressure,
        "runtime_contract_version": 2,
        "temp_store_mode": temp_store_mode,
        "cache_size_kb": cache_size_kb,
        "cache_size_pragma": -cache_size_kb,
        "mmap_requested_mb": requested_mmap_size_mb,
        "mmap_enabled": bool(mmap_explicitly_allowed and mmap_size_mb > 0),
        "mmap_disabled_reason": (
            "resource_pressure_or_unavailable_evidence"
            if pressure_level != "green" and requested_mmap_size_mb > 0
            else (
                ""
                if mmap_explicitly_allowed or requested_mmap_size_mb <= 0
                else "ops_sqlite_mmap_opt_in_required"
            )
        ),
        "mmap_size_mb": mmap_size_mb,
        "mmap_size_bytes": int(mmap_size_mb * 1024 * 1024),
        "busy_timeout_ms": busy_timeout_ms,
        "busy_timeout_scope": "configured_ceiling_capped_by_connection_timeout",
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
    conn.execute(
        f"PRAGMA busy_timeout={max(_safe_int(settings.get('busy_timeout_ms'), 30000), 0)}"
    )
    if not readonly:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        f"PRAGMA temp_store={normalize_temp_store_mode(settings.get('temp_store_mode'), 'MEMORY')}"
    )
    conn.execute(f"PRAGMA cache_size={int(settings.get('cache_size_pragma') or -8192)}")
    conn.execute(f"PRAGMA mmap_size={int(settings.get('mmap_size_bytes') or 0)}")
    conn.execute(
        f"PRAGMA cache_spill={1 if bool(settings.get('cache_spill', True)) else 0}"
    )
    wal_autocheckpoint_pages = max(
        _safe_int(settings.get("wal_autocheckpoint_pages"), 1000), 0
    )
    if not readonly:
        conn.execute(f"PRAGMA wal_autocheckpoint={wal_autocheckpoint_pages}")
    if query_only:
        conn.execute("PRAGMA query_only=ON")


def _validated_database_path(path: Path | str) -> Path:
    logical = Path(path).expanduser().absolute()
    route = inspect_storage_path(logical)
    if route["status"] not in {"present", "missing"}:
        raise PermissionError(f"sqlite_route_rejected:{route['status']}")
    resolved = Path(str(route["resolved_path"]))
    if ".." in resolved.parts:
        raise PermissionError("sqlite_route_rejected:unresolved_parent_traversal")
    for base in {logical, resolved}:
        for suffix in ("-wal", "-shm", "-journal"):
            sidecar = inspect_storage_path(Path(f"{base}{suffix}"))
            if sidecar["status"] not in {"present", "missing"}:
                raise PermissionError(
                    f"sqlite_sidecar_route_rejected:{sidecar['status']}"
                )
    return resolved


def connect_sqlite(
    path: Path | str,
    *,
    project_root: Path,
    timeout_seconds: float = 30.0,
    query_only: bool = False,
    readonly: bool = False,
    extra_pragmas: Sequence[str] | None = None,
) -> sqlite3.Connection:
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("SQLite timeout must be finite and nonnegative")
    memory_database = str(path) == ":memory:"
    if memory_database and readonly:
        raise ValueError(
            "A read-only SQLite connection requires an existing database file"
        )
    db_path = None if memory_database else _validated_database_path(path)
    settings = resolve_sqlite_runtime_settings(project_root)
    settings["busy_timeout_ms"] = min(settings["busy_timeout_ms"], int(timeout * 1000))
    if db_path is not None and not readonly:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    if memory_database:
        conn = sqlite3.connect(":memory:", timeout=timeout)
    elif readonly:
        uri = f"{db_path.as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=timeout)
    else:
        conn = sqlite3.connect(str(db_path), timeout=timeout)
    try:
        apply_sqlite_runtime_settings(
            conn, settings, query_only=query_only, readonly=readonly
        )
        for pragma in list(extra_pragmas or ()):
            conn.execute(str(pragma))
    except BaseException:
        conn.close()
        raise
    return conn


def sqlite_integrity_summary(
    path: Path | str,
    *,
    project_root: Path,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    started = time.monotonic()
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("SQLite timeout must be finite and nonnegative")
    deadline = started + timeout
    result: dict[str, Any] = {
        "db_path": str(path),
        "present": False,
        "ok": False,
        "quick_check": "missing",
        "timed_out": False,
        "db_size_bytes": 0,
        "wal_size_bytes": 0,
        "shm_size_bytes": 0,
    }
    conn = None

    def deadline_expired() -> int:
        return int(time.monotonic() >= deadline)

    try:
        db_path = _validated_database_path(path)
        result["db_path"] = str(db_path)
        try:
            result["db_size_bytes"] = db_path.stat().st_size
        except FileNotFoundError:
            return result
        result["present"] = True
        for key, suffix in (("wal_size_bytes", "-wal"), ("shm_size_bytes", "-shm")):
            try:
                result[key] = Path(f"{db_path}{suffix}").stat().st_size
            except FileNotFoundError:
                pass
        if deadline_expired():
            raise TimeoutError("sqlite_integrity_deadline_exceeded")
        conn = connect_sqlite(
            db_path,
            project_root=project_root,
            timeout_seconds=min(max(deadline - time.monotonic(), 0.0), 1.0),
            query_only=True,
            readonly=True,
        )
        # SQLite's busy timeout does not limit query execution. Interrupt VM work too.
        conn.set_progress_handler(deadline_expired, 1000)
        values = {}
        for key, pragma in (
            ("quick_check", "quick_check(1)"),
            ("page_count", "page_count"),
            ("freelist_count", "freelist_count"),
        ):
            if deadline_expired():
                raise TimeoutError("sqlite_integrity_deadline_exceeded")
            row = conn.execute(f"PRAGMA {pragma}").fetchone()
            values[key] = (
                str(row[0] or "unknown") if key == "quick_check" else int(row[0] or 0)
            )
        if deadline_expired():
            raise TimeoutError("sqlite_integrity_deadline_exceeded")
        result.update(values)
        result["ok"] = values["quick_check"] == "ok"
    except Exception as exc:
        timed_out = isinstance(exc, TimeoutError) or (
            isinstance(exc, sqlite3.OperationalError)
            and "interrupted" in str(exc).lower()
            and deadline_expired()
        )
        result.update(
            timed_out=bool(timed_out),
            quick_check="timeout" if timed_out else f"error:{type(exc).__name__}:{exc}",
        )
    finally:
        if conn is not None:
            conn.close()
        result["elapsed_seconds"] = round(time.monotonic() - started, 4)
    return result
