import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_maintenance import maintenance_hold_snapshot

DEFAULT_DB = PROJECT_ROOT / "data" / "jsonl_link.sqlite3"
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "sqlite_maintenance_latest.json"


class MaintenanceDeadlineExceeded(RuntimeError):
    pass


def _emit_progress(message: str, *, as_json: bool) -> None:
    stream = os.sys.stderr if as_json else os.sys.stdout
    print(message, file=stream, flush=True)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _truthy(raw: Any, default: bool = False) -> bool:
    text = str(raw or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _normalize_temp_store_mode(raw: Any, default: str = "MEMORY") -> str:
    mode = str(raw or default).strip().upper()
    if mode in {"DEFAULT", "FILE", "MEMORY"}:
        return mode
    return str(default or "MEMORY").strip().upper()


def _resource_guard_snapshot(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    payload = _read_json(project_root / "governance" / "health" / "resource_guard_latest.json")
    return payload if isinstance(payload, dict) else {}


def resolve_runtime_settings(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    resource_guard = _resource_guard_snapshot(project_root)
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
        "green": {"temp_store_mode": "MEMORY", "cache_size_kb": 20000, "mmap_size_mb": 0, "analyze_enabled": True},
        "yellow": {"temp_store_mode": "FILE", "cache_size_kb": 12000, "mmap_size_mb": 0, "analyze_enabled": True},
        "red": {"temp_store_mode": "FILE", "cache_size_kb": 4096, "mmap_size_mb": 0, "analyze_enabled": False},
    }[pressure_level]
    temp_store_mode = _normalize_temp_store_mode(
        os.getenv("SQLITE_TEMP_STORE_MODE", defaults["temp_store_mode"]),
        default=defaults["temp_store_mode"],
    )
    cache_size_kb = max(_safe_int(os.getenv("SQLITE_CACHE_SIZE_KB", str(defaults["cache_size_kb"])), defaults["cache_size_kb"]), 1024)
    requested_mmap_size_mb = max(
        _safe_int(os.getenv("SQLITE_MMAP_SIZE_MB", str(defaults["mmap_size_mb"])), defaults["mmap_size_mb"]),
        0,
    )
    mmap_explicitly_allowed = _truthy(os.getenv("SQLITE_ALLOW_MMAP", "0"), False)
    mmap_size_mb = requested_mmap_size_mb if mmap_explicitly_allowed else 0
    cache_spill = _truthy(os.getenv("SQLITE_CACHE_SPILL", "1"), True)
    analyze_enabled = _truthy(
        os.getenv("SQLITE_ANALYZE_ENABLED", "1" if defaults["analyze_enabled"] else "0"),
        defaults["analyze_enabled"],
    )
    optimize_enabled = _truthy(os.getenv("SQLITE_OPTIMIZE_ENABLED", "1"), True)
    auto_vacuum_allowed = pressure_level != "red" or not _truthy(os.getenv("SQLITE_SKIP_AUTO_VACUUM_ON_MEMORY_PRESSURE", "1"), True)
    return {
        "pressure_level": pressure_level,
        "memory_pressure_state": memory_state,
        "memory_pressure_kind": memory_kind,
        "memory_free_pct": round(memory_free_pct, 3),
        "swap_used_gb": round(swap_used_gb, 3),
        "temp_store_mode": temp_store_mode,
        "cache_size_kb": cache_size_kb,
        "cache_size_pragma": -cache_size_kb,
        "mmap_requested_mb": requested_mmap_size_mb,
        "mmap_enabled": bool(mmap_explicitly_allowed and mmap_size_mb > 0),
        "mmap_disabled_reason": "" if mmap_explicitly_allowed or requested_mmap_size_mb <= 0 else "sqlite_mmap_opt_in_required",
        "mmap_size_mb": mmap_size_mb,
        "mmap_size_bytes": int(mmap_size_mb * 1024 * 1024),
        "cache_spill": cache_spill,
        "analyze_enabled": analyze_enabled,
        "optimize_enabled": optimize_enabled,
        "auto_vacuum_allowed": auto_vacuum_allowed,
    }


def _apply_runtime_pragmas(conn: sqlite3.Connection, runtime_settings: dict[str, Any], *, timeout_seconds: float) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA temp_store={_normalize_temp_store_mode(runtime_settings.get('temp_store_mode'), 'MEMORY')}")
    conn.execute(f"PRAGMA cache_size={int(runtime_settings.get('cache_size_pragma') or -20000)}")
    conn.execute(f"PRAGMA mmap_size={int(runtime_settings.get('mmap_size_bytes') or 0)}")
    conn.execute(f"PRAGMA cache_spill={1 if bool(runtime_settings.get('cache_spill', True)) else 0}")
    conn.execute(f"PRAGMA busy_timeout={int(max(float(timeout_seconds), 1.0) * 1000)}")


def _default_db_path() -> Path:
    configured = str(os.getenv("SQL_LINK_SERVICE_PRIMARY_DB", "") or "").strip()
    if configured:
        return Path(configured)
    progress = _read_json(PROJECT_ROOT / "governance" / "health" / "sql_link_service_progress_latest.json")
    primary_db = str(progress.get("primary_db") or "").strip()
    if primary_db:
        return Path(primary_db)
    latest = _read_json(PROJECT_ROOT / "governance" / "health" / "sql_link_service_latest.json")
    latest_db = str(latest.get("db_path") or latest.get("primary_db") or "").strip()
    if latest_db:
        return Path(latest_db)
    return DEFAULT_DB


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def _sqlite_exec_with_retry(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple = (),
    *,
    lock_retries: int,
    lock_retry_delay_seconds: float,
):
    attempt = 0
    while True:
        try:
            return conn.execute(sql, params)
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            is_locked = ("database is locked" in msg) or ("database table is locked" in msg)
            if (not is_locked) or attempt >= max(lock_retries, 0):
                raise
            sleep_s = min(max(lock_retry_delay_seconds, 0.01) * (2 ** attempt), 5.0)
            print(
                f"SQLite busy during maintenance; retrying in {sleep_s:.2f}s "
                f"(attempt {attempt + 1}/{max(lock_retries, 0)})"
            )
            time.sleep(sleep_s)
            attempt += 1


def _raise_if_deadline_expired(deadline_monotonic: float | None) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise MaintenanceDeadlineExceeded("sqlite_maintenance_runtime_exceeded")


def _write_heartbeat(
    payload: dict[str, Any],
    out_path: Path,
    *,
    current_step: str,
    started_monotonic: float,
) -> None:
    payload.update(
        {
            "running": True,
            "current_step": str(current_step or ""),
            "pid": os.getpid(),
            "last_heartbeat_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(max(time.monotonic() - started_monotonic, 0.0), 3),
        }
    )
    heartbeat = dict(payload)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(heartbeat, ensure_ascii=True, indent=2), encoding="utf-8")
        tmp.replace(out_path)
    except Exception:
        pass


def _sqlite_sidecar_path(db_path: Path, suffix: str) -> Path:
    return Path(f"{db_path}{suffix}")


def _size_gb(path: Path) -> float:
    try:
        return float(path.stat().st_size) / (1024 ** 3)
    except Exception:
        return 0.0


def _disk_free_gb(path: Path) -> float:
    try:
        return float(shutil.disk_usage(path).free) / (1024**3)
    except Exception:
        return 0.0


def _vacuum_temp_dir_candidates(db_path: Path, project_root: Path, explicit: str = "") -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    if str(explicit or "").strip():
        candidates.append((Path(str(explicit).strip()).expanduser(), "vacuum_temp_dir_arg"))
    env_sqlite_tmp = str(os.getenv("SQLITE_TMPDIR", "") or "").strip()
    if env_sqlite_tmp:
        candidates.append((Path(env_sqlite_tmp).expanduser(), "sqlite_tmpdir_env"))
    candidates.append((db_path.parent / ".sqlite_tmp", "db_volume_tmpdir"))
    video_root = Path("/Volumes/VIDEO")
    if video_root.exists():
        candidates.append((video_root / "schwab_trading_bot_cold" / "sqlite_tmp", "video_volume_tmpdir"))
    env_tmpdir = str(os.getenv("TMPDIR", "") or "").strip()
    if env_tmpdir:
        candidates.append((Path(env_tmpdir).expanduser(), "tmpdir_env"))
    candidates.append((project_root / ".tmp" / "sqlite_vacuum", "project_tmpdir"))

    seen: set[str] = set()
    unique: list[tuple[Path, str]] = []
    for path, source in candidates:
        key = str(path)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((path, source))
    return unique


def _select_vacuum_temp_dir(
    *,
    db_path: Path,
    project_root: Path,
    db_size_gb: float,
    explicit: str = "",
    min_free_ratio: float = 1.15,
    min_free_gb: float = 8.0,
) -> dict[str, Any]:
    required_gb = max(float(db_size_gb) * max(float(min_free_ratio), 1.0), float(db_size_gb) + max(float(min_free_gb), 0.0))
    evaluations: list[dict[str, Any]] = []
    for candidate, source in _vacuum_temp_dir_candidates(db_path, project_root, explicit):
        row = {
            "path": str(candidate),
            "source": source,
            "exists": False,
            "is_dir": False,
            "free_gb": 0.0,
            "required_gb": round(required_gb, 3),
            "usable": False,
            "reason": "",
        }
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            row["exists"] = candidate.exists()
            row["is_dir"] = candidate.is_dir()
            free_gb = _disk_free_gb(candidate)
            row["free_gb"] = round(free_gb, 3)
            row["usable"] = bool(candidate.exists() and candidate.is_dir() and free_gb >= required_gb)
            if not row["usable"]:
                row["reason"] = "insufficient_free_space" if free_gb < required_gb else "not_a_directory"
        except Exception as exc:
            row["reason"] = f"unusable:{exc}"
        evaluations.append(row)
        if row["usable"]:
            return {
                "selected": True,
                "selected_dir": str(candidate),
                "selected_source": source,
                "required_gb": round(required_gb, 3),
                "free_gb": row["free_gb"],
                "candidate_evaluations": evaluations,
                "reason": "",
            }
    return {
        "selected": False,
        "selected_dir": "",
        "selected_source": "",
        "required_gb": round(required_gb, 3),
        "free_gb": 0.0,
        "candidate_evaluations": evaluations,
        "reason": "insufficient_vacuum_temp_headroom",
    }


def _normalize_checkpoint_mode(raw: str) -> str:
    mode = str(raw or "auto").strip().lower()
    if mode in {"auto", "passive", "truncate", "restart"}:
        return mode
    return "auto"


def _checkpoint_mode_for_wal(wal_size_gb: float, requested_mode: str, truncate_max_gb: float) -> str:
    mode = _normalize_checkpoint_mode(requested_mode)
    if wal_size_gb <= 0.0:
        return ""
    if mode != "auto":
        return mode
    return "truncate" if wal_size_gb <= max(float(truncate_max_gb), 0.0) else "passive"


def _row_count_skip_reason(
    *,
    checkpoint_only: bool,
    skip_row_count: bool,
    db_size_gb: float,
    skip_over_gb: float,
) -> str:
    if bool(skip_row_count):
        return "operator_skip_row_count"
    if bool(checkpoint_only):
        return "checkpoint_only"
    threshold = max(float(skip_over_gb), 0.0)
    if threshold > 0.0 and float(db_size_gb) >= threshold:
        return f"db_size_over_row_count_skip_threshold:{float(db_size_gb):.3f}>={threshold:.3f}"
    return ""


def _analyze_skip_reason(*, skip_analyze: bool, db_size_gb: float, skip_over_gb: float) -> str:
    if bool(skip_analyze):
        return "operator_skip_analyze"
    threshold = max(float(skip_over_gb), 0.0)
    if threshold > 0.0 and float(db_size_gb) >= threshold:
        return f"db_size_over_analyze_skip_threshold:{float(db_size_gb):.3f}>={threshold:.3f}"
    return ""


def _emit(payload: dict, out_path: Path, as_json: bool) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if as_json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "sqlite_maintenance "
            f"ok={int(bool(payload.get('ok', False)))} "
            f"vacuum_ran={int(bool(payload.get('vacuum_ran', False)))} "
            f"size_gb_after={payload.get('size_gb_after', 'n/a')}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply SQLite performance tuning and maintenance.")
    parser.add_argument("--db", default=str(_default_db_path()))
    parser.add_argument("--vacuum", action="store_true")
    parser.add_argument("--no-auto-vacuum", action="store_true")
    parser.add_argument("--checkpoint-only", action="store_true")
    parser.add_argument("--wal-checkpoint-threshold-gb", type=float, default=float(os.getenv("SQLITE_WAL_CHECKPOINT_THRESHOLD_GB", "0.25")))
    parser.add_argument("--wal-truncate-max-gb", type=float, default=float(os.getenv("SQLITE_WAL_TRUNCATE_MAX_GB", "8")))
    parser.add_argument("--wal-checkpoint-mode", choices=("auto", "passive", "truncate", "restart"), default=_normalize_checkpoint_mode(os.getenv("SQLITE_WAL_CHECKPOINT_MODE", "auto")))
    parser.add_argument("--auto-vacuum-over-gb", type=float, default=float(os.getenv("SQLITE_AUTO_VACUUM_OVER_GB", "24")))
    parser.add_argument("--vacuum-min-interval-hours", type=float, default=float(os.getenv("SQLITE_VACUUM_MIN_INTERVAL_HOURS", "24")))
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--analyze-skip-over-gb", type=float, default=float(os.getenv("SQLITE_ANALYZE_SKIP_OVER_GB", "50")))
    parser.add_argument("--skip-row-count", action="store_true")
    parser.add_argument("--row-count-skip-over-gb", type=float, default=float(os.getenv("SQLITE_ROW_COUNT_SKIP_OVER_GB", "50")))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--sqlite-timeout-seconds", type=float, default=float(os.getenv("SQLITE_TIMEOUT_SECONDS", "60")))
    parser.add_argument("--sqlite-lock-retries", type=int, default=int(os.getenv("SQLITE_LOCK_RETRIES", "8")))
    parser.add_argument("--sqlite-lock-retry-delay-seconds", type=float, default=float(os.getenv("SQLITE_LOCK_RETRY_DELAY_SECONDS", "0.25")))
    parser.add_argument("--max-runtime-seconds", type=float, default=float(os.getenv("SQLITE_MAINTENANCE_MAX_RUNTIME_SECONDS", "7200")))
    parser.add_argument("--vacuum-temp-dir", default=os.getenv("SQLITE_VACUUM_TMPDIR", ""))
    parser.add_argument(
        "--vacuum-temp-min-free-ratio",
        type=float,
        default=float(os.getenv("SQLITE_VACUUM_TMP_MIN_FREE_RATIO", "1.15")),
    )
    parser.add_argument(
        "--vacuum-temp-min-free-gb",
        type=float,
        default=float(os.getenv("SQLITE_VACUUM_TMP_MIN_FREE_GB", "8")),
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    out_path = Path(args.out_file)
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    max_runtime_seconds = max(float(args.max_runtime_seconds), 0.0)
    deadline_monotonic = started_monotonic + max_runtime_seconds if max_runtime_seconds > 0.0 else None

    maintenance_hold = maintenance_hold_snapshot(PROJECT_ROOT)
    if bool(maintenance_hold.get("active", False)):
        return _emit(
            {
                "timestamp_utc": timestamp_utc,
                "ok": True,
                "overall_status": "runtime_maintenance_hold",
                "db_path": str(db_path),
                "running": False,
                "checkpoint_only": bool(args.checkpoint_only),
                "route_mutation_performed": False,
                "runtime_maintenance_hold": maintenance_hold,
                "reason": "runtime_maintenance_hold_blocks_sqlite_maintenance",
            },
            out_path,
            args.json,
        )

    if not db_path.exists():
        payload = {
            "timestamp_utc": timestamp_utc,
            "ok": False,
            "db_path": str(db_path),
            "error": f"db_missing:{db_path}",
            "vacuum_ran": False,
            "indexes_touched": 0,
            "jsonl_records_rows": 0,
            "size_gb_before": 0.0,
            "size_gb_after": 0.0,
            "auto_vacuum_over_gb": float(args.auto_vacuum_over_gb),
            "vacuum_min_interval_hours": float(args.vacuum_min_interval_hours),
            "running": False,
            "timed_out": False,
            "max_runtime_seconds": max_runtime_seconds,
        }
        return _emit(payload, out_path, args.json)

    conn = None
    created_indexes = 0
    total_rows = 0
    do_vacuum = False
    size_gb_before = db_path.stat().st_size / (1024 ** 3)
    wal_path = _sqlite_sidecar_path(db_path, "-wal")
    wal_size_gb_before = _size_gb(wal_path)
    analyze_skip_over_gb = max(float(args.analyze_skip_over_gb), 0.0)
    row_count_skip_over_gb = max(float(args.row_count_skip_over_gb), 0.0)

    payload = {
        "timestamp_utc": timestamp_utc,
        "ok": False,
        "db_path": str(db_path),
        "vacuum_ran": False,
        "indexes_touched": 0,
        "jsonl_records_rows": 0,
        "size_gb_before": round(size_gb_before, 3),
        "size_gb_after": round(size_gb_before, 3),
        "wal_size_gb_before": round(wal_size_gb_before, 3),
        "wal_size_gb_after": round(wal_size_gb_before, 3),
        "checkpoint_only": bool(args.checkpoint_only),
        "checkpoint_ran": False,
        "checkpoint_mode_requested": _normalize_checkpoint_mode(args.wal_checkpoint_mode),
        "checkpoint_mode_applied": "",
        "checkpoint_result": {},
        "checkpoint_skipped_reason": "",
        "wal_checkpoint_threshold_gb": float(args.wal_checkpoint_threshold_gb),
        "wal_truncate_max_gb": float(args.wal_truncate_max_gb),
        "auto_vacuum_over_gb": float(args.auto_vacuum_over_gb),
        "no_auto_vacuum": bool(args.no_auto_vacuum),
        "vacuum_min_interval_hours": float(args.vacuum_min_interval_hours),
        "max_runtime_seconds": max_runtime_seconds,
        "analyze_skipped": False,
        "analyze_skipped_reason": "",
        "analyze_skip_over_gb": analyze_skip_over_gb,
        "row_count_skipped": False,
        "row_count_skipped_reason": "",
        "row_count_skip_over_gb": row_count_skip_over_gb,
        "running": True,
        "pid": os.getpid(),
        "current_step": "starting",
        "last_heartbeat_utc": timestamp_utc,
        "elapsed_seconds": 0.0,
        "timed_out": False,
        "vacuum_temp_safety": {},
        "vacuum_safety_skipped": False,
        "vacuum_skipped_reason": "",
    }
    runtime_settings = resolve_runtime_settings(PROJECT_ROOT)
    payload["sqlite_runtime_settings"] = {
        "pressure_level": str(runtime_settings.get("pressure_level") or ""),
        "temp_store_mode": str(runtime_settings.get("temp_store_mode") or ""),
        "cache_size_kb": int(runtime_settings.get("cache_size_kb") or 0),
        "mmap_size_mb": int(runtime_settings.get("mmap_size_mb") or 0),
        "cache_spill": bool(runtime_settings.get("cache_spill", True)),
        "analyze_enabled": bool(runtime_settings.get("analyze_enabled", True)),
        "optimize_enabled": bool(runtime_settings.get("optimize_enabled", True)),
        "auto_vacuum_allowed": bool(runtime_settings.get("auto_vacuum_allowed", True)),
    }
    payload["memory_snapshot"] = {
        "memory_pressure_state": str(runtime_settings.get("memory_pressure_state") or ""),
        "memory_pressure_kind": str(runtime_settings.get("memory_pressure_kind") or ""),
        "memory_free_pct": float(runtime_settings.get("memory_free_pct") or 0.0),
        "swap_used_gb": float(runtime_settings.get("swap_used_gb") or 0.0),
    }
    vacuum_may_run = bool(
        (not args.checkpoint_only)
        and (
            bool(args.vacuum)
            or (
                not bool(args.no_auto_vacuum)
                and bool(runtime_settings.get("auto_vacuum_allowed", True))
                and size_gb_before >= float(args.auto_vacuum_over_gb)
            )
        )
    )
    vacuum_temp_safety: dict[str, Any] = {}
    if vacuum_may_run:
        vacuum_temp_safety = _select_vacuum_temp_dir(
            db_path=db_path,
            project_root=PROJECT_ROOT,
            db_size_gb=size_gb_before,
            explicit=str(args.vacuum_temp_dir or ""),
            min_free_ratio=float(args.vacuum_temp_min_free_ratio),
            min_free_gb=float(args.vacuum_temp_min_free_gb),
        )
        payload["vacuum_temp_safety"] = vacuum_temp_safety
        selected_dir = str(vacuum_temp_safety.get("selected_dir") or "").strip()
        if selected_dir:
            os.environ["SQLITE_TMPDIR"] = selected_dir
            os.environ["TMPDIR"] = selected_dir

    try:
        _write_heartbeat(payload, out_path, current_step="starting", started_monotonic=started_monotonic)
        _emit_progress(
            f"sqlite_maintenance start db={db_path} size_gb={size_gb_before:.3f} "
            f"wal_gb={wal_size_gb_before:.3f} checkpoint_only={str(bool(args.checkpoint_only)).lower()} "
            f"no_auto_vacuum={str(bool(args.no_auto_vacuum)).lower()}",
            as_json=args.json,
        )
        conn = sqlite3.connect(str(db_path), timeout=max(float(args.sqlite_timeout_seconds), 1.0))
        deadline_state = {"expired": False}
        if deadline_monotonic is not None:
            def _progress_handler() -> int:
                if time.monotonic() >= float(deadline_monotonic):
                    deadline_state["expired"] = True
                    return 1
                return 0

            conn.set_progress_handler(_progress_handler, 100000)
        _apply_runtime_pragmas(conn, runtime_settings, timeout_seconds=float(args.sqlite_timeout_seconds))

        if (not args.checkpoint_only) and _table_exists(conn, "jsonl_records"):
            _raise_if_deadline_expired(deadline_monotonic)
            _write_heartbeat(payload, out_path, current_step="index_jsonl_records", started_monotonic=started_monotonic)
            _emit_progress("sqlite_maintenance step=index_jsonl_records", as_json=args.json)
            idx_sql = [
                "CREATE INDEX IF NOT EXISTS idx_jsonl_source_rel_ingested ON jsonl_records(source_rel, ingested_at)",
                "CREATE INDEX IF NOT EXISTS idx_jsonl_source_rel_line ON jsonl_records(source_rel, line_no)",
                "CREATE INDEX IF NOT EXISTS idx_jsonl_action_expr ON jsonl_records((json_extract(payload_json, '$.action')))",
                "CREATE INDEX IF NOT EXISTS idx_jsonl_symbol_expr ON jsonl_records((json_extract(payload_json, '$.symbol')))",
                "CREATE INDEX IF NOT EXISTS idx_jsonl_ts_expr ON jsonl_records((json_extract(payload_json, '$.timestamp_utc')))",
            ]
            for sql in idx_sql:
                _sqlite_exec_with_retry(
                    conn,
                    sql,
                    lock_retries=max(args.sqlite_lock_retries, 0),
                    lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
                )
                created_indexes += 1

        if (not args.checkpoint_only) and _table_exists(conn, "json_file_records"):
            _raise_if_deadline_expired(deadline_monotonic)
            _write_heartbeat(payload, out_path, current_step="index_json_file_records", started_monotonic=started_monotonic)
            _emit_progress("sqlite_maintenance step=index_json_file_records", as_json=args.json)
            idx_sql = [
                "CREATE INDEX IF NOT EXISTS idx_json_file_source_rel_ingested ON json_file_records(source_rel, ingested_at)",
                "CREATE INDEX IF NOT EXISTS idx_json_file_stream_ingested ON json_file_records(stream, ingested_at)",
            ]
            for sql in idx_sql:
                _sqlite_exec_with_retry(
                    conn,
                    sql,
                    lock_retries=max(args.sqlite_lock_retries, 0),
                    lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
                )
                created_indexes += 1

        if not args.checkpoint_only:
            _sqlite_exec_with_retry(
                conn,
                """
                CREATE TABLE IF NOT EXISTS db_maintenance_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    db_path TEXT NOT NULL,
                    vacuum_ran INTEGER NOT NULL,
                    indexes_touched INTEGER NOT NULL,
                    notes TEXT NOT NULL
                )
                """,
                lock_retries=max(args.sqlite_lock_retries, 0),
                lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
            )

        analyze_ran = False
        optimize_ran = False
        payload["analyze_ran"] = False
        payload["optimize_ran"] = False
        if not args.checkpoint_only and (bool(runtime_settings.get("analyze_enabled", True)) or bool(runtime_settings.get("optimize_enabled", True))):
            _raise_if_deadline_expired(deadline_monotonic)
            _write_heartbeat(payload, out_path, current_step="analyze_optimize", started_monotonic=started_monotonic)
            _emit_progress("sqlite_maintenance step=analyze_optimize", as_json=args.json)
            analyze_skip_reason = _analyze_skip_reason(
                skip_analyze=bool(args.skip_analyze),
                db_size_gb=size_gb_before,
                skip_over_gb=analyze_skip_over_gb,
            )
            if bool(runtime_settings.get("analyze_enabled", True)) and not analyze_skip_reason:
                _sqlite_exec_with_retry(
                    conn,
                    "ANALYZE",
                    lock_retries=max(args.sqlite_lock_retries, 0),
                    lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
                )
                analyze_ran = True
                payload["analyze_ran"] = True
            elif analyze_skip_reason:
                payload["analyze_skipped"] = True
                payload["analyze_skipped_reason"] = analyze_skip_reason
                _emit_progress(
                    f"sqlite_maintenance step=analyze_skipped reason={analyze_skip_reason}",
                    as_json=args.json,
                )
            if bool(runtime_settings.get("optimize_enabled", True)):
                _write_heartbeat(payload, out_path, current_step="optimize", started_monotonic=started_monotonic)
                _sqlite_exec_with_retry(
                    conn,
                    "PRAGMA optimize",
                    lock_retries=max(args.sqlite_lock_retries, 0),
                    lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
                )
                optimize_ran = True
                payload["optimize_ran"] = True
        payload["analyze_ran"] = analyze_ran
        payload["optimize_ran"] = optimize_ran

        checkpoint_threshold_gb = max(float(args.wal_checkpoint_threshold_gb), 0.0)
        checkpoint_mode_applied = ""
        if wal_size_gb_before <= 0.0:
            payload["checkpoint_skipped_reason"] = "no_wal"
        elif wal_size_gb_before < checkpoint_threshold_gb:
            payload["checkpoint_skipped_reason"] = "wal_below_threshold"
        else:
            _raise_if_deadline_expired(deadline_monotonic)
            _write_heartbeat(payload, out_path, current_step="wal_checkpoint", started_monotonic=started_monotonic)
            _emit_progress("sqlite_maintenance step=wal_checkpoint", as_json=args.json)
            checkpoint_mode_applied = _checkpoint_mode_for_wal(
                wal_size_gb=wal_size_gb_before,
                requested_mode=str(args.wal_checkpoint_mode),
                truncate_max_gb=float(args.wal_truncate_max_gb),
            )
            if checkpoint_mode_applied:
                row = _sqlite_exec_with_retry(
                    conn,
                    f"PRAGMA wal_checkpoint({checkpoint_mode_applied.upper()})",
                    lock_retries=max(args.sqlite_lock_retries, 0),
                    lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
                ).fetchone()
                payload["checkpoint_ran"] = True
                payload["checkpoint_mode_applied"] = checkpoint_mode_applied
                payload["checkpoint_result"] = {
                    "busy": int(row[0] if row and len(row) > 0 else 0),
                    "log_frames": int(row[1] if row and len(row) > 1 else 0),
                    "checkpointed_frames": int(row[2] if row and len(row) > 2 else 0),
                }
            else:
                payload["checkpoint_skipped_reason"] = "checkpoint_mode_unresolved"

        last_vacuum_ts = None
        if not args.checkpoint_only:
            try:
                row = _sqlite_exec_with_retry(
                    conn,
                    "SELECT timestamp_utc FROM db_maintenance_events WHERE vacuum_ran=1 ORDER BY id DESC LIMIT 1",
                    lock_retries=max(args.sqlite_lock_retries, 0),
                    lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
                ).fetchone()
                if row and row[0]:
                    last_vacuum_ts = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00")).astimezone(timezone.utc)
            except Exception:
                last_vacuum_ts = None

        do_vacuum = bool(args.vacuum and (not args.checkpoint_only))
        if (
            (not args.checkpoint_only)
            and (not args.no_auto_vacuum)
            and (not do_vacuum)
            and bool(runtime_settings.get("auto_vacuum_allowed", True))
            and size_gb_before >= float(args.auto_vacuum_over_gb)
        ):
            if last_vacuum_ts is None:
                do_vacuum = True
            else:
                elapsed_h = (datetime.now(timezone.utc) - last_vacuum_ts).total_seconds() / 3600.0
                do_vacuum = elapsed_h >= float(args.vacuum_min_interval_hours)
        if (
            (not args.checkpoint_only)
            and (not args.no_auto_vacuum)
            and (not bool(args.vacuum))
            and not bool(runtime_settings.get("auto_vacuum_allowed", True))
        ):
            payload["auto_vacuum_skipped_reason"] = "memory_pressure_red"

        if do_vacuum:
            if not bool((payload.get("vacuum_temp_safety") or {}).get("selected", False)):
                do_vacuum = False
                payload["vacuum_safety_skipped"] = True
                payload["vacuum_skipped_reason"] = str(
                    (payload.get("vacuum_temp_safety") or {}).get("reason") or "vacuum_temp_dir_not_selected"
                )
                _emit_progress(
                    f"sqlite_maintenance step=vacuum_skipped reason={payload['vacuum_skipped_reason']}",
                    as_json=args.json,
                )
            else:
                _raise_if_deadline_expired(deadline_monotonic)
                _write_heartbeat(payload, out_path, current_step="vacuum", started_monotonic=started_monotonic)
                _emit_progress("sqlite_maintenance step=vacuum", as_json=args.json)
                _sqlite_exec_with_retry(
                    conn,
                    "VACUUM",
                    lock_retries=max(args.sqlite_lock_retries, 0),
                    lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
                )

        row_count_skip_reason = _row_count_skip_reason(
            checkpoint_only=bool(args.checkpoint_only),
            skip_row_count=bool(args.skip_row_count),
            db_size_gb=size_gb_before,
            skip_over_gb=row_count_skip_over_gb,
        )
        if _table_exists(conn, "jsonl_records") and not row_count_skip_reason:
            _raise_if_deadline_expired(deadline_monotonic)
            _write_heartbeat(payload, out_path, current_step="count_jsonl_records", started_monotonic=started_monotonic)
            _emit_progress("sqlite_maintenance step=count_jsonl_records", as_json=args.json)
            row = _sqlite_exec_with_retry(
                conn,
                "SELECT COUNT(*) FROM jsonl_records",
                lock_retries=max(args.sqlite_lock_retries, 0),
                lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
            ).fetchone()
            total_rows = int(row[0] if row else 0)
        elif row_count_skip_reason:
            payload["row_count_skipped"] = True
            payload["row_count_skipped_reason"] = row_count_skip_reason
            _emit_progress(
                f"sqlite_maintenance step=count_jsonl_records_skipped reason={row_count_skip_reason}",
                as_json=args.json,
            )

        if not args.checkpoint_only:
            _sqlite_exec_with_retry(
                conn,
                "INSERT INTO db_maintenance_events(timestamp_utc, db_path, vacuum_ran, indexes_touched, notes) VALUES (?, ?, ?, ?, ?)",
                (
                    timestamp_utc,
                    str(db_path),
                    1 if do_vacuum else 0,
                    created_indexes,
                    "auto_maintenance",
                ),
                lock_retries=max(args.sqlite_lock_retries, 0),
                lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
            )
            conn.commit()

        size_gb_after = db_path.stat().st_size / (1024 ** 3)
        wal_size_gb_after = _size_gb(wal_path)
        payload.update(
            {
                "ok": True,
                "vacuum_ran": bool(do_vacuum),
                "analyze_ran": bool(analyze_ran),
                "optimize_ran": bool(optimize_ran),
                "indexes_touched": int(created_indexes),
                "jsonl_records_rows": int(total_rows),
                "size_gb_before": round(size_gb_before, 3),
                "size_gb_after": round(size_gb_after, 3),
                "wal_size_gb_after": round(wal_size_gb_after, 3),
                "running": False,
                "current_step": "complete",
                "last_heartbeat_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(max(time.monotonic() - started_monotonic, 0.0), 3),
            }
        )
        _emit_progress(
            f"sqlite_maintenance complete ok=true vacuum_ran={str(bool(do_vacuum)).lower()} "
            f"checkpoint_ran={str(bool(payload.get('checkpoint_ran', False))).lower()}",
            as_json=args.json,
        )
    except Exception as exc:
        size_gb_after = db_path.stat().st_size / (1024 ** 3) if db_path.exists() else 0.0
        wal_size_gb_after = _size_gb(wal_path)
        deadline_expired = isinstance(exc, MaintenanceDeadlineExceeded) or (
            "deadline_state" in locals()
            and bool(deadline_state.get("expired", False))
            and "interrupted" in str(exc).lower()
        )
        payload.update(
            {
                "ok": False,
                "error": "sqlite_maintenance_runtime_exceeded" if deadline_expired else str(exc),
                "vacuum_ran": bool(do_vacuum),
                "analyze_ran": bool(payload.get("analyze_ran", False)),
                "optimize_ran": bool(payload.get("optimize_ran", False)),
                "indexes_touched": int(created_indexes),
                "jsonl_records_rows": int(total_rows),
                "size_gb_after": round(size_gb_after, 3),
                "wal_size_gb_after": round(wal_size_gb_after, 3),
                "running": False,
                "timed_out": bool(deadline_expired),
                "current_step": str(payload.get("current_step") or "error"),
                "last_heartbeat_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(max(time.monotonic() - started_monotonic, 0.0), 3),
            }
        )
        _emit_progress(f"sqlite_maintenance error={exc}", as_json=args.json)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    return _emit(payload, out_path, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
