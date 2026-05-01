#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import eastern_off_hours_window, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, eastern_off_hours_window, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "backpressure_drainer_fleet_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "backpressure_drainer_fleet.lock"
SERVICE_REQUEST_PATH = PROJECT_ROOT / "governance" / "health" / "sql_link_service_request_latest.json"
WRITER_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "jsonl_sql_writer.lock"
MIN_MATERIAL_PENDING_LINES = 100
CORE_HARD_PENDING_LINES = 50_000


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


def _acquire_nonblocking_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate(0)
    handle.write(f"pid={os.getpid()} timestamp_utc={iso_now()}\n")
    handle.flush()
    return handle


def _writer_owner(lock_path: Path = WRITER_LOCK_PATH) -> str:
    try:
        return lock_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _rows_from(backpressure: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = backpressure.get(key)
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _collect_sources(backpressure: dict[str, Any], prefixes: tuple[str, ...], *, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, Any]] = {}
    for key in keys:
        for row in _rows_from(backpressure, key):
            source_rel = str(row.get("source_rel") or "").strip()
            if not source_rel or not source_rel.startswith(prefixes):
                continue
            pending_lines = max(_safe_int(row.get("pending_lines"), 0), 0)
            if pending_lines <= 0:
                continue
            age_seconds = max(_safe_float(row.get("oldest_pending_age_seconds"), 0.0), 0.0)
            current = by_source.get(source_rel)
            if current is None:
                by_source[source_rel] = {
                    "source_rel": source_rel,
                    "pending_lines": pending_lines,
                    "oldest_pending_age_seconds": round(age_seconds, 3),
                }
                continue
            current["pending_lines"] = max(_safe_int(current.get("pending_lines"), 0), pending_lines)
            current["oldest_pending_age_seconds"] = round(
                max(_safe_float(current.get("oldest_pending_age_seconds"), 0.0), age_seconds),
                3,
            )
    return sorted(
        by_source.values(),
        key=lambda row: (_safe_int(row.get("pending_lines"), 0), _safe_float(row.get("oldest_pending_age_seconds"), 0.0)),
        reverse=True,
    )


def _base_env(*, critical: bool) -> dict[str, str]:
    return {
        "INGEST_MAX_DEFERRED_FILES": "6" if critical else "4",
        "JSONL_SQL_MAX_COLD_LANE_FILES": "2" if critical else "1",
        "LOG_DATA_INGRESS": "0",
        "LOG_API_CALLS": "0",
        "LOG_LOOP_STATE": "0",
        "LOG_SHADOW_PNL_ATTRIBUTION": "0",
        "SQL_LINK_SERVICE_INTERVAL_SECONDS": "12" if critical else "15",
        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "25" if critical else "45",
        "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "30",
        "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "240000" if critical else "200000",
        "SQL_LINK_SERVICE_HOT_MAX_ROWS": "2400000" if critical else "1800000",
        "SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB": "0.25" if critical else "0.5",
        "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB": "0.25" if critical else "0.5",
    }


def _profile(
    *,
    name: str,
    reason: str,
    rows: list[dict[str, Any]],
    shards: list[str],
    env: dict[str, str],
    priority_boost: int,
    live_window_safe: bool,
) -> dict[str, Any]:
    pending_lines = sum(max(_safe_int(row.get("pending_lines"), 0), 0) for row in rows)
    oldest_age = max([_safe_float(row.get("oldest_pending_age_seconds"), 0.0) for row in rows] or [0.0])
    path_focus = [str(row.get("source_rel") or "").strip() for row in rows[:8] if str(row.get("source_rel") or "").strip()]
    priority_score = int(priority_boost + pending_lines + (oldest_age / 60.0))
    return {
        "name": name,
        "reason": reason,
        "status": "ready" if pending_lines >= MIN_MATERIAL_PENDING_LINES else "idle",
        "pending_lines": int(pending_lines),
        "oldest_pending_age_seconds": round(oldest_age, 3),
        "priority_score": priority_score,
        "shards": shards,
        "path_focus": path_focus,
        "live_window_safe": bool(live_window_safe),
        "env_overrides": env,
    }


def _is_crypto_decision_source(source_rel: str) -> bool:
    rel = str(source_rel or "")
    return any(
        part in rel
        for part in (
            "shadow_crypto/",
            "shadow_crypto_futures_crypto/",
            "default_crypto_coinbase",
            "crypto_futures_crypto_coinbase",
            "default_crypto_schwab",
            "crypto_futures_crypto_schwab",
        )
    )


def _is_aggressive_decision_source(source_rel: str) -> bool:
    rel = str(source_rel or "")
    return any(
        part in rel
        for part in (
            "shadow_aggressive_",
            "shadow_intraday_aggressive_",
            "shadow_swing_aggressive_",
        )
    )


def _decision_drainer_env(base: dict[str, str], core_rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, str]]:
    regular_focus: list[str] = []
    aggressive_focus: list[str] = []
    crypto_focus: list[str] = []
    for row in core_rows[:12]:
        source_rel = str(row.get("source_rel") or "").strip()
        if not source_rel:
            continue
        if _is_crypto_decision_source(source_rel):
            crypto_focus.append(source_rel)
        elif _is_aggressive_decision_source(source_rel):
            aggressive_focus.append(source_rel)
        else:
            regular_focus.append(source_rel)

    shards: list[str] = []
    env: dict[str, str] = {**base}
    if regular_focus:
        shards.append("trading")
        env["SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS"] = ",".join(regular_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_TRADING_MAX_FILES"] = "16"
        env["SQL_LINK_SERVICE_SHARD_TRADING_MAX_LINES_PER_FILE"] = "64000"
        # Keep the companion aggressive shard in the handoff for mixed equity decision pressure.
        # Some aggressive sleeves write through regular decision-channel paths rather than
        # shadow_aggressive-prefixed files, so this preserves the broader hot-lane sweep.
        shards.append("aggressive_trading")
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_FILES"] = "14"
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE"] = "24000"
    if aggressive_focus:
        if "aggressive_trading" not in shards:
            shards.append("aggressive_trading")
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_PATH_CONTAINS"] = ",".join(aggressive_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_FILES"] = "14"
        env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE"] = "24000"
    if crypto_focus:
        shards.append("crypto_trading")
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_PATH_CONTAINS"] = ",".join(crypto_focus[:8])
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MAX_FILES"] = "14"
        env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MAX_LINES_PER_FILE"] = "64000"

    if not shards:
        shards = ["trading", "aggressive_trading", "crypto_trading"]
    shards = ordered_unique([*shards, "health_fast", "support_watchdog"])
    env["SQL_LINK_SERVICE_SHARDS"] = ",".join(shards)
    return shards, env


def _candidate_drainers(backpressure: dict[str, Any], *, critical: bool) -> list[dict[str, Any]]:
    base = _base_env(critical=critical)
    core_rows = _collect_sources(
        backpressure,
        ("governance/channels/decision/", "decisions/"),
        keys=("top_pending_files",),
    )
    governance_rows = _collect_sources(
        backpressure,
        ("governance/execution_lanes/",),
        keys=("top_pending_files",),
    )
    runtime_rows = _collect_sources(
        backpressure,
        (
            "governance/channels/runtime/",
            "governance/channels/api/",
            "governance/channels/ingress/",
            "governance/channels/loop_state/",
        ),
        keys=("top_pending_files", "top_deferred_pending_files"),
    )
    support_rows = _collect_sources(
        backpressure,
        ("governance/watchdog/",),
        keys=("top_support_telemetry_pending_files", "top_deferred_pending_files"),
    )
    cold_rows = _collect_sources(
        backpressure,
        ("data/stale_stage/", "decision_explanations/"),
        keys=("top_cold_pending_files", "top_deferred_pending_files"),
    )
    decision_shards, decision_env = _decision_drainer_env(base, core_rows)

    profiles = [
        _profile(
            name="core_decision_drainer",
            reason="drain concentrated decision-channel backlog through the matching hot decision shards",
            rows=core_rows,
            shards=decision_shards,
            priority_boost=100_000 if _safe_int(backpressure.get("pending_lines"), 0) >= CORE_HARD_PENDING_LINES else 60_000,
            live_window_safe=True,
            env=decision_env,
        ),
        _profile(
            name="governance_execution_drainer",
            reason="drain stale execution-lane governance backlog before widening broad governance work",
            rows=governance_rows,
            shards=["governance", "health_fast", "support_watchdog"],
            priority_boost=80_000,
            live_window_safe=True,
            env={
                **base,
                "SQL_LINK_SERVICE_SHARDS": "governance,health_fast,support_watchdog",
                "SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS": ",".join(str(row["source_rel"]) for row in governance_rows[:8]),
                "SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_FILES": "14",
                "SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_LINES_PER_FILE": "64000",
            },
        ),
        _profile(
            name="runtime_channel_drainer",
            reason="drain runtime channel files without pulling cold analytics work forward",
            rows=runtime_rows,
            shards=["runtime", "crypto_runtime", "health_fast"],
            priority_boost=50_000,
            live_window_safe=True,
            env={
                **base,
                "SQL_LINK_SERVICE_SHARDS": "runtime,crypto_runtime,health_fast",
                "SQL_LINK_SERVICE_SHARD_RUNTIME_PATH_CONTAINS": ",".join(str(row["source_rel"]) for row in runtime_rows[:8]),
                "SQL_LINK_SERVICE_SHARD_RUNTIME_MAX_FILES": "16",
                "SQL_LINK_SERVICE_SHARD_RUNTIME_MAX_LINES_PER_FILE": "64000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_RUNTIME_MAX_FILES": "10",
            },
        ),
        _profile(
            name="support_watchdog_drainer",
            reason="drain failover, pager, and killswitch support telemetry off the main governance path",
            rows=support_rows,
            shards=["support_watchdog", "health_fast"],
            priority_boost=30_000,
            live_window_safe=True,
            env={
                **base,
                "SQL_LINK_SERVICE_SHARDS": "support_watchdog,health_fast",
                "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_PATH_CONTAINS": ",".join(str(row["source_rel"]) for row in support_rows[:8]),
                "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_MAX_FILES": "20",
                "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_MAX_LINES_PER_FILE": "96000",
                "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_STATE_CHECKPOINT_LINES": "4000",
            },
        ),
        _profile(
            name="cold_stage_drainer",
            reason="drain stale-stage and explanation backlog only during protected drain windows",
            rows=cold_rows,
            shards=["data", "explanations", "crypto_explanations", "health_fast"],
            priority_boost=20_000,
            live_window_safe=False,
            env={
                **base,
                "SQL_LINK_SERVICE_SHARDS": "data,explanations,crypto_explanations,health_fast",
                "JSONL_SQL_MAX_COLD_LANE_FILES": "4" if critical else "2",
                "SQL_LINK_SERVICE_SHARD_DATA_PATH_CONTAINS": ",".join(str(row["source_rel"]) for row in cold_rows[:8]),
                "SQL_LINK_SERVICE_SHARD_DATA_MAX_FILES": "10",
                "SQL_LINK_SERVICE_SHARD_DATA_MAX_LINES_PER_FILE": "64000",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_FILES": "8",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_FILES": "8",
            },
        ),
    ]
    ready = [row for row in profiles if str(row.get("status") or "") == "ready"]
    idle = [row for row in profiles if str(row.get("status") or "") != "ready"]
    ready.sort(key=lambda row: (_safe_int(row.get("priority_score"), 0), _safe_int(row.get("pending_lines"), 0)), reverse=True)
    return ready + idle


def _write_service_request(path: Path, *, active_drainer: dict[str, Any], now_utc: datetime, ttl_seconds: int) -> dict[str, Any]:
    expires_utc = now_utc.timestamp() + max(int(ttl_seconds), 300)
    env = active_drainer.get("env_overrides") if isinstance(active_drainer.get("env_overrides"), dict) else {}
    payload = {
        "timestamp_utc": now_utc.isoformat(),
        "active": True,
        "request_kind": "backpressure_drainer_fleet",
        "reason": f"backpressure_drainer_fleet:{active_drainer.get('name', '')}",
        "requested_at": now_utc.isoformat(),
        "expires_utc": datetime.fromtimestamp(expires_utc, tz=timezone.utc).isoformat(),
        "env_overrides": {str(key): str(value) for key, value in env.items() if str(key).strip()},
    }
    write_payload(path, payload)
    return payload


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    force_live_window: bool = False,
    ttl_seconds: int = 900,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    now = now_utc or datetime.now(timezone.utc)
    health_root = project_root / "governance" / "health"
    backpressure = load_json(health_root / "ingestion_backpressure_latest.json")
    storage_control = load_json(health_root / "ingestion_storage_control_latest.json")
    governor = load_json(health_root / "ingestion_storage_governor_latest.json")
    critical = bool(
        str(governor.get("profile") or "").strip() == "critical_backpressure"
        or str(storage_control.get("severity") or "").strip() == "critical"
        or _safe_int(backpressure.get("pending_lines"), 0) >= CORE_HARD_PENDING_LINES
    )
    window = eastern_off_hours_window(now=now)
    drainers = _candidate_drainers(backpressure, critical=critical)
    ready_drainers = [row for row in drainers if str(row.get("status") or "") == "ready"]
    active_drainer = ready_drainers[0] if ready_drainers else {}
    live_window_allowed = bool(
        force_live_window
        or window.get("active", False)
        or bool(active_drainer.get("live_window_safe", False))
    )
    service_request: dict[str, Any] = {}
    blocked_reasons: list[str] = []
    if not backpressure:
        blocked_reasons.append("missing_backpressure_artifact")
    if active_drainer and not live_window_allowed:
        blocked_reasons.append("market_hours_guard")

    if apply and active_drainer and not blocked_reasons:
        service_request = _write_service_request(
            project_root / "governance" / "health" / "sql_link_service_request_latest.json",
            active_drainer=active_drainer,
            now_utc=now,
            ttl_seconds=ttl_seconds,
        )

    active_env = active_drainer.get("env_overrides") if isinstance(active_drainer.get("env_overrides"), dict) else {}
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": bool(not blocked_reasons),
        "overall_status": "handoff_requested" if service_request else ("ready" if active_drainer and not blocked_reasons else ("idle" if not blocked_reasons else "blocked")),
        "apply_requested": bool(apply),
        "service_request": service_request,
        "service_request_path": str(project_root / "governance" / "health" / "sql_link_service_request_latest.json"),
        "writer_active": bool(_writer_owner(project_root / "governance" / "locks" / "jsonl_sql_writer.lock")),
        "writer_lock_owner": _writer_owner(project_root / "governance" / "locks" / "jsonl_sql_writer.lock"),
        "off_hours_window": window,
        "blocked_reasons": blocked_reasons,
        "active_drainer": {
            key: value
            for key, value in active_drainer.items()
            if key not in {"env_overrides"}
        },
        "ready_drainer_count": len(ready_drainers),
        "candidate_drainers": [
            {key: value for key, value in row.items() if key not in {"env_overrides"}}
            for row in drainers
        ],
        "active_env_override_count": len(active_env),
        "active_env_overrides": active_env,
        "metrics": {
            "core_pending_lines": _safe_int(backpressure.get("pending_lines"), 0),
            "total_pending_lines": _safe_int(backpressure.get("pending_lines_total"), 0),
            "deferred_pending_lines": _safe_int(backpressure.get("pending_lines_deferred"), 0),
            "cold_pending_lines": _safe_int(backpressure.get("pending_lines_cold"), 0),
            "support_pending_lines": _safe_int(backpressure.get("pending_lines_support_telemetry"), 0),
            "ready_drainer_count": len(ready_drainers),
        },
        "recommended_actions": ordered_unique(
            [
                "keep one SQL writer active; use these drainers as focused handoffs instead of parallel SQLite writers",
                "run the highest-priority drainer first, then let the next storage-autopilot cycle re-score the backlog",
                "use live-window-safe drainers for core, runtime, and support pressure; keep cold-stage drainers for protected windows",
                "keep the drainer fleet wired into storage-backpressure-autopilot so focused handoffs happen automatically",
            ]
        ),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Score and hand off focused backpressure drainers to the single SQL writer.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--force-live-window", action="store_true")
    parser.add_argument("--ttl-seconds", type=int, default=900)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_file = Path(args.out_file).expanduser()
    lock_file = Path(args.lock_file).expanduser()

    lock = _acquire_nonblocking_lock(lock_file)
    if lock is None:
        payload = {
            "timestamp_utc": iso_now(),
            "schema_version": 1,
            "ok": True,
            "overall_status": "already_running",
            "busy": True,
            "lock_file": str(lock_file),
        }
        write_payload(out_file, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print("backpressure_drainer_fleet overall_status=already_running")
        return 0

    with lock:
        payload = build_payload(
            project_root,
            apply=bool(args.apply),
            force_live_window=bool(args.force_live_window),
            ttl_seconds=int(args.ttl_seconds),
        )
        payload["lock_file"] = str(lock_file)
        write_payload(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "backpressure_drainer_fleet "
            f"overall_status={payload.get('overall_status', '')} "
            f"ready_drainers={int(payload.get('ready_drainer_count', 0) or 0)}"
        )
    return 0 if str(payload.get("overall_status") or "") in {"already_running", "ready", "idle", "handoff_requested"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
