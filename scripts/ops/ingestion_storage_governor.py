#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "ingestion_storage_governor_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.storage_pressure_override"


def _load_json(path: Path) -> dict[str, Any]:
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


def _override_lines(profile_name: str, env_overrides: dict[str, str]) -> list[str]:
    lines = [
        "# Auto-managed by scripts/ops/ingestion_storage_governor.py",
        f"BOT_INGESTION_STORAGE_PROFILE={profile_name}",
    ]
    for key, value in sorted(env_overrides.items()):
        lines.append(f"{key}={value}")
    return lines


def _write_override(path: Path, profile_name: str, env_overrides: dict[str, str]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(_override_lines(profile_name, env_overrides)) + "\n"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _storage_profile(
    *,
    hard_gate: bool,
    core_pending_lines: int,
    deferred_pending_lines: int,
    cold_pending_lines: int,
    support_pending_lines: int,
    stale_stage_pending_lines: int,
    retention_debt_gb: float,
    route_drift: bool,
) -> str:
    effective_deferred_pending_lines = max(
        int(deferred_pending_lines) - int(stale_stage_pending_lines) - int(support_pending_lines),
        0,
    )
    effective_cold_pending_lines = max(int(cold_pending_lines) - int(stale_stage_pending_lines), 0)
    if (
        hard_gate
        or core_pending_lines >= 50000
        or effective_deferred_pending_lines >= 250000
        or effective_cold_pending_lines >= 100000
        or retention_debt_gb >= 5.0
        or route_drift
    ):
        return "critical_backpressure"
    if (
        core_pending_lines >= 15000
        or effective_deferred_pending_lines >= 100000
        or effective_cold_pending_lines >= 10000
        or support_pending_lines >= 100000
        or stale_stage_pending_lines >= 100000
        or retention_debt_gb > 0.0
    ):
        return "elevated_backpressure"
    return "steady_state"


def _critical_deferred_budget(*, core_pending_lines: int, deferred_pending_lines: int, route_drift: bool) -> int:
    if route_drift or deferred_pending_lines <= 0:
        return 0
    if core_pending_lines <= 5000:
        return 2
    if core_pending_lines <= 15000:
        return 1
    return 0


def _profile_env(
    profile_name: str,
    project_root: Path,
    *,
    core_pending_lines: int = 0,
    deferred_pending_lines: int = 0,
    route_drift: bool = False,
) -> dict[str, str]:
    routed_primary_db = str(project_root / "data" / "jsonl_link.sqlite3")
    routed_queue_db = str(project_root / "data" / "bot_channel_queue.sqlite3")
    base = {
        "SQL_LINK_SERVICE_PRIMARY_DB": routed_primary_db,
        "BOT_CHANNEL_QUEUE_DB": routed_queue_db,
        "SQL_LINK_SERVICE_QUEUE_DB": routed_queue_db,
    }
    if profile_name == "critical_backpressure":
        deferred_budget = _critical_deferred_budget(
            core_pending_lines=core_pending_lines,
            deferred_pending_lines=deferred_pending_lines,
            route_drift=route_drift,
        )
        explanation_max_files = "4" if deferred_budget >= 2 else "3" if deferred_budget == 1 else "2"
        base.update(
            {
                "INGEST_MAX_DEFERRED_FILES": str(deferred_budget),
                "JSONL_SQL_MAX_COLD_LANE_FILES": "0",
                "LOG_DATA_INGRESS": "0",
                "LOG_API_CALLS": "0",
                "LOG_LOOP_STATE": "0",
                "LOG_GATE_EVALUATIONS": "0",
                "LOG_GATE_PASSES": "0",
                "LOG_SUB_BOT_DECISIONS": "0",
                "LOG_MASTER_VARIANT_DECISIONS": "0",
                "LOG_DECISION_EXPLANATIONS": "0",
                "LOG_SHADOW_PNL_ATTRIBUTION": "0",
                "CHANNEL_LOG_PRIMARY_MODE": "channel",
                "LEGACY_HOT_CHANNEL_MIRROR_ENABLED": "0",
                "SQL_LINK_SERVICE_INTERVAL_SECONDS": "20",
                "SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB": "0.5",
                "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB": "0.5",
                "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_ROWS": "250000",
                "SQL_LINK_SERVICE_WAL_CHECKPOINT_MIN_INTERVAL_SECONDS": "60",
                "SQL_LINK_SERVICE_WAL_TRUNCATE_MAX_GB": "2",
                "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "60",
                "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "180000",
                "SQL_LINK_SERVICE_HOT_MAX_ROWS": "1800000",
                "SQL_LINK_SERVICE_HOT_DAYS": "3",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_FILES": explanation_max_files,
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_FILES": explanation_max_files,
                "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_MAX_LINES_PER_FILE": "96000",
                "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_STATE_CHECKPOINT_LINES": "4000",
                "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_MAX_FILES": "1",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_MAX_FILES": "1",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_MAX_DB_GB": "4",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_TRIGGER_GROWTH_GB": "0.5",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_TRIGGER_ROWS": "120000",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_HOT_DAYS": "2",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_BATCH_SIZE": "180000",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_MAX_ROWS": "1200000",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_MIN_INTERVAL_SECONDS": "60",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_ARCHIVE_PERIOD": "day",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_ARCHIVE_RETENTION_DAYS": "30",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_VACUUM_THRESHOLD_GB": "2",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_MAX_DB_GB": "3",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_TRIGGER_GROWTH_GB": "0.4",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_TRIGGER_ROWS": "100000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_HOT_DAYS": "2",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_BATCH_SIZE": "160000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_MAX_ROWS": "900000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_MIN_INTERVAL_SECONDS": "60",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_ARCHIVE_PERIOD": "day",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_ARCHIVE_RETENTION_DAYS": "21",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_VACUUM_THRESHOLD_GB": "1.5",
                "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_HOT_RETENTION_BATCH_SIZE": "220000",
                "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_HOT_RETENTION_MAX_ROWS": "1500000",
                "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_HOT_RETENTION_MIN_INTERVAL_SECONDS": "60",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_BATCH_SIZE": "220000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_MAX_ROWS": "1500000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_MIN_INTERVAL_SECONDS": "60",
            }
        )
        return base
    if profile_name == "elevated_backpressure":
        base.update(
            {
                "INGEST_MAX_DEFERRED_FILES": "1",
                "JSONL_SQL_MAX_COLD_LANE_FILES": "0",
                "LOG_DATA_INGRESS": "1",
                "LOG_API_CALLS": "1",
                "LOG_LOOP_STATE": "1",
                "LOG_GATE_EVALUATIONS": "0",
                "LOG_GATE_PASSES": "0",
                "LOG_SHADOW_PNL_ATTRIBUTION": "0",
                "CHANNEL_LOG_PRIMARY_MODE": "channel",
                "LEGACY_HOT_CHANNEL_MIRROR_ENABLED": "0",
                "SQL_LINK_SERVICE_INTERVAL_SECONDS": "30",
                "SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB": "0.75",
                "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB": "0.75",
                "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_ROWS": "400000",
                "SQL_LINK_SERVICE_WAL_CHECKPOINT_MIN_INTERVAL_SECONDS": "120",
                "SQL_LINK_SERVICE_WAL_TRUNCATE_MAX_GB": "3",
                "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "120",
                "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "140000",
                "SQL_LINK_SERVICE_HOT_MAX_ROWS": "1200000",
                "SQL_LINK_SERVICE_HOT_DAYS": "4",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_FILES": "4",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_FILES": "4",
                "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_MAX_LINES_PER_FILE": "96000",
                "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_STATE_CHECKPOINT_LINES": "4000",
                "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_MAX_FILES": "2",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_MAX_FILES": "2",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_MAX_DB_GB": "6",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_TRIGGER_GROWTH_GB": "0.75",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_TRIGGER_ROWS": "160000",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_HOT_DAYS": "3",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_BATCH_SIZE": "140000",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_MAX_ROWS": "900000",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_MIN_INTERVAL_SECONDS": "90",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_ARCHIVE_PERIOD": "day",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_ARCHIVE_RETENTION_DAYS": "45",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_VACUUM_THRESHOLD_GB": "2.5",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_MAX_DB_GB": "4.5",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_TRIGGER_GROWTH_GB": "0.5",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_TRIGGER_ROWS": "120000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_HOT_DAYS": "3",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_BATCH_SIZE": "120000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_MAX_ROWS": "700000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_MIN_INTERVAL_SECONDS": "90",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_ARCHIVE_PERIOD": "day",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_ARCHIVE_RETENTION_DAYS": "30",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_VACUUM_THRESHOLD_GB": "2",
                "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_HOT_RETENTION_BATCH_SIZE": "180000",
                "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_HOT_RETENTION_MAX_ROWS": "1000000",
                "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_HOT_RETENTION_MIN_INTERVAL_SECONDS": "90",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_BATCH_SIZE": "180000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_MAX_ROWS": "1000000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_MIN_INTERVAL_SECONDS": "90",
            }
        )
        return base
    base.update(
        {
            "INGEST_MAX_DEFERRED_FILES": "2",
            "JSONL_SQL_MAX_COLD_LANE_FILES": "1",
            "LOG_DATA_INGRESS": "1",
            "LOG_API_CALLS": "1",
            "LOG_LOOP_STATE": "1",
            "LOG_SHADOW_PNL_ATTRIBUTION": "1",
            "SQL_LINK_SERVICE_INTERVAL_SECONDS": "45",
            "SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB": "2",
        }
    )
    return base


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
    action: str,
    changed: bool = False,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    storage_control = _load_json(health_root / "ingestion_storage_control_latest.json")
    backpressure = _load_json(health_root / "ingestion_backpressure_latest.json")
    queue = _load_json(health_root / "ingestion_priority_queue_latest.json")
    mount = _load_json(health_root / "storage_mount_guard_latest.json")
    failback = _load_json(health_root / "storage_failback_sync_latest.json")
    split_brain = _load_json(health_root / "storage_split_brain_reconciler_latest.json")
    sql_service = _load_json(health_root / "sql_link_service_latest.json")
    sql_progress = _load_json(health_root / "sql_link_service_progress_latest.json")
    health_gates = _load_json(health_root / "health_gates_latest.json")

    core_pending_lines = _safe_int(backpressure.get("pending_lines"), _safe_int(((queue.get("lane_counts") or {}).get("core") or {}).get("pending_lines"), 0))
    deferred_pending_lines = _safe_int(backpressure.get("pending_lines_deferred"), _safe_int(((queue.get("lane_counts") or {}).get("deferred") or {}).get("pending_lines"), 0))
    cold_pending_lines = _safe_int(backpressure.get("pending_lines_cold"), _safe_int(((queue.get("lane_counts") or {}).get("cold") or {}).get("pending_lines"), 0))
    support_pending_lines = _safe_int(backpressure.get("pending_lines_support_telemetry"), 0)
    stale_stage_pending_lines = _safe_int(backpressure.get("pending_lines_stale_stage"), 0)
    retention_debt_gb = _safe_float(((storage_control.get("storage") or {}).get("retention_debt_gb")), _safe_float(((health_gates.get("storage_pressure") or {}).get("retention_debt_gb")), 0.0))
    hard_gate_flags = health_gates.get("hard_gates") if isinstance(health_gates.get("hard_gates"), dict) else {}
    storage_hard_gate = any(
        bool(hard_gate_flags.get(key, False))
        for key in (
            "ingestion_pending_lines",
            "ingestion_oldest_age",
            "ingestion_invalid_lines",
            "ingestion_backpressure_overload",
            "priority_shard_storage",
            "sql_progress_stall",
            "sql_wal_pressure",
        )
    )
    hard_gate = bool(
        str(storage_control.get("overall_status") or "") == "blocked"
        or (storage_hard_gate if hard_gate_flags else bool(health_gates.get("hard_gate_triggered", False)))
    )

    current_primary_db = str(sql_progress.get("primary_db") or sql_service.get("primary_db") or "").strip()
    current_primary_db_realpath = str(sql_progress.get("primary_db_realpath") or sql_service.get("primary_db_realpath") or current_primary_db)
    split_brain_conflicts = max(
        _safe_int(failback.get("split_brain_conflicts"), 0),
        _safe_int(((split_brain.get("summary") or {}).get("unresolved_conflicts")), 0),
    )
    storage_external = bool(mount.get("external_available", False)) and str(mount.get("storage_mode") or failback.get("mode") or "") == "external"
    route_drift = bool(
        storage_external
        and split_brain_conflicts == 0
        and (
            "/local_fallback_storage/" in current_primary_db
            or "/local_fallback_storage/" in current_primary_db_realpath
        )
    )

    profile_name = _storage_profile(
        hard_gate=hard_gate,
        core_pending_lines=core_pending_lines,
        deferred_pending_lines=deferred_pending_lines,
        cold_pending_lines=cold_pending_lines,
        support_pending_lines=support_pending_lines,
        stale_stage_pending_lines=stale_stage_pending_lines,
        retention_debt_gb=retention_debt_gb,
        route_drift=route_drift,
    )
    env_overrides = _profile_env(
        profile_name,
        project_root,
        core_pending_lines=core_pending_lines,
        deferred_pending_lines=deferred_pending_lines,
        route_drift=route_drift,
    )
    top_actions: list[str] = []
    if route_drift:
        top_actions.append("normalize SQL linker back to the routed primary DB path and restart the writer service")
    if deferred_pending_lines > 0:
        top_actions.append("keep deferred ingestion quota-limited until core drain stays under 30 minutes")
    if support_pending_lines > 0:
        top_actions.append("route watchdog failover and pager telemetry through the support shard so it stops crowding governance backlog")
    if cold_pending_lines > 0:
        top_actions.append("hold shadow PnL attribution cold-lane ingestion at zero until the cold backlog clears")
    if stale_stage_pending_lines > 0:
        top_actions.append("treat stale-stage debt as archive or reap work instead of generic hot-path ingestion pressure")
    if retention_debt_gb > 0.0:
        top_actions.append("run aggressive explanation and attribution shard hot retention until retention debt is near zero")
    if hard_gate:
        top_actions.append("treat storage pressure as maintenance priority over new training and research work")

    notes = [
        "the governor writes a dedicated storage-pressure override so manual storage route switches can still live in config/.env.storage_override",
        "SQL_LINK_SERVICE_PRIMARY_DB is pinned to the routed repo path instead of a resolved fallback path so future failback transitions can follow symlink routing cleanly",
    ]
    if profile_name == "critical_backpressure":
        notes.append(
            "critical profile keeps cold lanes at zero and only reopens a small deferred trickle once the core queue is nearly clear and routing is healthy"
        )
    elif profile_name == "elevated_backpressure":
        notes.append("elevated profile keeps core ingestion moving while limiting shadow PnL attribution and reducing deferred fan-in")
    else:
        notes.append("steady-state profile preserves the routed primary DB path and relaxed deferred/cold quotas")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": True,
        "action": action,
        "profile": profile_name,
        "changed": bool(changed),
        "override_path": str(override_path),
        "override_exists": bool(override_path.exists()),
        "storage": {
            "external_available": bool(mount.get("external_available", False)),
            "storage_mode": str(mount.get("storage_mode") or failback.get("mode") or ""),
            "split_brain_conflicts": int(split_brain_conflicts),
        },
        "pressure": {
            "hard_gate": bool(hard_gate),
            "core_pending_lines": int(core_pending_lines),
            "deferred_pending_lines": int(deferred_pending_lines),
            "cold_pending_lines": int(cold_pending_lines),
            "support_pending_lines": int(support_pending_lines),
            "stale_stage_pending_lines": int(stale_stage_pending_lines),
            "retention_debt_gb": round(float(retention_debt_gb), 3),
        },
        "sql_primary_db": {
            "current_path": current_primary_db,
            "current_realpath": current_primary_db_realpath,
            "target_path": str(project_root / "data" / "jsonl_link.sqlite3"),
            "route_drift": bool(route_drift),
        },
        "throttle_controls": {
            "deferred_files_budget": _safe_int(env_overrides.get("INGEST_MAX_DEFERRED_FILES"), 0),
            "cold_files_budget": _safe_int(env_overrides.get("JSONL_SQL_MAX_COLD_LANE_FILES"), 0),
            "log_api_calls": env_overrides.get("LOG_API_CALLS"),
            "log_loop_state": env_overrides.get("LOG_LOOP_STATE"),
            "log_data_ingress": env_overrides.get("LOG_DATA_INGRESS"),
            "log_gate_evaluations": env_overrides.get("LOG_GATE_EVALUATIONS"),
            "log_gate_passes": env_overrides.get("LOG_GATE_PASSES"),
            "log_sub_bot_decisions": env_overrides.get("LOG_SUB_BOT_DECISIONS"),
            "log_master_variant_decisions": env_overrides.get("LOG_MASTER_VARIANT_DECISIONS"),
            "log_decision_explanations": env_overrides.get("LOG_DECISION_EXPLANATIONS"),
            "log_shadow_pnl_attribution": env_overrides.get("LOG_SHADOW_PNL_ATTRIBUTION"),
        },
        "env_overrides": env_overrides,
        "top_actions": top_actions,
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply authoritative ingestion/storage throttles and normalize SQL primary DB routing.")
    parser.add_argument("action", choices=("status", "apply"))
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    override_path = Path(args.override_file).expanduser()
    changed = False
    if args.action == "apply":
        payload_preview = build_payload(project_root, override_path=override_path, action=args.action, changed=False)
        changed = _write_override(override_path, str(payload_preview.get("profile") or "steady_state"), payload_preview.get("env_overrides") if isinstance(payload_preview.get("env_overrides"), dict) else {})
    payload = build_payload(project_root, override_path=override_path, action=args.action, changed=changed)

    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "ingestion_storage_governor "
            f"profile={payload.get('profile', '')} "
            f"changed={int(bool(payload.get('changed', False)))} "
            f"route_drift={int(bool(((payload.get('sql_primary_db') or {}).get('route_drift', False))))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
