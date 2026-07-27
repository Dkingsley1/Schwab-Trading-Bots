#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time as time_mod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python
from core.storage_mounts import resolve_external_storage


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "storage_maintenance_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "storage_maintenance.lock"
DEFAULT_SQL_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "jsonl_sql_writer.lock"
DEFAULT_SQL_MAINTENANCE_SHARDS = (
    "health_fast,crypto_trading_fast,trading_fast,crypto_explanations,explanations,"
    "crypto_shadow_attribution,shadow_attribution,crypto_governance,crypto_trading,governance,aggressive_trading,trading,data"
)
DEFAULT_PRIORITY_RETENTION_SHARDS = (
    "health_fast",
    "crypto_explanations",
    "explanations",
    "crypto_shadow_attribution",
    "shadow_attribution",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _parse_json_output(text: str) -> dict[str, Any]:
    for line in reversed([raw.strip() for raw in str(text or "").splitlines() if raw.strip()]):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _run_json_command(
    cmd: list[str],
    *,
    cwd: Path,
    payload_path: Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    env = os.environ.copy()
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items()})
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    payload = _parse_json_output(proc.stdout or "")
    if not payload and payload_path is not None:
        payload = _load_json(payload_path)
    duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
    return {
        "cmd": list(cmd),
        "rc": int(proc.returncode),
        "duration_ms": duration_ms,
        "payload": payload,
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-12:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-12:]),
    }


def _payload_reason_tokens(payload: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in ("reason", "skipped_reason"):
        value = str(payload.get(key) or "").strip()
        if value:
            tokens.add(value)
    resource_guard = payload.get("resource_guard") if isinstance(payload.get("resource_guard"), dict) else {}
    for key in ("reason", "skipped_reason"):
        value = str(resource_guard.get(key) or "").strip()
        if value:
            tokens.add(value)
    for raw in resource_guard.get("resource_guard_reasons") or []:
        value = str(raw or "").strip()
        if value:
            tokens.add(value)
    freeze_contract = (
        resource_guard.get("support_maintenance_freeze_contract")
        if isinstance(resource_guard.get("support_maintenance_freeze_contract"), dict)
        else {}
    )
    value = str(freeze_contract.get("reason") or "").strip()
    if value:
        tokens.add(value)
    return tokens


def _has_nonfatal_reason(payload: dict[str, Any], accepted: set[str]) -> bool:
    return bool(accepted and (_payload_reason_tokens(payload) & accepted))


def _support_maintenance_frozen(payload: dict[str, Any]) -> bool:
    return bool(
        payload.get("support_maintenance_frozen", False)
        or "support_maintenance_frozen_for_mac_fluidity" in _payload_reason_tokens(payload)
    )


def _step_status(result: dict[str, Any], *, nonfatal_reasons: set[str] | None = None) -> str:
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    accepted = nonfatal_reasons or set()
    reason = str(payload.get("reason") or "")
    if _has_nonfatal_reason(payload, accepted):
        return "busy"
    if int(result.get("rc", 1)) != 0:
        return "error"
    if bool(payload.get("busy", False)) or reason in accepted:
        return "busy"
    if bool(payload.get("skipped", False)):
        return "skipped"
    if payload.get("ok") is False:
        return "error"
    return "ok"


def _step_record(result: dict[str, Any], *, nonfatal_reasons: set[str] | None = None) -> dict[str, Any]:
    return {
        "status": _step_status(result, nonfatal_reasons=nonfatal_reasons),
        "rc": int(result.get("rc", 1)),
        "duration_ms": float(result.get("duration_ms", 0.0) or 0.0),
        "cmd": list(result.get("cmd") or []),
        "stdout_tail": str(result.get("stdout_tail") or ""),
        "stderr_tail": str(result.get("stderr_tail") or ""),
    }


def _normalize_sqlite_maintenance_result(result: dict[str, Any]) -> dict[str, Any]:
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    error = str(payload.get("error") or "")
    if not error.startswith("db_missing:"):
        return result
    normalized = dict(result)
    normalized_payload = dict(payload)
    normalized_payload.update(
        {
            "ok": True,
            "skipped": True,
            "reason": "sqlite_primary_db_missing",
            "original_error": error,
        }
    )
    normalized["rc"] = 0
    normalized["payload"] = normalized_payload
    return normalized


def _usage_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "exists": True,
        "free_gb": round(float(usage.free) / (1024.0 ** 3), 3),
        "used_gb": round(float(usage.used) / (1024.0 ** 3), 3),
        "total_gb": round(float(usage.total) / (1024.0 ** 3), 3),
    }


def _file_size_gb(path: Path) -> float:
    try:
        return round(float(path.stat().st_size) / (1024.0 ** 3), 3)
    except Exception:
        return 0.0


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


def _split_csv(raw: Any) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _ordered_unique(rows: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        token = str(row or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _priority_retention_focus(project_root: Path, base_env: dict[str, str]) -> dict[str, Any]:
    health_gates = _load_json(project_root / "governance" / "health" / "health_gates_latest.json")
    raw_priority_rows = health_gates.get("priority_shards") if isinstance(health_gates.get("priority_shards"), list) else []
    focus_rows: list[dict[str, Any]] = []
    allowed = set(DEFAULT_PRIORITY_RETENTION_SHARDS)
    for raw in raw_priority_rows:
        if not isinstance(raw, dict):
            continue
        shard = str(raw.get("shard") or "").strip()
        if shard not in allowed or shard == "health_fast":
            continue
        retention_debt_gb = max(_safe_float(raw.get("retention_debt_gb"), 0.0), 0.0)
        latency_multiplier = max(_safe_float(raw.get("latency_limit_multiplier"), 0.0), 0.0)
        storage_breached = bool(raw.get("storage_breached", False))
        latency_breached = bool(raw.get("latency_breached", False))
        if retention_debt_gb <= 0.0 and not storage_breached and not latency_breached:
            continue
        focus_rows.append(
            {
                "shard": shard,
                "retention_debt_gb": round(retention_debt_gb, 3),
                "latency_limit_multiplier": round(latency_multiplier, 3),
                "storage_breached": storage_breached,
                "latency_breached": latency_breached,
                "recommended_action": str(raw.get("recommended_action") or ""),
            }
        )

    focus_rows.sort(
        key=lambda row: (
            float(row.get("retention_debt_gb", 0.0) or 0.0),
            1 if bool(row.get("storage_breached", False)) else 0,
            float(row.get("latency_limit_multiplier", 0.0) or 0.0),
        ),
        reverse=True,
    )
    focus_shards = [str(row["shard"]) for row in focus_rows]
    targeted_retention_debt_gb = round(sum(float(row.get("retention_debt_gb", 0.0) or 0.0) for row in focus_rows), 3)
    severe_focus = bool(
        focus_rows
        and (
            targeted_retention_debt_gb >= 20.0
            or any(bool(row.get("storage_breached", False)) for row in focus_rows)
        )
    )
    current_shards = _split_csv(base_env.get("SQL_LINK_SERVICE_SHARDS") or os.getenv("SQL_LINK_SERVICE_SHARDS", DEFAULT_SQL_MAINTENANCE_SHARDS))
    if severe_focus:
        ordered_shards = _ordered_unique(list(DEFAULT_PRIORITY_RETENTION_SHARDS))
    else:
        ordered_shards = _ordered_unique(["health_fast", *focus_shards, *current_shards])

    env_overrides: dict[str, str] = {}
    if focus_rows:
        env_overrides["SQL_LINK_SERVICE_SHARDS"] = ",".join(ordered_shards)
        if severe_focus:
            env_overrides.update(
                {
                    "SQL_LINK_SERVICE_WAL_CHECKPOINT_MIN_INTERVAL_SECONDS": "60",
                    "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_ROWS": "150000",
                    "SQL_LINK_SERVICE_WAL_TRUNCATE_MAX_GB": "1.5",
                }
            )
        if "explanations" in focus_shards:
            env_overrides.update(
                {
                    "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_FILES": "8",
                    "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_MAX_DB_GB": "3.5" if severe_focus else "5",
                    "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_TRIGGER_GROWTH_GB": "0.35" if severe_focus else "0.6",
                    "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_TRIGGER_ROWS": "90000" if severe_focus else "140000",
                    "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_HOT_DAYS": "1" if severe_focus else "2",
                    "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_BATCH_SIZE": "260000",
                    "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_MAX_ROWS": "2600000" if severe_focus else "2200000",
                    "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_MIN_INTERVAL_SECONDS": "30" if severe_focus else "60",
                    "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_ARCHIVE_PERIOD": "day",
                    "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_ARCHIVE_RETENTION_DAYS": "21" if severe_focus else "30",
                    "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_VACUUM_THRESHOLD_GB": "1.5" if severe_focus else "2.0",
                }
            )
        if "crypto_explanations" in focus_shards:
            env_overrides.update(
                {
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_FILES": "8",
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_MAX_DB_GB": "2.5" if severe_focus else "4",
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_TRIGGER_GROWTH_GB": "0.3" if severe_focus else "0.5",
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_TRIGGER_ROWS": "75000" if severe_focus else "110000",
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_HOT_DAYS": "1" if severe_focus else "2",
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_BATCH_SIZE": "240000" if severe_focus else "220000",
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_MAX_ROWS": "1800000" if severe_focus else "1500000",
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_MIN_INTERVAL_SECONDS": "30" if severe_focus else "60",
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_ARCHIVE_PERIOD": "day",
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_ARCHIVE_RETENTION_DAYS": "14" if severe_focus else "21",
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_VACUUM_THRESHOLD_GB": "1.0" if severe_focus else "1.5",
                }
            )
        if "shadow_attribution" in focus_shards:
            env_overrides.update(
                {
                    "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_MAX_FILES": "4",
                    "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_HOT_RETENTION_BATCH_SIZE": "260000",
                    "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_HOT_RETENTION_MAX_ROWS": "1800000",
                    "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_HOT_RETENTION_MIN_INTERVAL_SECONDS": "60",
                }
            )
        if "crypto_shadow_attribution" in focus_shards:
            env_overrides.update(
                {
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_MAX_FILES": "4",
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_BATCH_SIZE": "260000",
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_MAX_ROWS": "1800000",
                    "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_MIN_INTERVAL_SECONDS": "60",
                }
            )

    top_actions: list[str] = []
    if focus_rows:
        top_actions.append("prioritize explanation and attribution shards ahead of trading shards during maintenance runs")
    if targeted_retention_debt_gb > 0.0:
        top_actions.append("keep retention maintenance focused on oversized explanation shards until retention debt is near zero")
    if any(bool(row.get("latency_breached", False)) for row in focus_rows):
        top_actions.append("treat priority shard latency breaches as a maintenance-first signal rather than widening ingestion fan-in")

    return {
        "enabled": bool(focus_rows),
        "severe_focus": severe_focus,
        "focus_shards": focus_shards,
        "ordered_shards": ordered_shards,
        "targeted_retention_debt_gb": targeted_retention_debt_gb,
        "priority_rows": focus_rows,
        "env_overrides": env_overrides,
        "top_actions": top_actions[:5],
    }


def _age_minutes_from_timestamp(raw: Any) -> float | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max((datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 60.0, 0.0)


def _lock_owner_pid(lock_path: Path) -> int | None:
    try:
        text = lock_path.read_text(encoding="utf-8")
    except Exception:
        return None
    for part in text.split():
        if part.startswith("pid="):
            raw = part.split("=", 1)[1].strip()
            try:
                return int(raw)
            except Exception:
                return None
    return None


def _retry_writer_maintenance(
    *,
    project_root: Path,
    health_root: Path,
    env_overrides: dict[str, str],
    poll_seconds: float,
    wait_timeout_seconds: float,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    deadline = started.timestamp() + max(float(wait_timeout_seconds), 0.0)
    attempts = 0
    last_result: dict[str, Any] | None = None
    observed_writer_pid = _lock_owner_pid(DEFAULT_SQL_LOCK_PATH)

    while datetime.now(timezone.utc).timestamp() <= deadline:
        attempts += 1
        result = _run_json_command(
            [str(PY), str(project_root / "scripts" / "ops" / "sql_link_shard_manager.py"), "--once", "--json"],
            cwd=project_root,
            payload_path=health_root / "sql_link_service_latest.json",
            env_overrides=env_overrides,
        )
        last_result = result
        status = _step_status(result, nonfatal_reasons={"writer_lock_busy"})
        if status != "busy":
            break
        observed_writer_pid = observed_writer_pid or _lock_owner_pid(DEFAULT_SQL_LOCK_PATH)
        remaining = max(deadline - datetime.now(timezone.utc).timestamp(), 0.0)
        if remaining <= 0.0:
            break
        time_mod.sleep(min(max(float(poll_seconds), 0.1), remaining))

    waited_seconds = max((datetime.now(timezone.utc) - started).total_seconds(), 0.0)
    final_status = _step_status(last_result or {}, nonfatal_reasons={"writer_lock_busy"})
    completed = bool(last_result is not None and final_status != "busy")
    return {
        "requested": True,
        "completed": completed,
        "timed_out": not completed,
        "attempts": int(attempts),
        "poll_seconds": round(float(poll_seconds), 3),
        "wait_timeout_seconds": round(float(wait_timeout_seconds), 3),
        "waited_seconds": round(waited_seconds, 3),
        "observed_writer_pid": observed_writer_pid,
        "status": "completed" if completed else "timed_out",
        "last_result": last_result or {},
    }


def _storage_roots(project_root: Path) -> tuple[Path, Path]:
    resolution = resolve_external_storage()
    return resolution.mount_root, resolution.external_root


def build_storage_maintenance_payload(
    project_root: Path,
    *,
    resource_profile: str,
    force: bool,
    vacuum: bool,
) -> dict[str, Any]:
    mount_root, external_root = _storage_roots(project_root)
    disk_before = {
        "project_root": _usage_snapshot(project_root),
        "external_mount": _usage_snapshot(mount_root),
        "external_project_root": _usage_snapshot(external_root),
    }
    governor = _run_json_command(
        [str(PY), str(project_root / "scripts" / "ops" / "ingestion_storage_governor.py"), "apply", "--json"],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "ingestion_storage_governor_latest.json",
    )
    governor_payload = governor.get("payload") if isinstance(governor.get("payload"), dict) else {}
    env_overrides = {
        str(key): str(value)
        for key, value in (governor_payload.get("env_overrides") or {}).items()
        if str(key).strip()
    } if isinstance(governor_payload.get("env_overrides"), dict) else {}
    maintenance_focus = _priority_retention_focus(project_root, env_overrides)
    if isinstance(maintenance_focus.get("env_overrides"), dict):
        env_overrides = {
            **env_overrides,
            **{str(key): str(value) for key, value in maintenance_focus["env_overrides"].items() if str(key).strip()},
        }
    strategy_reloader = _run_json_command(
        [str(PY), str(project_root / "scripts" / "ops" / "maintenance_strategy_reloader.py")],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "maintenance_strategy_reloader_latest.json",
        env_overrides=env_overrides,
    )
    resource_guard = _run_json_command(
        [str(PY), str(project_root / "scripts" / "resource_guard.py"), "--profile", str(resource_profile or "optional"), "--json"],
        cwd=project_root,
        env_overrides=env_overrides,
    )
    failback = _run_json_command(
        [str(PY), str(project_root / "scripts" / "ops" / "storage_failback_sync.py"), "--json"],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "storage_failback_sync_latest.json",
        env_overrides=env_overrides,
    )

    heavy_steps_skipped = False
    shard_manager: dict[str, Any] | None = None
    shard_follow_through: dict[str, Any] = {"requested": False, "completed": False, "timed_out": False, "attempts": 0, "waited_seconds": 0.0}
    sqlite_maintenance: dict[str, Any] | None = None
    stale_sweeper: dict[str, Any] | None = None
    stale_reaper: dict[str, Any] | None = None
    retention: dict[str, Any] | None = None
    content_store_gc: dict[str, Any] | None = None
    resource_guard_payload = resource_guard.get("payload") if isinstance(resource_guard.get("payload"), dict) else {}
    resource_support_frozen = _support_maintenance_frozen(resource_guard_payload)
    resource_ok = bool(resource_guard_payload.get("ok", resource_guard_payload.get("resource_guard_ok", False)))
    if (not resource_ok or resource_support_frozen) and not force:
        heavy_steps_skipped = True
    else:
        shard_manager = _run_json_command(
            [str(PY), str(project_root / "scripts" / "ops" / "sql_link_shard_manager.py"), "--once", "--json"],
            cwd=project_root,
            payload_path=project_root / "governance" / "health" / "sql_link_service_latest.json",
            env_overrides=env_overrides,
        )
        writer_busy = _step_status(shard_manager, nonfatal_reasons={"writer_lock_busy"}) == "busy"
        if writer_busy and bool(maintenance_focus.get("severe_focus", False)):
            shard_follow_through = _retry_writer_maintenance(
                project_root=project_root,
                health_root=project_root / "governance" / "health",
                env_overrides=env_overrides,
                poll_seconds=max(float(os.getenv("SQL_LINK_SERVICE_MAINTENANCE_LOCK_POLL_SECONDS", "5") or 5.0), 0.1),
                wait_timeout_seconds=max(float(os.getenv("SQL_LINK_SERVICE_MAINTENANCE_LOCK_WAIT_SECONDS", "900") or 900.0), 0.0),
            )
            if isinstance(shard_follow_through.get("last_result"), dict) and shard_follow_through.get("last_result"):
                shard_manager = shard_follow_through["last_result"]
        sqlite_cmd = [str(PY), str(project_root / "scripts" / "sqlite_performance_maintenance.py")]
        if vacuum:
            sqlite_cmd.append("--vacuum")
        else:
            sqlite_cmd.append("--checkpoint-only")
        sqlite_cmd.append("--json")
        sqlite_maintenance = _run_json_command(
            sqlite_cmd,
            cwd=project_root,
            payload_path=project_root / "governance" / "health" / "sqlite_maintenance_latest.json",
            env_overrides=env_overrides,
        )
        sqlite_maintenance = _normalize_sqlite_maintenance_result(sqlite_maintenance)
        stale_sweeper = _run_json_command(
            [str(PY), str(project_root / "scripts" / "ops" / "stale_artifact_sweeper_bot.py"), "--json"],
            cwd=project_root,
            payload_path=project_root / "governance" / "health" / "stale_artifact_sweeper_bot_latest.json",
            env_overrides=env_overrides,
        )
        stale_reaper = _run_json_command(
            [str(PY), str(project_root / "scripts" / "ops" / "stale_artifact_reaper_bot.py"), "--json"],
            cwd=project_root,
            payload_path=project_root / "governance" / "health" / "stale_artifact_reaper_bot_latest.json",
            env_overrides=env_overrides,
        )
        retention = _run_json_command(
            [
                str(PY),
                str(project_root / "scripts" / "data_retention_policy.py"),
                "--apply",
                "--no-stale-stage",
                "--no-stale-purge",
                "--json",
            ],
            cwd=project_root,
            payload_path=project_root / "governance" / "health" / "data_retention_latest.json",
            env_overrides=env_overrides,
        )
        content_store_gc = _run_json_command(
            [str(PY), str(project_root / "scripts" / "ops" / "content_addressed_artifact_store.py"), "--json"],
            cwd=project_root,
            payload_path=project_root / "governance" / "content_store" / "latest.json",
            env_overrides=env_overrides,
        )

    disk_after = {
        "project_root": _usage_snapshot(project_root),
        "external_mount": _usage_snapshot(mount_root),
        "external_project_root": _usage_snapshot(external_root),
    }

    steps = {
        "ingestion_storage_governor": _step_record(governor),
        "maintenance_strategy_reloader": _step_record(strategy_reloader),
        "resource_guard": _step_record(resource_guard, nonfatal_reasons={"support_maintenance_frozen_for_mac_fluidity"}),
        "storage_failback_sync": _step_record(failback),
    }
    if heavy_steps_skipped:
        skip_reason = "support_maintenance_frozen_for_mac_fluidity" if resource_support_frozen else "resource_guard_blocked"
        steps["resource_guard"]["status"] = "busy" if resource_support_frozen else "blocked"
        steps["sql_link_shard_manager"] = {"status": "skipped", "reason": skip_reason}
        steps["content_addressed_artifact_store"] = {"status": "skipped", "reason": skip_reason}
        steps["sqlite_maintenance"] = {"status": "skipped", "reason": skip_reason}
        steps["stale_artifact_sweeper_bot"] = {"status": "skipped", "reason": skip_reason}
        steps["stale_artifact_reaper_bot"] = {"status": "skipped", "reason": skip_reason}
        steps["data_retention_policy"] = {"status": "skipped", "reason": skip_reason}
    else:
        assert shard_manager is not None
        assert sqlite_maintenance is not None
        assert stale_sweeper is not None
        assert stale_reaper is not None
        assert retention is not None
        assert content_store_gc is not None
        steps["sql_link_shard_manager"] = _step_record(shard_manager, nonfatal_reasons={"writer_lock_busy"})
        if bool(shard_follow_through.get("requested", False)):
            if isinstance(shard_follow_through.get("last_result"), dict) and shard_follow_through.get("last_result"):
                steps["sql_link_shard_manager_follow_through"] = _step_record(
                    shard_follow_through["last_result"],
                    nonfatal_reasons={"writer_lock_busy"},
                )
            else:
                steps["sql_link_shard_manager_follow_through"] = {
                    "status": "busy",
                    "rc": 124,
                    "duration_ms": round(float(shard_follow_through.get("waited_seconds", 0.0) or 0.0) * 1000.0, 3),
                    "timed_out": True,
                    "cmd": [str(PY), str(project_root / "scripts" / "ops" / "sql_link_shard_manager.py"), "--once", "--json"],
                    "stdout_tail": "",
                    "stderr_tail": "",
                }
        steps["content_addressed_artifact_store"] = _step_record(content_store_gc)
        steps["sqlite_maintenance"] = _step_record(sqlite_maintenance)
        steps["stale_artifact_sweeper_bot"] = _step_record(stale_sweeper, nonfatal_reasons={"already_running", "lock_busy"})
        steps["stale_artifact_reaper_bot"] = _step_record(stale_reaper, nonfatal_reasons={"already_running", "lock_busy"})
        steps["data_retention_policy"] = _step_record(retention, nonfatal_reasons={"lock_busy"})

    ok = True
    for result, nonfatal in (
        (governor, set()),
        (strategy_reloader, set()),
        (resource_guard, {"support_maintenance_frozen_for_mac_fluidity"}),
        (failback, set()),
        (shard_manager, {"writer_lock_busy"}),
        (content_store_gc, set()),
        (sqlite_maintenance, set()),
        (stale_sweeper, {"already_running", "lock_busy"}),
        (stale_reaper, {"already_running", "lock_busy"}),
        (retention, {"lock_busy"}),
    ):
        if result is None:
            continue
        payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        if int(result.get("rc", 1)) != 0 and not _has_nonfatal_reason(payload, nonfatal):
            ok = False
            break
        reason = str(payload.get("reason") or "")
        if payload.get("ok") is False and not bool(payload.get("busy", False)) and reason not in nonfatal and not _has_nonfatal_reason(payload, nonfatal):
            ok = False
            break

    if heavy_steps_skipped:
        base_ok = all(
            int(result.get("rc", 1)) == 0 and _step_status(result) != "error"
            for result in (governor, strategy_reloader, failback)
        )
        if resource_support_frozen:
            ok = bool(
                base_ok
                and _step_status(resource_guard, nonfatal_reasons={"support_maintenance_frozen_for_mac_fluidity"})
                in {"ok", "busy", "skipped"}
            )
        else:
            ok = base_ok

    reason = "ok"
    overall_status = "ready"
    if heavy_steps_skipped and resource_support_frozen:
        reason = "support_maintenance_frozen_for_mac_fluidity"
        overall_status = "guarded_hold"
    elif heavy_steps_skipped:
        reason = "resource_guard_blocked"
        overall_status = "guarded_hold"
    elif not ok:
        overall_status = "blocked"
        for name, record in steps.items():
            if str(record.get("status") or "") == "error":
                reason = f"{name}_failed"
                break

    if heavy_steps_skipped and not resource_support_frozen:
        ok = all(
            int(result.get("rc", 1)) == 0 and _step_status(result) != "error"
            for result in (governor, strategy_reloader, failback)
        )
        overall_status = "guarded_hold"

    failback_payload = failback.get("payload") if isinstance(failback.get("payload"), dict) else {}
    shard_payload = (shard_manager or {}).get("payload") if isinstance((shard_manager or {}).get("payload"), dict) else {}
    sqlite_payload = (sqlite_maintenance or {}).get("payload") if isinstance((sqlite_maintenance or {}).get("payload"), dict) else {}
    retention_payload = (retention or {}).get("payload") if isinstance((retention or {}).get("payload"), dict) else {}
    stale_sweeper_payload = (stale_sweeper or {}).get("payload") if isinstance((stale_sweeper or {}).get("payload"), dict) else {}
    stale_reaper_payload = (stale_reaper or {}).get("payload") if isinstance((stale_reaper or {}).get("payload"), dict) else {}
    low_space = failback_payload.get("low_space_autoprune") if isinstance(failback_payload.get("low_space_autoprune"), dict) else {}
    sql_progress = _load_json(project_root / "governance" / "health" / "sql_link_service_progress_latest.json")
    sql_service = _load_json(project_root / "governance" / "health" / "sql_link_service_latest.json")
    primary_db = Path(str(sql_progress.get("primary_db") or sql_service.get("primary_db") or "")).expanduser()
    wal_path = Path(str(primary_db) + "-wal") if str(primary_db) else Path("")
    sql_progress_idle_minutes = _age_minutes_from_timestamp(sql_progress.get("timestamp_utc"))

    payload = {
        "timestamp_utc": _utc_now(),
        "project_root": str(project_root),
        "ok": bool(ok),
        "overall_status": overall_status,
        "reason": reason,
        "force": bool(force),
        "vacuum": bool(vacuum),
        "heavy_steps_skipped": bool(heavy_steps_skipped),
        "steps": steps,
        "artifacts": {
            "ingestion_storage_governor": str(project_root / "governance" / "health" / "ingestion_storage_governor_latest.json"),
            "maintenance_strategy_reloader": str(project_root / "governance" / "health" / "maintenance_strategy_reloader_latest.json"),
            "storage_failback_sync": str(project_root / "governance" / "health" / "storage_failback_sync_latest.json"),
            "health_gates": str(project_root / "governance" / "health" / "health_gates_latest.json"),
            "sql_link_service": str(project_root / "governance" / "health" / "sql_link_service_latest.json"),
            "content_addressed_artifact_store": str(project_root / "governance" / "content_store" / "latest.json"),
            "sqlite_maintenance": str(project_root / "governance" / "health" / "sqlite_maintenance_latest.json"),
            "stale_artifact_sweeper_bot": str(project_root / "governance" / "health" / "stale_artifact_sweeper_bot_latest.json"),
            "stale_artifact_reaper_bot": str(project_root / "governance" / "health" / "stale_artifact_reaper_bot_latest.json"),
            "data_retention": str(project_root / "governance" / "health" / "data_retention_latest.json"),
        },
        "maintenance_focus": maintenance_focus,
        "shard_follow_through": shard_follow_through,
        "resource_guard": resource_guard.get("payload") if isinstance(resource_guard.get("payload"), dict) else {},
        "disk_before": disk_before,
        "disk_after": disk_after,
        "summary": {
            "ingestion_storage_profile": str(governor_payload.get("profile") or ""),
            "governor_route_drift": bool(((governor_payload.get("sql_primary_db") or {}).get("route_drift", False))),
            "priority_retention_focus_enabled": bool(maintenance_focus.get("enabled", False)),
            "priority_retention_focus_shards": list(maintenance_focus.get("focus_shards") or []),
            "priority_retention_targeted_debt_gb": _safe_float(maintenance_focus.get("targeted_retention_debt_gb"), 0.0),
            "maintenance_reloader_changed": bool((strategy_reloader.get("payload") or {}).get("changed", False)),
            "maintenance_reloader_deferred": bool((strategy_reloader.get("payload") or {}).get("deferred", False)),
            "storage_mode": str(failback_payload.get("mode") or ""),
            "active_root": str(failback_payload.get("active_root") or ""),
            "autosync_copied_files": int((((failback_payload.get("autosync") or {}).get("copied_files", 0)) or 0)),
            "autoprune_deleted_count": int(low_space.get("deleted_count", 0) or 0),
            "shard_reason": str(shard_payload.get("reason") or ""),
            "shard_follow_through_completed": bool(shard_follow_through.get("completed", False)),
            "shard_follow_through_waited_seconds": round(float(shard_follow_through.get("waited_seconds", 0.0) or 0.0), 3),
            "sqlite_wal_gb_before": sqlite_payload.get("wal_size_gb_before"),
            "sqlite_wal_gb_after": sqlite_payload.get("wal_size_gb_after"),
            "stale_stage_candidate_files": int((((stale_sweeper_payload.get("summary") or {}).get("candidate_files", 0)) or 0)),
            "stale_stage_staged_files": int((((stale_sweeper_payload.get("summary") or {}).get("staged_files", 0)) or 0)),
            "stale_stage_reaped_files": int((((stale_reaper_payload.get("summary") or {}).get("deleted_files", 0)) or 0)),
            "retention_deleted": int(retention_payload.get("deleted", 0) or 0),
            "retention_delete_errors": int(retention_payload.get("delete_errors", 0) or 0),
            "content_store_deleted_blobs": int(((((content_store_gc or {}).get("payload") or {}).get("gc") or {}).get("deleted_blob_count", 0) or 0)),
            "content_store_deleted_bytes": int(((((content_store_gc or {}).get("payload") or {}).get("gc") or {}).get("deleted_bytes", 0) or 0)),
            "content_store_skipped_blobs": int((((content_store_gc or {}).get("payload") or {}).get("skipped_blob_count", 0) or 0)),
            "sql_sync_status": str(sql_progress.get("status") or ""),
            "sql_sync_step": str(sql_progress.get("current_step") or ""),
            "sql_progress_idle_minutes": round(float(sql_progress_idle_minutes), 3) if sql_progress_idle_minutes is not None else None,
            "primary_db_size_gb_live": _file_size_gb(primary_db) if str(primary_db) else 0.0,
            "primary_wal_size_gb_live": _file_size_gb(wal_path) if str(wal_path) else 0.0,
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run guarded storage hygiene and retention maintenance outside the trading registry.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--resource-profile", default="optional")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--vacuum", action="store_true")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_file = Path(args.out_file).expanduser()
    lock_file = Path(args.lock_file).expanduser()
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "project_root": str(project_root),
        "ok": True,
        "skipped": False,
        "reason": "pending",
    }

    with lock_file.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload.update({"skipped": True, "reason": "already_running"})
            _write_json(out_file, payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("storage_maintenance skipped=1 reason=already_running")
            return 0

        payload = build_storage_maintenance_payload(
            project_root,
            resource_profile=str(args.resource_profile or "optional"),
            force=bool(args.force),
            vacuum=bool(args.vacuum),
        )
        _write_json(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "storage_maintenance "
            f"ok={int(bool(payload.get('ok', False)))} "
            f"retention_deleted={int(((payload.get('summary') or {}).get('retention_deleted', 0) or 0))} "
            f"storage_mode={((payload.get('summary') or {}).get('storage_mode', '') or '')}"
        )
    return 0 if bool(payload.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
