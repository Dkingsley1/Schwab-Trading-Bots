#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import dotenv_values

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from .long_runtime_common import iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "unattended_soak_readiness_latest.json"
DEFAULT_TARGET_DAYS = 30.0
DEFAULT_EXTERNAL_ROOT = Path("/Volumes/BOT_LOGS/schwab_trading_bot")
BOUNDED_TRANSIENT_CORE_MAX_LINES = 10_000
BOUNDED_TRANSIENT_TOTAL_MAX_LINES = 15_000
BOUNDED_TRANSIENT_AGE_MAX_SECONDS = 300.0
DiskSnapshotFn = Callable[[Path], dict[str, Any]]
MANAGED_WARNING_NAMES = {
    "host_not_on_ac_power_operator_approved_battery_override",
    "storage_margin_managed_by_approved_cold_archive_spillover",
    "zero_touch_remote_pager_missing_mobile_operator_coverage_active",
}


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


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on", "operator_approved"}


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "operator_approved"}


def _battery_override_settings(project_root: Path) -> dict[str, Any]:
    override_path = project_root / "config" / ".env.unattended_soak_override"
    try:
        persisted = dotenv_values(override_path) if override_path.is_file() else {}
    except Exception:
        persisted = {}

    def resolve(name: str) -> tuple[str, str]:
        if name in os.environ:
            return str(os.getenv(name, "")), "environment"
        if name in persisted:
            return str(persisted.get(name) or ""), "config/.env.unattended_soak_override"
        return "", "default"

    enabled_raw, enabled_source = resolve("BOT_UNATTENDED_SOAK_ALLOW_BATTERY")
    reason, reason_source = resolve("BOT_UNATTENDED_SOAK_BATTERY_REASON")
    expires_raw, expires_source = resolve("BOT_UNATTENDED_SOAK_BATTERY_EXPIRES_AT_UTC")
    expires_at: datetime | None = None
    if expires_raw:
        try:
            expires_at = datetime.fromisoformat(expires_raw.replace("Z", "+00:00"))
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            expires_at = expires_at.astimezone(timezone.utc)
        except ValueError:
            expires_at = None
    expired = bool(expires_at and expires_at <= datetime.now(timezone.utc))
    enabled = _truthy(enabled_raw)
    reason_present = bool(reason.strip())
    expiry_valid = bool(not expires_raw or expires_at is not None)
    allowed = bool(enabled and reason_present and expiry_valid and not expired)
    return {
        "allowed": allowed,
        "enabled": enabled,
        "reason": reason.strip(),
        "reason_present": reason_present,
        "expires_at_utc": expires_at.isoformat() if expires_at else "",
        "expiry_configured": bool(expires_raw),
        "expiry_valid": expiry_valid,
        "expired": expired,
        "source": enabled_source,
        "reason_source": reason_source,
        "expiry_source": expires_source,
    }


def _warning_is_managed_for_soak(warning: Any) -> bool:
    text = str(warning or "")
    return bool(text in MANAGED_WARNING_NAMES or text.endswith("_overridden_by_caffeinate_guard"))


def _dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _grade(score: float) -> str:
    value = max(min(float(score), 100.0), 0.0)
    if value >= 97.0:
        return "A+"
    if value >= 93.0:
        return "A"
    if value >= 85.0:
        return "B"
    if value >= 75.0:
        return "C"
    if value >= 65.0:
        return "D"
    return "F"


def _disk_snapshot(path: Path) -> dict[str, Any]:
    probe = Path(path).expanduser()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except Exception:
        return {"path": str(path), "probe_path": str(probe), "exists": bool(path.exists()), "free_gb": 0.0, "used_pct": 100.0}
    used_pct = 100.0 * float(usage.used) / max(float(usage.total), 1.0)
    return {
        "path": str(path),
        "probe_path": str(probe),
        "exists": bool(path.exists()),
        "total_gb": round(float(usage.total) / (1024.0**3), 3),
        "free_gb": round(float(usage.free) / (1024.0**3), 3),
        "used_gb": round(float(usage.used) / (1024.0**3), 3),
        "used_pct": round(used_pct, 3),
    }


def _run_text(cmd: list[str], *, timeout_sec: int = 5) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=max(int(timeout_sec), 1))
    except Exception:
        return ""
    return (proc.stdout or "").strip()


def _parse_pmset_custom(text: str) -> dict[str, dict[str, str]]:
    profiles: dict[str, dict[str, str]] = {}
    current = ""
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.endswith(":"):
            current = line[:-1].strip()
            profiles.setdefault(current, {})
            continue
        parts = line.split()
        if len(parts) < 2 or not current:
            continue
        profiles.setdefault(current, {})[parts[0]] = parts[-1]
    return profiles


def _process_has_caffeinate(process_text: str | None = None) -> bool:
    text = process_text if process_text is not None else _run_text(["/bin/ps", "-axo", "command"], timeout_sec=5)
    lowered = str(text or "").lower()
    return "/usr/bin/caffeinate" in lowered or " caffeinate " in f" {lowered} "


def _host_power_contract(
    *,
    project_root: Path = PROJECT_ROOT,
    pmset_custom_text: str | None = None,
    pmset_batt_text: str | None = None,
    process_text: str | None = None,
) -> dict[str, Any]:
    system = platform.system()
    custom_text = pmset_custom_text if pmset_custom_text is not None else _run_text(["/usr/bin/pmset", "-g", "custom"])
    batt_text = pmset_batt_text if pmset_batt_text is not None else _run_text(["/usr/bin/pmset", "-g", "batt"])
    profiles = _parse_pmset_custom(custom_text)
    ac = profiles.get("AC Power", {})
    caffeinate_active = _process_has_caffeinate(process_text)
    ac_attached = "AC Power" in str(batt_text or "") or "AC attached" in str(batt_text or "")
    battery_override = _battery_override_settings(project_root)
    battery_override_allowed = bool(battery_override.get("allowed"))
    battery_override_reason = str(battery_override.get("reason") or "")

    blockers: list[str] = []
    warnings: list[str] = []
    managed_controls: list[str] = []
    if system == "Darwin" and not custom_text:
        warnings.append("pmset_custom_unavailable")
    if system == "Darwin" and not ac_attached:
        if battery_override_allowed:
            warnings.append("host_not_on_ac_power_operator_approved_battery_override")
            managed_controls.append("host_not_on_ac_power_operator_approved_battery_override")
        else:
            blockers.append("host_not_on_ac_power")
            if battery_override.get("enabled") and not battery_override.get("reason_present"):
                warnings.append("battery_override_missing_operator_reason")
            if battery_override.get("enabled") and not battery_override.get("expiry_valid"):
                warnings.append("battery_override_expiry_invalid")
            if battery_override.get("expired"):
                warnings.append("battery_override_expired")

    sleep = _safe_float(ac.get("sleep"), -1.0)
    disksleep = _safe_float(ac.get("disksleep"), -1.0)
    standby = _safe_float(ac.get("standby"), 0.0)
    autopoweroff = _safe_float(ac.get("autopoweroff"), 0.0)
    sleep_blockers: list[str] = []
    if system == "Darwin":
        if sleep < 0:
            warnings.append("host_sleep_setting_unknown")
        elif sleep != 0.0:
            sleep_blockers.append("host_sleep_not_disabled_on_ac")
        if disksleep > 0.0:
            sleep_blockers.append("disk_sleep_not_disabled_on_ac")
        if standby > 0.0:
            sleep_blockers.append("host_standby_not_disabled_on_ac")
        if autopoweroff > 0.0:
            sleep_blockers.append("host_autopoweroff_not_disabled_on_ac")

    if sleep_blockers and caffeinate_active:
        managed_controls.extend(f"{item}_overridden_by_caffeinate_guard" for item in sleep_blockers)
    else:
        blockers.extend(sleep_blockers)

    status = "ready" if not blockers else "blocked"
    scored_warnings = [item for item in warnings if not _warning_is_managed_for_soak(item)]
    score = max(100.0 - (18.0 * len(blockers)) - (4.0 * len(scored_warnings)), 0.0)
    return {
        "status": status,
        "ready": status == "ready",
        "score": round(score, 2),
        "grade": _grade(score),
        "system": system,
        "ac_attached": bool(ac_attached),
        "battery_override_allowed": bool(battery_override_allowed),
        "battery_override_reason": battery_override_reason,
        "battery_override_source": str(battery_override.get("source") or "default"),
        "battery_override_expires_at_utc": str(battery_override.get("expires_at_utc") or ""),
        "battery_override_expired": bool(battery_override.get("expired")),
        "caffeinate_active": bool(caffeinate_active),
        "pmset_profiles": profiles,
        "settings": {
            "sleep": ac.get("sleep", ""),
            "disksleep": ac.get("disksleep", ""),
            "standby": ac.get("standby", ""),
            "autopoweroff": ac.get("autopoweroff", ""),
            "ttyskeepawake": ac.get("ttyskeepawake", ""),
            "tcpkeepalive": ac.get("tcpkeepalive", ""),
        },
        "blockers": blockers,
        "warnings": warnings,
        "managed_warnings": [item for item in warnings if _warning_is_managed_for_soak(item)],
        "managed_controls": managed_controls,
        "recommended_command": (
            "scripts/install_caffeinate_launchd.sh or sudo pmset -a sleep 0 disksleep 0 standby 0 autopoweroff 0"
            if blockers
            else ""
        ),
    }


def _external_root(project_root: Path, health_root: Path) -> Path:
    mount_guard = load_json(health_root / "storage_mount_guard_latest.json")
    raw = str(mount_guard.get("external_root") or os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT") or "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_EXTERNAL_ROOT


def _managed_ingestion_soak_watch(ingestion: dict[str, Any], ingestion_contract: dict[str, Any]) -> bool:
    allowed_blockers = {
        "steady_state_targets_not_clear",
        "backlog_relief_contract_active",
        "drain_time_above_target",
    }
    contract_blockers = {str(item) for item in ingestion_contract.get("blockers", []) if str(item)}
    if contract_blockers and not contract_blockers.issubset(allowed_blockers):
        return False

    status = str(ingestion.get("overall_status") or "").lower()
    severity = str(ingestion.get("severity") or "").lower()
    if status not in {"ready", "ok"} or severity not in {"stable", "low", "ready", "normal", "watch", "elevated"}:
        return False

    raw_live = _dict(_dict(ingestion.get("backlog_truth")).get("raw_live"))
    backpressure = _dict(ingestion.get("backpressure"))
    if not raw_live:
        raw_live = backpressure
    raw_grade = str(raw_live.get("grade") or "").upper()
    total_pending = _safe_int(raw_live.get("total_pending_lines"), _safe_int(backpressure.get("total_pending_lines"), 0))
    core_pending = _safe_int(raw_live.get("core_pending_lines"), _safe_int(backpressure.get("core_pending_lines"), 0))
    oldest_age = _safe_float(
        raw_live.get("oldest_pending_age_seconds"),
        _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0),
    )
    pressure_index = _safe_float(ingestion.get("pressure_index"), 99.0)

    stale_locator = _dict(ingestion.get("stale_pending_locator"))
    stale_clear = str(stale_locator.get("status") or "clear").lower() in {"clear", "ready", ""}
    route = _dict(ingestion.get("external_route_verification"))
    route_ready = str(route.get("verification_state") or "").lower() in {
        "ready",
        "verified",
        "ok",
        "active_local_ready",
        "active_passthrough",
    }
    resilience = _dict(ingestion.get("storage_resilience"))
    resilience_ready = str(resilience.get("overall_status") or "").lower() in {"ready", "ok", ""}
    data_integrity = _dict(ingestion.get("data_integrity"))
    data_clean = all(
        _safe_int(data_integrity.get(key), 0) == 0
        for key in ("sql_invalid_lines", "sql_overlay_invalid_lines", "sql_overlay_oversize_payloads", "sql_overlay_ops_write_failures")
    )
    writer_shedding = _dict(ingestion.get("writer_shedding"))
    no_queue_breaches = not writer_shedding.get("hard_breaches") and not writer_shedding.get("elevated_breaches")
    steady = _dict(_dict(ingestion.get("steady_state")).get("target_status"))
    backlog_relief_ready = bool(steady.get("backlog_relief_a_plus_ready") or steady.get("backlog_relief_a_plus_plus_ready"))
    strict_ready = bool(
        raw_grade in {"A+", "A", ""}
        and pressure_index <= 0.6
        and total_pending <= 5000
        and core_pending <= 1000
        and oldest_age <= 240.0
        and stale_clear
        and route_ready
        and resilience_ready
        and data_clean
        and no_queue_breaches
        and backlog_relief_ready
    )
    if strict_ready:
        return True

    bounded = _dict(ingestion.get("bounded_recovery_contract"))
    storage = _dict(ingestion.get("storage"))
    efficiency = _dict(ingestion.get("storage_efficiency_contract"))
    inputs = _dict(ingestion_contract.get("inputs"))
    efficiency_status = str(
        efficiency.get("overall_status")
        or inputs.get("storage_efficiency_status")
        or ingestion.get("storage_efficiency_status")
        or ""
    ).strip().lower()
    efficiency_grade = str(
        efficiency.get("grade")
        or inputs.get("storage_efficiency_grade")
        or storage.get("efficiency_grade")
        or ingestion.get("storage_efficiency_grade")
        or ""
    ).strip().upper()
    plane_phase = str(
        storage.get("storage_plane_phase")
        or _dict(efficiency.get("storage_plane_phase_contract")).get("phase")
        or ""
    ).strip().lower()
    collector_status = str(
        inputs.get("collector_intake_status")
        or ingestion.get("collector_intake_status")
        or ""
    ).strip().lower()
    collector_safe = bool(
        inputs.get("collector_intake_soak_safe")
        or inputs.get("collector_partial_reserve_pressure_soak_safe")
        or ingestion.get("collector_intake_soak_safe")
        or collector_status in {"", "not_required", "ready", "enforced"}
    )
    bounded_progress = bool(bounded.get("active_drain_progress") or bounded.get("drain_delta_signal_observed"))
    hard_gate_clear = not bool(bounded.get("hard_gate_active")) and not bool(bounded.get("effective_hard_gate_active"))
    efficiency_ready = bool(
        efficiency_status in {"", "ready", "ok"}
        and (efficiency_grade in {"", "A", "A+"} or plane_phase in {"steady_state", "deep_cold_managed_steady_state"})
    )
    bounded_ready = bool(
        raw_grade in {"A+", "A", ""}
        and severity in {"stable", "low", "ready", "normal", "watch", "elevated"}
        and pressure_index <= 1.05
        and total_pending <= BOUNDED_TRANSIENT_TOTAL_MAX_LINES
        and core_pending <= BOUNDED_TRANSIENT_CORE_MAX_LINES
        and oldest_age <= BOUNDED_TRANSIENT_AGE_MAX_SECONDS
        and stale_clear
        and route_ready
        and resilience_ready
        and data_clean
        and no_queue_breaches
        and bounded_progress
        and hard_gate_clear
        and efficiency_ready
        and collector_safe
    )
    return bounded_ready


def _storage_contract(
    *,
    project_root: Path,
    target_days: float,
    disk_snapshot_fn: DiskSnapshotFn = _disk_snapshot,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    retention = load_json(health_root / "storage_retention_unison_latest.json")
    ingestion = load_json(health_root / "ingestion_storage_control_latest.json")
    cleanup = load_json(health_root / "bot_logs_cleanup_intelligence_latest.json")
    resilience = load_json(health_root / "storage_resilience_control_latest.json")
    mount_guard = load_json(health_root / "storage_mount_guard_latest.json")
    external_root = _external_root(project_root, health_root)
    disk = disk_snapshot_fn(external_root)
    local_disk = disk_snapshot_fn(project_root)

    retention_contract = retention.get("continuous_run_contract") if isinstance(retention.get("continuous_run_contract"), dict) else {}
    ingestion_contract = ingestion.get("continuous_run_soak_contract") if isinstance(ingestion.get("continuous_run_soak_contract"), dict) else {}
    effective_daily = max(
        _safe_float(retention_contract.get("effective_daily_growth_gb"), 0.0),
        _safe_float(retention_contract.get("min_daily_growth_gb"), 0.5),
        0.5,
    )
    pressure_free = max(_safe_float(retention_contract.get("pressure_free_gb"), 64.0), 0.0)
    buffer_gb = max(_safe_float(retention_contract.get("safety_buffer_gb"), 32.0), 0.0)
    required_free = round(pressure_free + buffer_gb + (effective_daily * max(float(target_days), 1.0)), 3)
    current_free = _safe_float(disk.get("free_gb"), _safe_float(retention_contract.get("current_external_free_gb"), 0.0))
    projected_free = _safe_float(cleanup.get("projected_free_gb"), current_free)
    margin = round(current_free - required_free, 3)
    projected_margin = round(projected_free - required_free, 3)
    cold_archive_spillover_ready = bool(retention_contract.get("cold_archive_spillover_ready", False))
    cold_archive_spillover_available = bool(retention_contract.get("cold_archive_spillover_available", False))
    cold_archive_spillover_status = str(retention_contract.get("cold_archive_spillover_status") or "")
    cold_archive_adjusted_margin = _safe_float(retention_contract.get("cold_archive_adjusted_margin_gb"), margin)
    cold_archive_capacity = _safe_float(retention_contract.get("cold_archive_spillover_capacity_gb"), 0.0)
    cold_archive_required = _safe_float(retention_contract.get("cold_archive_required_spillover_gb"), max(-margin, 0.0))
    cold_archive_shortfall = _safe_float(
        retention_contract.get("cold_archive_capacity_shortfall_gb"),
        max(-cold_archive_adjusted_margin, 0.0),
    )
    primary_pressure_buffer = _safe_float(retention_contract.get("cold_archive_primary_pressure_buffer_gb"), 16.0)
    primary_above_pressure_guard = bool(current_free >= pressure_free + primary_pressure_buffer)
    storage_margin_ready = bool(
        current_free >= required_free
        or (cold_archive_spillover_ready and cold_archive_adjusted_margin >= 0.0 and primary_above_pressure_guard)
    )

    blockers: list[str] = []
    warnings: list[str] = []
    managed_controls: list[str] = []
    local_target_free = max(_safe_float(os.getenv("BOT_LOCAL_STORAGE_TARGET_FREE_GB"), 64.0), 1.0)
    local_pressure_free = min(
        max(_safe_float(os.getenv("BOT_LOCAL_STORAGE_PRESSURE_FREE_GB"), 32.0), 1.0),
        local_target_free,
    )
    local_free = _safe_float(local_disk.get("free_gb"), 0.0)
    local_known = bool(local_free > 0.0 and _safe_float(local_disk.get("used_pct"), 100.0) < 100.0)
    if not local_known:
        blockers.append("local_hot_storage_free_space_unknown")
    elif local_free < local_pressure_free:
        blockers.append("local_hot_storage_pressure_reserve_breached")
    elif local_free < local_target_free:
        warnings.append("local_hot_storage_below_unattended_target")
    if current_free <= 0.0:
        blockers.append("external_free_space_unknown")
    if not storage_margin_ready:
        blockers.append("storage_margin_not_30_day_ready")
        if projected_margin >= 0.0 and _safe_float(cleanup.get("selected_reclaimable_gb"), 0.0) > 0.0:
            warnings.append("storage_cleanup_plan_available_not_applied")
    elif current_free < required_free and cold_archive_spillover_ready:
        warnings.append("storage_margin_managed_by_approved_cold_archive_spillover")
        managed_controls.append("approved_cold_archive_spillover")
    if not bool(retention_contract.get("ready", False)):
        blockers.append("storage_retention_contract_not_ready")
    ingestion_managed_watch = _managed_ingestion_soak_watch(ingestion, ingestion_contract)
    ingestion_soak_ready = bool(
        ingestion_contract.get("ready", False)
        or ingestion_contract.get("soak_ready", False)
        or ingestion_managed_watch
    )
    if not ingestion_soak_ready:
        blockers.append("ingestion_soak_contract_not_ready")
    elif ingestion_managed_watch and not bool(ingestion_contract.get("ready", False) or ingestion_contract.get("soak_ready", False)):
        warnings.append("ingestion_soak_contract_managed_by_bounded_backlog_watch")
        managed_controls.append("ingestion_soak_contract_managed_by_bounded_backlog_watch")
    if not bool(resilience.get("ok", False)):
        blockers.append("storage_resilience_not_ready")
    if bool(mount_guard.get("external_low_space", False)):
        warnings.append("storage_mount_guard_low_space")

    failed_db_checks = []
    for row in resilience.get("database_integrity_checks") if isinstance(resilience.get("database_integrity_checks"), list) else []:
        if isinstance(row, dict) and bool(row.get("present", False)) and not bool(row.get("ok", False)):
            failed_db_checks.append(str(row.get("db_path") or "unknown"))
    if failed_db_checks:
        blockers.append("storage_database_integrity_errors")

    status = "ready" if not blockers else "blocked"
    score = max(100.0 - (12.0 * len(blockers)) - (3.0 * len(warnings)), 0.0)
    return {
        "status": status,
        "ready": status == "ready",
        "score": round(score, 2),
        "grade": _grade(score),
        "target_days": round(float(target_days), 3),
        "external_root": str(external_root),
        "disk": disk,
        "local_hot_storage": {
            "disk": local_disk,
            "free_gb": round(local_free, 3),
            "target_free_gb": round(local_target_free, 3),
            "pressure_free_gb": round(local_pressure_free, 3),
            "ready": bool(local_known and local_free >= local_target_free),
        },
        "required_external_free_gb": required_free,
        "current_external_free_gb": round(current_free, 3),
        "available_margin_gb": margin,
        "projected_margin_gb": projected_margin,
        "cold_archive_spillover_available": cold_archive_spillover_available,
        "cold_archive_spillover_ready": cold_archive_spillover_ready,
        "cold_archive_spillover_status": cold_archive_spillover_status,
        "cold_archive_spillover_capacity_gb": round(cold_archive_capacity, 3),
        "cold_archive_required_spillover_gb": round(cold_archive_required, 3),
        "cold_archive_capacity_shortfall_gb": round(cold_archive_shortfall, 3),
        "cold_archive_adjusted_margin_gb": round(cold_archive_adjusted_margin, 3),
        "cold_archive_capacity_policy": str(retention_contract.get("cold_archive_capacity_policy") or ""),
        "primary_above_pressure_guard": primary_above_pressure_guard,
        "primary_pressure_buffer_gb": round(primary_pressure_buffer, 3),
        "effective_daily_growth_gb": round(effective_daily, 4),
        "retention_status": str(retention_contract.get("status") or ""),
        "ingestion_status": str(ingestion_contract.get("status") or ""),
        "ingestion_soak_ready": ingestion_soak_ready,
        "ingestion_managed_watch": ingestion_managed_watch,
        "resilience_status": str(resilience.get("overall_status") or ""),
        "failed_database_integrity_checks": failed_db_checks,
        "blockers": ordered_unique(blockers),
        "warnings": ordered_unique(warnings),
        "managed_controls": ordered_unique(managed_controls),
    }


def _paper_soak_auth_ready(auth: dict[str, Any], schwab_auth: dict[str, Any], broker: dict[str, Any]) -> bool:
    broker_state = _dict(auth.get("broker_state"))
    lease_budget = _dict(auth.get("lease_budget"))
    token = _dict(schwab_auth.get("token"))
    broker_preflight = _dict(broker.get("preflight_checks"))
    lease_expires = max(
        _safe_float(auth.get("expires_in_seconds"), 0.0),
        _safe_float(lease_budget.get("expires_in_seconds"), 0.0),
        _safe_float(token.get("expires_in_seconds"), 0.0),
        _safe_float(broker.get("token_expires_in_seconds"), 0.0),
    )
    ready_floor = max(
        _safe_float(schwab_auth.get("min_ready_expires_seconds"), 0.0),
        _safe_float(token.get("min_ready_expires_seconds"), 0.0),
        _safe_float(_dict(schwab_auth.get("regression_contract")).get("schwab_token_ready_floor_seconds"), 0.0),
        900.0,
    )
    critical_floor = max(_safe_float(lease_budget.get("critical_lease_seconds"), 0.0), 600.0)
    token_ready = bool(
        bool(schwab_auth.get("token_ready", False))
        or bool(token.get("ready", False))
        or bool(broker_preflight.get("token_ready_for_open", False))
        or bool(broker.get("ready_for_open", False))
    )
    readiness_refresh_needed = bool(
        bool(schwab_auth.get("readiness_refresh_needed", False))
        or bool(token.get("readiness_refresh_needed", False))
        or bool(broker_preflight.get("readiness_refresh_needed_after", False))
        or lease_expires < ready_floor
    )
    network_ok = bool(
        broker.get("network_ok", True) is not False
        and broker_state.get("network_ok", True) is not False
    )
    broker_operable = bool(bool(broker.get("ready_for_open", False)) or bool(broker_state.get("broker_operable", False)))
    configured_for_refresh = bool(
        broker_state.get("configured_for_refresh", True) is not False
        and (token_ready or bool(token) or bool(broker_preflight.get("token_exists", False)))
    )
    return bool(
        token_ready
        and not readiness_refresh_needed
        and network_ok
        and broker_operable
        and configured_for_refresh
        and lease_expires >= critical_floor
    )


def _runtime_contract(project_root: Path) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    process = load_json(health_root / "process_watchdog_latest.json")
    live_runtime = load_json(health_root / "live_runtime_separation_control_latest.json")
    auth = load_json(health_root / "auth_lease_manager_latest.json")
    schwab_auth = load_json(health_root / "schwab_auth_supervisor_latest.json")
    broker = load_json(health_root / "broker_readiness_latest.json")
    blockers: list[str] = []
    warnings: list[str] = []
    managed_controls: list[str] = []
    restart_storms = process.get("restart_storms") if isinstance(process.get("restart_storms"), list) else []
    isolation = process.get("restart_storm_isolation") if isinstance(process.get("restart_storm_isolation"), dict) else {}
    if not isolation:
        intelligence = process.get("watchdog_intelligence") if isinstance(process.get("watchdog_intelligence"), dict) else {}
        isolation = intelligence.get("restart_storm_isolation") if isinstance(intelligence.get("restart_storm_isolation"), dict) else {}
    isolated_read_only_storms = bool(
        restart_storms
        and isolation
        and bool(isolation.get("all_active_storms_isolated", False))
        and _safe_int(isolation.get("execution_blocking_count"), 0) == 0
    )
    if str(process.get("overall_status") or "").lower() != "ready":
        if isolated_read_only_storms:
            managed_controls.append("process_watchdog_degraded_only_by_isolated_read_only_collection")
        else:
            blockers.append("process_watchdog_not_ready")
    if restart_storms:
        if isolated_read_only_storms:
            managed_controls.append("read_only_collection_restart_storms_isolated")
        else:
            blockers.append("restart_storms_present")
    if process.get("alerts"):
        if isolated_read_only_storms:
            managed_controls.append("process_watchdog_alerts_isolated_read_only_collection")
        else:
            warnings.append("process_watchdog_alerts_present")
    live_runtime_status = str(live_runtime.get("overall_status") or "").lower()
    live_plane = live_runtime.get("live_plane") if isinstance(live_runtime.get("live_plane"), dict) else {}
    clearance_plan = live_runtime.get("clearance_plan") if isinstance(live_runtime.get("clearance_plan"), dict) else {}
    release_contract = live_runtime.get("release_contract") if isinstance(live_runtime.get("release_contract"), dict) else {}
    clearance_state = str(clearance_plan.get("clearance_state") or "").strip().lower()
    cold_lane_deferred = bool(
        live_runtime_status == "degraded"
        and bool(live_plane.get("ready", False))
        and bool(live_plane.get("broker_ready", False))
        and bool(live_plane.get("session_ready", False))
        and bool(live_plane.get("live_lane_running", False))
        and clearance_state in {"awaiting_cold_lane", "managed_cold_lane_deferred", "managed_coverage_stage_deferred", "protect_live", "ready"}
    )
    paper_soak_live_release_deferred = bool(
        live_runtime_status == "degraded"
        and bool(release_contract.get("live_lane_should_be_read_only", False))
        and bool(live_plane.get("broker_ready", False))
        and bool(live_plane.get("session_ready", False))
        and clearance_state in {"awaiting_cold_lane", "managed_cold_lane_deferred", "managed_coverage_stage_deferred", "protect_live"}
    )
    if live_runtime_status == "blocked":
        blockers.append("live_runtime_separation_blocked")
    elif cold_lane_deferred:
        managed_controls.append("live_plane_ready_cold_lane_refresh_deferred")
    elif paper_soak_live_release_deferred:
        managed_controls.append("paper_soak_live_money_locked_cold_lane_deferred")
    elif live_runtime_status == "degraded":
        warnings.append("live_runtime_separation_degraded")
    paper_auth_ready = _paper_soak_auth_ready(auth, schwab_auth, broker)
    if auth and not bool(auth.get("ok", False)) and not paper_auth_ready:
        blockers.append("auth_lease_not_ready")
    if broker and not bool(broker.get("ready_for_open", broker.get("ok", True))):
        warnings.append("broker_readiness_not_ready_for_open")
    status = "ready" if not blockers else "blocked"
    score = max(100.0 - (15.0 * len(blockers)) - (4.0 * len(warnings)), 0.0)
    return {
        "status": status,
        "ready": status == "ready",
        "score": round(score, 2),
        "grade": _grade(score),
        "process_watchdog_status": str(process.get("overall_status") or ""),
        "live_runtime_status": str(live_runtime.get("overall_status") or ""),
        "auth_status": str(auth.get("overall_status") or auth.get("status") or ""),
        "strict_auth_ready": bool(auth.get("ok", False)),
        "paper_soak_auth_ready": bool(paper_auth_ready),
        "broker_ready_for_open": bool(broker.get("ready_for_open", False)),
        "restart_storm_count": len(process.get("restart_storms") or []),
        "alert_count": len(process.get("alerts") or []),
        "isolated_read_only_restart_storms": bool(isolated_read_only_storms),
        "blockers": blockers,
        "warnings": warnings,
        "managed_controls": managed_controls,
    }


def _livefeed_remote_viewer_ready(livefeed_guard: dict[str, Any]) -> bool:
    if not livefeed_guard:
        return False
    health = livefeed_guard.get("health") if isinstance(livefeed_guard.get("health"), dict) else {}
    blockers = livefeed_guard.get("blockers") if isinstance(livefeed_guard.get("blockers"), list) else []
    status = str(livefeed_guard.get("overall_status") or livefeed_guard.get("status") or "").strip().lower()
    return bool(livefeed_guard.get("ok", health.get("ok", False))) and status in {"ready", "running"} and not blockers


def _alerting_contract(project_root: Path) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    ladder = load_json(health_root / "notification_escalation_ladder_latest.json")
    livefeed_guard = load_json(health_root / "livefeed_refresh_guard_latest.json")
    blockers: list[str] = []
    warnings: list[str] = []
    managed_controls: list[str] = []
    backlog = ladder.get("critical_backlog") if isinstance(ladder.get("critical_backlog"), dict) else {}
    grouped_unsent = _safe_int(backlog.get("grouped_unsent_count"), 0)
    grouped_unacked = _safe_int(backlog.get("grouped_unacked_count"), 0)
    attended_runtime_ready = bool(ladder.get("attended_runtime_ready", False))
    zero_touch_unattended_ready = bool(ladder.get("unattended_runtime_ready", False))
    phone_bridge_ready = bool(ladder.get("phone_bridge_ready", False))
    remote_pager_ready = bool(ladder.get("remote_pager_ready", False))
    livefeed_ready = bool(ladder.get("livefeed_remote_viewer_ready", False)) or _livefeed_remote_viewer_ready(livefeed_guard)
    mobile_operator_coverage_ready = bool(ladder.get("mobile_operator_coverage_ready", False)) or bool(
        phone_bridge_ready
        and livefeed_ready
        and grouped_unsent == 0
        and grouped_unacked == 0
    )
    if not ladder:
        blockers.append("notification_ladder_missing")
    if ladder and not attended_runtime_ready:
        blockers.append("attended_alert_path_not_ready")
    if ladder and not zero_touch_unattended_ready and not mobile_operator_coverage_ready:
        blockers.append("unattended_remote_pager_not_ready")
    if grouped_unsent > 0:
        blockers.append("critical_alerts_unsent")
    if grouped_unacked > 0:
        blockers.append("critical_alerts_unacked")
    if ladder and mobile_operator_coverage_ready and not zero_touch_unattended_ready:
        managed_controls.append("daily_mobile_operator_coverage_active_without_zero_touch_remote_pager")
    elif ladder and phone_bridge_ready and not remote_pager_ready:
        warnings.append("phone_bridge_ready_but_remote_pager_missing")
    status = "ready" if not blockers else "blocked"
    score = max(100.0 - (18.0 * len(blockers)) - (4.0 * len(warnings)), 0.0)
    return {
        "status": status,
        "ready": status == "ready",
        "score": round(score, 2),
        "grade": _grade(score),
        "attended_runtime_ready": attended_runtime_ready,
        "unattended_runtime_ready": zero_touch_unattended_ready,
        "zero_touch_unattended_ready": zero_touch_unattended_ready,
        "mobile_operator_coverage_ready": mobile_operator_coverage_ready,
        "livefeed_remote_viewer_ready": livefeed_ready,
        "operator_coverage_model": (
            "zero_touch_remote_pager"
            if zero_touch_unattended_ready
            else (
                "daily_supervised_mobile_operator"
                if mobile_operator_coverage_ready
                else ("attended_alerts_only" if attended_runtime_ready else "not_ready")
            )
        ),
        "remote_pager_ready": remote_pager_ready,
        "phone_bridge_ready": phone_bridge_ready,
        "critical_backlog": backlog,
        "blockers": blockers,
        "warnings": warnings,
        "managed_controls": managed_controls,
    }


def _freshness_contract(project_root: Path) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    paths = {
        "storage_retention_unison": health_root / "storage_retention_unison_latest.json",
        "ingestion_storage_control": health_root / "ingestion_storage_control_latest.json",
        "process_watchdog": health_root / "process_watchdog_latest.json",
        "notification_escalation_ladder": health_root / "notification_escalation_ladder_latest.json",
        "livefeed_refresh_guard": health_root / "livefeed_refresh_guard_latest.json",
        "storage_resilience_control": health_root / "storage_resilience_control_latest.json",
    }
    rows: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for name, path in paths.items():
        payload = load_json(path)
        age = payload_age_minutes(payload, path)
        rows[name] = {"path": str(path), "present": bool(payload), "age_minutes": round(float(age), 3) if age is not None else None}
        if not payload:
            warnings.append(f"{name}_missing")
        elif age is not None and age > 180.0:
            warnings.append(f"{name}_stale")
    status = "ready" if not warnings else "watch"
    score = max(100.0 - (4.0 * len(warnings)), 0.0)
    return {
        "status": status,
        "ready": status == "ready",
        "score": round(score, 2),
        "grade": _grade(score),
        "warnings": warnings,
        "artifacts": rows,
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    target_days: float = DEFAULT_TARGET_DAYS,
    pmset_custom_text: str | None = None,
    pmset_batt_text: str | None = None,
    process_text: str | None = None,
    disk_snapshot_fn: DiskSnapshotFn = _disk_snapshot,
) -> dict[str, Any]:
    target = max(float(target_days), 1.0)
    storage = _storage_contract(project_root=project_root, target_days=target, disk_snapshot_fn=disk_snapshot_fn)
    host = _host_power_contract(
        project_root=project_root,
        pmset_custom_text=pmset_custom_text,
        pmset_batt_text=pmset_batt_text,
        process_text=process_text,
    )
    runtime = _runtime_contract(project_root)
    alerting = _alerting_contract(project_root)
    freshness = _freshness_contract(project_root)
    sections = {
        "storage": storage,
        "host_power": host,
        "runtime_loops": runtime,
        "alerting": alerting,
        "artifact_freshness": freshness,
    }
    blockers = ordered_unique(
        list(storage.get("blockers") or [])
        + list(host.get("blockers") or [])
        + list(runtime.get("blockers") or [])
        + list(alerting.get("blockers") or [])
    )
    warnings = ordered_unique(
        list(storage.get("warnings") or [])
        + list(host.get("warnings") or [])
        + list(runtime.get("warnings") or [])
        + list(alerting.get("warnings") or [])
        + list(freshness.get("warnings") or [])
    )
    scored_warnings = [item for item in warnings if not _warning_is_managed_for_soak(item)]
    managed_warnings = [item for item in warnings if _warning_is_managed_for_soak(item)]
    managed_controls = ordered_unique(
        list(storage.get("managed_controls") or [])
        + list(host.get("managed_controls") or [])
        + list(runtime.get("managed_controls") or [])
        + list(alerting.get("managed_controls") or [])
    )
    section_scores = [_safe_float(row.get("score"), 92.0) for row in sections.values() if isinstance(row, dict)]
    base_score = sum(section_scores) / max(len(section_scores), 1)
    score = max(min(base_score - (6.0 * len(blockers)) - (1.5 * len(scored_warnings)), 100.0), 0.0)
    status = "ready" if not blockers else "blocked"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": status == "ready",
        "overall_status": status,
        "overall_score": round(score, 2),
        "overall_grade": _grade(score),
        "target_days": round(target, 3),
        "safe_to_leave_unattended": bool(status == "ready"),
        "live_money_readiness": "locked_not_evaluated_by_soak_readiness",
        "blockers": blockers,
        "warnings": warnings,
        "managed_warnings": managed_warnings,
        "managed_controls": managed_controls,
        "scored_warnings": scored_warnings,
        "sections": sections,
        "control_env": {
            "BOT_UNATTENDED_SOAK_ACTIVE": "1",
            "BOT_UNATTENDED_SOAK_TARGET_DAYS": str(round(target, 3)),
            "BOT_UNATTENDED_SOAK_READY": "1" if status == "ready" else "0",
            "BOT_LIVE_MONEY_LOCKED_DURING_SOAK": "1",
        },
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "storage-retention-unison", "--apply", "--soak-days", str(round(target, 3)), "--json"],
            ["./scripts/ops/opsctl.sh", "bot-logs-cleanup-intelligence", "--apply", "--target-free-gb", "125", "--max-tier", "2", "--json"],
            ["./scripts/install_caffeinate_launchd.sh"],
            ["./scripts/ops/opsctl.sh", "notification-escalation-ladder", "--json"],
            ["./scripts/ops/opsctl.sh", "storage-resilience-control", "--fast", "--json"],
            ["./scripts/ops/opsctl.sh", "local-storage-reserve-guard", "--apply", "--json"],
        ],
        "next_action": (
            "all unattended soak gates are ready; leave live money locked and let the soak run"
            if status == "ready"
            else "clear blockers before leaving the 30-day soak unattended"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a single 30-day unattended soak readiness contract.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--target-days", type=float, default=float(os.getenv("UNATTENDED_SOAK_TARGET_DAYS", str(DEFAULT_TARGET_DAYS))))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), target_days=float(args.target_days))
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "unattended_soak_readiness "
            f"status={payload.get('overall_status', '')} "
            f"grade={payload.get('overall_grade', '')} "
            f"target_days={payload.get('target_days', 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
