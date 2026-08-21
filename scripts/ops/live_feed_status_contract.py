#!/usr/bin/env python3
"""Build a freshness-aware, contradiction-resistant livefeed status snapshot."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 3
ARTIFACT_SPECS = {
    "health_fast": ("health_fast_latest.json", 10 * 60),
    "broker_readiness": ("broker_readiness_latest.json", 10 * 60),
    "auth_lease": ("auth_lease_manager_latest.json", 10 * 60),
    "schwab_auth": ("schwab_auth_supervisor_latest.json", 10 * 60),
    "storage": ("ingestion_storage_control_latest.json", 10 * 60),
    "throttle": ("runtime_throttle_control_latest.json", 10 * 60),
    "paper_ramp": ("paper_400_ramp_latest.json", 10 * 60),
    "unattended_soak": ("unattended_soak_readiness_latest.json", 45 * 60),
    "continuous_soak": ("continuous_soak_integrity_control_latest.json", 45 * 60),
    "production_excellence": ("production_excellence_control_latest.json", 15 * 60),
    "strategy_specialization": (
        "governance/research/sleeve_strategy_specialization_latest.json",
        90 * 60,
    ),
}
REQUIRED_SOURCES = {
    "health_fast",
    "broker_readiness",
    "auth_lease",
    "schwab_auth",
    "storage",
    "throttle",
    "paper_ramp",
    "unattended_soak",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _artifact(root: Path, filename: str, max_age_seconds: float, now: datetime) -> dict[str, Any]:
    relative = Path(filename)
    path = relative if relative.is_absolute() else (
        root / relative if len(relative.parts) > 1 else root / "governance" / "health" / relative
    )
    payload = _load_json(path)
    present = bool(payload)
    timestamp = _parse_timestamp(payload.get("timestamp_utc"))
    timestamp_source = "payload"
    if timestamp is None and path.exists():
        try:
            timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            timestamp_source = "mtime"
        except OSError:
            timestamp = None
    age_seconds = max(0.0, (now - timestamp).total_seconds()) if timestamp else None
    fresh = bool(present and age_seconds is not None and age_seconds <= max_age_seconds)
    return {
        "path": str(path),
        "payload": payload,
        "present": present,
        "fresh": fresh,
        "state": "fresh" if fresh else ("stale" if present else "missing"),
        "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
        "max_age_seconds": max_age_seconds,
        "timestamp_source": timestamp_source if timestamp else "missing",
    }


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _first(values: Any, default: str = "none") -> str:
    if isinstance(values, list):
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
    return default


def _status_text(payload: dict[str, Any], default: str = "unknown") -> str:
    return str(payload.get("overall_status") or payload.get("status") or default).strip().lower()


def _source_public(artifact: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in artifact.items() if key != "payload"}


def _auth_row(sources: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    broker_source = sources["broker_readiness"]
    lease_source = sources["auth_lease"]
    supervisor_source = sources["schwab_auth"]
    broker = broker_source["payload"]
    lease = lease_source["payload"]
    supervisor = supervisor_source["payload"]
    broker_state = lease.get("broker_state") if isinstance(lease.get("broker_state"), dict) else {}
    budget = lease.get("lease_budget") if isinstance(lease.get("lease_budget"), dict) else {}
    token = supervisor.get("token") if isinstance(supervisor.get("token"), dict) else {}
    preflight = broker.get("preflight_checks") if isinstance(broker.get("preflight_checks"), dict) else {}

    broker_ok = bool(broker.get("ready_for_open") and broker.get("network_ok") and broker.get("auth_ok"))
    lease_ok = bool(
        _status_text(lease) == "ready"
        and str(lease.get("lease_state") or "").lower() == "healthy"
        and broker_state.get("broker_ready", True)
        and broker_state.get("auth_ok", True)
        and broker_state.get("network_ok", True)
    )
    supervisor_ok = bool(_status_text(supervisor) == "ready" and token.get("ready"))
    expires_candidates = [
        _as_float(token.get("expires_in_seconds"), -1.0),
        _as_float(budget.get("expires_in_seconds"), -1.0),
        _as_float(broker.get("token_expires_in_seconds"), -1.0),
    ]
    valid_expires = [value for value in expires_candidates if value >= 0.0]
    expires_in_seconds = min(valid_expires) if valid_expires else 0.0
    ready_floor = _as_float(token.get("min_ready_expires_seconds"), 900.0)

    state_by_source = {
        "broker_readiness": broker_ok,
        "auth_lease": lease_ok,
        "schwab_auth": supervisor_ok,
    }
    fresh_states = {
        name: state
        for name, state in state_by_source.items()
        if sources[name]["fresh"]
    }
    fresh_failures = [name for name, state in fresh_states.items() if not state]
    stale_sources = [name for name in state_by_source if not sources[name]["fresh"]]
    missing_sources = [name for name in state_by_source if not sources[name]["present"]]
    active_warnings: list[str] = []
    superseded_warnings: list[str] = []
    for warning in broker.get("warnings") if isinstance(broker.get("warnings"), list) else []:
        text = str(warning or "").strip()
        expiring = text.startswith("token_expiring_soon:")
        refresh_resolved = bool(
            expiring
            and expires_in_seconds >= _as_float(token.get("min_expires_seconds"), 1500.0)
            and not preflight.get("refresh_needed_after", False)
            and not token.get("refresh_needed", False)
        )
        (superseded_warnings if refresh_resolved else active_warnings).append(text)

    if missing_sources:
        status = "blocked"
        reason = f"missing_{missing_sources[0]}"
    elif expires_in_seconds < ready_floor:
        status = "blocked"
        reason = "token_below_ready_floor"
    elif fresh_failures:
        status = "blocked"
        reason = f"{fresh_failures[0]}_not_ready"
    elif len(fresh_states) < 2:
        status = "degraded"
        reason = "insufficient_fresh_auth_quorum"
    elif active_warnings:
        status = "watch"
        reason = active_warnings[0]
    elif stale_sources:
        status = "watch"
        reason = f"stale_{stale_sources[0]}"
    else:
        status = "ready"
        reason = "none"

    unique_states = set(fresh_states.values())
    if len(unique_states) > 1:
        consistency = "conflict"
    elif superseded_warnings or stale_sources:
        consistency = "reconciled"
    else:
        consistency = "consistent"
    action = "none"
    if status == "blocked":
        action = "schwab-auth-supervisor"
    elif token.get("refresh_needed") or expires_in_seconds < _as_float(token.get("min_expires_seconds"), 1500.0):
        action = "token-refresh"

    auth = {
        "status": status,
        "reason": reason,
        "lease": str(lease.get("lease_state") or "unknown"),
        "broker_ready": broker_ok,
        "network_ok": bool(broker.get("network_ok") and broker_state.get("network_ok", True)),
        "auth_ok": bool(broker.get("auth_ok") and broker_state.get("auth_ok", True)),
        "probe_ok": bool(broker_state.get("auth_probe_ok", budget.get("probe_backed", False))),
        "token_ready": supervisor_ok,
        "expires_in_seconds": round(expires_in_seconds, 3),
        "freshness": "fresh" if not stale_sources else "partial",
        "consistency": consistency,
        "active_warning_count": len(active_warnings),
        "superseded_warning_count": len(superseded_warnings),
        "max_source_age_seconds": max(
            (_as_float(sources[name].get("age_seconds"), 0.0) for name in state_by_source),
            default=0.0,
        ),
        "impact": "paper_blocked" if status == "blocked" else "none",
        "action": action,
    }
    schwab_auth = {
        "status": status,
        "token_ready": supervisor_ok,
        "refresh_needed": bool(token.get("refresh_needed")),
        "expires_in_seconds": round(expires_in_seconds, 3),
        "token_age_seconds": _as_float(token.get("age_seconds"), 0.0),
        "artifact_age_seconds": supervisor_source.get("age_seconds"),
        "warning_count": len(active_warnings),
        "reconciled_warning_count": len(superseded_warnings),
        "action": action,
    }
    return auth, schwab_auth


def _system_and_collection_rows(sources: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = sources["health_fast"]
    health = source["payload"]
    operational = health.get("operational_readiness") if isinstance(health.get("operational_readiness"), dict) else {}
    paper = operational.get("guarded_paper") if isinstance(operational.get("guarded_paper"), dict) else {}
    live = operational.get("live_execution") if isinstance(operational.get("live_execution"), dict) else {}
    collection_payload = health.get("collection") if isinstance(health.get("collection"), dict) else {}
    process_watchdog = health.get("process_watchdog") if isinstance(health.get("process_watchdog"), dict) else {}
    sleeves = (
        process_watchdog.get("all_sleeves_effective_runtime")
        if isinstance(process_watchdog.get("all_sleeves_effective_runtime"), dict)
        else {}
    )
    paper_status = str(paper.get("status") or ("ready" if paper.get("ok") else "blocked")).lower()
    blockers = paper.get("blockers") if isinstance(paper.get("blockers"), list) else []
    status = _status_text(health)
    if not source["fresh"]:
        status = "degraded"
    cause = _first(blockers, "repair_backlog_active" if health.get("repair_backlog_active") else "none")
    system = {
        "status": status,
        "strict_all_clear": bool(health.get("strict_all_clear")),
        "paper_status": paper_status,
        "live_status": str(live.get("status") or "unknown"),
        "repair_active": bool(health.get("repair_backlog_active")),
        "cause": cause,
        "artifact_age_seconds": source.get("age_seconds"),
        "action": "none" if cause == "none" and status == "ready" else "storage-backpressure-autopilot" if cause.startswith("storage_") else "health-fast",
    }
    collector_count = _as_int(collection_payload.get("collector_count"))
    observing = _as_int(collection_payload.get("effective_bots_with_observations", collection_payload.get("bots_with_observations")))
    zero = _as_int(collection_payload.get("unmanaged_zero_observation_count"))
    launcher_ok = bool(sleeves.get("ok", sleeves.get("status") == "ready"))
    collection_status = "ready" if launcher_ok and collector_count > 0 and observing >= collector_count and zero == 0 else "degraded"
    if not source["fresh"]:
        collection_status = "degraded"
    if not source["fresh"]:
        collection_cause = "stale_health_fast_artifact"
    elif collector_count <= 0:
        collection_cause = "no_collectors_reported"
    elif observing < collector_count:
        collection_cause = "collector_observation_coverage_gap"
    elif zero > 0:
        collection_cause = "unmanaged_zero_observation_bots"
    elif not launcher_ok:
        collection_cause = "sleeve_fanout_not_ready"
    else:
        collection_cause = "none"
    collection = {
        "status": collection_status,
        "collectors": collector_count,
        "observing": observing,
        "unmanaged_zero": zero,
        "observations": _as_int(collection_payload.get("total_observations")),
        "sleeve_children": _as_int(sleeves.get("child_process_count")),
        "fanout_ok": bool(sleeves.get("child_fanout_ok")),
        "cause": collection_cause,
        "impact": "none" if collection_cause == "none" else "collection_incomplete",
        "artifact_age_seconds": source.get("age_seconds"),
        "action": "none" if collection_cause == "none" else "process-watchdog",
    }
    return system, collection


def _paper_ramp_row(sources: dict[str, dict[str, Any]], system: dict[str, Any]) -> dict[str, Any]:
    source = sources["paper_ramp"]
    ramp = source["payload"]
    stage = str(ramp.get("stage") or "unknown").strip().lower()
    blockers = ramp.get("blockers") if isinstance(ramp.get("blockers"), list) else []
    armed = bool(ramp.get("armed") or stage == "armed")
    ramp_ready = bool(ramp.get("ok") and armed and not blockers)
    health_paper_status = str(system.get("paper_status") or "unknown").strip().lower()
    health_ready = health_paper_status == "ready"

    if not source["present"]:
        status = "blocked"
        cause = "missing_paper_ramp_artifact"
    elif not source["fresh"]:
        status = "degraded"
        cause = "stale_paper_ramp_artifact"
    elif ramp_ready:
        status = "ready"
        cause = "none"
    else:
        status = "blocked"
        cause = _first(blockers, "paper_ramp_not_armed" if not armed else "paper_ramp_not_ready")

    sources_disagree = bool(source["fresh"] and sources["health_fast"]["fresh"] and health_ready != ramp_ready)
    if sources_disagree or status == "blocked" or health_paper_status == "blocked":
        effective_paper_status = "blocked"
    elif status != "ready" or health_paper_status != "ready":
        effective_paper_status = "degraded"
    else:
        effective_paper_status = "ready"
    return {
        "status": effective_paper_status,
        "source_status": status,
        "health_status": health_paper_status,
        "stage": stage,
        "armed": armed,
        "ok": ramp_ready,
        "blockers": [str(value) for value in blockers if str(value or "").strip()],
        "sources_disagree": sources_disagree,
        "cause": "paper_sources_disagree" if sources_disagree else cause,
        "impact": "none" if effective_paper_status == "ready" else "paper_blocked" if effective_paper_status == "blocked" else "paper_unverified",
        "artifact_age_seconds": source.get("age_seconds"),
        "action": "none" if effective_paper_status == "ready" else "paper-400-ramp",
    }


def _fx_provider_row(project_root: Path, now: datetime) -> dict[str, Any]:
    session_source = _artifact(project_root, "fx_shadow_session_latest.json", 10 * 60, now)
    ingress_source = _artifact(project_root, "data_ingress_latest_fx_equities_schwab.json", 10 * 60, now)
    guard_source = _artifact(project_root, "fx_twelve_data_guard_latest.json", 24 * 60 * 60, now)
    if not session_source["present"] and not ingress_source["present"] and not guard_source["present"]:
        return {}

    session_payload = session_source["payload"]
    ingress = ingress_source["payload"]
    guard = guard_source["payload"]
    session = session_payload.get("session") if isinstance(session_payload.get("session"), dict) else {}
    provider = session.get("provider") if isinstance(session.get("provider"), dict) else {}
    cooldown = provider.get("cooldown") if isinstance(provider.get("cooldown"), dict) else guard
    cooldown_until = _as_float(cooldown.get("cooldown_until_ts"), 0.0)
    cooldown_active = bool(cooldown.get("active", cooldown_until > now.timestamp()) and cooldown_until > now.timestamp())
    cooldown_kind = str(cooldown.get("kind") or "none").strip().lower()
    mode = str(session_payload.get("mode") or ingress.get("loop_state") or "unknown").strip().lower()
    error_rate = _as_float(ingress.get("iter_error_rate"), 0.0)
    request_count = _as_int(ingress.get("iter_total_requests"), 0)
    fallback_active = mode in {
        "forex_session_context_only",
        "live_proxy_market_hours",
        "forex_weekend_closed",
    }
    provider_available = bool(provider.get("available", not cooldown_active))
    evidence_fresh = bool(
        (not session_source["present"] or session_source["fresh"])
        and (not ingress_source["present"] or ingress_source["fresh"])
    )

    if request_count > 0 and error_rate >= 0.8 and not fallback_active:
        status = "blocked"
        cause = "fx_realtime_ingestion_error_rate_high"
        managed_fallback = False
        impact = "fx_observations_failed"
        action = "fx-provider-fallback"
    elif cooldown_active and fallback_active:
        status = "watch"
        cause = f"twelve_data_{cooldown_kind}_cooldown"
        managed_fallback = True
        impact = "fx_realtime_deferred"
        action = "renew-twelve-data-key" if cooldown_kind == "auth" else "none"
    elif cooldown_active:
        status = "blocked"
        cause = f"twelve_data_{cooldown_kind}_cooldown_without_fallback"
        managed_fallback = False
        impact = "fx_observations_failed"
        action = "fx-provider-fallback"
    elif not evidence_fresh:
        status = "degraded"
        cause = "stale_fx_provider_evidence"
        managed_fallback = False
        impact = "fx_provider_unverified"
        action = "fx-provider-refresh"
    else:
        status = "ready"
        cause = "none"
        managed_fallback = False
        impact = "none"
        action = "none"

    ages = [
        _as_float(source.get("age_seconds"), 0.0)
        for source in (session_source, ingress_source, guard_source)
        if source.get("age_seconds") is not None
    ]
    return {
        "status": status,
        "provider": "twelve_data",
        "enabled": bool(provider.get("enabled", True)),
        "available": provider_available,
        "mode": mode,
        "fallback_active": fallback_active,
        "managed_fallback": managed_fallback,
        "cooldown_kind": cooldown_kind,
        "cooldown_remaining_seconds": max(cooldown_until - now.timestamp(), 0.0) if cooldown_active else 0.0,
        "credential_action_required": bool(cooldown.get("credential_action_required", cooldown_kind == "auth")),
        "iter_error_rate": round(error_rate, 4),
        "iter_request_count": request_count,
        "cause": cause,
        "impact": impact,
        "artifact_age_seconds": max(ages, default=0.0),
        "action": action,
    }


def _storage_row(sources: dict[str, dict[str, Any]], system: dict[str, Any]) -> dict[str, Any]:
    source = sources["storage"]
    storage = source["payload"]
    backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    effective = backpressure.get("raw_live") if isinstance(backpressure.get("raw_live"), dict) else {}
    if not effective:
        candidate = storage.get("effective_raw_live")
        effective = candidate if isinstance(candidate, dict) else {}
    total = _as_int(backpressure.get("total_pending_lines"))
    core = _as_int(backpressure.get("core_pending_lines"))
    effective_total = _as_int(effective.get("total_pending_lines"), total)
    effective_core = _as_int(effective.get("core_pending_lines"), core)
    threshold = _as_int(storage.get("pending_lines_threshold"), 15000)
    pressure = _as_float(storage.get("pressure_index"))
    strict_status = "blocked" if pressure >= 1.0 or total > threshold else ("watch" if pressure >= 0.5 else "ready")
    storage_detail = storage.get("storage") if isinstance(storage.get("storage"), dict) else {}
    backlog_truth = storage.get("backlog_truth") if isinstance(storage.get("backlog_truth"), dict) else {}
    raw_truth = backlog_truth.get("raw_live") if isinstance(backlog_truth.get("raw_live"), dict) else {}
    overlay_truth = backlog_truth.get("sql_overlay") if isinstance(backlog_truth.get("sql_overlay"), dict) else {}
    stale_locator = storage.get("stale_pending_locator") if isinstance(storage.get("stale_pending_locator"), dict) else {}
    if not stale_locator:
        candidate = backlog_truth.get("stale_pending_locator")
        stale_locator = candidate if isinstance(candidate, dict) else {}
    oldest_sources = stale_locator.get("oldest_sources") if isinstance(stale_locator.get("oldest_sources"), list) else []
    oldest_source = oldest_sources[0] if oldest_sources and isinstance(oldest_sources[0], dict) else {}
    leader_path = str(oldest_source.get("source_rel") or "none")
    leader_parts = Path(leader_path).parts
    leader = "/".join(leader_parts[-2:]) if len(leader_parts) >= 2 else leader_path
    stale_source_count = _as_int(stale_locator.get("stale_source_count"), len(oldest_sources))
    drain = str(storage_detail.get("backlog_drain_status") or "idle")
    paper_status = str(system.get("paper_status") or "unknown")
    control_status = _status_text(storage)
    severity = str(storage.get("severity") or "unknown").lower()
    managed_bounded_backlog = bool(
        strict_status == "watch"
        and paper_status == "ready"
        and control_status in {"ready", "advisory"}
        and severity in {"stable", "normal", "clear", "ready"}
        and total <= threshold
        and effective_total <= threshold
        and stale_source_count == 0
    )
    if _status_text(storage) in {"blocked", "critical", "failed"}:
        status = "blocked"
    elif managed_bounded_backlog:
        status = "watch"
    elif drain in {"drain_active", "running", "active"} and (strict_status != "ready" or paper_status != "ready"):
        status = "recovering"
    elif strict_status == "watch":
        status = "watch"
    else:
        status = _status_text(storage)
    if not source["fresh"]:
        status = "degraded"
    cause = "none"
    if stale_source_count > 0 and pressure >= 1.0:
        cause = "stale_sql_overlay"
    elif total > threshold:
        cause = "pending_above_threshold"
    elif pressure >= 1.0:
        cause = "pressure_index_high"
    elif pressure >= 0.5:
        cause = "bounded_backlog"
    return {
        "status": status,
        "control_status": control_status,
        "severity": severity,
        "strict_status": strict_status,
        "managed_bounded_backlog": managed_bounded_backlog,
        "paper_status": paper_status,
        "truth_mode": str(backlog_truth.get("authoritative_mode") or "direct_backpressure"),
        "raw_grade": str(raw_truth.get("grade") or "unknown"),
        "overlay_grade": str(overlay_truth.get("grade") or "unknown"),
        "pressure_index": round(pressure, 3),
        "core_pending_lines": core,
        "total_pending_lines": total,
        "effective_core_pending_lines": effective_core,
        "effective_total_pending_lines": effective_total,
        "oldest_pending_age_seconds": _as_float(backpressure.get("oldest_pending_age_seconds")),
        "stale_source_count": stale_source_count,
        "oldest_source": leader,
        "oldest_source_shard": str(oldest_source.get("shard") or "none"),
        "oldest_source_pending_lines": _as_int(oldest_source.get("pending_lines")),
        "drain_status": drain,
        "recovery_state": str(storage.get("recovery_state") or ("draining" if drain == "drain_active" else "idle")),
        "estimated_drain_minutes": _as_float(storage.get("estimated_total_drain_minutes"), _as_float(backpressure.get("estimated_total_drain_minutes"))),
        "cause": cause,
        "impact": "paper_blocked" if paper_status != "ready" else "strict_live_gate_only" if strict_status == "blocked" else "none",
        "artifact_age_seconds": source.get("age_seconds"),
        "action": "storage-backpressure-autopilot" if cause != "none" else "none",
    }


def _throttle_row(sources: dict[str, dict[str, Any]], storage: dict[str, Any]) -> dict[str, Any]:
    source = sources["throttle"]
    throttle = source["payload"]
    soft_cap = throttle.get("soft_cap_advisory_reclassification") if isinstance(throttle.get("soft_cap_advisory_reclassification"), dict) else {}
    measurements = soft_cap.get("measurements") if isinstance(soft_cap.get("measurements"), dict) else {}
    governor = throttle.get("runtime_saturation_governor_v2") if isinstance(throttle.get("runtime_saturation_governor_v2"), dict) else {}
    paper_policy = governor.get("paper_live_data_policy") if isinstance(governor.get("paper_live_data_policy"), dict) else {}
    runtime_paper_policy = throttle.get("paper_execution_policy") if isinstance(throttle.get("paper_execution_policy"), dict) else {}
    host_attribution = throttle.get("host_pressure_attribution") if isinstance(throttle.get("host_pressure_attribution"), dict) else {}
    status = _status_text(throttle)
    if not source["fresh"]:
        status = "degraded"
    compute = str(throttle.get("compute_pressure_level") or "unknown")
    memory = str(throttle.get("memory_pressure_level") or "unknown")
    paper_paused = bool(
        measurements.get("paper_execution_paused")
        or paper_policy.get("paper_execution_consumer_paused")
        or runtime_paper_policy.get("pause_paper_execution")
    )
    paper_allowed = bool(
        runtime_paper_policy.get(
            "paper_execution_allowed",
            paper_policy.get("paper_execution_allowed", measurements.get("paper_execution_allowed", not paper_paused)),
        )
    )
    policy_reason = str(soft_cap.get("reason") or runtime_paper_policy.get("reason") or "none")
    compute_hot = compute not in {"normal", "clear", "green"}
    memory_clear = memory in {"normal", "clear", "green"}
    storage_paper_safe = bool(
        storage.get("paper_status") == "ready"
        and storage.get("status") not in {"blocked", "degraded", "recovering"}
        and storage.get("impact") != "paper_blocked"
    )
    managed_compute_pressure = bool(
        source["fresh"]
        and status in {"ready", "advisory"}
        and compute_hot
        and memory_clear
        and paper_allowed
        and not paper_paused
        and storage_paper_safe
        and bool(soft_cap.get("active"))
        and policy_reason != "none"
    )
    if memory not in {"normal", "clear", "green"}:
        cause = "memory_pressure"
    elif storage.get("managed_bounded_backlog"):
        cause = "bounded_storage_watch"
    elif str(storage.get("strict_status")) in {"blocked", "watch"}:
        cause = "storage_backlog" if paper_paused else "strict_storage_backlog"
    elif paper_paused:
        cause = str(
            paper_policy.get("paper_execution_pause_reason")
            or runtime_paper_policy.get("reason")
            or "paper_gate"
        )
    elif managed_compute_pressure:
        cause = "managed_compute_pressure"
    elif compute_hot:
        cause = "compute_pressure"
    elif status not in {"ready", "advisory"}:
        cause = str(soft_cap.get("reason") or "runtime_policy_guard")
    else:
        cause = "none"
    recovery = (
        "storage_drain_active"
        if storage.get("drain_status") == "drain_active"
        and cause in {"bounded_storage_watch", "storage_backlog", "strict_storage_backlog"}
        else "none"
    )
    collector_policy = governor.get("collector_policy") if isinstance(governor.get("collector_policy"), dict) else {}
    managed_advisory = bool(
        status in {"ready", "advisory"}
        and cause in {"none", "managed_compute_pressure"}
        and source["fresh"]
    )
    managed_control = bool(
        managed_advisory
        or (storage.get("managed_bounded_backlog") and status in {"ready", "advisory"} and not paper_paused)
    )
    pressure_owner = str(host_attribution.get("dominant_bucket") or "unknown")
    if host_attribution.get("external_pressure_dominant"):
        pressure_owner = "external"
    return {
        "status": status,
        "profile": str(throttle.get("throttle_profile") or "unknown"),
        "compute": compute,
        "memory": memory,
        "host_saturation_score": round(_as_float(throttle.get("host_saturation_score")), 2),
        "cause": cause,
        "paper_state": "paused" if paper_paused else "allowed",
        "paper_allowed": paper_allowed,
        "impact": "paper_paused" if paper_paused else "none" if managed_control else "paper_runtime_guard",
        "pressure_owner": pressure_owner,
        "research_low_priority": bool(host_attribution.get("research_hot_low_priority")),
        "paper_low_priority": bool(host_attribution.get("paper_hot_low_priority")),
        "writer_hot": bool(host_attribution.get("storage_writer_hot")),
        "collection_mode": str(collector_policy.get("mode") or "unknown"),
        "recovery": recovery,
        "managed_advisory": managed_advisory,
        "managed_control": managed_control,
        "policy_reason": (
            "bounded_backlog_under_paper_threshold"
            if storage.get("managed_bounded_backlog") and status == "ready"
            else policy_reason
            if status == "advisory" or managed_compute_pressure
            else "none"
        ),
        "artifact_age_seconds": source.get("age_seconds"),
        "action": (
            "none"
            if managed_control
            else "storage-backpressure-autopilot"
            if cause in {"bounded_storage_watch", "storage_backlog", "strict_storage_backlog"}
            else "runtime-throttle"
        ),
    }


def _soak_row(sources: dict[str, dict[str, Any]], system: dict[str, Any], storage: dict[str, Any]) -> dict[str, Any]:
    source = sources["unattended_soak"]
    soak = source["payload"]
    continuous_source = sources.get("continuous_soak", {})
    continuous_payload = continuous_source.get("payload")
    continuous = continuous_payload if isinstance(continuous_payload, dict) else {}
    historical_payload = continuous.get("historical_soak_evidence")
    historical = historical_payload if isinstance(historical_payload, dict) else {}
    declared_safe = bool(soak.get("safe_to_leave_unattended"))
    warnings = soak.get("warnings") if isinstance(soak.get("warnings"), list) else []
    paper_ready = system.get("paper_status") == "ready"
    strict_storage_ready = storage.get("strict_status") == "ready"
    managed_storage_watch = bool(storage.get("managed_bounded_backlog"))
    storage_acceptable = bool(strict_storage_ready or managed_storage_watch)
    effective_safe = bool(
        declared_safe
        and paper_ready
        and storage_acceptable
        and sources["health_fast"]["fresh"]
        and source["fresh"]
    )
    live_state = str(system.get("live_status") or "unknown")
    live_locked = live_state in {"blocked", "blocked_read_only", "locked", "disabled"}
    if effective_safe:
        status = "ready"
        cause = "none"
    elif storage.get("status") == "recovering":
        status = "recovering"
        cause = str(storage.get("cause") or "storage_recovery")
    elif not source["fresh"]:
        status = "degraded"
        cause = "stale_unattended_soak_artifact"
    else:
        status = "blocked"
        cause = str(system.get("cause") or "paper_not_ready")
    return {
        "status": status,
        "grade": str(soak.get("overall_grade") or "unknown"),
        "score": _as_float(soak.get("overall_score")),
        "declared_safe": declared_safe,
        "effective_safe": effective_safe,
        "paper_status": str(system.get("paper_status") or "unknown"),
        "strict_storage_ready": strict_storage_ready,
        "managed_storage_watch": managed_storage_watch,
        "storage_watch": str(storage.get("cause") or "none") if managed_storage_watch else "none",
        "live_locked": live_locked,
        "warning_count": len(warnings),
        "warning": _first(warnings),
        "cause": cause,
        "artifact_age_seconds": source.get("age_seconds"),
        "continuous_soak_fresh": bool(continuous_source.get("fresh", False)),
        "main_soak_elapsed_hours": _as_float(
            continuous.get("main_soak_elapsed_hours"),
            _as_float(historical.get("historical_segmented_wall_clock_hours")),
        ),
        "main_soak_elapsed_days": _as_float(
            continuous.get("main_soak_elapsed_days"),
            _as_float(historical.get("historical_segmented_wall_clock_days")),
        ),
        "main_soak_progress_percent": _as_float(continuous.get("main_soak_progress_percent")),
        "main_soak_includes_pre_reset_time": bool(continuous.get("main_soak_includes_pre_reset_time", False)),
        "main_soak_count_is_promotion_credit": bool(continuous.get("main_soak_count_is_promotion_credit", False)),
        "clean_window_elapsed_hours": _as_float(continuous.get("clean_window_elapsed_hours")),
        "observed_window_elapsed_hours": _as_float(continuous.get("observed_window_elapsed_hours")),
        "historical_segmented_hours": _as_float(historical.get("historical_segmented_wall_clock_hours")),
        "historical_segmented_days": _as_float(historical.get("historical_segmented_wall_clock_days")),
        "historical_segment_count": _as_int(historical.get("segment_count")),
        "historical_counts_toward_clean_720": bool(historical.get("counts_toward_current_clean_720_hours", False)),
        "action": (
            "storage-backpressure-autopilot"
            if managed_storage_watch
            else "none"
            if cause == "none"
            else "storage-backpressure-autopilot"
            if cause.startswith(("pending_", "storage_", "bounded_", "pressure_"))
            else "unattended-soak-readiness"
        ),
    }


def _paper_debt_recovery_row(source: dict[str, Any]) -> dict[str, Any]:
    payload = source.get("payload") if isinstance(source.get("payload"), dict) else {}
    contract = (
        payload.get("paper_debt_recovery_contract")
        if isinstance(payload.get("paper_debt_recovery_contract"), dict)
        else {}
    )
    attribution = contract.get("candidate_attribution") if isinstance(contract.get("candidate_attribution"), dict) else {}
    velocity = contract.get("recovery_velocity") if isinstance(contract.get("recovery_velocity"), dict) else {}
    risk = contract.get("risk_budget") if isinstance(contract.get("risk_budget"), dict) else {}
    if not source.get("present") or not contract:
        status = "missing"
    elif not source.get("fresh"):
        status = "stale"
    elif contract.get("live_promotion_ready", False):
        status = "ready"
    elif str(contract.get("state") or "") in {"evidence_unavailable", "paused_drawdown", "debt_worsening"}:
        status = "blocked"
    else:
        status = "recovering"
    return {
        "status": status,
        "state": str(contract.get("state") or "unknown"),
        "baseline_debt_amount": _as_float(contract.get("baseline_debt_amount")),
        "remaining_debt_amount": _as_float(contract.get("remaining_debt_amount")),
        "recovery_progress_norm": _as_float(contract.get("recovery_progress_norm")),
        "candidate_id": str(attribution.get("candidate_id") or "none"),
        "candidate_samples": _as_int(attribution.get("sample_count")),
        "candidate_days": _as_int(attribution.get("observed_days")),
        "candidate_attributed_pnl": _as_float(attribution.get("total_candidate_attributed_pnl")),
        "actual_daily_recovery": _as_float(velocity.get("actual_daily_net_improvement")),
        "entry_size_cap_norm": _as_float(
            (contract.get("runtime_enforcement") or {}).get("recovery_entry_size_multiplier_norm")
            if isinstance(contract.get("runtime_enforcement"), dict)
            else 0.0
        ),
        "new_entries_paused": bool(risk.get("new_entries_paused", True)),
        "live_proof": bool(contract.get("live_promotion_ready", False)),
        "live_execution": False,
        "artifact_age_seconds": source.get("age_seconds"),
        "action": "none" if status in {"ready", "recovering"} else "paper-profitability-control",
    }


def _profitability_assessment_row(source: dict[str, Any]) -> dict[str, Any]:
    payload = source.get("payload") if isinstance(source.get("payload"), dict) else {}
    binding = payload.get("candidate_binding") if isinstance(payload.get("candidate_binding"), dict) else {}
    grades = payload.get("grades") if isinstance(payload.get("grades"), dict) else {}
    measurement = payload.get("measurement") if isinstance(payload.get("measurement"), dict) else {}
    next_action = payload.get("next_safe_action") if isinstance(payload.get("next_safe_action"), dict) else {}
    if not source.get("present"):
        status = "missing"
    elif not source.get("fresh"):
        status = "stale"
    elif not binding.get("identity_consistent", False):
        status = "blocked"
    elif grades.get("economic_evidence_ready", False):
        status = "ready"
    else:
        status = "collecting"
    return {
        "status": status,
        "assessment_status": str(
            payload.get("assessment_status")
            or ("blocked" if not binding.get("identity_consistent", False) else "ready")
        ),
        "candidate_id": str(binding.get("candidate_id") or measurement.get("candidate_id") or "none"),
        "candidate_consistent": bool(binding.get("identity_consistent", False)),
        "implementation_grade": str(grades.get("implementation_grade") or "unknown"),
        "implementation_score": _as_float(grades.get("implementation_score")),
        "economic_grade": str(grades.get("economic_evidence_grade") or "unknown"),
        "economic_score": _as_float(grades.get("economic_evidence_score")),
        "economic_ready": bool(grades.get("economic_evidence_ready", False)),
        "candidate_samples": _as_int(measurement.get("candidate_post_cost_sample_count")),
        "candidate_minimum_samples": _as_int(measurement.get("candidate_post_cost_minimum_samples"), 30),
        "candidate_pnl": _as_float(measurement.get("candidate_post_cost_pnl")),
        "historical_book_pnl": _as_float(measurement.get("historical_active_book_net_pnl")),
        "historical_grades_candidate": bool(
            measurement.get("historical_active_book_candidate_grade_eligible", False)
        ),
        "ready_lanes": _as_int(grades.get("evidence_ready_lanes")),
        "lane_count": _as_int(grades.get("evidence_lane_count"), 8),
        "next_need": str(next_action.get("blocker") or "none"),
        "live_execution": False,
        "artifact_age_seconds": source.get("age_seconds"),
        "action": "none" if status == "ready" else "profitability-self-assessment",
    }


def _sleeve_strategy_scaling_row(source: dict[str, Any]) -> dict[str, Any]:
    payload = source.get("payload") if isinstance(source.get("payload"), dict) else {}
    contract = (
        payload.get("sleeve_strategy_profitability_scaling_contract")
        if isinstance(payload.get("sleeve_strategy_profitability_scaling_contract"), dict)
        else {}
    )
    binding = contract.get("candidate_binding") if isinstance(contract.get("candidate_binding"), dict) else {}
    tier_counts = contract.get("tier_counts") if isinstance(contract.get("tier_counts"), dict) else {}
    validated_count = sum(
        _as_int(tier_counts.get(tier))
        for tier in ("validated_baseline", "scale_tier_1", "scale_tier_2")
    )
    if not source.get("present") or not contract:
        status = "missing"
    elif not source.get("fresh"):
        status = "stale"
    elif not contract.get("source_ready", False) or not binding.get("candidate_binding_valid", False):
        status = "blocked"
    elif not contract.get("keep_sells_and_reduce_only_paths_open", False):
        status = "blocked"
    elif contract.get("scale_up_ready", False):
        status = "ready"
    elif validated_count > 0:
        status = "guarded"
    else:
        status = "collecting"
    return {
        "status": status,
        "mode": str(contract.get("mode") or "missing"),
        "candidate_id": str(binding.get("candidate_id") or "none"),
        "candidate_binding_valid": bool(binding.get("candidate_binding_valid", False)),
        "source_ready": bool(contract.get("source_ready", False)),
        "profile_control_count": _as_int(contract.get("profile_control_count")),
        "strategy_control_count": _as_int(contract.get("strategy_control_count")),
        "blocked_control_count": _as_int(contract.get("blocked_control_count")),
        "probationary_control_count": _as_int(contract.get("probationary_control_count")),
        "validated_control_count": validated_count,
        "above_baseline_ready_count": _as_int(contract.get("above_baseline_ready_count")),
        "global_entry_size_cap_norm": _as_float(contract.get("global_entry_size_cap_norm")),
        "maximum_entry_size_multiplier_norm": _as_float(
            contract.get("maximum_above_baseline_entry_size_multiplier_norm")
        ),
        "entry_only": bool(contract.get("entry_only", False)),
        "exit_paths_open": bool(contract.get("keep_sells_and_reduce_only_paths_open", False)),
        "live_execution": False,
        "tier_counts": tier_counts,
        "artifact_age_seconds": source.get("age_seconds"),
        "action": "none" if status in {"ready", "guarded", "collecting"} else "paper-profitability-control",
    }


def _production_excellence_row(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = sources["production_excellence"]
    payload = source["payload"]
    if not source["present"]:
        status = "missing"
    elif not source["fresh"]:
        status = "stale"
    elif payload.get("ten_out_of_ten_ready", False):
        status = "ready"
    else:
        status = "evidence_pending"
    blocked = payload.get("blocked_pillars") if isinstance(payload.get("blocked_pillars"), list) else []
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    return {
        "status": status,
        "grade": str(payload.get("overall_grade") or "unknown"),
        "score": _as_float(payload.get("overall_score")),
        "ready_pillars": _as_int(payload.get("ready_pillar_count")),
        "pillar_count": _as_int(payload.get("pillar_count"), 10),
        "candidate_id": str(candidate.get("candidate_id") or "none"),
        "candidate_drift": bool(candidate.get("candidate_drift", False)),
        "blocked_pillars": blocked,
        "live_consideration": bool(payload.get("live_money_consideration_ready", False)),
        "live_locked": bool(payload.get("live_orders_must_remain_disabled", True)),
        "paper_impact": "none",
        "artifact_age_seconds": source.get("age_seconds"),
        "action": "none" if status == "ready" else "production-excellence",
    }


def _strategy_specialization_row(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = sources["strategy_specialization"]
    payload = source["payload"]
    coverage = payload.get("contract_coverage") if isinstance(payload.get("contract_coverage"), dict) else {}
    binding = payload.get("candidate_binding") if isinstance(payload.get("candidate_binding"), dict) else {}
    lifecycles = payload.get("lifecycle_counts") if isinstance(payload.get("lifecycle_counts"), dict) else {}
    library = payload.get("strategy_library") if isinstance(payload.get("strategy_library"), dict) else {}
    families = payload.get("strategy_families") if isinstance(payload.get("strategy_families"), dict) else {}
    quality = payload.get("quality_summary") if isinstance(payload.get("quality_summary"), dict) else {}
    regime = payload.get("current_regime") if isinstance(payload.get("current_regime"), dict) else {}
    if not source["present"]:
        status = "missing"
    elif not source["fresh"]:
        status = "stale"
    elif payload.get("ok", False):
        status = "ready"
    else:
        status = "blocked"
    return {
        "status": status,
        "grade": str(coverage.get("grade") or "unknown"),
        "sleeves": _as_int(coverage.get("sleeve_count")),
        "strategies": _as_int(coverage.get("strategy_count")),
        "complete": _as_int(coverage.get("complete_contract_count")),
        "authority_violations": _as_int(coverage.get("authority_violation_count")),
        "candidate_id": str(binding.get("candidate_id") or "none"),
        "candidate_bound": bool(binding.get("bound", False)),
        "parked": _as_int(lifecycles.get("parked_candidate")),
        "probation": _as_int(lifecycles.get("probation")),
        "validated": _as_int(lifecycles.get("validated_candidate")),
        "control_only": _as_int(lifecycles.get("control_only")),
        "library_strategies": _as_int(library.get("strategy_count")),
        "library_target": _as_int(library.get("target_total_strategies"), 12000),
        "hot_strategies": _as_int(library.get("hot_strategy_count")),
        "cold_strategies": _as_int(library.get("cold_strategy_count")),
        "canonical_families": _as_int(families.get("canonical_record_count")),
        "native_hot_families": _as_int(families.get("native_hot_family_count")),
        "cold_parent_families": _as_int(families.get("cold_parent_family_count")),
        "family_lineage_covered": _as_int(families.get("lineage_covered_strategy_count")),
        "family_identity_changes": _as_int(families.get("runtime_identity_change_count")),
        "validated_good": _as_int(quality.get("validated_good_count")),
        "promising": _as_int(quality.get("promising_unconfirmed_count")),
        "weak": _as_int(quality.get("weak_count")),
        "retirement_candidates": _as_int(quality.get("retirement_candidate_count")),
        "current_regime": str(regime.get("current_regime") or "unknown"),
        "regime_activation_ready": bool(regime.get("activation_ready", False)),
        "paper_impact": "none",
        "live_authority": False,
        "artifact_age_seconds": source.get("age_seconds"),
        "action": "none" if status == "ready" else "sleeve-strategy-specialization",
    }


def _institutional_capability_row(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = sources["institutional_capabilities"]
    payload = source["payload"]
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    provider = payload.get("provider_policy") if isinstance(payload.get("provider_policy"), dict) else {}
    candidate = payload.get("candidate_binding") if isinstance(payload.get("candidate_binding"), dict) else {}
    entitlements = payload.get("conditional_external_entitlements") if isinstance(payload.get("conditional_external_entitlements"), list) else []
    if not source["present"]:
        status = "missing"
    elif not source["fresh"]:
        status = "stale"
    else:
        status = str(payload.get("overall_status") or "unknown")
    return {
        "status": status,
        "implementation_ready": _as_int(summary.get("implementation_ready_count")),
        "paper_ready": _as_int(summary.get("paper_soak_ready_count")),
        "candidate_evidence_ready": _as_int(summary.get("candidate_evidence_ready_count")),
        "live_ready": _as_int(summary.get("live_promotion_ready_count")),
        "pillar_count": _as_int(summary.get("pillar_count"), 6),
        "verified_source_bundles": _as_int(summary.get("verified_source_bundle_count")),
        "provider_target": provider.get("target_range") or [15, 30],
        "ten_thousand_sources_required": bool(provider.get("ten_thousand_sources_required", False)),
        "conditional_entitlements": len(entitlements),
        "candidate_id": str(candidate.get("candidate_id") or "none"),
        "candidate_bound": bool(candidate.get("bound", False)),
        "paper_soak_ready": bool(payload.get("paper_soak_ready", False)),
        "live_promotion_ready": bool(payload.get("live_promotion_ready", False)),
        "local_refresh_actions": _as_int(summary.get("local_refresh_action_count")),
        "external_actions": len(payload.get("external_or_human_actions") or []),
        "paper_impact": "none" if payload.get("paper_soak_ready", False) else "paper_attention",
        "live_execution": False,
        "artifact_age_seconds": source.get("age_seconds"),
        "action": "none" if status in {"ready", "ready_with_evidence_debt"} else "institutional-capability-control",
    }


def _authoritative_systems_row(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = sources["authoritative_systems"]
    payload = source["payload"]
    if not source["present"]:
        status = "missing"
    elif not source["fresh"]:
        status = "stale"
    elif payload.get("ok", False):
        status = "ready"
    else:
        status = "blocked"
    return {
        "status": status,
        "grade": str(payload.get("grade") or "unknown"),
        "grade_scope": str(payload.get("grade_scope") or "local structural implementation only"),
        "references": _as_int(payload.get("reference_count")),
        "ready_controls": _as_int(payload.get("ready_control_count")),
        "control_count": _as_int(payload.get("control_count"), 8),
        "live_authority": bool(payload.get("live_execution_authority", False)),
        "soak_reset": bool(
            (payload.get("soak_acceptance") or {}).get("reset_soak_clock", False)
            if isinstance(payload.get("soak_acceptance"), dict)
            else False
        ),
        "paper_impact": "none",
        "artifact_age_seconds": source.get("age_seconds"),
        "action": "none" if status == "ready" else "authoritative-systems",
    }


def _paper_live_equivalence_row(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = sources["paper_live_equivalence"]
    payload = source["payload"]
    if not source["present"]:
        status = "missing"
    elif not source["fresh"]:
        status = "stale"
    else:
        status = str(payload.get("status") or "unknown")
    return {
        "status": status,
        "structural_ready": bool(payload.get("structural_ready", False)),
        "empirical_ready": bool(payload.get("empirical_ready", False)),
        "paper_count": _as_int(payload.get("paper_count")),
        "live_count": _as_int(payload.get("live_count")),
        "paired_count": _as_int(payload.get("paired_count")),
        "mismatch_count": _as_int(payload.get("mismatch_count")),
        "missing_live_count": _as_int(payload.get("missing_live_count")),
        "unpaired_paper_count": _as_int(payload.get("unpaired_paper_count")),
        "missing_paper_count": _as_int(payload.get("missing_paper_count")),
        "paper_impact": "none",
        "live_evidence": "ready" if payload.get("empirical_ready", False) else "pending",
        "live_authority": False,
        "artifact_age_seconds": source.get("age_seconds"),
        "action": "none" if payload.get("empirical_ready", False) else "collect_live_shadow_pairs",
    }


def _capability_materialization_row(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = sources["capability_materialization"]
    payload = source["payload"]
    capabilities = [row for row in payload.get("capabilities", []) if isinstance(row, dict)]
    ready_rows = [
        row
        for row in capabilities
        if row.get("usable") is True
        and str(row.get("proof_semantics") or "") == "direct"
        and bool(str(row.get("proof_receipt_sha256") or ""))
    ]
    ready_ids = {str(row.get("capability_id") or "") for row in ready_rows}
    required_ids = {
        "trading_calendars",
        "market_session_state",
        "derivatives_contract_master",
        "stress_scenarios",
    }
    authority = payload.get("authority_contract") if isinstance(payload.get("authority_contract"), dict) else {}
    calendar = payload.get("calendar_materialization") if isinstance(payload.get("calendar_materialization"), dict) else {}
    derivatives = payload.get("derivative_contract_materialization") if isinstance(payload.get("derivative_contract_materialization"), dict) else {}
    stress = payload.get("stress_scenario_materialization") if isinstance(payload.get("stress_scenario_materialization"), dict) else {}
    if not source["present"]:
        status = "missing"
        cause = "capability_materialization_missing"
    elif not source["fresh"]:
        status = "stale"
        cause = "capability_materialization_stale"
    elif payload.get("live_promotion_ready") is not True or not required_ids.issubset(ready_ids):
        status = "blocked"
        cause = "capability_materialization_proof_incomplete"
    elif any(bool(value) for value in authority.values()):
        status = "blocked"
        cause = "capability_materialization_authority_contract_unsafe"
    else:
        status = "ready"
        cause = "none"
    return {
        "status": status,
        "grade": "A+" if status == "ready" else "F",
        "direct_proofs": len(required_ids & ready_ids),
        "required_proofs": len(required_ids),
        "contracts": _as_int(derivatives.get("contract_count")),
        "stress_scenarios": _as_int(stress.get("scenario_count")),
        "calendar_library": str(calendar.get("library_version") or "unknown"),
        "live_promotion_ready": bool(payload.get("live_promotion_ready", False)),
        "cause": cause,
        "impact": "none" if status == "ready" else "paper_soak_blocked",
        "artifact_age_seconds": source.get("age_seconds"),
        "action": "none" if status == "ready" else "capability-materialization",
    }


def _collector_capability_row(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = sources["collector_capabilities"]
    payload = source["payload"]
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    coverage = payload.get("coverage_debt") if isinstance(payload.get("coverage_debt"), dict) else {}
    routing = (
        payload.get("ingestion_routing_contract")
        if isinstance(payload.get("ingestion_routing_contract"), dict)
        else {}
    )
    transport = (
        routing.get("transport_contract")
        if isinstance(routing.get("transport_contract"), dict)
        else {}
    )
    blockers = list(payload.get("structural_blockers") or []) + list(payload.get("paper_soak_blockers") or [])
    if not source["present"]:
        status = "missing"
        cause = "collector_capability_control_missing"
    elif not source["fresh"]:
        status = "stale"
        cause = "collector_capability_control_stale"
    elif payload.get("ok") is not True:
        status = "blocked"
        cause = _first(blockers, "collector_capability_control_structurally_blocked")
    elif payload.get("paper_soak_ready") is not True:
        status = "blocked"
        cause = _first(blockers, "collector_capability_paper_soak_not_ready")
    else:
        status = "ready"
        cause = "none"
    return {
        "status": status,
        "planes": _as_int(summary.get("plane_count")),
        "capabilities": _as_int(summary.get("capability_count")),
        "bots": _as_int(summary.get("bot_binding_count")),
        "assignments": _as_int(summary.get("assignment_count")),
        "profiles": _as_int(summary.get("subscription_profile_count")),
        "ingestion_route_profiles": _as_int(
            summary.get("ingestion_route_profile_count")
        ),
        "routing_policy": str(routing.get("policy_id") or "unknown"),
        "decision_stage": str(routing.get("decision_stage") or "unknown"),
        "decision_families": _as_int(routing.get("decision_family_count")),
        "runtime_routes": _as_int(routing.get("runtime_route_count")),
        "runtime_paper_ready_routes": _as_int(
            routing.get("runtime_paper_ready_route_count")
        ),
        "runtime_live_ready_routes": _as_int(
            routing.get("runtime_live_ready_route_count")
        ),
        "paper_ready_profile_routes": _as_int(
            routing.get("paper_ready_profile_route_count")
        ),
        "live_ready_profile_routes": _as_int(
            routing.get("live_ready_profile_route_count")
        ),
        "average_route_quality": _as_float(
            routing.get("average_profile_route_quality")
        ),
        "independent_redundancy_ratio": _as_float(
            summary.get("required_capability_independent_redundancy_ratio")
        ),
        "transport_contract_complete": bool(
            transport and all(bool(value) for value in transport.values())
        ),
        "route_receipt": str(
            routing.get("routing_artifact_receipt_sha256") or ""
        )[:12],
        "collector_mapping_complete": bool(
            (payload.get("current_collector_mapping") or {}).get("complete", False)
        ),
        "coverage_gaps": _as_int(coverage.get("gap_count")),
        "candidate_blocking_gaps": _as_int(coverage.get("candidate_blocking_gap_count")),
        "optional_gaps": _as_int(coverage.get("optional_gap_count")),
        "coverage_debt_scope": "candidate_required_blocking_optional_advisory",
        "required_usable_ratio": _as_float(summary.get("required_capability_usable_ratio")),
        "required_redundancy_ratio": _as_float(
            summary.get("required_capability_redundancy_ratio")
        ),
        "full_catalog_coverage_ready": bool(summary.get("full_catalog_coverage_ready", False)),
        "paper_soak_ready": bool(payload.get("paper_soak_ready", False)),
        "live_promotion_ready": bool(payload.get("live_promotion_ready", False)),
        "cause": cause,
        "impact": "none" if status == "ready" else "paper_soak_blocked",
        "artifact_age_seconds": source.get("age_seconds"),
        "action": "none" if status == "ready" else "collector-capability-control",
    }


def _effective_operational_state(name: str, row: dict[str, Any]) -> tuple[str, bool]:
    status = str(row.get("status") or "unknown").strip().lower()
    if name == "throttle" and bool(row.get("managed_control")):
        no_paper_impact = bool(
            row.get("paper_allowed")
            and row.get("paper_state") == "allowed"
            and row.get("impact") == "none"
            and row.get("action") == "none"
        )
        if no_paper_impact:
            return "ready", status != "ready"
    if name == "storage" and bool(row.get("managed_bounded_backlog")):
        no_paper_impact = bool(
            row.get("paper_status") == "ready"
            and row.get("impact") == "none"
            and row.get("status") == "watch"
        )
        if no_paper_impact:
            return "ready", True
    if name == "fx_provider" and bool(row.get("managed_fallback")):
        fallback_safe = bool(
            row.get("fallback_active")
            and row.get("status") == "watch"
            and row.get("impact") == "fx_realtime_deferred"
        )
        if fallback_safe:
            return "ready", True
    return status, False


def _rollup_status(states: list[str]) -> str:
    normalized = {str(state or "unknown").strip().lower() for state in states}
    if normalized & {"blocked", "failed", "failure", "critical", "error"}:
        return "blocked"
    if normalized & {"degraded", "needs_work", "stale", "missing", "unknown"}:
        return "degraded"
    if normalized & {"advisory", "recovering", "watch", "warn", "warning"}:
        return "advisory"
    return "ready"


def _operator_summary(
    *,
    visibility_status: str,
    operational_status: str,
    guarded_paper_status: str,
    operational_rows: dict[str, dict[str, Any]],
    effective_states: dict[str, str],
    sources: dict[str, dict[str, Any]],
    missing: list[str],
    stale: list[str],
    contradictions: list[str],
) -> dict[str, Any]:
    headline_status = _rollup_status([visibility_status, operational_status])
    priority = ["auth", "paper_ramp", "fx_provider", "storage", "throttle", "collection", "system", "soak"]
    severity = {"blocked": 4, "failed": 4, "critical": 4, "error": 4, "degraded": 3, "needs_work": 3, "stale": 3, "advisory": 2, "recovering": 2, "watch": 2, "warning": 2, "warn": 2}
    active = [name for name, state in effective_states.items() if state != "ready"]
    active.sort(key=lambda name: (-severity.get(effective_states.get(name, ""), 1), priority.index(name) if name in priority else len(priority)))
    owner = active[0] if active else "none"
    owner_row = operational_rows.get(owner, {})
    cause = str(owner_row.get("cause") or owner_row.get("reason") or "none")
    action = str(owner_row.get("action") or "none")
    if owner == "none" and missing:
        cause = f"missing_{missing[0]}"
    elif owner == "none" and stale:
        cause = f"stale_{stale[0]}"
    elif owner == "none" and contradictions:
        cause = contradictions[0]
    if headline_status != "ready" and action == "none":
        action = "runtime-artifact-refresh"

    aged_sources = [
        (name, _as_float(source.get("age_seconds"), -1.0))
        for name, source in sources.items()
        if source.get("age_seconds") is not None
    ]
    oldest_source, oldest_age = max(aged_sources, key=lambda item: item[1]) if aged_sources else ("unknown", -1.0)
    soak = operational_rows.get("soak", {})
    system = operational_rows.get("system", {})
    safe_to_leave = bool(
        headline_status == "ready"
        and guarded_paper_status == "ready"
        and soak.get("effective_safe")
        and soak.get("live_locked")
        and system.get("live_status") in {"blocked", "blocked_read_only", "locked", "disabled"}
    )
    return {
        "headline_status": headline_status,
        "safe_to_leave_unattended": safe_to_leave,
        "attention_owner": owner,
        "root_cause": cause,
        "next_action": action,
        "domain_impact": str(owner_row.get("impact") or "none"),
        "paper_impact": "none" if guarded_paper_status == "ready" else "paper_blocked" if guarded_paper_status == "blocked" else "paper_unverified",
        "oldest_source": oldest_source,
        "oldest_source_age_seconds": None if oldest_age < 0.0 else round(oldest_age, 3),
    }


def build_status_snapshot(project_root: Path, source: str = "main", now: datetime | None = None) -> dict[str, Any]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    artifacts = {
        name: _artifact(project_root, filename, max_age, now)
        for name, (filename, max_age) in ARTIFACT_SPECS.items()
    }
    capability_configured = (
        project_root / "config" / "collector_capability_catalog_v1.json"
    ).is_file()
    materialization_configured = (
        project_root / "config" / "capability_materialization_v1.json"
    ).is_file()
    institutional_capabilities_configured = (
        project_root / "config" / "institutional_capability_control_v1.json"
    ).is_file()
    authoritative_systems_configured = (
        project_root / "config" / "authoritative_systems_v1.json"
    ).is_file()
    if materialization_configured:
        artifacts["capability_materialization"] = _artifact(
            project_root,
            "governance/collector_capabilities/materialized_capabilities_latest.json",
            30 * 60,
            now,
        )
    if capability_configured:
        artifacts["collector_capabilities"] = _artifact(
            project_root,
            "collector_capability_control_latest.json",
            30 * 60,
            now,
        )
    if institutional_capabilities_configured:
        artifacts["institutional_capabilities"] = _artifact(
            project_root,
            "institutional_capability_control_latest.json",
            90 * 60,
            now,
        )
    if authoritative_systems_configured:
        artifacts["authoritative_systems"] = _artifact(
            project_root,
            "authoritative_systems_control_latest.json",
            24 * 60 * 60,
            now,
        )
        artifacts["paper_live_equivalence"] = _artifact(
            project_root,
            "paper_live_equivalence_latest.json",
            24 * 60 * 60,
            now,
        )
    profitability_assessment_configured = (
        project_root / "config" / "profitability_self_assessment_v1.json"
    ).is_file()
    if profitability_assessment_configured:
        artifacts["profitability_assessment"] = _artifact(
            project_root,
            "profitability_self_assessment_latest.json",
            4 * 60 * 60,
            now,
        )
    auth, schwab_auth = _auth_row(artifacts)
    system, collection = _system_and_collection_rows(artifacts)
    paper_ramp = _paper_ramp_row(artifacts, system)
    system["health_paper_status"] = system.get("paper_status")
    system["paper_ramp_status"] = paper_ramp.get("source_status")
    system["paper_status"] = paper_ramp.get("status")
    if paper_ramp.get("status") == "blocked" and system.get("status") == "ready":
        system["status"] = "blocked"
        system["cause"] = paper_ramp.get("cause")
        system["action"] = paper_ramp.get("action")
    storage = _storage_row(artifacts, system)
    throttle = _throttle_row(artifacts, storage)
    runtime_paper_blocked = bool(not throttle.get("paper_allowed") or throttle.get("paper_state") == "paused")
    paper_runtime_policy_disagrees = bool(runtime_paper_blocked and system.get("paper_status") == "ready")
    if runtime_paper_blocked:
        system["paper_status"] = "blocked"
        if system.get("status") == "ready":
            system["status"] = "blocked"
            system["cause"] = throttle.get("cause") or "runtime_paper_policy_blocked"
            system["action"] = throttle.get("action") or "runtime-throttle"
        storage["paper_status"] = "blocked"
    soak = _soak_row(artifacts, system, storage)
    profitability_runtime_source = _artifact(
        project_root,
        "paper_runtime_profitability_controls_latest.json",
        15 * 60,
        now,
    )
    paper_debt_recovery = _paper_debt_recovery_row(profitability_runtime_source)
    profitability_assessment = (
        _profitability_assessment_row(artifacts["profitability_assessment"])
        if profitability_assessment_configured
        else {}
    )
    sleeve_strategy_scaling = _sleeve_strategy_scaling_row(profitability_runtime_source)
    production_excellence = _production_excellence_row(artifacts)
    strategy_specialization = _strategy_specialization_row(artifacts)
    institutional_capabilities = (
        _institutional_capability_row(artifacts)
        if institutional_capabilities_configured
        else {}
    )
    authoritative_systems = (
        _authoritative_systems_row(artifacts)
        if authoritative_systems_configured
        else {}
    )
    paper_live_equivalence = (
        _paper_live_equivalence_row(artifacts)
        if authoritative_systems_configured
        else {}
    )
    collector_capabilities = (
        _collector_capability_row(artifacts) if capability_configured else {}
    )
    capability_materialization = (
        _capability_materialization_row(artifacts) if materialization_configured else {}
    )
    fx_provider = _fx_provider_row(project_root, now)
    source_states = {name: artifact["state"] for name, artifact in artifacts.items()}
    required_sources = set(REQUIRED_SOURCES)
    if capability_configured:
        required_sources.add("collector_capabilities")
    if materialization_configured:
        required_sources.add("capability_materialization")
    if institutional_capabilities_configured:
        required_sources.add("institutional_capabilities")
    missing = sorted(name for name in required_sources if source_states[name] == "missing")
    stale = sorted(name for name in required_sources if source_states[name] == "stale")
    contradictions = []
    if auth["consistency"] == "conflict":
        contradictions.append("auth_sources_disagree")
    if paper_ramp["sources_disagree"]:
        contradictions.append("paper_sources_disagree")
    if paper_runtime_policy_disagrees:
        contradictions.append("paper_runtime_policy_disagrees")
    if soak["declared_safe"] and not soak["effective_safe"]:
        contradictions.append("soak_snapshot_superseded_by_current_health")
    contract_status = "blocked" if missing else ("degraded" if stale or contradictions else "ready")
    operational_rows = {
        "system": system,
        "collection": collection,
        "auth": auth,
        "paper_ramp": paper_ramp,
        "storage": storage,
        "throttle": throttle,
        "soak": soak,
        **(
            {"capability_materialization": capability_materialization}
            if capability_materialization
            else {}
        ),
        **({"collector_capabilities": collector_capabilities} if collector_capabilities else {}),
    }
    if fx_provider:
        operational_rows["fx_provider"] = fx_provider
    effective_operational_states: dict[str, str] = {}
    managed_operational_watches: list[str] = []
    for name, row in operational_rows.items():
        effective_state, managed = _effective_operational_state(name, row)
        effective_operational_states[name] = effective_state
        if managed:
            managed_operational_watches.append(name)
    operational_status = _rollup_status(list(effective_operational_states.values()))
    operator = _operator_summary(
        visibility_status=contract_status,
        operational_status=operational_status,
        guarded_paper_status=str(system.get("paper_status") or "unknown"),
        operational_rows=operational_rows,
        effective_states=effective_operational_states,
        sources=artifacts,
        missing=missing,
        stale=stale,
        contradictions=contradictions,
    )
    return {
        "timestamp_utc": now.isoformat(),
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "overall_status": contract_status,
        "ok": contract_status == "ready",
        "visibility_status": contract_status,
        "operational_status": operational_status,
        "headline_status": operator["headline_status"],
        "guarded_paper_status": system.get("paper_status"),
        "safe_to_leave_unattended": operator["safe_to_leave_unattended"],
        "operator_summary": operator,
        "active_operational_rows": {
            name: state
            for name, state in effective_operational_states.items()
            if state != "ready"
        },
        "managed_operational_watches": managed_operational_watches,
        "source_count": len(artifacts),
        "fresh_source_count": sum(1 for artifact in artifacts.values() if artifact["fresh"]),
        "missing_sources": missing,
        "stale_sources": stale,
        "contradictions": contradictions,
        "sources": {name: _source_public(artifact) for name, artifact in artifacts.items()},
        "rows": {
            "system": system,
            "collection": collection,
            "auth": auth,
            "schwab_auth": schwab_auth,
            "paper_ramp": paper_ramp,
            "storage": storage,
            "throttle": throttle,
            "soak": soak,
            "paper_debt_recovery": paper_debt_recovery,
            **(
                {"profitability_assessment": profitability_assessment}
                if profitability_assessment
                else {}
            ),
            "sleeve_strategy_scaling": sleeve_strategy_scaling,
            "production_excellence": production_excellence,
            "strategy_specialization": strategy_specialization,
            **(
                {"institutional_capabilities": institutional_capabilities}
                if institutional_capabilities
                else {}
            ),
            **(
                {"authoritative_systems": authoritative_systems}
                if authoritative_systems
                else {}
            ),
            **(
                {"paper_live_equivalence": paper_live_equivalence}
                if paper_live_equivalence
                else {}
            ),
            **(
                {"capability_materialization": capability_materialization}
                if capability_materialization
                else {}
            ),
            **({"collector_capabilities": collector_capabilities} if collector_capabilities else {}),
            **({"fx_provider": fx_provider} if fx_provider else {}),
        },
        "contract": {
            "freshness_aware": True,
            "contradiction_resistant": True,
            "strict_and_paper_storage_views_separate": True,
            "visibility_and_operational_status_separate": True,
            "live_execution_authority": False,
            "mutation_authority": False,
            "policy": "report source age, root cause, recovery, PAPER impact, and one bounded next action without changing runtime state",
        },
    }


def _token(value: Any, max_len: int = 96) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "unknown"
    if isinstance(value, float):
        text = f"{value:.3f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    text = re.sub(r"\s+", "_", text.strip()) or "none"
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _age(value: Any) -> str:
    if value is None:
        return "unknown"
    return f"{max(0, _as_int(value))}s"


def _line(label: str, values: list[tuple[str, Any]]) -> str:
    return f"[{label}] " + " ".join(f"{key}={_token(value)}" for key, value in values)


def _level(status: Any) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"blocked", "failed", "failure", "critical", "error"}:
        return "alert"
    if normalized in {
        "advisory",
        "degraded",
        "evidence_pending",
        "paper_attention",
        "ready_with_evidence_debt",
        "needs_work",
        "recovering",
        "watch",
        "warn",
        "warning",
        "stale",
    }:
        return "watch"
    return "ok"


def _live_display_status(status: Any) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"blocked", "blocked_read_only", "locked", "disabled"}:
        return "locked_read_only"
    return normalized or "unknown"


def status_exit_code(snapshot: dict[str, Any]) -> int:
    return 2 if str(snapshot.get("headline_status") or "").strip().lower() == "blocked" else 0


def format_status_lines(snapshot: dict[str, Any]) -> list[str]:
    rows = snapshot.get("rows") if isinstance(snapshot.get("rows"), dict) else {}
    system = rows.get("system") if isinstance(rows.get("system"), dict) else {}
    collection = rows.get("collection") if isinstance(rows.get("collection"), dict) else {}
    auth = rows.get("auth") if isinstance(rows.get("auth"), dict) else {}
    schwab = rows.get("schwab_auth") if isinstance(rows.get("schwab_auth"), dict) else {}
    storage = rows.get("storage") if isinstance(rows.get("storage"), dict) else {}
    throttle = rows.get("throttle") if isinstance(rows.get("throttle"), dict) else {}
    soak = rows.get("soak") if isinstance(rows.get("soak"), dict) else {}
    paper_debt_recovery = rows.get("paper_debt_recovery") if isinstance(rows.get("paper_debt_recovery"), dict) else {}
    profitability_assessment = rows.get("profitability_assessment") if isinstance(rows.get("profitability_assessment"), dict) else {}
    sleeve_strategy_scaling = rows.get("sleeve_strategy_scaling") if isinstance(rows.get("sleeve_strategy_scaling"), dict) else {}
    production_excellence = rows.get("production_excellence") if isinstance(rows.get("production_excellence"), dict) else {}
    strategy_specialization = rows.get("strategy_specialization") if isinstance(rows.get("strategy_specialization"), dict) else {}
    institutional_capabilities = rows.get("institutional_capabilities") if isinstance(rows.get("institutional_capabilities"), dict) else {}
    authoritative_systems = rows.get("authoritative_systems") if isinstance(rows.get("authoritative_systems"), dict) else {}
    paper_live_equivalence = rows.get("paper_live_equivalence") if isinstance(rows.get("paper_live_equivalence"), dict) else {}
    capability_materialization = rows.get("capability_materialization") if isinstance(rows.get("capability_materialization"), dict) else {}
    collector_capabilities = rows.get("collector_capabilities") if isinstance(rows.get("collector_capabilities"), dict) else {}
    fx_provider = rows.get("fx_provider") if isinstance(rows.get("fx_provider"), dict) else {}
    operator = snapshot.get("operator_summary") if isinstance(snapshot.get("operator_summary"), dict) else {}
    lines = [
        _line(
            "status-contract",
            [
                ("level", _level(snapshot.get("headline_status"))),
                ("schema", snapshot.get("schema_version")),
                ("status", snapshot.get("headline_status")),
                ("visibility", snapshot.get("visibility_status")),
                ("operational", snapshot.get("operational_status")),
                ("paper", snapshot.get("guarded_paper_status")),
                ("walkaway", snapshot.get("safe_to_leave_unattended")),
                ("active_issues", len(snapshot.get("active_operational_rows") or {})),
                ("managed_watches", ",".join(snapshot.get("managed_operational_watches") or []) or "none"),
                ("fresh", f"{snapshot.get('fresh_source_count', 0)}/{snapshot.get('source_count', 0)}"),
                ("oldest", f"{_token(operator.get('oldest_source'))}:{_age(operator.get('oldest_source_age_seconds'))}"),
                ("stale", len(snapshot.get("stale_sources") or [])),
                ("missing", len(snapshot.get("missing_sources") or [])),
                ("contradictions", len(snapshot.get("contradictions") or [])),
                ("cause", operator.get("root_cause")),
                ("owner", operator.get("attention_owner")),
                ("impact", operator.get("domain_impact")),
                ("paper_impact", operator.get("paper_impact")),
                ("action", operator.get("next_action")),
            ],
        ),
        _line(
            "system",
            [
                ("level", _level(system.get("status"))),
                ("status", system.get("status")),
                ("strict", system.get("strict_all_clear")),
                ("paper", system.get("paper_status")),
                ("paper_health", system.get("health_paper_status")),
                ("paper_ramp", system.get("paper_ramp_status")),
                ("live", _live_display_status(system.get("live_status"))),
                ("repair", system.get("repair_active")),
                ("cause", system.get("cause")),
                ("age", _age(system.get("artifact_age_seconds"))),
                ("action", system.get("action")),
            ],
        ),
        _line(
            "collection",
            [
                ("level", _level(collection.get("status"))),
                ("status", collection.get("status")),
                ("observing", f"{collection.get('observing', 0)}/{collection.get('collectors', 0)}"),
                ("observations", collection.get("observations")),
                ("sleeve_children", collection.get("sleeve_children")),
                ("fanout", collection.get("fanout_ok")),
                ("zero", collection.get("unmanaged_zero")),
                ("cause", collection.get("cause")),
                ("impact", collection.get("impact")),
                ("age", _age(collection.get("artifact_age_seconds"))),
                ("action", collection.get("action")),
            ],
        ),
        _line(
            "auth",
            [
                ("level", _level(auth.get("status"))),
                ("status", auth.get("status")),
                ("lease", auth.get("lease")),
                ("token", "ready" if auth.get("token_ready") else "not_ready"),
                ("expires", _age(auth.get("expires_in_seconds"))),
                ("broker", auth.get("broker_ready")),
                ("network", auth.get("network_ok")),
                ("probe", auth.get("probe_ok")),
                ("freshness", auth.get("freshness")),
                ("consistency", auth.get("consistency")),
                ("reason", auth.get("reason")),
                ("age", _age(auth.get("max_source_age_seconds"))),
                ("impact", auth.get("impact")),
                ("action", auth.get("action")),
            ],
        ),
        _line(
            "schwab-auth",
            [
                ("level", _level(schwab.get("status"))),
                ("status", schwab.get("status")),
                ("token_ready", schwab.get("token_ready")),
                ("refresh", "needed" if schwab.get("refresh_needed") else "not_needed"),
                ("expires", _age(schwab.get("expires_in_seconds"))),
                ("token_age", _age(schwab.get("token_age_seconds"))),
                ("warnings", schwab.get("warning_count")),
                ("reconciled", schwab.get("reconciled_warning_count")),
                ("artifact_age", _age(schwab.get("artifact_age_seconds"))),
            ],
        ),
        _line(
            "storage",
            [
                ("level", _level(storage.get("status"))),
                ("status", storage.get("status")),
                ("control", storage.get("control_status")),
                ("strict", storage.get("strict_status")),
                ("managed", storage.get("managed_bounded_backlog")),
                ("paper", storage.get("paper_status")),
                ("severity", storage.get("severity")),
                ("truth", storage.get("truth_mode")),
                ("raw_grade", storage.get("raw_grade")),
                ("overlay_grade", storage.get("overlay_grade")),
                ("pressure", storage.get("pressure_index")),
                ("pending", storage.get("total_pending_lines")),
                ("effective", storage.get("effective_total_pending_lines")),
                ("oldest", _age(storage.get("oldest_pending_age_seconds"))),
                ("stale_sources", storage.get("stale_source_count")),
                ("leader", storage.get("oldest_source")),
                ("leader_shard", storage.get("oldest_source_shard")),
                ("leader_pending", storage.get("oldest_source_pending_lines")),
                ("drain", storage.get("drain_status")),
                ("recovery", storage.get("recovery_state")),
                ("eta", f"{_token(storage.get('estimated_drain_minutes'))}m"),
                ("cause", storage.get("cause")),
                ("impact", storage.get("impact")),
                ("age", _age(storage.get("artifact_age_seconds"))),
                ("action", storage.get("action")),
            ],
        ),
        _line(
            "throttle",
            [
                ("level", _level(throttle.get("status"))),
                ("status", throttle.get("status")),
                ("profile", throttle.get("profile")),
                ("host", throttle.get("host_saturation_score")),
                ("compute", throttle.get("compute")),
                ("memory", throttle.get("memory")),
                ("cause", throttle.get("cause")),
                ("paper", throttle.get("paper_state")),
                ("impact", throttle.get("impact")),
                ("owner", throttle.get("pressure_owner")),
                ("collection", throttle.get("collection_mode")),
                ("recovery", throttle.get("recovery")),
                ("managed", throttle.get("managed_control")),
                ("reason", throttle.get("policy_reason")),
                ("age", _age(throttle.get("artifact_age_seconds"))),
                ("action", throttle.get("action")),
            ],
        ),
        _line(
            "soak",
            [
                ("level", _level(soak.get("status"))),
                ("status", soak.get("status")),
                ("grade", soak.get("grade")),
                ("score", soak.get("score")),
                ("declared_safe", soak.get("declared_safe")),
                ("effective_safe", soak.get("effective_safe")),
                ("paper", soak.get("paper_status")),
                ("managed", soak.get("managed_storage_watch")),
                ("watch", soak.get("storage_watch")),
                ("live_locked", soak.get("live_locked")),
                ("main_h", soak.get("main_soak_elapsed_hours")),
                ("main_days", soak.get("main_soak_elapsed_days")),
                ("main_pct", soak.get("main_soak_progress_percent")),
                ("main_includes_resets", soak.get("main_soak_includes_pre_reset_time")),
                ("main_is_promotion_credit", soak.get("main_soak_count_is_promotion_credit")),
                ("clean_h", soak.get("clean_window_elapsed_hours")),
                ("observed_h", soak.get("observed_window_elapsed_hours")),
                ("historical_h", soak.get("historical_segmented_hours")),
                ("historical_days", soak.get("historical_segmented_days")),
                ("segments", soak.get("historical_segment_count")),
                ("history_counts_clean", soak.get("historical_counts_toward_clean_720")),
                ("warnings", soak.get("warning_count")),
                ("warning", soak.get("warning")),
                ("cause", soak.get("cause")),
                ("age", _age(soak.get("artifact_age_seconds"))),
                ("action", soak.get("action")),
            ],
        ),
        _line(
            "paper-debt",
            [
                ("level", _level(paper_debt_recovery.get("status"))),
                ("status", paper_debt_recovery.get("status")),
                ("state", paper_debt_recovery.get("state")),
                ("baseline", paper_debt_recovery.get("baseline_debt_amount")),
                ("remaining", paper_debt_recovery.get("remaining_debt_amount")),
                ("progress", paper_debt_recovery.get("recovery_progress_norm")),
                ("candidate", paper_debt_recovery.get("candidate_id")),
                ("samples", paper_debt_recovery.get("candidate_samples")),
                ("days", paper_debt_recovery.get("candidate_days")),
                ("attributed_pnl", paper_debt_recovery.get("candidate_attributed_pnl")),
                ("velocity", paper_debt_recovery.get("actual_daily_recovery")),
                ("entry_size_cap", paper_debt_recovery.get("entry_size_cap_norm")),
                ("paused", paper_debt_recovery.get("new_entries_paused")),
                ("live_proof", paper_debt_recovery.get("live_proof")),
                ("live_execution", paper_debt_recovery.get("live_execution")),
                ("age", _age(paper_debt_recovery.get("artifact_age_seconds"))),
                ("action", paper_debt_recovery.get("action")),
            ],
        ),
        _line(
            "profit-scaling",
            [
                ("level", _level(sleeve_strategy_scaling.get("status"))),
                ("status", sleeve_strategy_scaling.get("status")),
                ("candidate", sleeve_strategy_scaling.get("candidate_id")),
                ("binding", sleeve_strategy_scaling.get("candidate_binding_valid")),
                ("source", sleeve_strategy_scaling.get("source_ready")),
                ("profiles", sleeve_strategy_scaling.get("profile_control_count")),
                ("strategies", sleeve_strategy_scaling.get("strategy_control_count")),
                ("blocked", sleeve_strategy_scaling.get("blocked_control_count")),
                ("probation", sleeve_strategy_scaling.get("probationary_control_count")),
                ("validated", sleeve_strategy_scaling.get("validated_control_count")),
                ("scale_ready", sleeve_strategy_scaling.get("above_baseline_ready_count")),
                ("entry_cap", sleeve_strategy_scaling.get("global_entry_size_cap_norm")),
                ("max_scale", sleeve_strategy_scaling.get("maximum_entry_size_multiplier_norm")),
                ("entry_only", sleeve_strategy_scaling.get("entry_only")),
                ("exits_open", sleeve_strategy_scaling.get("exit_paths_open")),
                ("live_execution", sleeve_strategy_scaling.get("live_execution")),
                ("age", _age(sleeve_strategy_scaling.get("artifact_age_seconds"))),
                ("action", sleeve_strategy_scaling.get("action")),
            ],
        ),
        _line(
            "production-excellence",
            [
                ("level", "ok" if production_excellence.get("status") == "ready" else "watch"),
                ("status", production_excellence.get("status")),
                ("grade", production_excellence.get("grade")),
                ("score", production_excellence.get("score")),
                ("pillars", f"{production_excellence.get('ready_pillars', 0)}/{production_excellence.get('pillar_count', 10)}"),
                ("candidate", production_excellence.get("candidate_id")),
                ("drift", production_excellence.get("candidate_drift")),
                ("blocked", ",".join(production_excellence.get("blocked_pillars") or []) or "none"),
                ("live_consideration", production_excellence.get("live_consideration")),
                ("live_locked", production_excellence.get("live_locked")),
                ("paper_impact", production_excellence.get("paper_impact")),
                ("age", _age(production_excellence.get("artifact_age_seconds"))),
                ("action", production_excellence.get("action")),
            ],
        ),
        _line(
            "strategy-specialization",
            [
                ("level", "ok" if strategy_specialization.get("status") == "ready" else "watch"),
                ("status", strategy_specialization.get("status")),
                ("grade", strategy_specialization.get("grade")),
                ("contracts", f"{strategy_specialization.get('complete', 0)}/{strategy_specialization.get('strategies', 0)}"),
                ("library", f"{strategy_specialization.get('library_strategies', 0)}/{strategy_specialization.get('library_target', 12000)}"),
                ("families", f"{strategy_specialization.get('canonical_families', 0)}/1989"),
                ("lineage", f"{strategy_specialization.get('family_lineage_covered', 0)}/12000"),
                ("identity_changes", strategy_specialization.get("family_identity_changes")),
                ("sleeves", strategy_specialization.get("sleeves")),
                ("candidate", strategy_specialization.get("candidate_id")),
                ("bound", strategy_specialization.get("candidate_bound")),
                ("parked", strategy_specialization.get("parked")),
                ("probation", strategy_specialization.get("probation")),
                ("validated", strategy_specialization.get("validated")),
                ("good", strategy_specialization.get("validated_good")),
                ("promising", strategy_specialization.get("promising")),
                ("weak", strategy_specialization.get("weak")),
                ("retire", strategy_specialization.get("retirement_candidates")),
                ("control_only", strategy_specialization.get("control_only")),
                ("regime", strategy_specialization.get("current_regime")),
                ("regime_ready", strategy_specialization.get("regime_activation_ready")),
                ("authority_violations", strategy_specialization.get("authority_violations")),
                ("paper_impact", strategy_specialization.get("paper_impact")),
                ("live_authority", strategy_specialization.get("live_authority")),
                ("age", _age(strategy_specialization.get("artifact_age_seconds"))),
                ("action", strategy_specialization.get("action")),
            ],
        ),
    ]
    if profitability_assessment:
        lines.insert(
            -1,
            _line(
                "profitability-truth",
                [
                    ("level", "ok" if profitability_assessment.get("status") == "ready" else "watch"),
                    ("status", profitability_assessment.get("status")),
                    ("assessor", profitability_assessment.get("assessment_status")),
                    ("candidate", profitability_assessment.get("candidate_id")),
                    ("binding", profitability_assessment.get("candidate_consistent")),
                    ("implementation", profitability_assessment.get("implementation_grade")),
                    ("implementation_score", profitability_assessment.get("implementation_score")),
                    ("economic", profitability_assessment.get("economic_grade")),
                    ("economic_score", profitability_assessment.get("economic_score")),
                    (
                        "samples",
                        f"{profitability_assessment.get('candidate_samples', 0)}/{profitability_assessment.get('candidate_minimum_samples', 30)}",
                    ),
                    ("candidate_pnl", profitability_assessment.get("candidate_pnl")),
                    ("historical_book_pnl", profitability_assessment.get("historical_book_pnl")),
                    ("historical_grades_candidate", profitability_assessment.get("historical_grades_candidate")),
                    (
                        "evidence_lanes",
                        f"{profitability_assessment.get('ready_lanes', 0)}/{profitability_assessment.get('lane_count', 8)}",
                    ),
                    ("next_need", profitability_assessment.get("next_need")),
                    ("live_execution", profitability_assessment.get("live_execution")),
                    ("age", _age(profitability_assessment.get("artifact_age_seconds"))),
                    ("action", profitability_assessment.get("action")),
                ],
            ),
        )
    if capability_materialization:
        lines.insert(
            2,
            _line(
                "capability-materialization",
                [
                    ("level", _level(capability_materialization.get("status"))),
                    ("status", capability_materialization.get("status")),
                    (
                        "direct_proofs",
                        f"{capability_materialization.get('direct_proofs', 0)}/{capability_materialization.get('required_proofs', 4)}",
                    ),
                    ("contracts", capability_materialization.get("contracts")),
                    ("stress_scenarios", capability_materialization.get("stress_scenarios")),
                    ("calendar_library", capability_materialization.get("calendar_library")),
                    ("live_promotion", capability_materialization.get("live_promotion_ready")),
                    ("cause", capability_materialization.get("cause")),
                    ("impact", capability_materialization.get("impact")),
                    ("age", _age(capability_materialization.get("artifact_age_seconds"))),
                    ("action", capability_materialization.get("action")),
                ],
            ),
        )
    if institutional_capabilities:
        target = institutional_capabilities.get("provider_target")
        target_token = "15-30"
        if isinstance(target, list) and len(target) >= 2:
            target_token = f"{target[0]}-{target[1]}"
        lines.insert(
            -1,
            _line(
                "institutional-capabilities",
                [
                    ("level", _level(institutional_capabilities.get("status"))),
                    ("status", institutional_capabilities.get("status")),
                    (
                        "implementation",
                        f"{institutional_capabilities.get('implementation_ready', 0)}/{institutional_capabilities.get('pillar_count', 6)}",
                    ),
                    (
                        "paper",
                        f"{institutional_capabilities.get('paper_ready', 0)}/{institutional_capabilities.get('pillar_count', 6)}",
                    ),
                    (
                        "evidence",
                        f"{institutional_capabilities.get('candidate_evidence_ready', 0)}/{institutional_capabilities.get('pillar_count', 6)}",
                    ),
                    (
                        "live",
                        f"{institutional_capabilities.get('live_ready', 0)}/{institutional_capabilities.get('pillar_count', 6)}",
                    ),
                    ("verified_bundles", institutional_capabilities.get("verified_source_bundles")),
                    ("provider_target", target_token),
                    ("need_10000", institutional_capabilities.get("ten_thousand_sources_required")),
                    ("conditional_entitlements", institutional_capabilities.get("conditional_entitlements")),
                    ("candidate", institutional_capabilities.get("candidate_id")),
                    ("bound", institutional_capabilities.get("candidate_bound")),
                    ("local_refreshes", institutional_capabilities.get("local_refresh_actions")),
                    ("external", institutional_capabilities.get("external_actions")),
                    ("paper_impact", institutional_capabilities.get("paper_impact")),
                    ("live_execution", institutional_capabilities.get("live_execution")),
                    ("age", _age(institutional_capabilities.get("artifact_age_seconds"))),
                    ("action", institutional_capabilities.get("action")),
                ],
            ),
        )
    if authoritative_systems:
        lines.insert(
            -1,
            _line(
                "authoritative-systems",
                [
                    ("level", "ok" if authoritative_systems.get("status") == "ready" else "watch"),
                    ("status", authoritative_systems.get("status")),
                    ("grade", authoritative_systems.get("grade")),
                    ("grade_scope", authoritative_systems.get("grade_scope")),
                    ("references", f"{authoritative_systems.get('references', 0)}/20"),
                    (
                        "controls",
                        f"{authoritative_systems.get('ready_controls', 0)}/{authoritative_systems.get('control_count', 8)}",
                    ),
                    ("soak_reset", authoritative_systems.get("soak_reset")),
                    ("paper_impact", authoritative_systems.get("paper_impact")),
                    ("live_authority", authoritative_systems.get("live_authority")),
                    ("age", _age(authoritative_systems.get("artifact_age_seconds"))),
                    ("action", authoritative_systems.get("action")),
                ],
            ),
        )
    if paper_live_equivalence:
        lines.insert(
            -1,
            _line(
                "paper-live-equivalence",
                [
                    (
                        "level",
                        "ok" if paper_live_equivalence.get("empirical_ready") else "watch",
                    ),
                    ("status", paper_live_equivalence.get("status")),
                    ("structural", paper_live_equivalence.get("structural_ready")),
                    ("empirical", paper_live_equivalence.get("empirical_ready")),
                    ("paper", paper_live_equivalence.get("paper_count")),
                    ("live_shadow", paper_live_equivalence.get("live_count")),
                    ("pairs", paper_live_equivalence.get("paired_count")),
                    ("mismatches", paper_live_equivalence.get("mismatch_count")),
                    ("missing_live", paper_live_equivalence.get("missing_live_count")),
                    ("paper_not_promoted", paper_live_equivalence.get("unpaired_paper_count")),
                    ("orphan_live", paper_live_equivalence.get("missing_paper_count")),
                    ("paper_impact", paper_live_equivalence.get("paper_impact")),
                    ("live_evidence", paper_live_equivalence.get("live_evidence")),
                    ("live_authority", paper_live_equivalence.get("live_authority")),
                    ("age", _age(paper_live_equivalence.get("artifact_age_seconds"))),
                    ("action", paper_live_equivalence.get("action")),
                ],
            ),
        )
    if collector_capabilities:
        lines.insert(
            2,
            _line(
                "collector-capabilities",
                [
                    ("level", _level(collector_capabilities.get("status"))),
                    ("status", collector_capabilities.get("status")),
                    ("planes", collector_capabilities.get("planes")),
                    ("capabilities", collector_capabilities.get("capabilities")),
                    (
                        "bots",
                        f"{collector_capabilities.get('bots', 0)}/{collector_capabilities.get('assignments', 0)}",
                    ),
                    ("profiles", collector_capabilities.get("profiles")),
                    ("route_profiles", collector_capabilities.get("ingestion_route_profiles")),
                    ("routing_policy", collector_capabilities.get("routing_policy")),
                    ("decision_stage", collector_capabilities.get("decision_stage")),
                    ("families", collector_capabilities.get("decision_families")),
                    (
                        "paper_routes",
                        f"{collector_capabilities.get('runtime_paper_ready_routes', 0)}/"
                        f"{collector_capabilities.get('runtime_routes', 0)}",
                    ),
                    (
                        "live_routes",
                        f"{collector_capabilities.get('runtime_live_ready_routes', 0)}/"
                        f"{collector_capabilities.get('runtime_routes', 0)}",
                    ),
                    ("route_quality", collector_capabilities.get("average_route_quality")),
                    (
                        "independent_redundancy",
                        collector_capabilities.get("independent_redundancy_ratio"),
                    ),
                    ("transport_contract", collector_capabilities.get("transport_contract_complete")),
                    ("route_receipt", collector_capabilities.get("route_receipt")),
                    ("mapped", collector_capabilities.get("collector_mapping_complete")),
                    ("coverage_gaps", collector_capabilities.get("coverage_gaps")),
                    ("candidate_blocking", collector_capabilities.get("candidate_blocking_gaps")),
                    ("optional_gaps", collector_capabilities.get("optional_gaps")),
                    ("debt_scope", collector_capabilities.get("coverage_debt_scope")),
                    ("required_usable", collector_capabilities.get("required_usable_ratio")),
                    ("required_redundancy", collector_capabilities.get("required_redundancy_ratio")),
                    ("full_catalog", collector_capabilities.get("full_catalog_coverage_ready")),
                    ("paper", collector_capabilities.get("paper_soak_ready")),
                    ("live_promotion", collector_capabilities.get("live_promotion_ready")),
                    ("cause", collector_capabilities.get("cause")),
                    ("impact", collector_capabilities.get("impact")),
                    ("age", _age(collector_capabilities.get("artifact_age_seconds"))),
                    ("action", collector_capabilities.get("action")),
                ],
            ),
        )
    if fx_provider:
        lines.insert(
            3,
            _line(
                "fx-provider",
                [
                    ("level", _level(fx_provider.get("status"))),
                    ("status", fx_provider.get("status")),
                    ("provider", fx_provider.get("provider")),
                    ("available", fx_provider.get("available")),
                    ("mode", fx_provider.get("mode")),
                    ("fallback", fx_provider.get("fallback_active")),
                    ("managed", fx_provider.get("managed_fallback")),
                    ("cooldown", fx_provider.get("cooldown_kind")),
                    ("retry", _age(fx_provider.get("cooldown_remaining_seconds"))),
                    ("credential", fx_provider.get("credential_action_required")),
                    ("errors", fx_provider.get("iter_error_rate")),
                    ("cause", fx_provider.get("cause")),
                    ("impact", fx_provider.get("impact")),
                    ("paper_impact", "none" if fx_provider.get("managed_fallback") else fx_provider.get("impact")),
                    ("age", _age(fx_provider.get("artifact_age_seconds"))),
                    ("action", fx_provider.get("action")),
                ],
            ),
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source", default="main")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    snapshot = build_status_snapshot(args.project_root.resolve(), source=args.source)
    if args.json:
        print(json.dumps(snapshot, ensure_ascii=True))
    else:
        print("\n".join(format_status_lines(snapshot)))
    return status_exit_code(snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
