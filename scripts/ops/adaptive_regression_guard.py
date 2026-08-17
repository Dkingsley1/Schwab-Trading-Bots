#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops import grade_regression_guard
    from scripts.ops.long_runtime_common import (
        PROJECT_ROOT,
        iso_now,
        load_json,
        ordered_unique,
        payload_age_minutes,
        payload_timestamp,
        utc_now,
        write_payload,
    )
else:
    from . import grade_regression_guard
    from .long_runtime_common import (
        PROJECT_ROOT,
        iso_now,
        load_json,
        ordered_unique,
        payload_age_minutes,
        payload_timestamp,
        utc_now,
        write_payload,
    )


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "adaptive_regression_guard_latest.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "adaptive_regression_guard_state.json"
DEFAULT_FEEDBACK_PATH = PROJECT_ROOT / "governance" / "health" / "adaptive_regression_guard_feedback.jsonl"

GradeGuardBuilder = Callable[[Path], dict[str, Any]]
ArtifactLoader = Callable[[Path], dict[str, Any]]

READY_STATES = {"ready", "ok", "active", "at_floor", "clear"}
DEGRADED_STATES = {"degraded", "needs_attention", "warn", "warning", "protected_by_floor", "advisory"}
BLOCKED_STATES = {"blocked", "critical", "missing", "failed", "below_floor"}
BOUNDED_TRANSIENT_STORAGE_PRESSURE_MAX = 1.0
BOUNDED_TRANSIENT_STORAGE_CORE_MAX = 10_000
BOUNDED_TRANSIENT_STORAGE_TOTAL_MAX = 15_000
BOUNDED_TRANSIENT_STORAGE_SUPPORT_MAX = 12_000
BOUNDED_TRANSIENT_STORAGE_AGE_MAX_SECONDS = 300.0
STORAGE_ROUTE_READY_STATES = {"ready", "verified", "ok", "active_local_ready", "active_passthrough"}
HEAVY_SURFACES = {
    "grade:storage_control",
    "grade:security_audit",
    "section:data_ingestion_and_storage",
    "guard:ingestion_storage_degradation_floor",
    "guard:stateful_storage_regression_guard",
    "guard:one_numbers_regression_guard",
}

SPECIALIZED_GUARDS: list[dict[str, Any]] = [
    {
        "surface_id": "guard:runtime_paper_regression_guard",
        "surface": "runtime_paper_regression_guard",
        "artifact": "runtime_paper_regression_guard_latest.json",
        "source": "runtime_paper_regression_guard",
        "max_age_minutes": 30,
        "recommended_command": ["./scripts/ops/opsctl.sh", "runtime-paper-regression-guard", "--json"],
    },
    {
        "surface_id": "guard:stateful_storage_regression_guard",
        "surface": "stateful_storage_regression_guard",
        "artifact": "stateful_storage_regression_guard_latest.json",
        "source": "stateful_storage_regression_guard",
        "max_age_minutes": 45,
        "recommended_command": ["./scripts/ops/opsctl.sh", "stateful-storage-regression-guard", "--apply", "--json"],
    },
    {
        "surface_id": "guard:one_numbers_regression_guard",
        "surface": "one_numbers_regression_guard",
        "artifact": "one_numbers_regression_guard_latest.json",
        "source": "one_numbers_regression_guard",
        "max_age_minutes": 45,
        "recommended_command": ["./scripts/ops/opsctl.sh", "one-numbers-regression-guard", "--apply", "--json"],
    },
]

CRITICAL_CONTRACT_IDS = {
    "guard:broker_auth_contract",
    "guard:livefeed_visibility_contract",
    "guard:source_verification_contract",
    "guard:memory_truth_contract",
    "guard:runtime_storage_contract",
    "guard:ingestion_storage_degradation_floor",
    "guard:backlog_pcore_contract",
}
OPTIONAL_CONTEXT_SOURCE_IDS = {
    "fx_market_context",
    "options_context_mesh",
    "macro_crossstack",
    "crypto_market_context",
    "free_equity_reference_context",
    "public_macro_feeds",
    "schwab_symbol_news",
    "ticker_news_context",
    "market_micro_context",
    "sec_edgar_context",
    "extended_quant_context",
    "public_policy_context",
}
PAPER_SOAK_QUALITY_DEBT_SURFACES = {
    "grade:training_quality",
    "section:training_and_model_quality",
}
PAPER_SOAK_ADVISORY_SECTION_SURFACES = {
    "section:data_ingestion_and_storage",
    "section:live_trading_readiness",
    "section:training_and_model_quality",
    "section:ops_and_autonomy",
}
PAPER_SOAK_PROMOTION_GATE_SURFACES = {
    "grade:training_lineage",
    "grade:live_canary",
    "grade:autonomy_control",
    "grade:promotion_autopilot",
}
GUARDED_READ_ONLY_RUNTIME_STATES = {
    "guarded_live_read_only",
    "managed_cold_lane_deferred",
    "managed_coverage_stage_deferred",
}
OPTIONAL_SOURCE_DEBT_WARNING_TOKENS = {
    "source_verification_stale_artifacts",
    "source_verification_low_confidence_sources",
    "source_verification_min_confidence_below_floor",
    "source_verification_optional_context_debt_for_guarded_paper",
    "source_verification_not_ready",
    "source_verification_unverified_sources",
    "source_verification_degraded_artifacts",
    "source_verification_not_all_verified",
}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


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


def _bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return raw != 0
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on", "ready", "ok"}


def _lower(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _source_ids(raw: list[Any]) -> set[str]:
    return {str(item).strip() for item in raw if str(item or "").strip()}


def _optional_source_ids_only(raw: list[Any]) -> bool:
    source_ids = _source_ids(raw)
    return bool(source_ids) and source_ids.issubset(OPTIONAL_CONTEXT_SOURCE_IDS)


def _state_rank(state: str) -> int:
    normalized = _lower(state)
    if normalized in BLOCKED_STATES:
        return 3
    if normalized in DEGRADED_STATES:
        return 2
    if normalized in READY_STATES:
        return 0
    return 1 if normalized else 2


def _canonical_state(state: Any, *, ok: Any = None) -> str:
    normalized = _lower(state)
    if normalized in READY_STATES:
        return "ready"
    if normalized in DEGRADED_STATES:
        return "degraded"
    if normalized in BLOCKED_STATES:
        return "blocked"
    if ok is not None:
        return "ready" if _bool(ok) else "blocked"
    return normalized or "missing"


def _command_key(cmd: list[str]) -> str:
    return " ".join(str(part) for part in cmd if str(part or "").strip())


def _newer_than(primary: dict[str, Any], secondary: dict[str, Any], *, min_seconds: float = 60.0) -> bool:
    primary_ts = payload_timestamp(primary)
    secondary_ts = payload_timestamp(secondary)
    if primary_ts is None or secondary_ts is None:
        return False
    return (primary_ts - secondary_ts).total_seconds() >= min_seconds


def _runtime_pressure_context(project_root: Path, loader: ArtifactLoader) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    health_fast = loader(health_root / "health_fast_latest.json")
    runtime = loader(health_root / "runtime_throttle_control_latest.json")
    health_ignored_as_stale = bool(runtime and health_fast and _newer_than(runtime, health_fast))
    pressure = {} if health_ignored_as_stale else _as_dict(health_fast.get("runtime_pressure"))
    if not pressure:
        pressure = runtime
    compute_level = _lower(pressure.get("compute_pressure_level") or runtime.get("compute_pressure_level"))
    memory_level = _lower(pressure.get("memory_pressure_level") or runtime.get("memory_pressure_level"))
    runtime_status = _lower(pressure.get("overall_status") or runtime.get("overall_status"))
    host_saturation = max(
        _safe_float(pressure.get("host_saturation_score"), 0.0),
        _safe_float(runtime.get("host_saturation_score"), 0.0),
    )
    high_pressure = (
        host_saturation >= 60.0
        or compute_level in {"high", "critical"}
        or memory_level in {"high", "critical"}
        or runtime_status in {"blocked", "critical"}
    )
    return {
        "overall_status": runtime_status or "unknown",
        "host_saturation_score": round(host_saturation, 2),
        "compute_pressure_level": compute_level or "unknown",
        "memory_pressure_level": memory_level or "unknown",
        "high_pressure": bool(high_pressure),
        "source": "runtime_throttle_control" if health_ignored_as_stale else "health_fast" if health_fast else "runtime_throttle_control" if runtime else "missing",
        "health_fast_ignored_as_stale": health_ignored_as_stale,
    }


def _guarded_paper_operational(project_root: Path, loader: ArtifactLoader) -> bool:
    health_root = project_root / "governance" / "health"
    health_fast = loader(health_root / "health_fast_latest.json")
    runtime = loader(health_root / "live_runtime_separation_control_latest.json")
    operational = _as_dict(health_fast.get("operational_readiness"))
    guarded_paper = _as_dict(operational.get("guarded_paper"))
    live_execution = _as_dict(operational.get("live_execution"))
    clearance_state = _lower(_as_dict(runtime.get("clearance_plan")).get("clearance_state"))
    guarded_ok = _bool(guarded_paper.get("ok")) and _lower(guarded_paper.get("status")) in {"ready", "armed", "guarded_ready"}
    live_locked = (
        clearance_state in GUARDED_READ_ONLY_RUNTIME_STATES
        or _lower(live_execution.get("status")) in {"blocked_read_only", "read_only", "operator_gated"}
        or "live_execution_requires_explicit_operator_control" in set(_as_list(live_execution.get("blockers")))
    )
    return bool(guarded_ok and live_locked)


def _health_fast_strict_clear(project_root: Path, loader: ArtifactLoader) -> bool:
    health_root = project_root / "governance" / "health"
    health_fast = loader(health_root / "health_fast_latest.json")
    return bool(
        health_fast
        and _bool(health_fast.get("ok", False))
        and _lower(health_fast.get("overall_status")) in {"ready", "ok"}
        and _bool(health_fast.get("strict_all_clear", health_fast.get("ok", False)))
    )


def _read_only_source_advisory_context(project_root: Path, loader: ArtifactLoader) -> bool:
    health_root = project_root / "governance" / "health"
    health_fast = loader(health_root / "health_fast_latest.json")
    operational = _as_dict(health_fast.get("operational_readiness"))
    live_execution = _as_dict(operational.get("live_execution"))
    live_locked = (
        _lower(live_execution.get("status")) in {"blocked_read_only", "read_only", "operator_gated"}
        or "live_execution_requires_explicit_operator_control" in set(_as_list(live_execution.get("blockers")))
        or bool(health_fast.get("read_only", False))
    )
    return bool(health_fast and live_locked)


def _grade_surfaces(grade_guard_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _as_list(grade_guard_payload.get("surfaces")):
        if not isinstance(row, dict):
            continue
        surface = str(row.get("surface") or "").strip()
        if not surface:
            continue
        state = _canonical_state(row.get("state"), ok=row.get("ok"))
        command = row.get("recommended_command") if isinstance(row.get("recommended_command"), list) else []
        rows.append(
            {
                "surface_id": f"grade:{surface}",
                "surface": surface,
                "source": "grade_regression_guard",
                "state": state,
                "base_severity": _lower(row.get("severity")) or ("critical" if state == "blocked" else "warning"),
                "summary": str(row.get("summary") or ""),
                "metrics": _as_dict(row.get("metrics")),
                "retry_budget": _as_dict(row.get("retry_budget")),
                "quiet_hours_preferred": bool(row.get("quiet_hours_preferred", False)),
                "recommended_command": [str(part) for part in command],
            }
        )
    return rows


def _section_surfaces(section_guard_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    advisory_sections = {
        str(section)
        for section in _as_list(section_guard_payload.get("advisory_below_floor_sections"))
        if str(section or "").strip()
    }
    paper_soak_advisory = bool(section_guard_payload.get("paper_soak_advisory_below_floor", False))
    for row in _as_list(section_guard_payload.get("sections")):
        if not isinstance(row, dict):
            continue
        section = str(row.get("section") or "").strip()
        if not section:
            continue
        raw_state = _lower(row.get("state") or row.get("floor_state"))
        state = "ready"
        if raw_state == "below_floor":
            state = "blocked"
        elif raw_state == "protected_by_floor":
            state = "degraded"
        if paper_soak_advisory and section in advisory_sections and state == "blocked":
            state = "degraded"
        command: list[str] = []
        for candidate in _as_list(row.get("recommended_commands")):
            if isinstance(candidate, list) and candidate:
                command = [str(part) for part in candidate]
                break
        rows.append(
            {
                "surface_id": f"section:{section}",
                "surface": section,
                "source": "section_grade_guard",
                "state": state,
                "base_severity": "critical" if state == "blocked" else "warning",
                "summary": str(row.get("floor_reason") or f"section_state={raw_state or state}"),
                "metrics": {
                    "score": _safe_float(row.get("score"), 0.0),
                    "raw_score": _safe_float(row.get("raw_score"), 0.0),
                    "letter_grade": str(row.get("letter_grade") or ""),
                    "raw_letter_grade": str(row.get("raw_letter_grade") or ""),
                    "paper_soak_advisory": bool(paper_soak_advisory and section in advisory_sections),
                },
                "retry_budget": {},
                "quiet_hours_preferred": section in {"data_ingestion_and_storage", "security_governance_and_auditability"},
                "recommended_command": command or ["./scripts/ops/opsctl.sh", "section-grade-autopilot", "--apply", "--json"],
            }
        )
    return rows


def _soften_paper_soak_quality_debt(
    surface: dict[str, Any],
    section_guard_payload: dict[str, Any],
    project_root: Path,
    loader: ArtifactLoader,
) -> dict[str, Any]:
    surface_id = str(surface.get("surface_id") or "")
    metrics = dict(_as_dict(surface.get("metrics")))
    guarded_operational = _guarded_paper_operational(project_root, loader)
    section_training_advisory = bool(
        section_guard_payload.get("paper_soak_advisory_below_floor", False)
        and bool(section_guard_payload.get("guarded_paper_ready", False))
        and bool(section_guard_payload.get("live_execution_locked", False))
    )
    section_paper_soak_advisory = bool(
        surface_id in PAPER_SOAK_ADVISORY_SECTION_SURFACES
        and metrics.get("paper_soak_advisory", False)
        and section_guard_payload.get("paper_soak_advisory_below_floor", False)
    )
    if (
        surface_id not in PAPER_SOAK_QUALITY_DEBT_SURFACES
        and not section_paper_soak_advisory
        or _canonical_state(surface.get("state")) == "ready"
        or not (section_training_advisory or guarded_operational)
    ):
        return surface
    original_state = _canonical_state(surface.get("state"))
    softened = dict(surface)
    softened["state"] = "ready"
    softened["base_severity"] = "info"
    summary = str(softened.get("summary") or "").strip()
    softened["summary"] = (
        f"{summary}; advisory-only during guarded paper soak while live execution remains locked"
        if summary
        else "training quality debt is advisory-only during guarded paper soak while live execution remains locked"
    )
    metrics["paper_soak_advisory"] = True
    metrics["paper_soak_quality_advisory_only"] = True
    metrics["paper_soak_section_advisory_only"] = section_paper_soak_advisory
    metrics["guarded_paper_operational"] = guarded_operational
    metrics["original_state"] = original_state
    softened["metrics"] = metrics
    return softened


def _soften_guarded_paper_incident_closeout(surface: dict[str, Any], project_root: Path, loader: ArtifactLoader) -> dict[str, Any]:
    if (
        str(surface.get("surface_id") or "") != "grade:incident_closeout"
        or _canonical_state(surface.get("state")) != "blocked"
        or not _guarded_paper_operational(project_root, loader)
        or not _health_fast_strict_clear(project_root, loader)
    ):
        return surface
    softened = dict(surface)
    softened["state"] = "degraded"
    softened["base_severity"] = "warning"
    summary = str(softened.get("summary") or "").strip()
    softened["summary"] = (
        f"{summary}; historical closeout debt is advisory while health-fast is strict-clear and live execution is locked"
        if summary
        else "incident closeout debt is advisory while health-fast is strict-clear and live execution is locked"
    )
    metrics = dict(_as_dict(softened.get("metrics")))
    metrics["guarded_paper_soak_advisory"] = True
    metrics["health_fast_strict_clear"] = True
    softened["metrics"] = metrics
    return softened


def _soften_paper_soak_promotion_gate(surface: dict[str, Any], project_root: Path, loader: ArtifactLoader) -> dict[str, Any]:
    surface_id = str(surface.get("surface_id") or "")
    if (
        surface_id not in PAPER_SOAK_PROMOTION_GATE_SURFACES
        or _canonical_state(surface.get("state")) == "ready"
        or not _guarded_paper_operational(project_root, loader)
    ):
        return surface
    original_state = _canonical_state(surface.get("state"))
    softened = dict(surface)
    softened["state"] = "ready"
    softened["base_severity"] = "info"
    summary = str(softened.get("summary") or "").strip()
    softened["summary"] = (
        f"{summary}; advisory-only while guarded paper soak is ready and live execution is locked"
        if summary
        else "promotion evidence gate is advisory-only while guarded paper soak is ready and live execution is locked"
    )
    metrics = dict(_as_dict(softened.get("metrics")))
    metrics["paper_soak_promotion_gate_advisory_only"] = True
    metrics["does_not_block_guarded_paper_soak"] = True
    metrics["original_state"] = original_state
    softened["metrics"] = metrics
    return softened


def _artifact_surface(
    project_root: Path,
    spec: dict[str, Any],
    *,
    loader: ArtifactLoader,
    max_artifact_age_minutes: int,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    path = health_root / str(spec["artifact"])
    payload = loader(path)
    now = utc_now()
    age = payload_age_minutes(payload, path, now=now) if payload else None
    max_age = min(_safe_int(spec.get("max_age_minutes"), max_artifact_age_minutes), int(max_artifact_age_minutes))
    stale = bool(age is not None and age > max_age)
    if not payload:
        state = "blocked"
        summary = f"{spec['artifact']} is missing"
        ok = False
    else:
        state = _canonical_state(payload.get("overall_status") or payload.get("status"), ok=payload.get("ok"))
        ok = state == "ready"
        if stale and ok:
            state = "degraded"
        summary = f"overall_status={payload.get('overall_status', payload.get('status', 'unknown'))}"
        if stale:
            summary = f"{summary} stale age_minutes={age:.1f}"
    return {
        "surface_id": str(spec["surface_id"]),
        "surface": str(spec["surface"]),
        "source": str(spec["source"]),
        "state": state,
        "base_severity": "critical" if state == "blocked" else "warning",
        "summary": summary,
        "metrics": {
            "artifact": str(path.relative_to(project_root)) if path.is_absolute() else str(path),
            "artifact_age_minutes": None if age is None else round(float(age), 2),
            "artifact_max_age_minutes": int(max_age),
            "artifact_stale": stale,
            "artifact_ok": ok,
        },
        "retry_budget": {},
        "quiet_hours_preferred": str(spec["surface_id"]) in HEAVY_SURFACES,
        "recommended_command": [str(part) for part in _as_list(spec.get("recommended_command"))],
    }


def _artifact_payload(
    project_root: Path,
    artifact: str,
    *,
    loader: ArtifactLoader,
    max_age_minutes: int,
) -> tuple[Path, dict[str, Any], float | None, bool]:
    path = project_root / "governance" / "health" / artifact
    payload = loader(path)
    age = payload_age_minutes(payload, path, now=utc_now()) if payload else None
    stale = bool(age is not None and age > max(int(max_age_minutes), 1))
    return path, payload, age, stale


def _artifact_metric(path: Path, project_root: Path, age: float | None, stale: bool, max_age_minutes: int) -> dict[str, Any]:
    return {
        "artifact": str(path.relative_to(project_root)) if path.is_absolute() else str(path),
        "artifact_age_minutes": None if age is None else round(float(age), 2),
        "artifact_max_age_minutes": int(max_age_minutes),
        "artifact_stale": bool(stale),
    }


def _contract_row(
    *,
    surface_id: str,
    surface: str,
    state: str,
    summary: str,
    metrics: dict[str, Any],
    recommended_command: list[str],
    quiet_hours_preferred: bool = False,
) -> dict[str, Any]:
    return {
        "surface_id": surface_id,
        "surface": surface,
        "source": "adaptive_regression_contract",
        "state": _canonical_state(state),
        "base_severity": "critical" if state == "blocked" else "warning",
        "summary": summary,
        "metrics": metrics,
        "retry_budget": {},
        "quiet_hours_preferred": bool(quiet_hours_preferred),
        "recommended_command": recommended_command,
        "critical_contract": surface_id in CRITICAL_CONTRACT_IDS,
    }


def _broker_auth_contract_surface(project_root: Path, loader: ArtifactLoader, max_artifact_age_minutes: int) -> dict[str, Any]:
    max_age = min(int(max_artifact_age_minutes), 30)
    auth_path, auth, auth_age, auth_stale = _artifact_payload(project_root, "auth_lease_manager_latest.json", loader=loader, max_age_minutes=max_age)
    broker_path, broker, broker_age, broker_stale = _artifact_payload(project_root, "broker_readiness_latest.json", loader=loader, max_age_minutes=max_age)
    supervisor_path, supervisor, supervisor_age, supervisor_stale = _artifact_payload(
        project_root,
        "schwab_auth_supervisor_latest.json",
        loader=loader,
        max_age_minutes=max_age,
    )
    broker_state = _as_dict(auth.get("broker_state"))
    lease_budget = _as_dict(auth.get("lease_budget"))
    expires_in = _safe_float(lease_budget.get("expires_in_seconds"), 0.0)
    critical_seconds = max(_safe_float(lease_budget.get("critical_lease_seconds"), 600.0), 60.0)
    auth_state = _canonical_state(auth.get("overall_status"), ok=auth.get("ok")) if auth else "blocked"
    supervisor_state = _canonical_state(supervisor.get("overall_status"), ok=supervisor.get("ok")) if supervisor else "blocked"
    broker_ready = _bool(broker.get("ready_for_open", broker_state.get("broker_ready", False))) if broker else False
    auth_ok = _bool(broker.get("auth_ok", broker_state.get("auth_ok", False))) if broker else _bool(broker_state.get("auth_ok", False))
    network_ok = _bool(broker.get("network_ok", broker_state.get("network_ok", False))) if broker else _bool(broker_state.get("network_ok", False))
    probe_ok = _bool(broker.get("probe_ok", broker_state.get("auth_probe_ok", True))) if broker or broker_state else False
    paper_soak_auth_operable = bool(
        supervisor.get("paper_soak_auth_operable", False)
        or (
            bool(lease_budget.get("token_lease_grace", False))
            and bool(broker_state.get("broker_operable", broker_ready))
            and bool(network_ok)
            and expires_in >= max(critical_seconds, 900.0)
        )
    )

    raw_blockers = ordered_unique(
        [
            "auth_lease_manager_missing" if not auth else "",
            "auth_lease_not_ready" if auth and auth_state != "ready" else "",
            "auth_lease_not_healthy" if auth and _lower(auth.get("lease_state")) not in {"healthy", "ready", "ok"} else "",
            "auth_lease_expiring_inside_critical_window" if expires_in and expires_in < critical_seconds else "",
            "broker_readiness_missing" if not broker else "",
            "broker_not_ready_for_open" if broker and not broker_ready else "",
            "broker_auth_not_ok" if broker and not auth_ok else "",
            "broker_network_not_ok" if broker and not network_ok else "",
            "broker_auth_probe_not_ok" if (broker or broker_state) and not probe_ok else "",
            "schwab_auth_supervisor_not_ready" if supervisor and supervisor_state != "ready" else "",
        ]
    )
    paper_soak_managed_blocker_set = {
        "auth_lease_not_ready",
        "auth_lease_not_healthy",
        "broker_auth_not_ok",
        "broker_auth_probe_not_ok",
        "schwab_auth_supervisor_not_ready",
    }
    managed_auth_blockers = [
        item
        for item in raw_blockers
        if paper_soak_auth_operable and item in paper_soak_managed_blocker_set
    ]
    blockers = [item for item in raw_blockers if item not in set(managed_auth_blockers)]
    warnings = ordered_unique(
        [
            "auth_lease_stale" if auth_stale else "",
            "broker_readiness_stale" if broker_stale else "",
            "schwab_auth_supervisor_missing" if not supervisor else "",
            "schwab_auth_supervisor_stale" if supervisor_stale else "",
            "paper_soak_auth_grace_active" if managed_auth_blockers else "",
        ]
    )
    state = "blocked" if blockers else "ready" if managed_auth_blockers else "degraded" if warnings else "ready"
    return _contract_row(
        surface_id="guard:broker_auth_contract",
        surface="broker_auth_contract",
        state=state,
        summary=(
            f"auth={auth_state} broker_ready={broker_ready} auth_ok={auth_ok} network_ok={network_ok} "
            f"blockers={len(blockers)} managed_auth_blockers={len(managed_auth_blockers)} warnings={len(warnings)}"
        ),
        metrics={
            "auth_lease": _artifact_metric(auth_path, project_root, auth_age, auth_stale, max_age),
            "broker_readiness": _artifact_metric(broker_path, project_root, broker_age, broker_stale, max_age),
            "schwab_auth_supervisor": _artifact_metric(supervisor_path, project_root, supervisor_age, supervisor_stale, max_age),
            "lease_state": str(auth.get("lease_state") or ""),
            "expires_in_seconds": round(expires_in, 3),
            "critical_lease_seconds": round(critical_seconds, 3),
            "broker_ready": broker_ready,
            "auth_ok": auth_ok,
            "network_ok": network_ok,
            "probe_ok": probe_ok,
            "paper_soak_auth_operable": paper_soak_auth_operable,
            "blockers": blockers,
            "raw_blockers": raw_blockers,
            "managed_auth_blockers": managed_auth_blockers,
            "warnings": warnings,
        },
        recommended_command=["./scripts/ops/opsctl.sh", "schwab-auth-supervisor", "--apply", "--json"],
    )


def _source_verification_contract_surface(project_root: Path, loader: ArtifactLoader, max_artifact_age_minutes: int) -> dict[str, Any]:
    max_age = min(int(max_artifact_age_minutes), 90)
    path, payload, age, stale = _artifact_payload(project_root, "source_verification_latest.json", loader=loader, max_age_minutes=max_age)
    overall = _as_dict(payload.get("overall"))
    counts = _as_dict(overall.get("counts"))
    degraded = _as_list(payload.get("degraded_artifacts"))
    stale_artifacts = _as_list(payload.get("stale_artifacts"))
    unverified = _as_list(payload.get("unverified_sources") or overall.get("unverified_sources"))
    low_confidence = _as_list(overall.get("low_confidence_sources"))
    source_rows = [row for row in _as_list(payload.get("sources")) if isinstance(row, dict)]
    verified_source_ids = {
        str(row.get("source_id") or "").strip()
        for row in source_rows
        if _lower(row.get("verification_status")) in {"cross_verified", "single_source_verified"}
        and str(row.get("source_id") or "").strip()
    }
    unverified_degraded = [item for item in degraded if str(item or "").strip() not in verified_source_ids]
    total_sources = _safe_int(overall.get("total_sources"), _safe_int(payload.get("total_sources"), 0))
    min_confidence = _safe_float(overall.get("min_source_confidence_score"), 0.0)
    all_verified = bool(overall.get("all_verified", False))
    status = _canonical_state(payload.get("overall_status"), ok=payload.get("ok")) if payload else "blocked"
    verified_source_count = _safe_int(counts.get("cross_verified"), 0) + _safe_int(counts.get("single_source_verified"), 0)
    optional_context_source_debt = bool(
        payload
        and (
            _guarded_paper_operational(project_root, loader)
            or _read_only_source_advisory_context(project_root, loader)
        )
        and unverified
        and _optional_source_ids_only(unverified)
        and (not unverified_degraded or _optional_source_ids_only(unverified_degraded))
        and (not stale_artifacts or _optional_source_ids_only(stale_artifacts))
        and (not low_confidence or _optional_source_ids_only(low_confidence))
        and verified_source_count >= 6
        and total_sources >= verified_source_count
    )
    verified_redundancy_advisory = bool(
        payload
        and status == "ready"
        and payload.get("ok") is not False
        and all_verified
        and degraded
        and not unverified
        and not unverified_degraded
        and not stale_artifacts
        and not low_confidence
        and min_confidence >= 0.70
        and total_sources > 0
        and verified_source_count == total_sources
    )

    blockers = ordered_unique(
        [
            "source_verification_missing" if not payload else "",
            "source_verification_not_ready" if payload and status != "ready" else "",
            "source_verification_unverified_sources" if unverified else "",
            "source_verification_degraded_artifacts" if degraded else "",
            "source_verification_not_all_verified" if payload and not all_verified else "",
            "source_verification_no_sources" if payload and total_sources <= 0 else "",
        ]
    )
    warnings = ordered_unique(
        [
            "source_verification_stale" if stale else "",
            "source_verification_stale_artifacts" if stale_artifacts else "",
            "source_verification_low_confidence_sources" if low_confidence else "",
            "source_verification_min_confidence_below_floor" if payload and min_confidence and min_confidence < 0.70 else "",
            "source_verification_optional_context_debt_for_guarded_paper" if optional_context_source_debt else "",
        ]
    )
    hard_blockers = blockers
    if optional_context_source_debt:
        hard_blockers = [item for item in blockers if item in {"source_verification_missing", "source_verification_no_sources"}]
        warnings = ordered_unique(warnings + [item for item in blockers if item not in set(hard_blockers)])
    elif verified_redundancy_advisory:
        hard_blockers = [item for item in blockers if item != "source_verification_degraded_artifacts"]
        warnings = ordered_unique(warnings + ["source_verification_verified_redundancy_warning"])
    optional_context_advisory_only = bool(
        optional_context_source_debt
        and not hard_blockers
        and set(warnings).issubset(OPTIONAL_SOURCE_DEBT_WARNING_TOKENS)
    )
    source_warning_advisory_only = bool(optional_context_advisory_only or verified_redundancy_advisory)
    state = "blocked" if hard_blockers else "ready" if source_warning_advisory_only else "degraded" if warnings else "ready"
    return _contract_row(
        surface_id="guard:source_verification_contract",
        surface="source_verification_contract",
        state=state,
        summary=f"status={status} total_sources={total_sources} unverified={len(unverified)} degraded={len(degraded)} stale={len(stale_artifacts)}",
        metrics={
            **_artifact_metric(path, project_root, age, stale, max_age),
            "total_sources": total_sources,
            "cross_verified_count": _safe_int(counts.get("cross_verified"), 0),
            "single_source_verified_count": _safe_int(counts.get("single_source_verified"), 0),
            "unverified_count": len(unverified),
            "degraded_artifact_count": len(degraded),
            "verified_warning_artifact_count": max(len(degraded) - len(unverified_degraded), 0),
            "unverified_degraded_artifact_count": len(unverified_degraded),
            "stale_artifact_count": len(stale_artifacts),
            "min_source_confidence_score": min_confidence,
            "optional_context_source_debt": optional_context_source_debt,
            "optional_context_advisory_only": optional_context_advisory_only,
            "verified_redundancy_advisory": verified_redundancy_advisory,
            "source_warning_advisory_only": source_warning_advisory_only,
            "read_only_source_advisory_context": _read_only_source_advisory_context(project_root, loader),
            "blockers": hard_blockers,
            "warnings": warnings,
        },
        recommended_command=["./scripts/ops/opsctl.sh", "source-verification-refresh", "--apply", "--json"],
    )


def _livefeed_contract_surface(project_root: Path, loader: ArtifactLoader, max_artifact_age_minutes: int) -> dict[str, Any]:
    guard_max_age = min(int(max_artifact_age_minutes), 20)
    local_max_age = min(int(max_artifact_age_minutes), 10)
    guard_path, guard, guard_age, guard_stale = _artifact_payload(project_root, "livefeed_refresh_guard_latest.json", loader=loader, max_age_minutes=guard_max_age)
    local_path, local, local_age, local_stale = _artifact_payload(project_root, "livefeed_local_latest.json", loader=loader, max_age_minutes=local_max_age)
    guard_state = _canonical_state(guard.get("overall_status"), ok=guard.get("ok")) if guard else "blocked"
    route_checks = _as_list(guard.get("route_checks"))
    route_failed = [row for row in route_checks if isinstance(row, dict) and not bool(row.get("ok", False))]
    running = _lower(local.get("status")) == "running"
    alive = _bool(local.get("alive", False))
    health_writer = _bool(local.get("health_writer", False))
    writer_mode = str(local.get("writer_mode") or "")
    skipped = _safe_int(local.get("skipped_file_count", local.get("skipped_unreadable_count")), 0)
    stale_sources = _safe_int(local.get("stale_count"), 0)
    refresh_guard_maintenance_due = bool(guard and guard_stale)
    refresh_guard_staleness_managed = bool(
        refresh_guard_maintenance_due
        and guard_state == "ready"
        and not route_failed
        and guard_age is not None
        and guard_age <= 60.0
        and local
        and not local_stale
        and running
        and alive
        and health_writer
        and writer_mode == "local_mirror"
        and skipped == 0
        and stale_sources == 0
    )

    blockers = ordered_unique(
        [
            "livefeed_refresh_guard_missing" if not guard else "",
            "livefeed_refresh_guard_not_ready" if guard and guard_state != "ready" else "",
            "livefeed_refresh_routes_failed" if route_failed else "",
            "livefeed_local_health_missing" if not local else "",
            "livefeed_local_not_running" if local and not running else "",
            "livefeed_local_not_alive" if local and not alive else "",
            "livefeed_health_writer_not_local_mirror" if local and (not health_writer or writer_mode != "local_mirror") else "",
        ]
    )
    warnings = ordered_unique(
        [
            "livefeed_refresh_guard_stale" if guard_stale and not refresh_guard_staleness_managed else "",
            "livefeed_local_health_stale" if local_stale else "",
            "livefeed_skipped_unreadable_files" if skipped > 0 else "",
            "livefeed_stale_sources" if stale_sources > 0 else "",
        ]
    )
    state = "blocked" if blockers else "degraded" if warnings else "ready"
    return _contract_row(
        surface_id="guard:livefeed_visibility_contract",
        surface="livefeed_visibility_contract",
        state=state,
        summary=f"guard={guard_state} running={running} alive={alive} writer_mode={writer_mode or 'missing'} failed_routes={len(route_failed)}",
        metrics={
            "livefeed_refresh_guard": _artifact_metric(guard_path, project_root, guard_age, guard_stale, guard_max_age),
            "livefeed_local": _artifact_metric(local_path, project_root, local_age, local_stale, local_max_age),
            "route_check_count": len(route_checks),
            "route_failed_count": len(route_failed),
            "running": running,
            "alive": alive,
            "health_writer": health_writer,
            "writer_mode": writer_mode,
            "skipped_unreadable_count": skipped,
            "stale_count": stale_sources,
            "refresh_guard_maintenance_due": refresh_guard_maintenance_due,
            "refresh_guard_staleness_managed": refresh_guard_staleness_managed,
            "refresh_guard_hard_age_minutes": 60,
            "blockers": blockers,
            "warnings": warnings,
        },
        recommended_command=["./scripts/ops/opsctl.sh", "livefeed-refresh-guard", "--apply", "--json"],
    )


def _memory_truth_contract_surface(project_root: Path, loader: ArtifactLoader, max_artifact_age_minutes: int) -> dict[str, Any]:
    max_age = min(int(max_artifact_age_minutes), 30)
    efficiency_path, efficiency, efficiency_age, efficiency_stale = _artifact_payload(
        project_root,
        "memory_efficiency_control_latest.json",
        loader=loader,
        max_age_minutes=max_age,
    )
    pressure_path, pressure, pressure_age, pressure_stale = _artifact_payload(
        project_root,
        "memory_pressure_intelligence_latest.json",
        loader=loader,
        max_age_minutes=max_age,
    )
    swap_path, swap, swap_age, swap_stale = _artifact_payload(project_root, "swap_pressure_governor_latest.json", loader=loader, max_age_minutes=max_age)
    efficiency_state = _canonical_state(efficiency.get("overall_status"), ok=efficiency.get("ok")) if efficiency else "blocked"
    pressure_state = _canonical_state(pressure.get("overall_status"), ok=pressure.get("ok")) if pressure else "blocked"
    swap_state = _canonical_state(swap.get("overall_status"), ok=swap.get("ok")) if swap else "blocked"
    memory_snapshot = _as_dict(efficiency.get("memory_snapshot"))
    raw_memory = _as_dict(efficiency.get("raw_memory_snapshot"))
    reconciliation = _as_dict(efficiency.get("memory_truth_reconciliation"))
    pressure_snapshot = _as_dict(pressure.get("snapshot"))
    pressure_reconciliation = _as_dict(pressure_snapshot.get("memory_truth_reconciliation"))
    swap_pressure = _as_dict(swap.get("swap_pressure"))
    swap_thresholds = _as_dict(swap_pressure.get("thresholds"))
    swap_pressure_tier = _lower(swap_pressure.get("tier"))
    calm_swap_limit = max(_safe_float(swap_thresholds.get("calm_swap_gb"), 10.0), 4.0)
    reopen_gate = _as_dict(pressure.get("reopen_gate"))
    classification = _as_dict(pressure.get("classification"))
    reasons = [str(item) for item in _as_list(efficiency.get("reasons"))]
    effective_swap = _safe_float(memory_snapshot.get("swap_used_gb"), 0.0)
    raw_swap = _safe_float(raw_memory.get("swap_used_gb"), effective_swap)
    effective_compressed = _safe_float(memory_snapshot.get("compressed_store_gb"), 0.0)
    raw_compressed = _safe_float(raw_memory.get("compressed_store_gb"), effective_compressed)
    high_water_gap = bool(raw_swap > effective_swap + 1.0 or raw_compressed > effective_compressed + 4.0)
    reconciliation_active = bool(reconciliation.get("active", False))
    pressure_reconciliation_active = bool(pressure_reconciliation.get("active", False))
    stale_high_water_guarded = bool(not high_water_gap or (reconciliation_active and pressure_reconciliation_active))
    bad_reason_tokens = {"compressed_memory_high", "swap_usage_high", "memory_pressure_red", "memory_pressure_yellow"}
    stale_reason_regression = any(reason in bad_reason_tokens for reason in reasons) and _lower(memory_snapshot.get("memory_pressure_state")) == "green"
    memory_telemetry_green = bool(
        _lower(memory_snapshot.get("memory_pressure_state")) == "green"
        and _lower(memory_snapshot.get("memory_pressure_kind")) in {"", "none", "normal"}
        and swap_pressure_tier in {"", "normal", "calm", "ready"}
        and raw_swap <= calm_swap_limit
        and effective_swap <= calm_swap_limit
        and _safe_float(pressure_snapshot.get("pages_throttled"), 0.0) <= 0.0
    )
    classification_status = _lower(classification.get("status"))
    classification_soft_or_clear = bool(
        classification_status in {
            "soft_guard",
            "foreground_headroom",
            "clear",
            "ready",
            "advisory",
        }
        or (_lower(pressure.get("overall_status")) == "advisory" and memory_telemetry_green)
    )
    memory_efficiency_advisory_ready = bool(
        efficiency_state == "degraded"
        and _lower(efficiency.get("overall_status")) == "advisory"
        and _bool(efficiency.get("ok", False))
        and memory_telemetry_green
        and swap_state == "ready"
        and classification_soft_or_clear
        and not any(reason in bad_reason_tokens for reason in reasons)
    )
    memory_pressure_advisory_ready = bool(
        pressure
        and pressure_state == "degraded"
        and _lower(pressure.get("overall_status")) == "advisory"
        and efficiency_state == "ready"
        and swap_state == "ready"
        and memory_telemetry_green
        and classification_soft_or_clear
        and (
            _bool(reopen_gate.get("safe_to_widen_p_core_workers", False))
            or _bool(reopen_gate.get("safe_for_training", False))
            or _bool(reopen_gate.get("small_batch_training_safe", False))
            or _bool(reopen_gate.get("small_canary_training_safe", False))
        )
    )
    memory_soft_guard_for_paper_soak = bool(
        _guarded_paper_operational(project_root, loader)
        and efficiency
        and pressure
        and swap
        and swap_state == "ready"
        and memory_telemetry_green
        and classification_soft_or_clear
    )

    blockers = ordered_unique(
        [
            "memory_efficiency_missing" if not efficiency else "",
            "memory_pressure_intelligence_missing" if not pressure else "",
            "swap_pressure_governor_missing" if not swap else "",
            "memory_efficiency_not_ready"
            if efficiency and efficiency_state != "ready" and not memory_efficiency_advisory_ready
            else "",
            "memory_pressure_intelligence_not_ready" if pressure and pressure_state != "ready" and not memory_pressure_advisory_ready else "",
            "swap_pressure_governor_not_ready" if swap and swap_state != "ready" else "",
            "stale_high_water_memory_not_reconciled" if high_water_gap and not stale_high_water_guarded else "",
            "green_memory_has_high_pressure_reason" if stale_reason_regression else "",
        ]
    )
    warnings = ordered_unique(
        [
            "memory_efficiency_stale" if efficiency_stale else "",
            "memory_pressure_intelligence_stale" if pressure_stale else "",
            "swap_pressure_governor_stale" if swap_stale else "",
            "memory_pressure_not_clear"
            if pressure and _lower(classification.get("status")) not in {"clear", "ready"} and not memory_pressure_advisory_ready
            else "",
            "memory_pressure_advisory_ready" if memory_pressure_advisory_ready else "",
            "memory_efficiency_advisory_ready" if memory_efficiency_advisory_ready else "",
            "training_not_open_despite_ready_memory"
            if pressure and pressure_state == "ready" and not _bool(reopen_gate.get("safe_for_training", False))
            else "",
            "memory_soft_guard_for_guarded_paper" if memory_soft_guard_for_paper_soak else "",
        ]
    )
    hard_blockers = blockers
    if memory_soft_guard_for_paper_soak:
        soft_guard_blockers = {
            "memory_efficiency_not_ready",
            "memory_pressure_intelligence_not_ready",
            "green_memory_has_high_pressure_reason",
        }
        hard_blockers = [item for item in blockers if item not in soft_guard_blockers]
        warnings = ordered_unique(warnings + [item for item in blockers if item in soft_guard_blockers])
    paper_soak_soft_guard_warning_tokens = {
        "memory_efficiency_stale",
        "swap_pressure_governor_stale",
        "memory_pressure_advisory_ready",
        "memory_efficiency_advisory_ready",
        "memory_pressure_not_clear",
        "memory_soft_guard_for_guarded_paper",
        "memory_efficiency_not_ready",
        "memory_pressure_intelligence_not_ready",
        "green_memory_has_high_pressure_reason",
        "training_not_open_despite_ready_memory",
    }
    memory_advisory_ready_warning_tokens = {
        "memory_pressure_advisory_ready",
        "memory_efficiency_advisory_ready",
    }
    paper_soak_soft_guard_advisory_only = bool(
        memory_soft_guard_for_paper_soak
        and not hard_blockers
        and set(warnings).issubset(paper_soak_soft_guard_warning_tokens)
    )
    memory_advisory_ready_only = bool(
        (memory_pressure_advisory_ready or memory_efficiency_advisory_ready)
        and not hard_blockers
        and set(warnings).issubset(memory_advisory_ready_warning_tokens)
    )
    state = "blocked" if hard_blockers else "ready" if (paper_soak_soft_guard_advisory_only or memory_advisory_ready_only) else "degraded" if warnings else "ready"
    return _contract_row(
        surface_id="guard:memory_truth_contract",
        surface="memory_truth_contract",
        state=state,
        summary=(
            f"efficiency={'advisory_ready' if memory_efficiency_advisory_ready else efficiency_state} "
            f"pressure={pressure_state} swap={swap_state} "
            f"raw_swap={raw_swap:.3f} effective_swap={effective_swap:.3f} reconciliation={reconciliation_active}"
        ),
        metrics={
            "memory_efficiency": _artifact_metric(efficiency_path, project_root, efficiency_age, efficiency_stale, max_age),
            "memory_pressure_intelligence": _artifact_metric(pressure_path, project_root, pressure_age, pressure_stale, max_age),
            "swap_pressure_governor": _artifact_metric(swap_path, project_root, swap_age, swap_stale, max_age),
            "recommended_profile": str(efficiency.get("recommended_profile") or ""),
            "memory_pressure_state": str(memory_snapshot.get("memory_pressure_state") or ""),
            "memory_pressure_kind": str(memory_snapshot.get("memory_pressure_kind") or ""),
            "raw_swap_used_gb": raw_swap,
            "effective_swap_used_gb": effective_swap,
            "raw_compressed_store_gb": raw_compressed,
            "effective_compressed_store_gb": effective_compressed,
            "high_water_gap": high_water_gap,
            "memory_truth_reconciliation_active": reconciliation_active,
            "pressure_reconciliation_active": pressure_reconciliation_active,
            "classification_status": str(classification.get("status") or ""),
            "classification_soft_or_clear": classification_soft_or_clear,
            "safe_to_widen_p_core_workers": _bool(reopen_gate.get("safe_to_widen_p_core_workers", False)),
            "safe_for_training": _bool(reopen_gate.get("safe_for_training", False)),
            "memory_pressure_advisory_ready": memory_pressure_advisory_ready,
            "memory_efficiency_advisory_ready": memory_efficiency_advisory_ready,
            "swap_pressure_tier": swap_pressure_tier,
            "calm_swap_limit_gb": calm_swap_limit,
            "memory_soft_guard_for_paper_soak": memory_soft_guard_for_paper_soak,
            "paper_soak_soft_guard_advisory_only": paper_soak_soft_guard_advisory_only,
            "memory_advisory_ready_only": memory_advisory_ready_only,
            "blockers": hard_blockers,
            "warnings": warnings,
        },
        recommended_command=["./scripts/ops/opsctl.sh", "memory-efficiency", "apply", "--json"],
    )


def _bounded_transient_storage_drain_managed(
    storage: dict[str, Any],
    *,
    storage_state: str,
    severity: str,
    pressure_index: float,
    total_pending: int,
    core_pending: int,
    support_pending: int,
    oldest: float,
    pending_threshold: int,
    oldest_threshold: float,
    backpressure_quality: float,
) -> bool:
    bounded = _as_dict(storage.get("bounded_recovery_contract"))
    route = _as_dict(storage.get("external_route_verification"))
    integrity = _as_dict(storage.get("data_integrity"))
    writer = _as_dict(storage.get("writer_shedding"))
    return bool(
        storage_state == "ready"
        and severity in {"", "stable", "ready", "low", "normal"}
        and 0.5 <= pressure_index < BOUNDED_TRANSIENT_STORAGE_PRESSURE_MAX
        and total_pending <= min(pending_threshold, BOUNDED_TRANSIENT_STORAGE_TOTAL_MAX)
        and core_pending <= min(pending_threshold, BOUNDED_TRANSIENT_STORAGE_CORE_MAX)
        and support_pending <= BOUNDED_TRANSIENT_STORAGE_SUPPORT_MAX
        and oldest <= min(oldest_threshold, BOUNDED_TRANSIENT_STORAGE_AGE_MAX_SECONDS)
        and bool(bounded.get("route_verified", False))
        and _lower(route.get("verification_state")) in STORAGE_ROUTE_READY_STATES
        and bool(bounded.get("active_drain_progress", False) or bounded.get("drain_delta_signal_observed", False))
        and not bool(bounded.get("hard_gate_active", False))
        and not bool(bounded.get("effective_hard_gate_active", False))
        and not _as_list(writer.get("hard_breaches"))
        and not _as_list(writer.get("elevated_breaches"))
        and all(
            _safe_int(integrity.get(key), 0) == 0
            for key in (
                "sql_invalid_lines",
                "sql_overlay_invalid_lines",
                "sql_overlay_oversize_payloads",
                "sql_overlay_ops_write_failures",
            )
        )
        and backpressure_quality >= 95.0
    )


def _bounded_steady_state_storage_managed(
    storage: dict[str, Any],
    *,
    storage_state: str,
    severity: str,
    pressure_index: float,
    total_pending: int,
    core_pending: int,
    support_pending: int,
    oldest: float,
    backpressure_quality: float,
) -> bool:
    continuous_soak = _as_dict(storage.get("continuous_run_soak_contract"))
    route = _as_dict(storage.get("external_route_verification"))
    integrity = _as_dict(storage.get("data_integrity"))
    writer = _as_dict(storage.get("writer_shedding"))
    disk = _as_dict(_as_dict(storage.get("storage_plane_contract")).get("disk_contract"))
    return bool(
        storage_state == "ready"
        and _bool(storage.get("ok", False))
        and severity in {"", "stable", "ready", "low", "normal", "calm"}
        and 0.35 <= pressure_index <= 0.85
        and total_pending <= 5_000
        and core_pending <= 2_500
        and support_pending <= 2_500
        and oldest <= BOUNDED_TRANSIENT_STORAGE_AGE_MAX_SECONDS
        and bool(continuous_soak)
        and _bool(continuous_soak.get("soak_ready", False))
        and not _as_list(continuous_soak.get("blockers"))
        and _lower(route.get("verification_state")) in STORAGE_ROUTE_READY_STATES
        and bool(integrity)
        and all(
            _safe_int(integrity.get(key), 0) == 0
            for key in (
                "sql_invalid_lines",
                "sql_overlay_invalid_lines",
                "sql_overlay_oversize_payloads",
                "sql_overlay_ops_write_failures",
            )
        )
        and not _as_list(writer.get("hard_breaches"))
        and not _as_list(writer.get("elevated_breaches"))
        and not bool(disk.get("emergency_disk_guard", False))
        and backpressure_quality >= 95.0
    )


def _runtime_storage_contract_surface(project_root: Path, loader: ArtifactLoader, max_artifact_age_minutes: int) -> dict[str, Any]:
    max_age = min(int(max_artifact_age_minutes), 30)
    runtime_path, runtime, runtime_age, runtime_stale = _artifact_payload(project_root, "runtime_throttle_control_latest.json", loader=loader, max_age_minutes=max_age)
    paper_path, paper, paper_age, paper_stale = _artifact_payload(project_root, "paper_400_ramp_latest.json", loader=loader, max_age_minutes=max_age)
    storage_path, storage, storage_age, storage_stale = _artifact_payload(
        project_root,
        "ingestion_storage_control_latest.json",
        loader=loader,
        max_age_minutes=max_age,
    )
    runtime_state = _canonical_state(runtime.get("overall_status"), ok=runtime.get("ok")) if runtime else "blocked"
    storage_state = _canonical_state(storage.get("overall_status"), ok=storage.get("ok")) if storage else "blocked"
    backpressure = _as_dict(storage.get("backpressure"))
    storage_section = _as_dict(storage.get("storage"))
    storage_route = _as_dict(storage.get("external_route_verification"))
    storage_integrity = _as_dict(storage.get("data_integrity"))
    writer_shedding = _as_dict(storage.get("writer_shedding"))
    raw_total_pending = _safe_int(backpressure.get("total_pending_lines"), _safe_int(backpressure.get("core_pending_lines"), 0))
    raw_core_pending = _safe_int(backpressure.get("core_pending_lines"), raw_total_pending)
    raw_deferred_pending = _safe_int(backpressure.get("deferred_pending_lines"), 0)
    raw_support_pending = _safe_int(backpressure.get("support_pending_lines"), 0)
    raw_oldest = _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0)
    small_residual_drain_managed = bool(
        storage_state == "ready"
        and _lower(storage.get("severity")) in {"", "stable", "ready"}
        and raw_deferred_pending > 0
        and _lower(storage_section.get("backlog_drain_status") or storage.get("backlog_drain_status"))
        in {"drain_active", "ready", "steady_state"}
        and raw_total_pending <= 1000
        and raw_core_pending <= 5000
        and raw_support_pending <= 12000
        and raw_oldest <= 300.0
        and _lower(storage_route.get("verification_state")) in {"ready", "verified", "ok"}
        and all(
            _safe_int(storage_integrity.get(key), 0) == 0
            for key in (
                "sql_invalid_lines",
                "sql_overlay_invalid_lines",
                "sql_overlay_oversize_payloads",
                "sql_overlay_ops_write_failures",
            )
        )
        and not _as_list(writer_shedding.get("hard_breaches"))
    )
    managed_pressure_view = bool(
        _bool(backpressure.get("managed_support_overlay_backlog", False))
        or _bool(backpressure.get("overlay_pressure_clear", False))
        or _bool(_as_dict(backpressure.get("managed_tiny_hot_tail")).get("active", False))
        or small_residual_drain_managed
    )
    total_pending = (
        _safe_int(backpressure.get("pressure_total_pending_lines"), raw_total_pending)
        if managed_pressure_view
        else raw_total_pending
    )
    pending_threshold = max(_safe_int(backpressure.get("pending_lines_threshold"), 15000), 1)
    oldest = (
        _safe_float(backpressure.get("pressure_oldest_pending_age_seconds"), raw_oldest)
        if managed_pressure_view
        else raw_oldest
    )
    oldest_threshold = max(_safe_float(backpressure.get("oldest_age_threshold_seconds"), 240.0), 1.0)
    storage_contract = _as_dict(storage.get("storage_plane_contract"))
    disk_contract = _as_dict(storage_contract.get("disk_contract"))
    pressure_index = _safe_float(storage.get("pressure_index"), 0.0)
    storage_severity = _lower(storage.get("severity"))
    backpressure_quality = _safe_float(
        storage.get("backpressure_quality_score"),
        100.0 if storage_state == "ready" else 0.0,
    )
    bounded_transient_drain_managed = _bounded_transient_storage_drain_managed(
        storage,
        storage_state=storage_state,
        severity=storage_severity,
        pressure_index=pressure_index,
        total_pending=raw_total_pending,
        core_pending=raw_core_pending,
        support_pending=raw_support_pending,
        oldest=raw_oldest,
        pending_threshold=pending_threshold,
        oldest_threshold=oldest_threshold,
        backpressure_quality=backpressure_quality,
    )
    bounded_steady_state_storage_managed = _bounded_steady_state_storage_managed(
        storage,
        storage_state=storage_state,
        severity=storage_severity,
        pressure_index=pressure_index,
        total_pending=raw_total_pending,
        core_pending=raw_core_pending,
        support_pending=raw_support_pending,
        oldest=raw_oldest,
        backpressure_quality=backpressure_quality,
    )
    compute_level = _lower(runtime.get("compute_pressure_level"))
    memory_level = _lower(runtime.get("memory_pressure_level"))
    host_saturation = _safe_float(runtime.get("host_saturation_score"), 0.0)
    storage_overlay = _as_dict(_as_dict(runtime.get("soft_cap_advisory_reclassification")).get("measurements")).get("storage_overlay_relief")
    soft_cap = _as_dict(runtime.get("soft_cap_advisory_reclassification"))
    runtime_measurements = _as_dict(soft_cap.get("measurements"))
    host_attribution = _as_dict(runtime.get("host_pressure_attribution"))
    if not host_attribution:
        host_attribution = _as_dict(runtime_measurements.get("host_pressure_attribution"))
    paper_policy = _as_dict(runtime.get("paper_execution_policy"))
    if not paper_policy:
        paper_policy = _as_dict(runtime_measurements.get("paper_execution_policy"))
    live_policy = _as_dict(_as_dict(runtime.get("runtime_saturation_governor_v2")).get("paper_live_data_policy"))
    paper_gate = _as_dict(_as_dict(paper.get("gates")).get("runtime")) if paper else {}
    paper_gate_blockers = _as_list(paper_gate.get("blockers"))
    paper_armed_clean = bool(
        paper
        and _bool(paper.get("ok", False))
        and _bool(paper.get("armed", False))
        and _lower(paper.get("stage")) == "armed"
        and not _as_list(paper.get("blockers"))
        and not paper_stale
    )
    paper_execution_open = bool(
        bool(paper_policy.get("paper_execution_allowed", runtime_measurements.get("paper_execution_allowed", False)))
        and not bool(paper_policy.get("pause_paper_execution", runtime_measurements.get("paper_execution_paused", False)))
        and (
            not live_policy
            or (
                bool(live_policy.get("paper_execution_allowed", True))
                and not bool(live_policy.get("paper_execution_consumer_paused", False))
            )
        )
    )
    ramp_capacity_limited_armed = bool(
        paper_armed_clean
        and _bool(paper_gate.get("capacity_limited_armed", False))
        and _bool(paper_gate.get("runtime_capacity_ready", False))
        and _bool(paper_gate.get("paper_execution_clean", False))
        and _bool(paper_gate.get("live_execution_locked", False))
        and not paper_gate_blockers
    )
    overlay_active = bool(_as_dict(storage_overlay).get("active", False) or managed_pressure_view)
    storage_clear = bool(
        storage_state == "ready"
        and (
            pressure_index < 0.5
            or small_residual_drain_managed
            or bounded_transient_drain_managed
            or bounded_steady_state_storage_managed
        )
        and total_pending <= pending_threshold
        and oldest <= oldest_threshold
    )
    external_pressure_advisory = bool(
        runtime
        and compute_level == "high"
        and memory_level == "normal"
        and storage_clear
        and not runtime_stale
        and bool(host_attribution.get("external_pressure_dominant", runtime_measurements.get("external_pressure_dominant", False)))
        and not bool(host_attribution.get("bot_owned_pressure_dominant", runtime_measurements.get("bot_owned_pressure_dominant", False)))
        and bool(paper_policy.get("paper_execution_allowed", runtime_measurements.get("paper_execution_allowed", False)))
        and not bool(paper_policy.get("pause_paper_execution", runtime_measurements.get("paper_execution_paused", False)))
    )
    external_paper_soak_advisory = bool(
        external_pressure_advisory
        and paper_armed_clean
        and paper_execution_open
        and ramp_capacity_limited_armed
        and _lower(paper_gate.get("status")) == "ready"
        and not paper_gate_blockers
    )
    paper_capacity_runtime_state = runtime_state in {"ready", "advisory", "degraded"}
    armed_paper_capacity_advisory = bool(
        runtime
        and paper_capacity_runtime_state
        and compute_level in {"normal", "elevated", "high"}
        and memory_level == "normal"
        and storage_clear
        and not runtime_stale
        and paper_execution_open
        and ramp_capacity_limited_armed
        and host_saturation < 75.0
    )
    capacity_limited_paper_advisory = bool(
        armed_paper_capacity_advisory
        or (
            runtime
            and paper_capacity_runtime_state
            and compute_level in {"elevated", "high"}
            and memory_level == "normal"
            and storage_clear
            and not runtime_stale
            and paper_execution_open
            and bool(paper_policy.get("capacity_limited_paper_execution", runtime_measurements.get("capacity_limited_paper_execution", False)))
        )
    )
    managed_high_compute_advisory = bool(
        runtime
        and (runtime_state == "degraded" or (runtime_state == "blocked" and external_pressure_advisory))
        and compute_level == "high"
        and memory_level == "normal"
        and storage_clear
        and not runtime_stale
        and (
            (
                _bool(runtime.get("ok"))
                and _lower(runtime.get("overall_status")) == "advisory"
                and host_saturation < 62.0
                and (
                    bool(soft_cap.get("active", False))
                    or _lower(soft_cap.get("reason"))
                    in {
                        "research_training_pressure_is_already_niced_and_guarded_advisory",
                        "external_high_compute_pressure_is_capacity_limited_advisory_not_bot_runtime_degradation",
                    }
                )
            )
            or external_pressure_advisory
        )
    )
    runtime_advisory_ready = bool(
        external_pressure_advisory
        or capacity_limited_paper_advisory
        or (
            runtime
            and _bool(runtime.get("ok"))
            and _lower(runtime.get("overall_status")) == "advisory"
            and runtime_state == "degraded"
            and (compute_level in {"normal", "elevated"} or managed_high_compute_advisory)
            and memory_level == "normal"
            and storage_clear
            and not runtime_stale
        )
    )
    guarded_runtime_ready_saturation = bool(
        runtime
        and runtime_state == "ready"
        and _bool(runtime.get("ok"))
        and not runtime_stale
        and memory_level == "normal"
        and storage_clear
        and _bool(soft_cap.get("active", False))
        and _lower(soft_cap.get("to_status")) == "ready"
        and _bool(runtime_measurements.get("runtime_ready_guarded", False))
        and host_saturation < 75.0
    )

    blockers = ordered_unique(
        [
            "runtime_throttle_missing" if not runtime else "",
            "ingestion_storage_control_missing" if not storage else "",
            "runtime_throttle_blocked" if runtime and runtime_state == "blocked" and not external_pressure_advisory else "",
            "ingestion_storage_blocked" if storage and storage_state == "blocked" else "",
            "runtime_compute_pressure_high"
            if compute_level in {"high", "critical"} and not (managed_high_compute_advisory or capacity_limited_paper_advisory)
            else "",
            "runtime_memory_pressure_high" if memory_level in {"high", "critical"} else "",
            "storage_pending_above_threshold_without_overlay" if total_pending > pending_threshold and not overlay_active else "",
            "storage_oldest_pending_above_threshold_without_overlay" if oldest > oldest_threshold and not overlay_active else "",
            "storage_disk_emergency_guard" if bool(disk_contract.get("emergency_disk_guard", False)) else "",
        ]
    )
    warnings = ordered_unique(
        [
            "runtime_throttle_stale" if runtime_stale else "",
            "ingestion_storage_control_stale" if storage_stale else "",
            "runtime_throttle_degraded" if runtime and runtime_state == "degraded" and not runtime_advisory_ready else "",
            "ingestion_storage_degraded" if storage and storage_state == "degraded" else "",
            "storage_pressure_index_elevated"
            if pressure_index > 0.50
            and not (
                small_residual_drain_managed
                or bounded_transient_drain_managed
                or bounded_steady_state_storage_managed
            )
            else "",
            "host_saturation_elevated"
            if host_saturation > 60.0
            and not (
                capacity_limited_paper_advisory
                or guarded_runtime_ready_saturation
                or external_paper_soak_advisory
            )
            else "",
        ]
    )
    state = "blocked" if blockers else "degraded" if warnings else "ready"
    return _contract_row(
        surface_id="guard:runtime_storage_contract",
        surface="runtime_storage_contract",
        state=state,
        summary=(
            f"runtime={runtime_state} storage={storage_state} pending={total_pending}/{pending_threshold} "
            f"compute={compute_level or 'unknown'} memory={memory_level or 'unknown'}"
        ),
        metrics={
            "runtime_throttle": _artifact_metric(runtime_path, project_root, runtime_age, runtime_stale, max_age),
            "paper_400_ramp": _artifact_metric(paper_path, project_root, paper_age, paper_stale, max_age),
            "ingestion_storage_control": _artifact_metric(storage_path, project_root, storage_age, storage_stale, max_age),
            "throttle_profile": str(runtime.get("throttle_profile") or ""),
            "host_saturation_score": host_saturation,
            "compute_pressure_level": compute_level,
            "memory_pressure_level": memory_level,
            "runtime_advisory_ready": runtime_advisory_ready,
            "guarded_runtime_ready_saturation": guarded_runtime_ready_saturation,
            "managed_high_compute_advisory": managed_high_compute_advisory,
            "external_pressure_advisory": external_pressure_advisory,
            "external_paper_soak_advisory": external_paper_soak_advisory,
            "capacity_limited_paper_advisory": capacity_limited_paper_advisory,
            "armed_paper_capacity_advisory": armed_paper_capacity_advisory,
            "paper_armed_clean": paper_armed_clean,
            "paper_execution_open": paper_execution_open,
            "paper_capacity_limited_armed": ramp_capacity_limited_armed,
            "paper_runtime_gate_status": str(paper_gate.get("status") or ""),
            "storage_clear": storage_clear,
            "storage_severity": storage_severity,
            "storage_pressure_index": pressure_index,
            "backpressure_quality_score": backpressure_quality,
            "total_pending_lines": total_pending,
            "raw_total_pending_lines": raw_total_pending,
            "pending_lines_threshold": pending_threshold,
            "oldest_pending_age_seconds": oldest,
            "raw_oldest_pending_age_seconds": raw_oldest,
            "oldest_age_threshold_seconds": oldest_threshold,
            "storage_overlay_relief_active": overlay_active,
            "managed_pressure_view": managed_pressure_view,
            "small_residual_drain_managed": small_residual_drain_managed,
            "bounded_transient_drain_managed": bounded_transient_drain_managed,
            "bounded_steady_state_storage_managed": bounded_steady_state_storage_managed,
            "external_available_gb": _safe_float(disk_contract.get("external_available_gb"), 0.0),
            "external_used_percent": _safe_float(disk_contract.get("external_used_percent"), 0.0),
            "blockers": blockers,
            "warnings": warnings,
        },
        recommended_command=["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"],
        quiet_hours_preferred=True,
    )


def _ingestion_storage_degradation_floor_surface(project_root: Path, loader: ArtifactLoader, max_artifact_age_minutes: int) -> dict[str, Any]:
    max_age = min(int(max_artifact_age_minutes), 30)
    storage_path, storage, storage_age, storage_stale = _artifact_payload(
        project_root,
        "ingestion_storage_control_latest.json",
        loader=loader,
        max_age_minutes=max_age,
    )
    fleet_path, fleet, fleet_age, fleet_stale = _artifact_payload(
        project_root,
        "backpressure_drainer_fleet_latest.json",
        loader=loader,
        max_age_minutes=60,
    )
    storage_state = _canonical_state(storage.get("overall_status"), ok=storage.get("ok")) if storage else "blocked"
    severity = _lower(storage.get("severity"))
    backpressure = _as_dict(storage.get("backpressure"))
    raw_total_pending = _safe_int(backpressure.get("total_pending_lines"), _safe_int(backpressure.get("core_pending_lines"), 0))
    raw_core_pending = _safe_int(backpressure.get("core_pending_lines"), raw_total_pending)
    raw_deferred_pending = _safe_int(backpressure.get("deferred_pending_lines"), 0)
    raw_support_pending = _safe_int(backpressure.get("support_pending_lines"), 0)
    managed_pressure_view = bool(
        _bool(backpressure.get("managed_support_overlay_backlog", False))
        or _bool(backpressure.get("overlay_pressure_clear", False))
        or _bool(_as_dict(backpressure.get("managed_tiny_hot_tail")).get("active", False))
    )
    total_pending = (
        _safe_int(backpressure.get("pressure_total_pending_lines"), raw_total_pending)
        if managed_pressure_view
        else raw_total_pending
    )
    core_pending = (
        _safe_int(backpressure.get("pressure_core_pending_lines"), raw_core_pending)
        if managed_pressure_view
        else raw_core_pending
    )
    deferred_pending = (
        _safe_int(backpressure.get("pressure_deferred_pending_lines"), raw_deferred_pending)
        if managed_pressure_view
        else raw_deferred_pending
    )
    support_pending = (
        _safe_int(backpressure.get("pressure_support_pending_lines"), raw_support_pending)
        if managed_pressure_view
        else raw_support_pending
    )
    pending_threshold = max(_safe_int(backpressure.get("pending_lines_threshold"), 15000), 1)
    raw_oldest = _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0)
    oldest = (
        _safe_float(backpressure.get("pressure_oldest_pending_age_seconds"), raw_oldest)
        if managed_pressure_view
        else raw_oldest
    )
    oldest_threshold = max(_safe_float(backpressure.get("oldest_age_threshold_seconds"), 240.0), 1.0)
    pressure_index = _safe_float(storage.get("pressure_index"), 0.0)
    recovery_quality = _safe_float(storage.get("recovery_quality_score"), 100.0 if storage_state == "ready" else 0.0)
    backpressure_quality = _safe_float(storage.get("backpressure_quality_score"), 100.0 if storage_state == "ready" else 0.0)

    collector = _as_dict(storage.get("collector_intake_enforcement_audit"))
    collector_status = _lower(collector.get("status"))
    collector_required = _bool(collector.get("required", False))
    collector_mismatches = _safe_int(collector.get("mismatch_count"), 0)
    backlog_relief = _as_dict(storage.get("backlog_relief_contract"))
    pcore_contract = _as_dict(backlog_relief.get("p_core_backlog_allocation_contract"))
    if not pcore_contract:
        request = _as_dict(fleet.get("service_request"))
        pcore_contract = _as_dict(request.get("p_core_backlog_allocation_contract"))
    if not pcore_contract:
        pcore_contract = _as_dict(_as_dict(fleet.get("active_drainer")).get("p_core_backlog_allocation_contract"))
    control_env = _as_dict(pcore_contract.get("control_env"))
    selected_workers = _safe_int(
        _as_dict(pcore_contract.get("p_core_burst_intelligence")).get("selected_workers"),
        _safe_int(pcore_contract.get("preprocess_worker_budget"), 0),
    )
    pcore_active = _bool(pcore_contract.get("active", False))
    single_writer = bool(
        _bool(pcore_contract.get("single_writer_only", False))
        or control_env.get("SQL_LINK_SERVICE_SINGLE_WRITER_ONLY") == "1"
        or _safe_int(pcore_contract.get("sqlite_writer_count"), 0) == 1
    )
    pcore_env_active = control_env.get("BACKLOG_PCORE_ALLOCATION_ACTIVE") == "1"

    storage_contract = _as_dict(storage.get("storage_plane_contract"))
    disk_contract = _as_dict(storage_contract.get("disk_contract"))
    storage_section = _as_dict(storage.get("storage"))
    storage_route = _as_dict(storage.get("external_route_verification"))
    storage_integrity = _as_dict(storage.get("data_integrity"))
    writer_shedding = _as_dict(storage.get("writer_shedding"))
    small_residual_drain_managed = bool(
        storage_state == "ready"
        and severity in {"", "stable", "ready"}
        and deferred_pending > 0
        and _lower(storage_section.get("backlog_drain_status") or storage.get("backlog_drain_status"))
        in {"drain_active", "ready", "steady_state"}
        and total_pending <= 1000
        and core_pending <= 5000
        and support_pending <= 12000
        and oldest <= 300.0
        and _lower(storage_route.get("verification_state")) in {"ready", "verified", "ok"}
        and all(
            _safe_int(storage_integrity.get(key), 0) == 0
            for key in (
                "sql_invalid_lines",
                "sql_overlay_invalid_lines",
                "sql_overlay_oversize_payloads",
                "sql_overlay_ops_write_failures",
            )
        )
        and not _as_list(writer_shedding.get("hard_breaches"))
        and recovery_quality >= 95.0
        and backpressure_quality >= 95.0
    )
    bounded_transient_drain_managed = _bounded_transient_storage_drain_managed(
        storage,
        storage_state=storage_state,
        severity=severity,
        pressure_index=pressure_index,
        total_pending=total_pending,
        core_pending=core_pending,
        support_pending=support_pending,
        oldest=oldest,
        pending_threshold=pending_threshold,
        oldest_threshold=oldest_threshold,
        backpressure_quality=backpressure_quality,
    )
    bounded_steady_state_storage_managed = _bounded_steady_state_storage_managed(
        storage,
        storage_state=storage_state,
        severity=severity,
        pressure_index=pressure_index,
        total_pending=total_pending,
        core_pending=core_pending,
        support_pending=support_pending,
        oldest=oldest,
        backpressure_quality=backpressure_quality,
    )
    raw_live = _as_dict(backpressure.get("raw_live"))
    overlay_adjusted = _bool(backpressure.get("overlay_adjusted", False))
    overlay_pressure_clear = _bool(backpressure.get("overlay_pressure_clear", False))
    raw_live_stale = _bool(raw_live.get("artifact_stale_for_overlay_reconciliation", False))
    storage_operationally_clear = bool(
        storage_state == "ready"
        and severity not in {"critical", "high", "blocked"}
        and (
            pressure_index < 0.5
            or small_residual_drain_managed
            or bounded_transient_drain_managed
            or bounded_steady_state_storage_managed
        )
        and total_pending <= pending_threshold
        and core_pending <= pending_threshold
        and oldest <= oldest_threshold
    )
    backpressure_quality_hard_floor = bool(backpressure_quality < 75.0 and not storage_operationally_clear)
    hard_pressure_breach = bool(
        storage_state == "blocked"
        or severity in {"critical", "high"}
        or pressure_index >= 1.5
        or total_pending > pending_threshold
        or core_pending > pending_threshold
        or oldest > oldest_threshold
        or backpressure_quality_hard_floor
        or bool(disk_contract.get("emergency_disk_guard", False))
    )
    quality_pressure_present = bool(
        not storage_operationally_clear
        and (recovery_quality < 95.0 or backpressure_quality < 95.0)
    )
    pressure_present = bool(
        hard_pressure_breach
        or (
            pressure_index >= 0.5
            and not (
                small_residual_drain_managed
                or bounded_transient_drain_managed
                or bounded_steady_state_storage_managed
            )
        )
        or quality_pressure_present
    )
    collector_intake_optional_safe = bool(
        collector_status == "not_required"
        and not collector_required
        and collector_mismatches == 0
        and not hard_pressure_breach
        and storage_state == "ready"
        and total_pending <= pending_threshold
        and core_pending <= pending_threshold
        and oldest <= oldest_threshold
    )

    blockers = ordered_unique(
        [
            "ingestion_storage_control_missing" if not storage else "",
            "ingestion_storage_control_stale" if storage_stale else "",
            "ingestion_storage_blocked" if storage and storage_state == "blocked" else "",
            "storage_pressure_index_beyond_floor" if pressure_index >= 1.5 else "",
            "storage_total_pending_beyond_floor" if total_pending > pending_threshold else "",
            "storage_core_pending_beyond_floor" if core_pending > pending_threshold else "",
            "storage_oldest_pending_beyond_floor" if oldest > oldest_threshold else "",
            "storage_recovery_quality_below_floor" if recovery_quality < 75.0 and hard_pressure_breach else "",
            "storage_backpressure_quality_below_floor" if backpressure_quality_hard_floor else "",
            "collector_intake_not_enforced_during_pressure"
            if pressure_present
            and not collector_intake_optional_safe
            and (collector_status != "enforced" or collector_mismatches > 0)
            else "",
            "backlog_relief_not_active_during_hard_pressure" if hard_pressure_breach and not _bool(backlog_relief.get("active", False)) else "",
            "p_core_contract_missing_during_hard_pressure" if hard_pressure_breach and not pcore_contract else "",
            "p_core_contract_not_active_during_hard_pressure" if hard_pressure_breach and pcore_contract and not pcore_active else "",
            "p_core_env_not_active_during_hard_pressure" if hard_pressure_breach and pcore_contract and not pcore_env_active else "",
            "single_writer_not_enforced_during_hard_pressure" if hard_pressure_breach and pcore_contract and not single_writer else "",
            "p_core_workers_below_pressure_floor" if hard_pressure_breach and pcore_contract and selected_workers < 4 else "",
            "storage_disk_emergency_guard" if bool(disk_contract.get("emergency_disk_guard", False)) else "",
        ]
    )
    warnings = ordered_unique(
        [
            "ingestion_storage_degraded" if storage and storage_state == "degraded" else "",
            "storage_pressure_index_elevated"
            if 0.5 <= pressure_index < 1.5
            and not (
                small_residual_drain_managed
                or bounded_transient_drain_managed
                or bounded_steady_state_storage_managed
            )
            else "",
            "storage_recovery_quality_below_floor_advisory" if recovery_quality < 75.0 and not hard_pressure_breach else "",
            "storage_recovery_quality_below_target" if 75.0 <= recovery_quality < 95.0 else "",
            "storage_backpressure_quality_below_floor_advisory" if backpressure_quality < 75.0 and not backpressure_quality_hard_floor else "",
            "storage_backpressure_quality_below_target" if 75.0 <= backpressure_quality < 95.0 else "",
            "collector_intake_enforcement_partial" if not pressure_present and collector_status not in {"", "enforced"} else "",
            "backpressure_drainer_fleet_stale" if fleet_stale else "",
            "raw_live_backpressure_stale_without_overlay_reconciliation"
            if raw_live_stale and not (overlay_adjusted and overlay_pressure_clear)
            else "",
        ]
    )
    storage_quality_advisory_only = bool(
        warnings
        and not blockers
        and storage_operationally_clear
        and set(warnings).issubset(
            {
                "storage_recovery_quality_below_floor_advisory",
                "storage_recovery_quality_below_target",
                "storage_backpressure_quality_below_floor_advisory",
                "storage_backpressure_quality_below_target",
                "collector_intake_enforcement_partial",
                "raw_live_backpressure_stale_without_overlay_reconciliation",
                "backpressure_drainer_fleet_stale",
            }
        )
    )
    collector_intake_advisory_only = bool(
        warnings
        and set(warnings) == {"collector_intake_enforcement_partial"}
        and storage_state == "ready"
        and severity in {"", "stable", "ready"}
        and pressure_index < 0.5
        and recovery_quality >= 95.0
        and backpressure_quality >= 95.0
        and total_pending <= pending_threshold
        and core_pending <= pending_threshold
        and oldest <= oldest_threshold
    )
    state = "blocked" if blockers else "ready" if (collector_intake_advisory_only or storage_quality_advisory_only) else "degraded" if warnings else "ready"
    return _contract_row(
        surface_id="guard:ingestion_storage_degradation_floor",
        surface="ingestion_storage_degradation_floor",
        state=state,
        summary=(
            f"storage={storage_state} pressure={pressure_index:.3f} pending={total_pending}/{pending_threshold} "
            f"oldest={oldest:.1f}/{oldest_threshold:.1f}s quality={backpressure_quality:.1f}"
        ),
        metrics={
            "ingestion_storage_control": _artifact_metric(storage_path, project_root, storage_age, storage_stale, max_age),
            "backpressure_drainer_fleet": _artifact_metric(fleet_path, project_root, fleet_age, fleet_stale, 60),
            "storage_state": storage_state,
            "storage_severity": severity,
            "pressure_index": pressure_index,
            "hard_pressure_floor": 1.5,
            "warning_pressure_floor": 0.5,
            "recovery_quality_score": recovery_quality,
            "backpressure_quality_score": backpressure_quality,
            "quality_floor": 75.0,
            "quality_target": 95.0,
            "total_pending_lines": total_pending,
            "core_pending_lines": core_pending,
            "deferred_pending_lines": deferred_pending,
            "support_pending_lines": support_pending,
            "raw_total_pending_lines": raw_total_pending,
            "raw_core_pending_lines": raw_core_pending,
            "raw_deferred_pending_lines": raw_deferred_pending,
            "raw_support_pending_lines": raw_support_pending,
            "pending_lines_threshold": pending_threshold,
            "oldest_pending_age_seconds": oldest,
            "raw_oldest_pending_age_seconds": raw_oldest,
            "oldest_age_threshold_seconds": oldest_threshold,
            "collector_intake_status": collector_status,
            "collector_intake_required": collector_required,
            "collector_intake_mismatch_count": collector_mismatches,
            "collector_intake_optional_safe": collector_intake_optional_safe,
            "backlog_relief_active": _bool(backlog_relief.get("active", False)),
            "p_core_contract_active": pcore_active,
            "p_core_env_active": pcore_env_active,
            "single_writer_only": single_writer,
            "selected_workers": selected_workers,
            "raw_live_artifact_stale": raw_live_stale,
            "overlay_adjusted": overlay_adjusted,
            "overlay_pressure_clear": overlay_pressure_clear,
            "managed_pressure_view": managed_pressure_view,
            "small_residual_drain_managed": small_residual_drain_managed,
            "bounded_transient_drain_managed": bounded_transient_drain_managed,
            "bounded_steady_state_storage_managed": bounded_steady_state_storage_managed,
            "storage_operationally_clear": storage_operationally_clear,
            "storage_quality_advisory_only": storage_quality_advisory_only,
            "external_available_gb": _safe_float(disk_contract.get("external_available_gb"), 0.0),
            "external_used_percent": _safe_float(disk_contract.get("external_used_percent"), 0.0),
            "blockers": blockers,
            "warnings": warnings,
            "collector_intake_advisory_only": collector_intake_advisory_only,
        },
        recommended_command=["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
        quiet_hours_preferred=True,
    )


def _backlog_pcore_contract_surface(project_root: Path, loader: ArtifactLoader, max_artifact_age_minutes: int) -> dict[str, Any]:
    max_age = min(int(max_artifact_age_minutes), 60)
    fleet_path, fleet, fleet_age, fleet_stale = _artifact_payload(project_root, "backpressure_drainer_fleet_latest.json", loader=loader, max_age_minutes=max_age)
    accelerator_path, accelerator, accelerator_age, accelerator_stale = _artifact_payload(
        project_root,
        "backlog_pcore_accelerator_latest.json",
        loader=loader,
        max_age_minutes=max_age,
    )
    request = _as_dict(fleet.get("service_request"))
    contract = _as_dict(request.get("p_core_backlog_allocation_contract"))
    if not contract:
        contract = _as_dict(_as_dict(fleet.get("active_drainer")).get("p_core_backlog_allocation_contract"))
    control_env = _as_dict(contract.get("control_env"))
    selected = _safe_int(_as_dict(contract.get("p_core_burst_intelligence")).get("selected_workers"), _safe_int(contract.get("preprocess_worker_budget"), 0))
    shard_lanes = _safe_int(contract.get("shard_link_writer_lanes"), 0)
    max_shard_lanes = _safe_int(contract.get("max_shard_link_writer_lanes"), shard_lanes)
    fleet_state_raw = _lower(fleet.get("overall_status"))
    fleet_ok = bool(fleet.get("ok", False)) or fleet_state_raw in {"ready", "active", "handoff_requested", "advisory"}
    accelerator_state_raw = _lower(accelerator.get("overall_status"))
    accelerator_ok = bool(accelerator.get("ok", False)) or accelerator_state_raw in {"ready", "active", "advisory"}
    accelerator_storage = _as_dict(accelerator.get("storage_contract"))
    backlog_green = bool(
        _bool(accelerator_storage.get("green", False))
        or (
            _bool(accelerator_storage.get("line_green", False))
            and _bool(accelerator_storage.get("age_green", False))
            and _bool(accelerator_storage.get("overlay_green", False))
        )
    )
    active_catchup_target_required = bool(accelerator and not backlog_green)
    host_lane = _as_dict(accelerator.get("host_lane_contract"))
    memory_worker_cap = _safe_int(host_lane.get("memory_worker_cap"), 0)
    # The accelerator's maintenance posture intentionally parks at three P-core
    # workers once lines, age, and overlays are green.  Reserve the wider floor
    # for active catch-up so an idle maintenance contract cannot fail itself.
    idle_worker_floor = min(3, memory_worker_cap) if backlog_green and memory_worker_cap > 0 else 3 if backlog_green else 4
    operational_worker_floor = 6 if active_catchup_target_required else idle_worker_floor
    safe_idle_worker_floor_relief = bool(
        backlog_green
        and not active_catchup_target_required
        and selected >= operational_worker_floor
        and operational_worker_floor < 4
    )

    blockers = ordered_unique(
        [
            "backpressure_drainer_fleet_missing" if not fleet else "",
            "backlog_pcore_accelerator_missing" if not accelerator else "",
            "backpressure_drainer_fleet_not_ok" if fleet and not fleet_ok else "",
            "backlog_pcore_accelerator_not_ok" if accelerator and not accelerator_ok else "",
            "p_core_allocation_contract_missing" if fleet and not contract else "",
            "p_core_allocation_not_active" if contract and not _bool(contract.get("active", False)) else "",
            "p_core_allocation_env_not_active" if contract and control_env.get("BACKLOG_PCORE_ALLOCATION_ACTIVE") != "1" else "",
            "sql_writer_not_single_writer" if contract and not _bool(contract.get("single_writer_only", False)) else "",
            "sql_writer_background_policy_enabled" if contract and control_env.get("SQL_LINK_WRITER_BACKGROUND_POLICY") not in {"0", 0} else "",
            "p_core_workers_below_floor" if contract and selected < operational_worker_floor else "",
            "shard_writer_lanes_exceed_max" if contract and max_shard_lanes and shard_lanes > max_shard_lanes else "",
        ]
    )
    warnings = ordered_unique(
        [
            "backpressure_drainer_fleet_stale" if fleet_stale else "",
            "backlog_pcore_accelerator_stale" if accelerator_stale else "",
            "p_core_worker_budget_below_active_catchup_target"
            if contract and active_catchup_target_required and selected < 6
            else "",
        ]
    )
    pcore_contract_operationally_clear = bool(
        contract
        and _bool(contract.get("active", False))
        and control_env.get("BACKLOG_PCORE_ALLOCATION_ACTIVE") == "1"
        and _bool(contract.get("single_writer_only", False))
        and selected >= operational_worker_floor
        and (not max_shard_lanes or shard_lanes <= max_shard_lanes)
        and not blockers
    )
    stale_support_advisory_only = bool(
        warnings
        and pcore_contract_operationally_clear
        and set(warnings).issubset({"backpressure_drainer_fleet_stale", "backlog_pcore_accelerator_stale"})
    )
    state = "blocked" if blockers else "ready" if stale_support_advisory_only else "degraded" if warnings else "ready"
    return _contract_row(
        surface_id="guard:backlog_pcore_contract",
        surface="backlog_pcore_contract",
        state=state,
        summary=f"contract_active={_bool(contract.get('active', False))} workers={selected} shard_lanes={shard_lanes}/{max_shard_lanes} blockers={len(blockers)}",
        metrics={
            "backpressure_drainer_fleet": _artifact_metric(fleet_path, project_root, fleet_age, fleet_stale, max_age),
            "backlog_pcore_accelerator": _artifact_metric(accelerator_path, project_root, accelerator_age, accelerator_stale, max_age),
            "fleet_status": str(fleet.get("overall_status") or ""),
            "accelerator_status": str(accelerator.get("overall_status") or ""),
            "contract_active": _bool(contract.get("active", False)),
            "single_writer_only": _bool(contract.get("single_writer_only", False)),
            "performance_core_primary": _bool(contract.get("performance_core_primary", False)),
            "selected_workers": selected,
            "backlog_green": backlog_green,
            "active_catchup_target_required": active_catchup_target_required,
            "operational_worker_floor": operational_worker_floor,
            "safe_idle_worker_floor_relief": safe_idle_worker_floor_relief,
            "memory_worker_cap": memory_worker_cap,
            "shard_link_writer_lanes": shard_lanes,
            "max_shard_link_writer_lanes": max_shard_lanes,
            "backlog_pcore_allocation_env": control_env.get("BACKLOG_PCORE_ALLOCATION_ACTIVE"),
            "sql_link_writer_background_policy": control_env.get("SQL_LINK_WRITER_BACKGROUND_POLICY"),
            "pcore_contract_operationally_clear": pcore_contract_operationally_clear,
            "stale_support_advisory_only": stale_support_advisory_only,
            "blockers": blockers,
            "warnings": warnings,
        },
        recommended_command=["./scripts/ops/opsctl.sh", "backlog-pcore-accelerator", "--apply", "--json"],
        quiet_hours_preferred=True,
    )


def _critical_contract_surfaces(project_root: Path, loader: ArtifactLoader, max_artifact_age_minutes: int) -> list[dict[str, Any]]:
    return [
        _broker_auth_contract_surface(project_root, loader, max_artifact_age_minutes),
        _source_verification_contract_surface(project_root, loader, max_artifact_age_minutes),
        _livefeed_contract_surface(project_root, loader, max_artifact_age_minutes),
        _memory_truth_contract_surface(project_root, loader, max_artifact_age_minutes),
        _runtime_storage_contract_surface(project_root, loader, max_artifact_age_minutes),
        _ingestion_storage_degradation_floor_surface(project_root, loader, max_artifact_age_minutes),
        _backlog_pcore_contract_surface(project_root, loader, max_artifact_age_minutes),
    ]


def _load_section_guard(project_root: Path, loader: ArtifactLoader) -> dict[str, Any]:
    return loader(project_root / "governance" / "health" / "section_grade_guard_latest.json")


def _load_state(path: Path) -> dict[str, Any]:
    return load_json(path)


def _project_surface_state(surface: dict[str, Any], previous: dict[str, Any], *, now_iso: str) -> dict[str, Any]:
    state = _canonical_state(surface.get("state"))
    prior_state = _canonical_state(previous.get("last_state")) if previous else ""
    prior_non_ready = _safe_int(previous.get("consecutive_non_ready_count"), 0)
    prior_blocked = _safe_int(previous.get("consecutive_blocked_count"), 0)
    prior_ready = _safe_int(previous.get("ready_streak"), 0)

    if state == "ready":
        non_ready_count = 0
        blocked_count = 0
        ready_streak = prior_ready + 1 if prior_state == "ready" else 1
        recovered = prior_non_ready > 0 or prior_blocked > 0
        first_non_ready = ""
    else:
        non_ready_count = prior_non_ready + 1 if prior_state != "ready" else 1
        blocked_count = prior_blocked + 1 if state == "blocked" and prior_state == "blocked" else (1 if state == "blocked" else 0)
        ready_streak = 0
        recovered = False
        first_non_ready = str(previous.get("first_non_ready_utc") or now_iso) if previous else now_iso

    return {
        "last_state": state,
        "previous_state": prior_state or "",
        "consecutive_non_ready_count": non_ready_count,
        "consecutive_blocked_count": blocked_count,
        "ready_streak": ready_streak,
        "recovered_this_run": recovered,
        "first_non_ready_utc": first_non_ready,
        "last_seen_utc": now_iso,
    }


def _adaptive_row(
    surface: dict[str, Any],
    projected: dict[str, Any],
    *,
    pressure_context: dict[str, Any],
    persistence_threshold: int,
    blocked_escalation_threshold: int,
) -> dict[str, Any]:
    state = _canonical_state(surface.get("state"))
    surface_id = str(surface.get("surface_id") or surface.get("surface") or "")
    non_ready_count = _safe_int(projected.get("consecutive_non_ready_count"), 0)
    blocked_count = _safe_int(projected.get("consecutive_blocked_count"), 0)
    persistent = state != "ready" and non_ready_count >= max(int(persistence_threshold), 1)
    repeated_block = state == "blocked" and blocked_count >= max(int(blocked_escalation_threshold), 1)
    hard_surface = bool(surface.get("critical_contract", False)) and state == "blocked"
    hard_surface = hard_surface or (surface_id.startswith("section:") and state == "blocked")
    heavy = bool(surface.get("quiet_hours_preferred", False)) or surface_id in HEAVY_SURFACES
    high_pressure = bool(pressure_context.get("high_pressure", False))

    severity = "info"
    if repeated_block or hard_surface:
        severity = "critical"
    elif state == "blocked":
        severity = "high"
    elif persistent:
        severity = "high"
    elif state == "degraded":
        severity = "warning"

    if state == "ready":
        action = "watch_recovery" if bool(projected.get("recovered_this_run", False)) else "no_action"
    elif high_pressure and heavy and severity != "critical":
        action = "defer_heavy_repair_until_pressure_cools"
    elif severity == "critical":
        action = "run_guarded_repair"
    elif persistent:
        action = "run_targeted_repair"
    else:
        action = "watch_and_refresh"

    return {
        **surface,
        "state": state,
        "adaptive_severity": severity,
        "adaptive_action": action,
        "persistent_regression": persistent,
        "repeated_blocked_regression": repeated_block,
        "heavy_repair": heavy,
        "pressure_deferred": action == "defer_heavy_repair_until_pressure_cools",
        "memory": projected,
    }


def _next_state(previous_state: dict[str, Any], adaptive_rows: list[dict[str, Any]], *, now_iso: str) -> dict[str, Any]:
    previous_surfaces = _as_dict(previous_state.get("surfaces"))
    current_surfaces = {
        str(row.get("surface_id") or ""): dict(row.get("memory") or {})
        for row in adaptive_rows
        if str(row.get("surface_id") or "")
    }
    retired = sorted(set(previous_surfaces) - set(current_surfaces))
    return {
        "timestamp_utc": now_iso,
        "schema_version": 1,
        "surfaces": current_surfaces,
        "retired_surfaces": retired[-100:],
    }


def _feedback_event(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp_utc": payload.get("timestamp_utc"),
        "event": "adaptive_regression_guard_publish",
        "overall_status": payload.get("overall_status"),
        "active_regression_count": payload.get("active_regression_count"),
        "persistent_regression_count": payload.get("persistent_regression_count"),
        "critical_regression_count": payload.get("critical_regression_count"),
        "recommended_command_count": len(_as_list(payload.get("recommended_commands"))),
    }


def _append_feedback(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    persistence_threshold: int = 3,
    blocked_escalation_threshold: int = 2,
    max_artifact_age_minutes: int = 60,
    grade_guard_builder: GradeGuardBuilder | None = None,
    artifact_loader: ArtifactLoader | None = None,
    state_path: Path | None = None,
    feedback_path: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    loader = artifact_loader or load_json
    now_iso = iso_now()
    state_file = state_path or (project_root / "governance" / "health" / "adaptive_regression_guard_state.json")
    feedback_file = feedback_path or (project_root / "governance" / "health" / "adaptive_regression_guard_feedback.jsonl")
    previous_state = _load_state(state_file)

    build_grade_guard = grade_guard_builder or grade_regression_guard.build_payload
    grade_guard_payload = build_grade_guard(project_root)
    section_guard_payload = _load_section_guard(project_root, loader)
    pressure_context = _runtime_pressure_context(project_root, loader)

    source_guards = [
        {
            "name": "grade_regression_guard",
            "overall_status": str(grade_guard_payload.get("overall_status") or ""),
            "blocked_surface_count": _safe_int(grade_guard_payload.get("blocked_surface_count"), 0),
            "degraded_surface_count": _safe_int(grade_guard_payload.get("degraded_surface_count"), 0),
        },
        {
            "name": "section_grade_guard",
            "overall_status": str(section_guard_payload.get("overall_status") or "missing"),
            "below_floor_count": _safe_int(section_guard_payload.get("below_floor_count"), 0),
            "protected_by_floor_count": _safe_int(section_guard_payload.get("protected_by_floor_count"), 0),
        },
        {
            "name": "critical_contract_surfaces",
            "surface_count": len(CRITICAL_CONTRACT_IDS),
            "contract_ids": sorted(CRITICAL_CONTRACT_IDS),
        },
    ]

    raw_surfaces: list[dict[str, Any]] = []
    raw_surfaces.extend(_grade_surfaces(grade_guard_payload))
    raw_surfaces.extend(_section_surfaces(section_guard_payload))
    for spec in SPECIALIZED_GUARDS:
        raw_surfaces.append(
            _artifact_surface(
                project_root,
                spec,
                loader=loader,
                max_artifact_age_minutes=max_artifact_age_minutes,
            )
        )
    raw_surfaces.extend(_critical_contract_surfaces(project_root, loader, max_artifact_age_minutes))
    raw_surfaces = [_soften_paper_soak_quality_debt(surface, section_guard_payload, project_root, loader) for surface in raw_surfaces]
    raw_surfaces = [_soften_guarded_paper_incident_closeout(surface, project_root, loader) for surface in raw_surfaces]
    raw_surfaces = [_soften_paper_soak_promotion_gate(surface, project_root, loader) for surface in raw_surfaces]

    prior_surfaces = _as_dict(previous_state.get("surfaces"))
    adaptive_rows: list[dict[str, Any]] = []
    for surface in raw_surfaces:
        surface_id = str(surface.get("surface_id") or "")
        projected = _project_surface_state(surface, _as_dict(prior_surfaces.get(surface_id)), now_iso=now_iso)
        adaptive_rows.append(
            _adaptive_row(
                surface,
                projected,
                pressure_context=pressure_context,
                persistence_threshold=persistence_threshold,
                blocked_escalation_threshold=blocked_escalation_threshold,
            )
        )

    active_rows = [row for row in adaptive_rows if row.get("state") != "ready"]
    persistent_rows = [row for row in active_rows if bool(row.get("persistent_regression", False))]
    critical_rows = [row for row in active_rows if row.get("adaptive_severity") == "critical"]
    recovered_rows = [row for row in adaptive_rows if bool(_as_dict(row.get("memory")).get("recovered_this_run", False))]
    pressure_deferred_rows = [row for row in active_rows if bool(row.get("pressure_deferred", False))]

    overall_status = "ready"
    if critical_rows:
        overall_status = "blocked"
    elif active_rows:
        overall_status = "degraded"

    recommended_commands: list[list[str]] = []
    if active_rows or recovered_rows:
        recommended_commands.append(["./scripts/ops/opsctl.sh", "adaptive-regression-guard", "--apply", "--json"])
    if any(row["surface_id"].startswith("grade:") for row in active_rows):
        recommended_commands.append(["./scripts/ops/opsctl.sh", "grade-regression-autopilot", "--apply", "--json"])
    if any(row["surface_id"].startswith("section:") for row in active_rows):
        recommended_commands.append(["./scripts/ops/opsctl.sh", "section-grade-autopilot", "--apply", "--json"])
    if pressure_deferred_rows or bool(pressure_context.get("high_pressure", False)):
        recommended_commands.append(["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"])
    for row in active_rows:
        cmd = row.get("recommended_command") if isinstance(row.get("recommended_command"), list) else []
        if cmd:
            recommended_commands.append([str(part) for part in cmd])

    deduped_commands: list[list[str]] = []
    seen_commands: set[str] = set()
    for cmd in recommended_commands:
        key = _command_key(cmd)
        if not key or key in seen_commands:
            continue
        seen_commands.add(key)
        deduped_commands.append(cmd)

    next_state = _next_state(previous_state, adaptive_rows, now_iso=now_iso)
    if apply:
        write_payload(state_file, next_state)

    payload = {
        "timestamp_utc": now_iso,
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply": bool(apply),
        "state_updated": bool(apply),
        "state_path": str(state_file),
        "feedback_path": str(feedback_file),
        "adaptive_surface_count": len(adaptive_rows),
        "active_regression_count": len(active_rows),
        "persistent_regression_count": len(persistent_rows),
        "critical_regression_count": len(critical_rows),
        "pressure_deferred_count": len(pressure_deferred_rows),
        "recovered_surface_count": len(recovered_rows),
        "persistence_threshold": max(int(persistence_threshold), 1),
        "blocked_escalation_threshold": max(int(blocked_escalation_threshold), 1),
        "pressure_context": pressure_context,
        "source_guards": source_guards,
        "surfaces": adaptive_rows,
        "adaptive_regression_guard_contract": {
            "generation": "adaptive_regression_guard_v1",
            "learns_from_persistent_surface_state": True,
            "pressure_aware_heavy_repair_deferral": True,
            "does_not_enable_live_execution": True,
            "does_not_run_repairs_directly": True,
            "state_update_requires_apply": True,
            "co_managed_with": [
                "grade_regression_guard",
                "grade_regression_autopilot",
                "section_grade_guard",
                "runtime_paper_regression_guard",
                "broker_auth_contract",
                "source_verification_contract",
                "livefeed_visibility_contract",
                "memory_truth_contract",
                "runtime_storage_contract",
                "backlog_pcore_contract",
                "system_drift_registry",
            ],
        },
        "recommended_commands": deduped_commands,
        "recommended_actions": ordered_unique(
            [
                "run adaptive-regression-guard with --apply so repeated degraded surfaces become persistent signals instead of isolated snapshots"
                if active_rows and not apply
                else "",
                "pressure is high, so heavy repairs should wait behind runtime-throttle unless a critical regression is active"
                if pressure_deferred_rows
                else "",
                "run guarded repair commands for critical repeated regressions"
                if critical_rows
                else "",
                "keep watching recovered surfaces until they build a ready streak"
                if recovered_rows
                else "",
            ]
            + [str(row.get("summary") or "") for row in active_rows[:8]]
        ),
        "upgrade_track": {
            "family": "infrabots",
            "upgradeable": True,
            "current_generation": "adaptive_regression_guard_v1",
            "future_upgrade_paths": [
                "learn per-surface persistence thresholds from historical repair success",
                "feed adaptive severity into system-drift-autopilot routing",
                "publish tenant-facing regression notifications only after adaptive persistence is proven",
            ],
        },
    }
    if apply:
        _append_feedback(feedback_file, _feedback_event(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish an adaptive regression guard that learns persistence across grade, section, and runtime guard surfaces.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--feedback-file", default=str(DEFAULT_FEEDBACK_PATH))
    parser.add_argument("--apply", action="store_true", help="Persist adaptive memory for future guard runs.")
    parser.add_argument(
        "--persistence-threshold",
        type=int,
        default=_safe_int(os.getenv("ADAPTIVE_REGRESSION_PERSISTENCE_THRESHOLD"), 3),
    )
    parser.add_argument(
        "--blocked-escalation-threshold",
        type=int,
        default=_safe_int(os.getenv("ADAPTIVE_REGRESSION_BLOCKED_ESCALATION_THRESHOLD"), 2),
    )
    parser.add_argument(
        "--max-artifact-age-minutes",
        type=int,
        default=_safe_int(os.getenv("ADAPTIVE_REGRESSION_MAX_ARTIFACT_AGE_MINUTES"), 60),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        apply=bool(args.apply),
        persistence_threshold=int(args.persistence_threshold),
        blocked_escalation_threshold=int(args.blocked_escalation_threshold),
        max_artifact_age_minutes=int(args.max_artifact_age_minutes),
        state_path=Path(args.state_file).expanduser(),
        feedback_path=Path(args.feedback_file).expanduser(),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "adaptive_regression_guard "
            f"overall_status={payload.get('overall_status', '')} "
            f"active_regression_count={payload.get('active_regression_count', 0)} "
            f"persistent_regression_count={payload.get('persistent_regression_count', 0)} "
            f"critical_regression_count={payload.get('critical_regression_count', 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
