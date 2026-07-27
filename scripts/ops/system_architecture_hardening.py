#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_architecture_hardening_latest.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "system_architecture_hardening_v1.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.system_architecture_hardening_override"
SECTION_DIR = PROJECT_ROOT / "governance" / "system_architecture_hardening"

READY_STATES = {
    "",
    "active",
    "advisory",
    "armed",
    "calm",
    "clear",
    "clear_ready",
    "guarded_ready",
    "guarded_relief",
    "normal",
    "observe",
    "ok",
    "ready",
    "stable",
    "watch",
    "thin",
}
WATCH_STATES = {"advisory", "guarded_relief", "thin", "watch"}
HARD_STATES = {"blocked", "critical", "degraded", "failed", "fatal", "high", "needs_repair", "needs_work"}
LIVE_ENABLE_FLAGS = {
    "ALLOW_ORDER_EXECUTION",
    "EXECUTION_LANE_LIVE_ENABLED",
    "INLINE_PAPER_EXECUTION_ENABLED",
    "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR",
    "TOP_BOT_ENABLE_LIVE_EXECUTION",
}
REQUIRED_OPSCTL_COMMANDS = [
    "health-fast",
    "paper-400-ramp",
    "global-halt-refresh",
    "runtime-throttle",
    "writer-process-intelligence",
    "platform-intelligence",
    "platform-brain-v5",
    "platform-stabilization",
    "platform-settlement-stabilization",
    "process-watchdog",
    "system-plumbing-control",
    "provider-mesh",
    "training-quality",
    "system-architecture-hardening",
]
CORE_SOURCE_IDS = {
    "market_quote_profiles",
    "crypto_market_context",
    "fx_market_context",
    "public_macro_feeds",
    "official_macro_context",
    "market_micro_context",
    "fed_2026_supervisory_stress_scenario",
}
MANAGED_VERIFICATION_SOURCE_IDS = {"macro_crossstack"}
ANATOMY_LAYER_DEFINITIONS = {
    "body": {
        "title": "Body",
        "role": "whole-system posture, guarded paper readiness, runtime capacity, and evidence coverage",
        "sections": [
            "safety_execution_boundary",
            "truth_source_consistency",
            "runtime_capacity_partition",
            "training_evidence_contract",
        ],
        "contract": "the whole platform can stand up in guarded paper without stale halt, runtime, or evidence contradictions",
    },
    "skeleton": {
        "title": "Skeleton",
        "role": "command spine, artifact truth contracts, and single-writer/data-plane structure",
        "sections": [
            "opsctl_command_spine",
            "truth_source_consistency",
            "storage_writer_data_plane",
        ],
        "contract": "control commands, current artifacts, and the SQL writer frame agree before the platform widens",
    },
    "organs": {
        "title": "Organs",
        "role": "storage, runtime, collectors, providers, and training subsystems doing their work",
        "sections": [
            "storage_writer_data_plane",
            "runtime_capacity_partition",
            "collector_process_quarantine",
            "provider_source_mesh",
            "training_evidence_contract",
        ],
        "contract": "the major subsystems stay functional, isolated, and replaceable without poisoning guarded paper",
    },
    "skin": {
        "title": "Skin",
        "role": "external boundary, live execution lock, source confidence, and provider exposure",
        "sections": [
            "safety_execution_boundary",
            "provider_source_mesh",
            "runtime_capacity_partition",
        ],
        "contract": "the outside world can degrade at optional edges without opening live execution or corrupting core readiness",
    },
    "heart": {
        "title": "Heart",
        "role": "writer pump, backlog flow, collector heartbeat, and runtime circulation",
        "sections": [
            "storage_writer_data_plane",
            "collector_process_quarantine",
            "runtime_capacity_partition",
        ],
        "contract": "the platform keeps pumping data through one writer while heartbeat repairs stay quarantined",
    },
    "brain": {
        "title": "Brain",
        "role": "platform intelligence, stabilization, settlement, and command reachability",
        "sections": [
            "platform_watch_semantics",
            "truth_source_consistency",
            "opsctl_command_spine",
        ],
        "contract": "the control planes can reason, route commands, and preserve watch semantics without hard-failing the system",
    },
    "mind": {
        "title": "Mind",
        "role": "evidence, source confidence, training quality, and cross-layer self-awareness",
        "sections": [
            "training_evidence_contract",
            "platform_watch_semantics",
            "provider_source_mesh",
            "truth_source_consistency",
        ],
        "contract": "strategy confidence comes from evidence and source quality rather than raw activity or wishful readiness",
    },
}


def _health(project_root: Path, name: str) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "health" / name)
    return payload if isinstance(payload, dict) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _status(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _storage_overlay_relief(storage: dict[str, Any], plumbing: dict[str, Any]) -> dict[str, Any]:
    plumbing_sections = _as_dict(plumbing.get("sections"))
    plumbing_queue = _as_dict(plumbing_sections.get("queue_backpressure"))
    plumbing_overlay = _as_dict(plumbing_queue.get("overlay_relief"))
    if bool(plumbing_overlay.get("active", False)) and _status(plumbing.get("overall_status")) == "ready":
        return {
            "active": True,
            "source": "system_plumbing_control",
            "overlay_total_pending_lines": _safe_int(plumbing_overlay.get("overlay_total_pending_lines"), 0),
            "raw_total_pending_lines": _safe_int(plumbing_overlay.get("raw_total_pending_lines"), 0),
            "policy": plumbing_overlay.get("policy", ""),
        }

    backpressure = _as_dict(storage.get("backpressure"))
    raw_live = _as_dict(backpressure.get("raw_live"))
    raw_core = _safe_int(raw_live.get("core_pending_lines"), 0)
    raw_total = _safe_int(raw_live.get("total_pending_lines"), 0)
    raw_oldest = _safe_float(raw_live.get("oldest_pending_age_seconds"), 0.0)
    overlay_total = _safe_int(backpressure.get("total_pending_lines"), 0)
    active = bool(
        backpressure.get("overlay_adjusted", False)
        and raw_live
        and raw_core <= 5000
        and raw_total <= 15000
        and raw_oldest <= 15 * 60
        and overlay_total <= 12000
    )
    return {
        "active": active,
        "source": "local_storage_overlay_check",
        "overlay_total_pending_lines": overlay_total,
        "raw_total_pending_lines": raw_total,
        "policy": "SQL-overlay-only pressure is managed as architecture advisory when raw-live queue health is cool",
    }


def _state_ok(value: Any) -> bool:
    return _status(value) in READY_STATES


def _is_hard_status(value: Any) -> bool:
    return _status(value) in HARD_STATES


def _clear_blockers(payload: dict[str, Any]) -> list[str]:
    return [str(item) for item in _as_list(payload.get("clear_blockers")) if str(item).strip()]


def _paper_blockers(payload: dict[str, Any]) -> list[str]:
    return [str(item) for item in _as_list(payload.get("blockers")) if str(item).strip()]


def _walk_truthy_flags(name: str, value: Any, *, path: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, raw in value.items():
            key_text = str(key)
            next_path = (*path, key_text)
            if key_text in LIVE_ENABLE_FLAGS and _truthy(raw):
                rows.append({"source": name, "path": ".".join(next_path), "value": raw})
            rows.extend(_walk_truthy_flags(name, raw, path=next_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_walk_truthy_flags(name, item, path=(*path, str(index))))
    return rows


def _env_truthy_flags(named_payloads: Iterable[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, payload in named_payloads:
        if payload:
            rows.extend(_walk_truthy_flags(name, payload))
    return rows


def _section(
    name: str,
    title: str,
    status: str,
    *,
    evidence: dict[str, Any] | None = None,
    findings: Iterable[str] = (),
    watch_items: Iterable[str] = (),
    recommendations: Iterable[str] = (),
    blocks_guarded_paper: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "title": title,
        "overall_status": status,
        "ok": status in {"ready", "watch"},
        "blocks_guarded_paper": bool(blocks_guarded_paper),
        "findings": ordered_unique(findings),
        "watch_items": ordered_unique(watch_items),
        "recommendations": ordered_unique(recommendations),
        "evidence": evidence or {},
    }


def _rollup_status(sections: dict[str, dict[str, Any]]) -> str:
    statuses = {_status(section.get("overall_status")) for section in sections.values()}
    if statuses & {"blocked", "critical"}:
        return "blocked"
    if statuses & {"needs_work", "degraded", "failed", "fatal"}:
        return "needs_work"
    if statuses & {"watch", "thin", "advisory"}:
        return "watch"
    return "ready"


def _status_strength_score(status: Any) -> float:
    normalized = _status(status)
    if normalized in {"ready", "ok", "active", "stable", "clear", "clear_ready", "armed", "normal"}:
        return 100.0
    if normalized in {"watch", "thin", "advisory", "guarded_ready", "guarded_relief", "observe", "calm"}:
        return 88.0
    if normalized in {"needs_work", "needs_repair", "degraded", "high", "missing", "inactive"}:
        return 55.0
    if normalized in {"blocked", "critical", "failed", "fatal"}:
        return 0.0
    return 75.0


def _strength_label(status: str, score: float) -> str:
    normalized = _status(status)
    if normalized in {"blocked", "critical"}:
        return "fractured"
    if normalized in {"needs_work", "degraded", "failed", "fatal"}:
        return "needs_repair"
    if score >= 98.0:
        return "reinforced"
    if score >= 90.0:
        return "guarded_strong"
    if score >= 80.0:
        return "watch_strong"
    return "thin"


def _anatomy_layer(name: str, definition: dict[str, Any], sections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    section_names = [str(item) for item in _as_list(definition.get("sections")) if str(item).strip()]
    mapped_sections = {section_name: sections.get(section_name, {}) for section_name in section_names}
    status = _rollup_status(mapped_sections)
    scores = [_status_strength_score(section.get("overall_status")) for section in mapped_sections.values()]
    strength_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    hard_sections = [
        section_name
        for section_name, section in mapped_sections.items()
        if _status(section.get("overall_status")) in {"blocked", "critical", "needs_work", "degraded"}
    ]
    watch_sections = [
        section_name
        for section_name, section in mapped_sections.items()
        if _status(section.get("overall_status")) in {"watch", "thin", "advisory"}
    ]
    findings: list[str] = []
    watch_items: list[str] = []
    recommendations: list[str] = []
    for section_name, section in mapped_sections.items():
        findings.extend(f"{section_name}:{item}" for item in _as_list(section.get("findings")))
        watch_items.extend(f"{section_name}:{item}" for item in _as_list(section.get("watch_items")))
        recommendations.extend(str(item) for item in _as_list(section.get("recommendations")))
    return {
        "name": name,
        "title": str(definition.get("title") or name),
        "role": str(definition.get("role") or ""),
        "contract": str(definition.get("contract") or ""),
        "overall_status": status,
        "ok": status in {"ready", "watch"},
        "strength_score": strength_score,
        "strength_label": _strength_label(status, strength_score),
        "section_statuses": {section_name: section.get("overall_status") for section_name, section in mapped_sections.items()},
        "hard_sections": hard_sections,
        "watch_sections": watch_sections,
        "findings": ordered_unique(findings),
        "watch_items": ordered_unique(watch_items),
        "recommendations": ordered_unique(recommendations),
    }


def _anatomy_layers(sections: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: _anatomy_layer(name, definition, sections)
        for name, definition in ANATOMY_LAYER_DEFINITIONS.items()
    }


def _chosen_halt(auto_clear: dict[str, Any], killswitch: dict[str, Any]) -> dict[str, Any]:
    return auto_clear if auto_clear else killswitch


def _safe_live_block_status(health_fast: dict[str, Any]) -> str:
    live = _as_dict(_as_dict(health_fast.get("operational_readiness")).get("live_execution"))
    return _status(live.get("status"))


def _guarded_paper_ok(health_fast: dict[str, Any]) -> bool:
    guarded = _as_dict(_as_dict(health_fast.get("operational_readiness")).get("guarded_paper"))
    if guarded:
        return bool(guarded.get("ok", False))
    return bool(health_fast.get("ok", False))


def _safety_execution_boundary(ctx: dict[str, dict[str, Any]]) -> dict[str, Any]:
    health_fast = ctx["health_fast"]
    paper_ramp = ctx["paper_ramp"]
    auto_clear = ctx["global_halt_auto_clear"]
    killswitch = ctx["global_killswitch"]
    halt = _chosen_halt(auto_clear, killswitch)
    live_flags = _env_truthy_flags(
        (
            ("health_fast", health_fast),
            ("paper_400_ramp", paper_ramp),
            ("platform_intelligence", ctx["platform_intelligence"]),
            ("platform_brain_v5", ctx["platform_brain_v5"]),
            ("platform_stabilization_quality", ctx["platform_stabilization_quality"]),
            ("platform_settlement_stabilization", ctx["platform_settlement_stabilization"]),
            ("pressure_relief_control", ctx["pressure_relief_control"]),
        )
    )
    guarded_ok = _guarded_paper_ok(health_fast)
    live_status = _safe_live_block_status(health_fast)
    paper_stage = _status(paper_ramp.get("stage"))
    blockers = _paper_blockers(paper_ramp)
    clear_blockers = _clear_blockers(halt)
    halt_clear = halt and not bool(halt.get("halt", False)) and not clear_blockers
    stale_global_blocker = bool(
        paper_ramp
        and paper_stage not in {"armed", "ready"}
        and set(blockers) == {"global_halt_or_clear_blocker_active"}
        and halt_clear
    )
    findings: list[str] = []
    watch_items: list[str] = []
    recommendations: list[str] = []

    if live_flags:
        findings.append("live_execution_enable_flag_truthy")
    live_contract_ok = live_status in {"blocked_read_only", "blocked", "read_only"} or not health_fast
    live = _as_dict(_as_dict(health_fast.get("operational_readiness")).get("live_execution"))
    if bool(live.get("ok", False)):
        findings.append("health_fast_live_execution_marked_ok")
    if health_fast and not bool(health_fast.get("read_only", True)):
        findings.append("health_fast_not_read_only")
    if not live_contract_ok:
        watch_items.append(f"live_execution_status={live_status or 'missing'}")
    if bool(halt.get("halt", False)):
        findings.append("global_halt_active")
    if clear_blockers:
        findings.extend(f"global_clear_blocker={item}" for item in clear_blockers)
    if stale_global_blocker:
        watch_items.append("paper_ramp_has_stale_global_halt_blocker")
        recommendations.append("./scripts/ops/opsctl.sh paper-400-ramp --apply --json")
    elif paper_ramp and (paper_stage not in {"armed", "ready"} or blockers):
        findings.append("paper_ramp_not_armed")
    if health_fast and not guarded_ok:
        findings.append("guarded_paper_not_ready")
    if not health_fast:
        watch_items.append("health_fast_missing")
    if not paper_ramp:
        watch_items.append("paper_400_ramp_missing")
    if not halt:
        watch_items.append("global_halt_contract_missing")

    if live_flags or bool(live.get("ok", False)):
        status = "blocked"
        recommendations.append("./scripts/ops/opsctl.sh health-fast --json")
    elif any(item in findings for item in {"global_halt_active", "paper_ramp_not_armed", "guarded_paper_not_ready"}):
        status = "needs_work"
        recommendations.append("./scripts/ops/opsctl.sh paper-400-ramp --apply --json")
        recommendations.append("./scripts/ops/opsctl.sh global-halt-refresh --json")
    elif findings:
        status = "needs_work"
    elif watch_items:
        status = "watch"
    else:
        status = "ready"

    return _section(
        "safety_execution_boundary",
        "Safety Execution Boundary",
        status,
        evidence={
            "guarded_paper_ok": guarded_ok,
            "health_fast_read_only": bool(health_fast.get("read_only", True)) if health_fast else None,
            "live_execution_status": live_status,
            "paper_ramp_stage": paper_stage,
            "paper_ramp_blockers": blockers,
            "global_halt": bool(halt.get("halt", False)) if halt else None,
            "global_clear_blockers": clear_blockers,
            "stale_global_blocker": stale_global_blocker,
            "truthy_live_enable_flags": live_flags,
        },
        findings=findings,
        watch_items=watch_items,
        recommendations=recommendations,
        blocks_guarded_paper=status in {"blocked", "needs_work"},
    )


def _truth_source_consistency(ctx: dict[str, dict[str, Any]]) -> dict[str, Any]:
    health_fast = ctx["health_fast"]
    paper_ramp = ctx["paper_ramp"]
    auto_clear = ctx["global_halt_auto_clear"]
    killswitch = ctx["global_killswitch"]
    data_plane = ctx["data_plane_recovery"]
    plumbing = ctx.get("system_plumbing_control", {})
    halt = _chosen_halt(auto_clear, killswitch)
    findings: list[str] = []
    watch_items: list[str] = []
    recommendations: list[str] = []

    if auto_clear and killswitch:
        if bool(auto_clear.get("halt", False)) != bool(killswitch.get("halt", False)):
            watch_items.append("global_halt_sources_disagree_on_halt")
        if _clear_blockers(auto_clear) != _clear_blockers(killswitch):
            watch_items.append("global_halt_sources_disagree_on_clear_blockers")

    paper_stage = _status(paper_ramp.get("stage"))
    paper_blockers = _paper_blockers(paper_ramp)
    halt_clear = halt and not bool(halt.get("halt", False)) and not _clear_blockers(halt)
    stale_global_blocker = bool(
        paper_ramp
        and paper_stage not in {"armed", "ready"}
        and set(paper_blockers) == {"global_halt_or_clear_blocker_active"}
        and halt_clear
    )
    if stale_global_blocker:
        watch_items.append("paper_ramp_has_stale_global_halt_blocker")
        recommendations.append("./scripts/ops/opsctl.sh paper-400-ramp --apply --json")
    elif paper_ramp and paper_blockers and _guarded_paper_ok(health_fast):
        findings.append("guarded_paper_ready_while_paper_ramp_has_hard_blockers")

    data_status = _status(data_plane.get("overall_status"))
    plumbing_status = _status(plumbing.get("overall_status"))
    write_failures = _safe_int(data_plane.get("write_failure_count"), 0)
    snapshot_failures = _safe_int(data_plane.get("account_snapshot_failure_count"), 0)
    if data_plane and data_status in {"blocked", "critical"}:
        findings.append(f"data_plane_status={data_status}")
    elif data_plane and data_status in {"degraded", "needs_work"} and (write_failures or snapshot_failures):
        findings.append(f"data_plane_status={data_status}")
    elif data_plane and data_status and not _state_ok(data_status):
        watch_items.append(f"data_plane_status={data_status}")
    elif not data_plane:
        watch_items.append("data_plane_recovery_missing")

    if findings:
        recommendations.append("./scripts/ops/opsctl.sh global-halt-refresh --json")
        recommendations.append("./scripts/ops/opsctl.sh paper-400-ramp --apply --json")
        status = "needs_work"
    elif watch_items:
        status = "watch"
    else:
        status = "ready"

    return _section(
        "truth_source_consistency",
        "Truth Source Consistency",
        status,
        evidence={
            "global_halt_auto_clear_present": bool(auto_clear),
            "global_killswitch_present": bool(killswitch),
            "chosen_halt_state": halt.get("halt_state") if halt else None,
            "chosen_halt": bool(halt.get("halt", False)) if halt else None,
            "chosen_clear_blockers": _clear_blockers(halt),
            "paper_ramp_stage": paper_stage,
            "paper_ramp_blockers": paper_blockers,
            "stale_global_blocker": stale_global_blocker,
            "data_plane_status": data_status,
            "write_failure_count": write_failures,
            "account_snapshot_failure_count": snapshot_failures,
        },
        findings=findings,
        watch_items=watch_items,
        recommendations=recommendations,
        blocks_guarded_paper=bool(findings),
    )


def _storage_writer_data_plane(ctx: dict[str, dict[str, Any]]) -> dict[str, Any]:
    writer = ctx["writer_process_intelligence"]
    storage = ctx["ingestion_storage"]
    drainer = ctx["backpressure_drainer_fleet"]
    process = ctx["process_watchdog"]
    data_plane = ctx["data_plane_recovery"]
    plumbing = ctx.get("system_plumbing_control", {})
    writer_health = _as_dict(writer.get("writer_health"))
    shard_contract = _as_dict(writer_health.get("shard_writer_lane_contract"))
    process_rows = _as_list(process.get("status"))
    raw_writer_running = sum(_safe_int(_as_dict(row).get("running"), 0) for row in process_rows if _as_dict(row).get("name") == "sql_link_writer")
    primary_count = _safe_int(shard_contract.get("primary_merge_writer_count"), 0)
    sqlite_count = _safe_int(shard_contract.get("sqlite_primary_writer_count"), 0)
    single_primary = bool(shard_contract.get("single_primary_merge_writer", False)) or (primary_count == 1 and sqlite_count <= 1)
    writer_lock_held = bool(writer_health.get("writer_lock_held", False)) or bool(drainer.get("writer_lock_held", False))
    risk_flags = {str(item) for item in _as_list(writer.get("risk_flags"))}
    storage_status = _status(storage.get("severity") or storage.get("overall_status") or "missing")
    pressure_index = _safe_float(storage.get("pressure_index"), 0.0)
    backpressure = _as_dict(storage.get("backpressure"))
    total_pending = _safe_int(backpressure.get("total_pending_lines"), 0)
    pending_threshold = max(_safe_int(backpressure.get("pending_lines_threshold"), 15000), 1)
    data_status = _status(data_plane.get("overall_status"))
    plumbing_status = _status(plumbing.get("overall_status"))
    overlay_relief = _storage_overlay_relief(storage, plumbing)
    overlay_relief_active = bool(overlay_relief.get("active", False))
    findings: list[str] = []
    watch_items: list[str] = []
    recommendations: list[str] = []

    duplicate_writer = bool("duplicate_sql_writer_processes" in risk_flags or (raw_writer_running > 1 and not single_primary))
    if duplicate_writer:
        findings.append("duplicate_sql_writer_processes")
    if writer and _is_hard_status(writer.get("overall_status")):
        findings.append(f"writer_status={_status(writer.get('overall_status'))}")
    if writer and not single_primary:
        watch_items.append("single_primary_merge_writer_proof_missing")
    if writer and not writer_lock_held:
        watch_items.append("writer_lock_not_held")
    if not writer:
        watch_items.append("writer_process_intelligence_missing")
    if storage_status in {"blocked", "critical", "high"} and not overlay_relief_active:
        findings.append(f"storage_status={storage_status}")
    elif storage_status in {"blocked", "critical", "high"} and overlay_relief_active:
        watch_items.append("storage_status_managed_by_sql_overlay_relief")
    if pressure_index >= 0.35 and not overlay_relief_active:
        findings.append("storage_pressure_index_high")
    elif pressure_index >= 0.35 and overlay_relief_active:
        watch_items.append("storage_pressure_index_managed_by_sql_overlay_relief")
    if total_pending >= pending_threshold:
        findings.append("storage_pending_above_threshold")
    if data_plane and data_status in {"blocked", "critical"}:
        findings.append(f"data_plane_status={data_status}")
    elif data_plane and data_status in {"degraded", "needs_work"}:
        watch_items.append(f"data_plane_status={data_status}")
    elif not data_plane:
        watch_items.append("data_plane_recovery_missing")
    if plumbing and plumbing_status in {"blocked", "critical"}:
        findings.append(f"system_plumbing_status={plumbing_status}")
    elif plumbing and plumbing_status in {"advisory", "watch"}:
        watch_items.append(f"system_plumbing_status={plumbing_status}")

    if duplicate_writer or (storage_status in {"blocked", "critical"} and not overlay_relief_active):
        status = "blocked"
    elif findings:
        status = "needs_work"
    elif watch_items:
        status = "watch"
    else:
        status = "ready"
    if status != "ready":
        recommendations.append("./scripts/ops/opsctl.sh writer-process-intelligence --apply --json")
        recommendations.append("./scripts/ops/opsctl.sh ingestion-storage-control --json")
        recommendations.append("./scripts/ops/opsctl.sh system-plumbing-control --json")

    return _section(
        "storage_writer_data_plane",
        "Storage Writer Data Plane",
        status,
        evidence={
            "writer_status": _status(writer.get("overall_status")),
            "writer_lock_held": writer_lock_held,
            "single_primary_merge_writer": single_primary,
            "primary_merge_writer_count": primary_count,
            "sqlite_primary_writer_count": sqlite_count,
            "raw_sql_link_writer_running_count": raw_writer_running,
            "risk_flags": sorted(risk_flags),
            "storage_status": storage_status,
            "storage_pressure_index": pressure_index,
            "storage_overlay_relief": overlay_relief,
            "total_pending_lines": total_pending,
            "pending_lines_threshold": pending_threshold,
            "data_plane_status": data_status,
            "system_plumbing_status": plumbing_status,
            "system_plumbing_score": _safe_int(plumbing.get("plumbing_score"), 0),
            "system_plumbing_blockers": plumbing.get("blockers", []),
            "system_plumbing_warnings": plumbing.get("warnings", []),
        },
        findings=findings,
        watch_items=watch_items,
        recommendations=recommendations,
        blocks_guarded_paper=status in {"blocked", "needs_work"},
    )


def _runtime_capacity_partition(ctx: dict[str, dict[str, Any]]) -> dict[str, Any]:
    runtime = ctx["runtime_throttle"]
    memory = ctx["memory_efficiency"]
    storage = ctx["ingestion_storage"]
    swap = _as_dict(ctx["swap_pressure"].get("swap_pressure"))
    pressure = ctx["pressure_relief_control"]
    live_separation = ctx["live_runtime_separation"]
    health_fast = ctx["health_fast"]
    plumbing = ctx.get("system_plumbing_control", {})
    plumbing_runtime = _as_dict(_as_dict(plumbing.get("sections")).get("runtime_memory"))
    guarded_paper = _as_dict(_as_dict(health_fast.get("operational_readiness")).get("guarded_paper"))
    runtime_status = _status(runtime.get("overall_status"))
    host_score = _safe_float(runtime.get("host_saturation_score"), 0.0)
    compute_level = _status(runtime.get("compute_pressure_level") or runtime.get("cpu_pressure_level") or "normal")
    memory_level = _status(runtime.get("memory_pressure_level") or "normal")
    memory_status = _status(memory.get("overall_status"))
    storage_status = _status(storage.get("severity") or storage.get("overall_status"))
    storage_pressure = _safe_float(storage.get("pressure_index"), 0.0)
    storage_backpressure = _as_dict(storage.get("backpressure"))
    storage_total_pending = _safe_int(storage_backpressure.get("total_pending_lines"), 0)
    storage_pending_threshold = _safe_int(storage_backpressure.get("pending_lines_threshold"), 0)
    storage_clear = (
        bool(storage)
        and storage_status in {"stable", "ready", "normal", "calm"}
        and storage_pressure <= 0.25
        and (storage_pending_threshold <= 0 or storage_total_pending <= storage_pending_threshold)
    )
    swap_tier = _status(swap.get("tier") or "normal")
    pressure_tier = _status(pressure.get("tier") or pressure.get("overall_status"))
    runtime_soft_reclassification = _as_dict(runtime.get("soft_cap_advisory_reclassification"))
    runtime_soft_measurements = _as_dict(runtime_soft_reclassification.get("measurements"))
    runtime_soft_to_status = _status(runtime_soft_reclassification.get("to_status"))
    runtime_soft_reason = str(runtime_soft_reclassification.get("reason") or "")
    live_status = _status(live_separation.get("overall_status"))
    live_release = _as_dict(live_separation.get("release_contract"))
    live_clearance = _as_dict(live_separation.get("clearance_plan"))
    live_clearance_state = _status(live_clearance.get("clearance_state"))
    live_read_only_policy = bool(
        live_release.get("live_lane_should_be_read_only", False)
        or live_clearance_state in {"protect_live", "read_only", "operator_gated"}
    )
    mac_fluidity = _as_dict(runtime.get("mac_fluidity_contract"))
    mac_fluidity_status = _status(mac_fluidity.get("overall_status"))
    mac_fluidity_band = _status(mac_fluidity.get("fluidity_band"))
    mac_fluidity_score = _safe_float(mac_fluidity.get("fluidity_score"), 0.0)
    plumbing_runtime_memory_relief = bool(
        _status(plumbing.get("overall_status")) in {"ready", "guarded_ready", "advisory"}
        and bool(plumbing_runtime.get("ok", False))
        and bool(plumbing_runtime.get("paper_only_runtime_memory_relief", False))
        and _status(plumbing_runtime.get("memory_pressure_level")) not in {"high", "critical"}
    )
    findings: list[str] = []
    watch_items: list[str] = []
    recommendations: list[str] = []

    if runtime_status in {"blocked", "critical"} or host_score >= 85.0 or compute_level in {"critical", "high"}:
        findings.append(f"runtime_pressure={runtime_status or compute_level or host_score}")
    elif runtime_status in {"degraded", "needs_work"} or host_score >= 65.0 or compute_level == "elevated":
        watch_items.append("runtime_capacity_elevated")
    elif runtime_status in {"advisory", "guarded_ready"} and host_score >= 50.0:
        watch_items.append("runtime_advisory_active")
    if memory_status in {"blocked", "critical", "needs_work", "degraded"}:
        findings.append(f"memory_status={memory_status}")
    if memory_level in {"critical", "high"}:
        findings.append(f"memory_pressure_level={memory_level}")
    elif memory_level == "elevated":
        watch_items.append("memory_pressure_elevated")
    if swap_tier not in {"normal", "calm", ""}:
        findings.append(f"swap_tier={swap_tier}")
    if live_status and live_status not in READY_STATES:
        if live_status in {"blocked", "degraded"} and live_read_only_policy:
            pass
        else:
            watch_items.append(f"live_runtime_separation_status={live_status}")
    if mac_fluidity:
        if mac_fluidity_status in {"blocked", "critical", "needs_work"} or mac_fluidity_score < 75.0:
            findings.append(f"mac_fluidity_status={mac_fluidity_status or 'thin'}")
        elif mac_fluidity_status in {"watch", "thin", "advisory"} or mac_fluidity_score < 90.0:
            watch_items.append(f"mac_fluidity_status={mac_fluidity_status or 'watch'}")
    if not runtime:
        watch_items.append("runtime_throttle_missing")
    if not memory:
        watch_items.append("memory_efficiency_missing")

    runtime_capacity_debt = {
        "runtime_capacity_elevated",
        "runtime_advisory_active",
        f"mac_fluidity_status={mac_fluidity_status or 'watch'}",
    }
    mac_fluidity_managed = (
        not mac_fluidity
        or (
            mac_fluidity_status in {"ready", "watch", "advisory"}
            and mac_fluidity_band in {"", "guarded_smooth", "smooth", "ready", "normal"}
            and mac_fluidity_score >= 85.0
        )
    )
    managed_capacity_contract = {
        "active": bool(
            watch_items
            and not findings
            and set(watch_items).issubset(runtime_capacity_debt)
            and runtime_status in {"ready", "guarded_ready"}
            and host_score < 65.0
            and compute_level in {"normal", "elevated"}
            and memory_status in {"ready", "normal", "stable", ""}
            and memory_level in {"normal", ""}
            and swap_tier in {"normal", "calm", ""}
            and storage_clear
            and mac_fluidity_managed
            and bool(health_fast.get("read_only", True))
            and bool(guarded_paper.get("ok", False))
            and _status(guarded_paper.get("status")) == "ready"
            and (not live_status or live_status in READY_STATES or live_read_only_policy)
        ),
        "policy": "ready_runtime_soft_cap_and_guarded_mac_fluidity_are_capacity_governed_not_architecture_debt",
        "runtime_ready": runtime_status in {"ready", "guarded_ready"},
        "host_below_capacity_watch_ceiling": host_score < 65.0,
        "memory_clear": memory_status in {"ready", "normal", "stable", ""} and memory_level in {"normal", ""},
        "swap_clear": swap_tier in {"normal", "calm", ""},
        "storage_clear": storage_clear,
        "mac_fluidity_managed": mac_fluidity_managed,
        "read_only": bool(health_fast.get("read_only", True)),
        "guarded_paper_ready": bool(guarded_paper.get("ok", False)) and _status(guarded_paper.get("status")) == "ready",
        "managed_watch_items": sorted(set(watch_items) & runtime_capacity_debt),
    }
    managed_runtime_ready_contract = {
        "active": bool(
            (findings or watch_items)
            and runtime_status in {"ready", "guarded_ready"}
            and bool(runtime_soft_reclassification.get("active", False))
            and runtime_soft_to_status == "ready"
            and bool(runtime_soft_measurements.get("runtime_ready_guarded", False))
            and host_score < 75.0
            and compute_level in {"normal", "elevated", "high"}
            and memory_status in {"ready", "normal", "stable", ""}
            and memory_level in {"normal", ""}
            and swap_tier in {"normal", "calm", ""}
            and storage_clear
            and bool(health_fast.get("read_only", True))
            and bool(guarded_paper.get("ok", False))
            and _status(guarded_paper.get("status")) == "ready"
            and (not live_status or live_status in READY_STATES or live_read_only_policy)
            and mac_fluidity_status not in {"blocked", "critical"}
            and (not mac_fluidity or mac_fluidity_score >= 65.0)
        ),
        "policy": "runtime_ready_guarded_contract_manages_bounded_writer_or_downshift_pressure_without_opening_live_execution",
        "runtime_soft_to_status": runtime_soft_to_status,
        "runtime_soft_reason": runtime_soft_reason,
        "runtime_ready_guarded": bool(runtime_soft_measurements.get("runtime_ready_guarded", False)),
        "storage_writer_cooling_guarded_ready": bool(runtime_soft_measurements.get("storage_writer_cooling_guarded_ready", False)),
        "bounded_writer_with_paper_shadow_guarded_ready": bool(runtime_soft_measurements.get("bounded_writer_with_paper_shadow_guarded_ready", False)),
        "support_low_priority_guarded_ready": bool(runtime_soft_measurements.get("support_low_priority_guarded_ready", False)),
        "mac_fluidity_score_floor": 65.0,
        "managed_findings": list(findings),
        "managed_watch_items": list(watch_items),
    }
    managed_plumbing_runtime_contract = {
        "active": bool(
            (findings or watch_items)
            and plumbing_runtime_memory_relief
            and host_score < 75.0
            and compute_level in {"normal", "elevated"}
            and memory_level not in {"high", "critical"}
            and swap_tier in {"normal", "calm", ""}
            and storage_clear
            and bool(health_fast.get("read_only", True))
            and bool(guarded_paper.get("ok", False))
            and _status(guarded_paper.get("status")) == "ready"
            and (not live_status or live_status in READY_STATES or live_read_only_policy)
            and mac_fluidity_status not in {"blocked", "critical"}
            and (not mac_fluidity or mac_fluidity_score >= 65.0)
        ),
        "policy": "system_plumbing_paper_only_runtime_memory_relief_turns_elevated_soft_cap_capacity_into_architecture_watch",
        "plumbing_status": _status(plumbing.get("overall_status")),
        "plumbing_runtime_status": _status(plumbing_runtime.get("status")),
        "paper_only_runtime_memory_relief": plumbing_runtime_memory_relief,
        "host_below_guarded_ceiling": host_score < 75.0,
        "memory_pressure_level": memory_level,
        "managed_findings": list(findings),
        "managed_watch_items": list(watch_items),
    }
    if managed_capacity_contract["active"] or managed_runtime_ready_contract["active"] or managed_plumbing_runtime_contract["active"]:
        findings = []
        watch_items = []

    if findings:
        recommendations.append("./scripts/ops/opsctl.sh runtime-throttle --apply --json")
        recommendations.append("./scripts/ops/opsctl.sh memory-efficiency apply --json")
        status = "needs_work"
    elif watch_items:
        recommendations.append("./scripts/ops/opsctl.sh runtime-throttle --apply --json")
        status = "watch"
    else:
        status = "ready"

    return _section(
        "runtime_capacity_partition",
        "Runtime Capacity Partition",
        status,
        evidence={
            "runtime_status": runtime_status,
            "host_saturation_score": host_score,
            "compute_pressure_level": compute_level,
            "memory_pressure_level": memory_level,
            "memory_status": memory_status,
            "storage_status": storage_status,
            "storage_pressure_index": storage_pressure,
            "storage_total_pending_lines": storage_total_pending,
            "storage_pending_lines_threshold": storage_pending_threshold,
            "storage_clear_for_runtime_capacity": storage_clear,
            "swap_tier": swap_tier,
            "pressure_relief_tier": pressure_tier,
            "runtime_soft_reclassification_active": bool(runtime_soft_reclassification.get("active", False)),
            "runtime_soft_reclassification_to_status": runtime_soft_to_status,
            "runtime_soft_reclassification_reason": runtime_soft_reason,
            "live_runtime_separation_status": live_status,
            "live_runtime_separation_clearance_state": live_clearance_state,
            "live_runtime_separation_read_only_policy": live_read_only_policy,
            "mac_fluidity_status": mac_fluidity_status,
            "mac_fluidity_band": mac_fluidity.get("fluidity_band") if mac_fluidity else "",
            "mac_fluidity_score": mac_fluidity_score if mac_fluidity else None,
            "managed_capacity_contract": managed_capacity_contract,
            "managed_runtime_ready_contract": managed_runtime_ready_contract,
            "managed_plumbing_runtime_contract": managed_plumbing_runtime_contract,
        },
        findings=findings,
        watch_items=watch_items,
        recommendations=recommendations,
    )


def _collector_process_quarantine(ctx: dict[str, dict[str, Any]]) -> dict[str, Any]:
    process = ctx["process_watchdog"]
    health_fast = ctx["health_fast"]
    guarded_paper = _as_dict(_as_dict(health_fast.get("operational_readiness")).get("guarded_paper"))
    health_process = _as_dict(health_fast.get("process_watchdog"))
    alert_summary = _as_dict(health_process.get("alert_summary"))
    if not alert_summary:
        alert_summary = _as_dict(process.get("alert_summary"))
    restart_storm = _as_dict(health_process.get("restart_storm_isolation")) or _as_dict(process.get("restart_storm_isolation"))
    safety_pause = _as_dict(health_process.get("safety_pause")) or _as_dict(process.get("safety_pause"))
    critical_count = _safe_int(alert_summary.get("critical_count"), 0)
    warning_count = _safe_int(alert_summary.get("warning_count"), 0)
    isolated_count = _safe_int(restart_storm.get("isolated_count"), 0)
    execution_blocking = _safe_int(restart_storm.get("execution_blocking_count"), 0)
    isolated_targets = [str(item) for item in _as_list(restart_storm.get("isolated_targets")) if str(item).strip()]
    isolated_target_set = set(isolated_targets)
    alert_rows = [_as_dict(row) for row in _as_list(alert_summary.get("rows"))]
    warning_rows = [
        row
        for row in alert_rows
        if _status(row.get("severity")) in {"warn", "warning"}
        or _status(row.get("type")) == "restart_storm"
        or str(row.get("target") or "").strip() in isolated_target_set
    ]
    warning_rows_cover_count = bool(warning_rows) and len(warning_rows) >= warning_count
    warning_rows_are_isolated = all(str(row.get("target") or "").strip() in isolated_target_set for row in warning_rows)
    warning_rows_are_nonblocking = all(not bool(row.get("blocks_guarded_paper", False)) for row in warning_rows)
    findings: list[str] = []
    watch_items: list[str] = []
    recommendations: list[str] = []

    if bool(safety_pause.get("active", False)):
        findings.append("process_safety_pause_active")
    if critical_count > 0:
        findings.append("critical_process_alerts_active")
    if execution_blocking > 0:
        findings.append("execution_blocking_restart_storms")
    if warning_count > 0:
        watch_items.append("warning_process_alerts_active")
    if isolated_count > 0:
        watch_items.append("read_only_restart_storms_isolated")
    if not process and not health_fast:
        watch_items.append("process_watchdog_missing")

    managed_quarantine_contract = {
        "active": bool(
            watch_items
            and not findings
            and set(watch_items).issubset({"warning_process_alerts_active", "read_only_restart_storms_isolated"})
            and isolated_count > 0
            and execution_blocking == 0
            and critical_count == 0
            and (
                warning_count == 0
                or (warning_rows_cover_count and warning_rows_are_isolated and warning_rows_are_nonblocking)
            )
            and bool(health_fast.get("read_only", True))
            and bool(guarded_paper.get("ok", False))
            and _status(guarded_paper.get("status")) == "ready"
        ),
        "policy": "isolated_read_only_collection_restart_storms_do_not_weaken_architecture_when_execution_is_blocked",
        "warning_rows_cover_count": warning_rows_cover_count,
        "warning_rows_are_isolated": warning_rows_are_isolated,
        "warning_rows_are_nonblocking": warning_rows_are_nonblocking,
        "read_only": bool(health_fast.get("read_only", True)),
        "guarded_paper_ready": bool(guarded_paper.get("ok", False)) and _status(guarded_paper.get("status")) == "ready",
    }
    if managed_quarantine_contract["active"]:
        watch_items = []

    if findings:
        status = "needs_work"
    elif watch_items:
        status = "watch"
    else:
        status = "ready"
    if status != "ready":
        recommendations.append("./scripts/ops/opsctl.sh process-watchdog --json")
        recommendations.append("./scripts/ops/opsctl.sh coinbase-api-health --snapshot --json")

    return _section(
        "collector_process_quarantine",
        "Collector Process Quarantine",
        status,
        evidence={
            "critical_alert_count": critical_count,
            "warning_alert_count": warning_count,
            "isolated_restart_storm_count": isolated_count,
            "execution_blocking_restart_storm_count": execution_blocking,
            "isolated_targets": isolated_targets,
            "safety_pause_active": bool(safety_pause.get("active", False)),
            "managed_quarantine_contract": managed_quarantine_contract,
        },
        findings=findings,
        watch_items=watch_items,
        recommendations=recommendations,
        blocks_guarded_paper=bool(findings),
    )


def _health_fast_strict_clear(ctx: dict[str, dict[str, Any]]) -> bool:
    health_fast = ctx.get("health_fast") if isinstance(ctx.get("health_fast"), dict) else {}
    guarded_paper = _as_dict(_as_dict(health_fast.get("operational_readiness")).get("guarded_paper"))
    return bool(
        health_fast.get("ok", False)
        and health_fast.get("strict_all_clear", False)
        and guarded_paper.get("ok", False)
        and _status(guarded_paper.get("status")) == "ready"
    )


def _health_fast_guarded_paper_ready(ctx: dict[str, dict[str, Any]]) -> bool:
    health_fast = ctx.get("health_fast") if isinstance(ctx.get("health_fast"), dict) else {}
    guarded_paper = _as_dict(_as_dict(health_fast.get("operational_readiness")).get("guarded_paper"))
    global_halt = _as_dict(health_fast.get("global_halt"))
    return bool(
        health_fast.get("ok", False)
        and health_fast.get("read_only", True)
        and bool(guarded_paper.get("ok", False))
        and _status(guarded_paper.get("status")) == "ready"
        and not bool(global_halt.get("halt", False))
        and not _clear_blockers(global_halt)
    )


def _platform_watch_semantics(ctx: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_names = [
        "platform_intelligence",
        "platform_brain_v5",
        "platform_stabilization_quality",
        "platform_settlement_stabilization",
    ]
    findings: list[str] = []
    watch_items: list[str] = []
    statuses: dict[str, str] = {}
    for name in source_names:
        payload = ctx[name]
        status = _status(payload.get("overall_status"))
        statuses[name] = status
        if not payload:
            watch_items.append(f"{name}_missing")
        elif status in {"blocked", "critical", "needs_work", "degraded"}:
            findings.append(f"{name}_status={status}")
        elif status in WATCH_STATES:
            watch_items.append(f"{name}_status={status}")

    managed_watch_contract = {
        "active": False,
        "reason": "",
        "strict_all_clear": _health_fast_strict_clear(ctx),
        "guarded_paper_ready": _health_fast_guarded_paper_ready(ctx),
        "watch_count": len(watch_items),
    }
    if findings:
        status = "needs_work"
    elif watch_items and _health_fast_strict_clear(ctx):
        managed_watch_contract.update(
            {
                "active": True,
                "reason": "platform_watch_states_are_nonblocking_under_strict_all_clear",
            }
        )
        status = "ready"
    elif watch_items and _health_fast_guarded_paper_ready(ctx):
        managed_watch_contract.update(
            {
                "active": True,
                "reason": "platform_watch_states_are_nonblocking_under_guarded_paper_ready",
            }
        )
        status = "ready"
    elif watch_items:
        status = "watch"
    else:
        status = "ready"
    return _section(
        "platform_watch_semantics",
        "Platform Watch Semantics",
        status,
        evidence={"source_statuses": statuses, "managed_watch_contract": managed_watch_contract},
        findings=findings,
        watch_items=[] if bool(managed_watch_contract.get("active", False)) else watch_items,
        recommendations=["./scripts/ops/opsctl.sh platform-stabilization --apply --json"] if findings else [],
    )


def _training_evidence_contract(ctx: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rollup = ctx["collection_rollup"]
    quality = ctx["training_quality"]
    runtime = ctx["training_runtime"]
    collector_count = _safe_int(rollup.get("collector_count"), 0)
    observed_count = _safe_int(rollup.get("bots_with_observations"), 0)
    zero_count = _safe_int(rollup.get("zero_observation_count"), 0)
    coverage_ratio = float(observed_count / collector_count) if collector_count else 0.0
    quality_status = _status(quality.get("overall_status"))
    quality_score = _safe_float(quality.get("training_quality_score", quality.get("training_quality_index")), 0.0)
    launch_blockers = [str(item) for item in _as_list(runtime.get("launch_blockers")) if str(item).strip()]
    findings: list[str] = []
    watch_items: list[str] = []

    if rollup and collector_count and coverage_ratio < 0.9:
        findings.append("collector_observation_coverage_low")
    elif rollup and collector_count and coverage_ratio < 0.98:
        watch_items.append("collector_observation_coverage_thin")
    if rollup and zero_count > max(5, int(collector_count * 0.05)):
        findings.append("zero_observation_count_high")
    elif rollup and zero_count > 0:
        watch_items.append("zero_observation_collectors_present")
    if quality and quality_status in {"blocked", "critical", "degraded", "needs_work"} and quality_score < 75.0:
        findings.append(f"training_quality_status={quality_status}")
    elif quality and (quality_status in WATCH_STATES or quality_score < 80.0):
        watch_items.append("training_quality_watch")
    elif not quality:
        watch_items.append("training_quality_missing")
    if launch_blockers:
        watch_items.append("training_runtime_launch_blockers_present")
    if not rollup:
        watch_items.append("collection_rollup_missing")

    managed_training_evidence_contract = {
        "active": False,
        "reason": "",
        "guarded_paper_ready": _health_fast_guarded_paper_ready(ctx),
        "strict_all_clear": _health_fast_strict_clear(ctx),
        "collection_flowing": bool(collector_count > 0 and observed_count > 0 and _safe_int(rollup.get("total_observations"), 0) > 0),
        "training_quality_score": quality_score,
    }
    if findings:
        if (
            _health_fast_guarded_paper_ready(ctx)
            and managed_training_evidence_contract["collection_flowing"]
            and quality_score >= 75.0
        ):
            managed_training_evidence_contract.update(
                {
                    "active": True,
                    "reason": "collection_maturity_debt_is_nonblocking_for_guarded_paper_soak",
                }
            )
            status = "watch"
        else:
            status = "needs_work"
    elif watch_items:
        status = "watch"
    else:
        status = "ready"
    return _section(
        "training_evidence_contract",
        "Training Evidence Contract",
        status,
        evidence={
            "collector_count": collector_count,
            "bots_with_observations": observed_count,
            "coverage_ratio": round(coverage_ratio, 4),
            "zero_observation_count": zero_count,
            "training_quality_status": quality_status,
            "training_quality_score": quality_score,
            "training_runtime_launch_allowed": bool(runtime.get("launch_allowed", False)) if runtime else None,
            "training_runtime_launch_blockers": launch_blockers,
            "managed_training_evidence_contract": managed_training_evidence_contract,
        },
        findings=findings,
        watch_items=watch_items,
        recommendations=["./scripts/ops/opsctl.sh training-quality --json"] if status != "ready" else [],
    )


def _provider_source_mesh(ctx: dict[str, dict[str, Any]]) -> dict[str, Any]:
    provider = ctx["provider_mesh"]
    source = ctx["source_verification"]
    summary = _as_dict(provider.get("summary"))
    required_ok = _safe_int(summary.get("required_contract_ok"), 0)
    required_total = _safe_int(summary.get("required_collectors"), 0)
    cooldown_count = len(_as_list(provider.get("cooldowns")))
    required_failures = [str(item) for item in _as_list(provider.get("required_failures")) if str(item).strip()]
    soft_failures = [str(item) for item in _as_list(provider.get("soft_failures")) if str(item).strip()]
    source_status = _status(source.get("overall_status") or _as_dict(source.get("overall")).get("overall_status"))
    provider_status = _status(provider.get("overall_status"))
    autorefresh_contract = _as_dict(source.get("autorefresh_contract"))
    unverified_sources = {str(item) for item in _as_list(source.get("unverified_sources")) if str(item).strip()}
    stale_sources = {str(item) for item in _as_list(source.get("stale_artifacts")) if str(item).strip()}
    degraded_sources = {str(item) for item in _as_list(source.get("degraded_artifacts")) if str(item).strip()}
    critical_source_debt = sorted(item for item in degraded_sources | unverified_sources if item in CORE_SOURCE_IDS)
    managed_verification_debt = sorted(item for item in degraded_sources | unverified_sources if item in MANAGED_VERIFICATION_SOURCE_IDS)
    optional_source_debt = sorted(
        item
        for item in degraded_sources | stale_sources | unverified_sources
        if item not in CORE_SOURCE_IDS and item not in MANAGED_VERIFICATION_SOURCE_IDS
    )
    required_provider_ready = bool(required_total > 0 and required_ok >= required_total and not required_failures)
    optional_provider_debt = bool(provider and provider_status in {"degraded", "needs_work"} and required_provider_ready)
    optional_source_debt_isolated = bool(
        source
        and source_status in {"degraded", "needs_work", "thin", "watch", "advisory"}
        and not critical_source_debt
        and bool(autorefresh_contract.get("enabled", False))
    )
    source_mesh_debt_contract = {
        "active": bool(optional_provider_debt or optional_source_debt_isolated),
        "required_provider_ready": required_provider_ready,
        "optional_provider_debt": optional_provider_debt,
        "optional_source_debt_isolated": optional_source_debt_isolated,
        "critical_source_debt": critical_source_debt,
        "managed_verification_debt": managed_verification_debt,
        "optional_source_debt": optional_source_debt,
        "autorefresh_enabled": bool(autorefresh_contract.get("enabled", False)),
        "policy": "required_source_mesh_blocks_architecture_optional_source_debt_is_governed_by_bounded_refresh",
    }
    managed_guarded_paper_source_debt = bool(
        _health_fast_guarded_paper_ready(ctx)
        and required_provider_ready
        and bool(autorefresh_contract.get("enabled", False))
        and not required_failures
        and provider_status not in {"blocked", "critical"}
        and critical_source_debt
    )
    if managed_guarded_paper_source_debt:
        source_mesh_debt_contract.update(
            {
                "active": True,
                "guarded_paper_source_debt_advisory": True,
                "managed_reason": "core_source_verification_refresh_debt_is_nonblocking_for_guarded_paper_soak",
            }
        )
    findings: list[str] = []
    watch_items: list[str] = []

    if provider and provider_status in {"blocked", "critical"}:
        findings.append(f"provider_mesh_status={provider_status}")
    elif provider and provider_status in {"degraded", "needs_work"} and not optional_provider_debt:
        watch_items.append(f"provider_mesh_status={provider_status}")
    if required_total > 0 and required_ok < required_total:
        findings.append("required_provider_contract_incomplete")
    if required_failures:
        findings.append("required_provider_failures_present")
    if cooldown_count > 0:
        watch_items.append("provider_cooldowns_present")
    if source and source_status in {"blocked", "critical"}:
        findings.append(f"source_verification_status={source_status}")
    elif critical_source_debt and managed_guarded_paper_source_debt:
        watch_items.append("core_source_verification_debt_managed_by_guarded_paper_autorefresh")
    elif critical_source_debt:
        findings.append("critical_source_verification_debt_present")
    elif source and source_status in {"degraded", "needs_work", "thin", "watch", "advisory"} and not optional_source_debt_isolated:
        watch_items.append(f"source_verification_status={source_status}")
    if not provider:
        watch_items.append("provider_mesh_missing")
    if not source:
        watch_items.append("source_verification_missing")

    if findings:
        status = "needs_work"
    elif watch_items:
        status = "watch"
    else:
        status = "ready"
    return _section(
        "provider_source_mesh",
        "Provider Source Mesh",
        status,
        evidence={
            "provider_mesh_status": provider_status,
            "required_contract_ok": required_ok,
            "required_collectors": required_total,
            "provider_cooldown_count": cooldown_count,
            "source_verification_status": source_status,
            "required_provider_failures": required_failures,
            "soft_provider_failures": soft_failures,
            "source_mesh_debt_contract": source_mesh_debt_contract,
        },
        findings=findings,
        watch_items=watch_items,
        recommendations=["./scripts/ops/opsctl.sh provider-mesh --json", "./scripts/ops/opsctl.sh source-verification --json"] if status != "ready" else [],
    )


def _opsctl_command_spine(project_root: Path) -> dict[str, Any]:
    opsctl = project_root / "scripts" / "ops" / "opsctl.sh"
    try:
        text = opsctl.read_text(encoding="utf-8")
    except Exception:
        text = ""
    missing = [cmd for cmd in REQUIRED_OPSCTL_COMMANDS if cmd not in text]
    status = "ready" if text and not missing else "needs_work"
    return _section(
        "opsctl_command_spine",
        "Opsctl Command Spine",
        status,
        evidence={
            "opsctl_path": str(opsctl),
            "required_command_count": len(REQUIRED_OPSCTL_COMMANDS),
            "missing_commands": missing,
        },
        findings=[f"missing_opsctl_command={cmd}" for cmd in missing],
        recommendations=["./scripts/ops/opsctl.sh commands-verify --json"] if missing else [],
    )


def _recommended_commands(sections: dict[str, dict[str, Any]]) -> list[list[str]]:
    commands: list[str] = []
    for section in sections.values():
        commands.extend(str(item) for item in _as_list(section.get("recommendations")))
    commands.extend(
        [
            "./scripts/ops/opsctl.sh system-architecture-hardening --apply --json",
            "./scripts/ops/opsctl.sh health-fast --json",
        ]
    )
    return [cmd.split(" ") for cmd in ordered_unique(commands)]


def _recommended_env_overrides(overall_status: str) -> dict[str, str]:
    return {
        "SYSTEM_ARCHITECTURE_HARDENING_ENABLED": "1",
        "SYSTEM_ARCHITECTURE_HARDENING_STATUS": overall_status,
        "ALLOW_ORDER_EXECUTION": "0",
        "PAPER_TRADE_LOCK": "1",
        "LIVE_EXECUTION_OPERATOR_GATED": "1",
        "TOP_BOT_ENABLE_LIVE_EXECUTION": "0",
        "EXECUTION_LANE_LIVE_ENABLED": "0",
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    ctx = {
        "health_fast": _health(project_root, "health_fast_latest.json"),
        "paper_ramp": _health(project_root, "paper_400_ramp_latest.json"),
        "global_halt_auto_clear": _health(project_root, "global_halt_auto_clear_latest.json"),
        "global_killswitch": _health(project_root, "global_killswitch_latest.json"),
        "data_plane_recovery": _health(project_root, "data_plane_recovery_controller_latest.json"),
        "writer_process_intelligence": _health(project_root, "writer_process_intelligence_latest.json"),
        "backpressure_drainer_fleet": _health(project_root, "backpressure_drainer_fleet_latest.json"),
        "ingestion_storage": _health(project_root, "ingestion_storage_control_latest.json"),
        "runtime_throttle": _health(project_root, "runtime_throttle_control_latest.json"),
        "memory_efficiency": _health(project_root, "memory_efficiency_control_latest.json"),
        "swap_pressure": _health(project_root, "swap_pressure_governor_latest.json"),
        "pressure_relief_control": _health(project_root, "pressure_relief_control_latest.json"),
        "process_watchdog": _health(project_root, "process_watchdog_latest.json"),
        "platform_intelligence": _health(project_root, "platform_intelligence_expansion_latest.json"),
        "platform_brain_v5": _health(project_root, "platform_brain_v5_latest.json"),
        "platform_stabilization_quality": _health(project_root, "platform_stabilization_quality_latest.json"),
        "platform_settlement_stabilization": _health(project_root, "platform_settlement_stabilization_latest.json"),
        "collection_rollup": _health(project_root, "data_collection_observation_rollup_latest.json"),
        "training_quality": _health(project_root, "training_quality_control_latest.json"),
        "training_runtime": _health(project_root, "training_runtime_control_latest.json"),
        "provider_mesh": _health(project_root, "provider_mesh_latest.json"),
        "source_verification": _health(project_root, "source_verification_latest.json"),
        "live_runtime_separation": _health(project_root, "live_runtime_separation_control_latest.json"),
        "system_plumbing_control": _health(project_root, "system_plumbing_control_latest.json"),
    }
    sections = {
        "safety_execution_boundary": _safety_execution_boundary(ctx),
        "truth_source_consistency": _truth_source_consistency(ctx),
        "storage_writer_data_plane": _storage_writer_data_plane(ctx),
        "runtime_capacity_partition": _runtime_capacity_partition(ctx),
        "collector_process_quarantine": _collector_process_quarantine(ctx),
        "platform_watch_semantics": _platform_watch_semantics(ctx),
        "training_evidence_contract": _training_evidence_contract(ctx),
        "provider_source_mesh": _provider_source_mesh(ctx),
        "opsctl_command_spine": _opsctl_command_spine(project_root),
    }
    overall_status = _rollup_status(sections)
    anatomy_layers = _anatomy_layers(sections)
    anatomy_status = _rollup_status(anatomy_layers)
    anatomy_scores = [_safe_float(layer.get("strength_score"), 0.0) for layer in anatomy_layers.values()]
    anatomy_strength_score = round(sum(anatomy_scores) / len(anatomy_scores), 2) if anatomy_scores else 0.0
    hard_sections = [
        name
        for name, section in sections.items()
        if _status(section.get("overall_status")) in {"blocked", "critical", "needs_work", "degraded"}
    ]
    watch_sections = [
        name
        for name, section in sections.items()
        if _status(section.get("overall_status")) in {"watch", "thin", "advisory"}
    ]
    anatomy_hard_layers = [
        name
        for name, layer in anatomy_layers.items()
        if _status(layer.get("overall_status")) in {"blocked", "critical", "needs_work", "degraded"}
    ]
    anatomy_watch_layers = [
        name
        for name, layer in anatomy_layers.items()
        if _status(layer.get("overall_status")) in {"watch", "thin", "advisory"}
    ]
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status in {"ready", "watch"},
        "overall_status": overall_status,
        "read_only": True,
        "started_heavy_reports": False,
        "section_count": len(sections),
        "hard_section_count": len(hard_sections),
        "watch_section_count": len(watch_sections),
        "hard_sections": hard_sections,
        "watch_sections": watch_sections,
        "section_statuses": {name: section.get("overall_status") for name, section in sections.items()},
        "sections": sections,
        "anatomy_status": anatomy_status,
        "anatomy_strength_score": anatomy_strength_score,
        "anatomy_strength_label": _strength_label(anatomy_status, anatomy_strength_score),
        "anatomy_layer_count": len(anatomy_layers),
        "anatomy_hard_layers": anatomy_hard_layers,
        "anatomy_watch_layers": anatomy_watch_layers,
        "anatomy_layers": anatomy_layers,
        "architecture_invariants": [
            "live_order_execution_requires_explicit_operator_gate",
            "guarded_paper_readiness_must_match_current_global_halt_and_paper_ramp_truth",
            "sql_writer_must_have_single_primary_merge_writer_proof",
            "runtime_capacity_debt_must_not_be_promoted_to_live_readiness",
            "isolated_read_only_collector_repairs_must_not_block_guarded_paper",
            "platform_watch_states_must_not_roll_up_as_hard_degradation",
            "training_and_provider_evidence_must_back_promotion_readiness",
            "opsctl_command_spine_must_reach_all_core_control_planes",
            "body_strength_requires_guarded_paper_runtime_and_training_evidence_alignment",
            "skeleton_strength_requires_command_spine_truth_source_and_single_writer_proof",
            "heart_strength_requires_writer_pump_collector_quarantine_and_runtime_circulation",
            "mind_strength_requires_evidence_source_confidence_and_watch_semantics",
        ],
        "recommended_env_overrides": _recommended_env_overrides(overall_status),
        "recommended_commands": _recommended_commands(sections),
        "next_best_command": "./scripts/ops/opsctl.sh system-architecture-hardening --apply --json",
        "policy": "advisory/read-only architecture hardening; live execution remains operator gated",
    }


def write_section_artifacts(project_root: Path, payload: dict[str, Any]) -> dict[str, str]:
    section_dir = project_root / "governance" / "system_architecture_hardening"
    written: dict[str, str] = {}
    for name, section in _as_dict(payload.get("sections")).items():
        path = section_dir / f"{name}.json"
        write_payload(path, section)
        written[name] = str(path)
    return written


def write_anatomy_artifacts(project_root: Path, payload: dict[str, Any]) -> dict[str, str]:
    anatomy_dir = project_root / "governance" / "system_architecture_hardening" / "anatomy"
    written: dict[str, str] = {}
    for name, layer in _as_dict(payload.get("anatomy_layers")).items():
        path = anatomy_dir / f"{name}.json"
        write_payload(path, layer)
        written[name] = str(path)
    return written


def write_config(project_root: Path, payload: dict[str, Any]) -> Path:
    path = project_root / "config" / "system_architecture_hardening_v1.json"
    config = {
        "schema_version": 1,
        "updated_at_utc": payload.get("timestamp_utc"),
        "enabled": True,
        "read_only": True,
        "invariants": payload.get("architecture_invariants", []),
        "anatomy_layers": {
            name: {
                "role": definition.get("role"),
                "sections": definition.get("sections"),
                "contract": definition.get("contract"),
            }
            for name, definition in ANATOMY_LAYER_DEFINITIONS.items()
        },
        "required_opsctl_commands": REQUIRED_OPSCTL_COMMANDS,
        "live_enable_flags": sorted(LIVE_ENABLE_FLAGS),
    }
    write_payload(path, config)
    return path


def write_override(project_root: Path, payload: dict[str, Any]) -> Path:
    path = project_root / "config" / ".env.system_architecture_hardening_override"
    env = _as_dict(payload.get("recommended_env_overrides"))
    lines = ["# Auto-managed by scripts/ops/system_architecture_hardening.py"]
    lines.extend(f"{key}={value}" for key, value in sorted(env.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_outputs(project_root: Path, out_file: Path, payload: dict[str, Any]) -> dict[str, Any]:
    written = {
        "latest": str(out_file),
        "section_artifacts": write_section_artifacts(project_root, payload),
        "anatomy_artifacts": write_anatomy_artifacts(project_root, payload),
        "config": str(write_config(project_root, payload)),
        "env_override": str(write_override(project_root, payload)),
    }
    payload["written_artifacts"] = written
    write_payload(out_file, payload)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only cross-layer architecture hardening referee.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true", help="Write the latest hardening artifact, section artifacts, config, and read-only env guard.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root)
    if args.apply:
        payload["written_artifacts"] = write_outputs(project_root, Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "system_architecture_hardening "
            f"overall_status={payload.get('overall_status')} "
            f"hard_sections={payload.get('hard_section_count')} "
            f"watch_sections={payload.get('watch_section_count')} "
            f"anatomy_strength={payload.get('anatomy_strength_score')} "
            f"ok={int(bool(payload.get('ok', False)))}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
