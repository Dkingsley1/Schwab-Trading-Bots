#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOVERNOR_VERSION = "whole_system_governor_v1"
OUT_DIR = PROJECT_ROOT / "governance" / "whole_system_governor"
HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "whole_system_governor_latest.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "whole_system_governor_v1.json"
MARKDOWN_PATH = PROJECT_ROOT / "exports" / "reports" / "operator" / "whole_system_governor_latest.md"


LAYERS: list[dict[str, Any]] = [
    {
        "layer_id": "system_governor",
        "title": "System Governor",
        "objective": "Score every sleeve by health, evidence, cost, risk, freshness, and dependency posture.",
        "primary_artifact": "governance/whole_system_governor/governor_decision_packet.json",
    },
    {
        "layer_id": "evidence_court",
        "title": "Evidence Court",
        "objective": "Require point-in-time promotion packets before training, paper promotion, or budget expansion.",
        "primary_artifact": "governance/whole_system_governor/evidence_court_packets.json",
    },
    {
        "layer_id": "memory_storage_triage",
        "title": "Memory And Storage Triage",
        "objective": "Downgrade collection capture from raw trace to digest, heartbeat, or parked as pressure rises.",
        "primary_artifact": "governance/whole_system_governor/memory_triage_policy.json",
    },
    {
        "layer_id": "backlog_outcome_learning",
        "title": "Backlog Intelligence",
        "objective": "Track whether drainer and organizer actions reduce pressure instead of only moving backlog around.",
        "primary_artifact": "governance/whole_system_governor/backlog_outcome_learning.json",
    },
    {
        "layer_id": "sleeve_economy",
        "title": "Sleeve Economy",
        "objective": "Give sleeves pressure-aware CPU, memory, storage, freshness, and operator-attention budgets.",
        "primary_artifact": "governance/whole_system_governor/sleeve_budgets.json",
    },
    {
        "layer_id": "self_model_upgrade",
        "title": "Self-Model Upgrade",
        "objective": "Keep a living map of bots, surfaces, stale signals, dependency edges, and unknowns.",
        "primary_artifact": "governance/whole_system_governor/self_model_upgrade.json",
    },
    {
        "layer_id": "operator_interface",
        "title": "Operator Interface",
        "objective": "Compress blockers, safe actions, risk, evidence gaps, and next commands into one packet.",
        "primary_artifact": "governance/whole_system_governor/operator_decision_packet.json",
    },
    {
        "layer_id": "clean_scaling_control",
        "title": "Clean Scaling Control",
        "objective": "Require raw/live headroom, overlay debt, storage mode, runtime, provider quality, and admission evidence to agree before growth.",
        "primary_artifact": "governance/whole_system_governor/clean_scaling_contract.json",
    },
]


SURFACE_FILES: dict[str, str] = {
    "system_self_model": "governance/health/system_self_model_latest.json",
    "whole_system_intelligence": "governance/health/whole_system_intelligence_latest.json",
    "distributed_cell_architecture": "governance/health/distributed_cell_architecture_latest.json",
    "cell_federation_intelligence": "governance/health/cell_federation_intelligence_latest.json",
    "quant_operational_intelligence": "governance/health/quant_operational_intelligence_latest.json",
    "memory_efficiency": "governance/health/memory_efficiency_control_latest.json",
    "runtime_throttle": "governance/health/runtime_throttle_control_latest.json",
    "ingestion_storage": "governance/health/ingestion_storage_control_latest.json",
    "backpressure_drainer_fleet": "governance/health/backpressure_drainer_fleet_latest.json",
    "backpressure_super_drainer": "governance/health/backpressure_super_drainer_latest.json",
    "backlog_organizer": "governance/health/backlog_organizer_latest.json",
    "operator_cockpit": "governance/health/operator_cockpit_latest.json",
    "codex_handoff": "governance/health/codex_handoff_latest.json",
    "artifact_freshness": "governance/health/artifact_freshness_slo_latest.json",
    "core_materialization": "governance/health/core_bot_materialization_guard_latest.json",
    "live_runtime_separation": "governance/health/live_runtime_separation_control_latest.json",
    "global_halt": "governance/health/global_killswitch_latest.json",
    "auth_lease": "governance/health/auth_lease_manager_latest.json",
    "process_fanout": "governance/health/process_fanout_guard_latest.json",
    "training_quality": "governance/health/training_quality_control_latest.json",
    "bot_quality": "governance/health/bot_quality_autopilot_latest.json",
    "expansion_capacity": "governance/health/expansion_capacity_planner_latest.json",
    "provider_mesh": "governance/health/provider_mesh_latest.json",
    "source_verification": "governance/health/source_verification_latest.json",
    "data_collection_observation_rollup": "governance/health/data_collection_observation_rollup_latest.json",
}


STATUS_WEIGHT = {
    "ready": 0,
    "ok": 0,
    "applied": 0,
    "steady_state": 0,
    "advisory": 1,
    "thin": 1,
    "waiting_for_writer": 2,
    "needs_work": 3,
    "degraded": 4,
    "blocked": 5,
    "critical": 6,
    "missing": 3,
}


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
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


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


def _status(payload: dict[str, Any], default: str = "missing") -> str:
    if not payload:
        return default
    for key in ("overall_status", "status"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    overall = payload.get("overall")
    if isinstance(overall, dict):
        value = overall.get("status")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    if payload.get("ok") is True:
        return "ready"
    if payload.get("ok") is False:
        return "blocked"
    return default


def _parse_iso(raw: Any) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _payload_time(path: Path, payload: dict[str, Any], now: datetime) -> tuple[str, float | None]:
    for key in ("generated_at_utc", "updated_at_utc", "timestamp_utc", "created_at"):
        parsed = _parse_iso(payload.get(key))
        if parsed is not None:
            return parsed.isoformat(), round(max((now - parsed).total_seconds() / 60.0, 0.0), 3)
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return "", None
    return modified.isoformat(), round(max((now - modified).total_seconds() / 60.0, 0.0), 3)


def _registry_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else registry.get("bots")
    return [row for row in rows or [] if isinstance(row, dict)]


def _bot_version(row: dict[str, Any]) -> int | None:
    match = re.match(r"^brain_refinery_v(\d+)", str(row.get("bot_id") or ""))
    return int(match.group(1)) if match else None


def _walk_numbers(payload: Any, key_fragments: tuple[str, ...]) -> list[float]:
    found: list[float] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key).lower()
            if any(fragment in key_text for fragment in key_fragments):
                if isinstance(value, (int, float, str)):
                    parsed = _safe_float(value, -1.0)
                    if parsed >= 0:
                        found.append(parsed)
            found.extend(_walk_numbers(value, key_fragments))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_walk_numbers(item, key_fragments))
    return found


def _surface_snapshot(project_root: Path, now: datetime) -> dict[str, dict[str, Any]]:
    surfaces: dict[str, dict[str, Any]] = {}
    for name, rel in SURFACE_FILES.items():
        path = project_root / rel
        payload = _load_json(path)
        timestamp, age_minutes = _payload_time(path, payload, now) if payload else ("", None)
        surfaces[name] = {
            "surface": name,
            "path": rel,
            "status": _status(payload),
            "exists": bool(payload),
            "timestamp_utc": timestamp,
            "age_minutes": age_minutes,
            "payload": payload,
        }
    return surfaces


def _registry_identity(registry: dict[str, Any]) -> dict[str, Any]:
    rows = _registry_rows(registry)
    summary = registry.get("summary") if isinstance(registry.get("summary"), dict) else {}
    versions = [version for row in rows for version in [_bot_version(row)] if version is not None]
    active = [row for row in rows if bool(row.get("active"))]
    collection = [row for row in rows if bool(row.get("data_collection_active"))]
    training_excluded = [
        row for row in rows if bool(row.get("training_excluded")) or bool(row.get("exclude_from_training"))
    ]
    return {
        "total_bots": len(rows) or _safe_int(summary.get("total_bots"), 0),
        "active_bots": len(active) or _safe_int(summary.get("active_bots"), 0),
        "inactive_bots": max(len(rows) - len(active), _safe_int(summary.get("inactive_bots"), 0)),
        "data_collection_active_bots": len(collection) or _safe_int(summary.get("data_collection_active_bots"), 0),
        "training_excluded_bots": len(training_excluded) or _safe_int(summary.get("training_excluded_bots"), 0),
        "max_bot_version": max(versions) if versions else summary.get("max_bot_version"),
        "target_platform_total_bots": summary.get("target_platform_total_bots"),
        "target_platform_total_bots_met": bool(summary.get("target_platform_total_bots_met", False)),
    }


def _group_key(row: dict[str, Any]) -> str:
    for key in ("capability_pack_slug", "sleeve_family", "capability_pack_version", "strategy_family"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    slot = str(row.get("slot_kind") or "").strip()
    return slot.split("_")[0] if slot else "legacy_unclassified"


def _pack_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)
    groups: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        active = [row for row in members if bool(row.get("active"))]
        collection = [row for row in members if bool(row.get("data_collection_active"))]
        guarded = [row for row in members if bool(row.get("data_collection_storage_guarded"))]
        training_excluded = [
            row for row in members if bool(row.get("training_excluded")) or bool(row.get("exclude_from_training"))
        ]
        quality_values = [_safe_float(row.get("quality_score"), 0.0) for row in members if row.get("quality_score") is not None]
        live_flags = [
            row
            for row in members
            if bool(row.get("live_trading_enabled")) or bool(row.get("execution_enabled")) or bool(row.get("allocation_enabled"))
        ]
        groups.append(
            {
                "group": key,
                "bot_count": len(members),
                "active_count": len(active),
                "collection_count": len(collection),
                "training_excluded_count": len(training_excluded),
                "storage_guarded_count": len(guarded),
                "live_authority_count": len(live_flags),
                "collection_ratio": round(len(collection) / max(len(active), 1), 4),
                "guarded_ratio": round(len(guarded) / max(len(collection), 1), 4),
                "average_quality_score": round(sum(quality_values) / max(len(quality_values), 1), 4) if quality_values else 0.0,
            }
        )
    return groups


def _pressure_snapshot(surfaces: dict[str, dict[str, Any]], identity: dict[str, Any]) -> dict[str, Any]:
    bad_surfaces = [
        name
        for name, surface in surfaces.items()
        if STATUS_WEIGHT.get(str(surface["status"]), 3) >= STATUS_WEIGHT["degraded"]
    ]
    payloads = [surface["payload"] for surface in surfaces.values()]
    pending_lines = max([0.0, *[number for payload in payloads for number in _walk_numbers(payload, ("pending_lines",))]])
    swap_gb = max([0.0, *[number for payload in payloads for number in _walk_numbers(payload, ("swap_gb", "swap_used_gb"))]])
    memory_pressure = max(
        [0.0, *[number for payload in payloads for number in _walk_numbers(payload, ("memory_pressure", "pressure_index"))]]
    )
    collection_count = _safe_int(identity.get("data_collection_active_bots"), 0)
    if bad_surfaces or pending_lines >= 250000 or swap_gb >= 24 or memory_pressure >= 85:
        tier = "protective"
    elif pending_lines >= 50000 or swap_gb >= 12 or collection_count >= 1200:
        tier = "constrained"
    else:
        tier = "steady"
    return {
        "pressure_tier": tier,
        "bad_surface_count": len(bad_surfaces),
        "bad_surfaces": bad_surfaces,
        "pending_lines_estimate": int(pending_lines),
        "swap_gb_estimate": round(swap_gb, 3),
        "memory_pressure_estimate": round(memory_pressure, 3),
        "collection_active_bots": collection_count,
    }


def _score_group(group: dict[str, Any], pressure_tier: str) -> dict[str, Any]:
    count = _safe_int(group.get("bot_count"), 0)
    active_count = _safe_int(group.get("active_count"), 0)
    collection_ratio = _safe_float(group.get("collection_ratio"), 0.0)
    guarded_ratio = _safe_float(group.get("guarded_ratio"), 0.0)
    quality = _safe_float(group.get("average_quality_score"), 0.0)
    live_authority = _safe_int(group.get("live_authority_count"), 0)
    group_name = str(group.get("group") or "")
    strategic_bonus = 0.18 if group_name in {"quant_operational_intelligence", "system_self_intelligence"} else 0.0
    strategic_bonus += 0.14 if "intelligence" in group_name or "governance" in group_name else 0.0
    value_score = max(0.0, min(1.0, 0.22 + quality * 0.35 + strategic_bonus + min(active_count / 500.0, 0.18)))
    cost_score = max(0.0, min(1.0, min(count / 260.0, 0.45) + collection_ratio * 0.35 + (0.15 if pressure_tier != "steady" else 0.0)))
    risk_score = max(0.0, min(1.0, live_authority * 0.25 + (1.0 - guarded_ratio) * 0.25 + (0.2 if collection_ratio > 0.95 else 0.0)))
    if pressure_tier == "protective" and value_score < 0.46:
        capture_tier = "heartbeat"
    elif pressure_tier in {"protective", "constrained"} or collection_ratio > 0.85:
        capture_tier = "thin_digest"
    else:
        capture_tier = "normal_digest"
    if risk_score >= 0.45:
        action = "quarantine_until_policy_review"
    elif capture_tier == "heartbeat":
        action = "sleep_low_value_collectors"
    elif capture_tier == "thin_digest":
        action = "run_thin_digest_only"
    else:
        action = "normal_guarded_run"
    return {
        **group,
        "value_score": round(value_score, 4),
        "cost_score": round(cost_score, 4),
        "risk_score": round(risk_score, 4),
        "capture_tier": capture_tier,
        "governor_action": action,
        "daily_storage_budget_mb": max(4, min(180, int(count * (2 if capture_tier == "normal_digest" else 1)))),
        "max_parallel_jobs": max(1, min(5, 1 + count // 220)),
        "freshness_slo_minutes": 20 if value_score >= 0.62 else 60 if capture_tier != "heartbeat" else 240,
    }


def _sleeve_budgets(groups: list[dict[str, Any]], pressure_tier: str) -> list[dict[str, Any]]:
    return [_score_group(group, pressure_tier) for group in groups]


def _evidence_court(groups: list[dict[str, Any]]) -> dict[str, Any]:
    required_sections = [
        "sample_size_and_regime_coverage",
        "point_in_time_label_lineage",
        "leakage_and_duplicate_alpha_review",
        "slippage_capacity_and_execution_realism",
        "paper_live_separation_attestation",
        "drawdown_tail_and_correlation_profile",
        "operator_reason_codes",
    ]
    packets = []
    for group in groups[:40]:
        collection_ratio = _safe_float(group.get("collection_ratio"), 0.0)
        packets.append(
            {
                "group": group["group"],
                "promotion_gate": "blocked_until_packet_complete" if collection_ratio > 0.5 else "review_required",
                "required_sections": required_sections,
                "minimum_observations": 65000,
                "minimum_collection_days": 160,
                "packet_status": "template_ready",
            }
        )
    return {
        "court_version": "evidence_court_whole_system_v1",
        "required_sections": required_sections,
        "packet_count": len(packets),
        "packets": packets,
    }


def _memory_triage(pressure: dict[str, Any], budgets: list[dict[str, Any]]) -> dict[str, Any]:
    tier = str(pressure["pressure_tier"])
    heartbeat_groups = [budget["group"] for budget in budgets if budget["capture_tier"] == "heartbeat"]
    thin_groups = [budget["group"] for budget in budgets if budget["capture_tier"] == "thin_digest"]
    return {
        "triage_version": "memory_storage_triage_v1",
        "pressure_tier": tier,
        "default_capture_tier": "thin_digest" if tier != "steady" else "normal_digest",
        "tier_rules": [
            {"tier": "raw_trace", "allowed_when": "low_pressure_and_promoted_high_value_only"},
            {"tier": "thin_digest", "allowed_when": "default_for_new_intelligence_and_collect_only_bots"},
            {"tier": "heartbeat", "allowed_when": "pressure_high_or_low_value_collection_only_group"},
            {"tier": "parked", "allowed_when": "duplicate_stale_unsafe_or_unowned"},
        ],
        "heartbeat_groups": heartbeat_groups,
        "thin_digest_groups": thin_groups,
        "hard_limits": {
            "new_collectors_default_sample_rate_max": 0.012 if tier == "protective" else 0.02,
            "new_collectors_default_daily_mb_max": 2 if tier == "protective" else 4,
            "raw_trace_requires_governor_exception": True,
        },
    }


def _clean_scaling_control(surfaces: dict[str, dict[str, Any]], identity: dict[str, Any]) -> dict[str, Any]:
    expansion = surfaces.get("expansion_capacity", {})
    payload = expansion.get("payload") if isinstance(expansion.get("payload"), dict) else {}
    contract = payload.get("clean_scaling_contract") if isinstance(payload.get("clean_scaling_contract"), dict) else {}
    status = str(contract.get("overall_status") or expansion.get("status") or "missing").strip().lower()
    blocked = contract.get("blocked_dimensions") if isinstance(contract.get("blocked_dimensions"), list) else []
    watch = contract.get("watch_dimensions") if isinstance(contract.get("watch_dimensions"), list) else []
    max_wave = _safe_int(contract.get("max_clean_wave_size_now"), 0)
    total_bots = _safe_int(identity.get("total_bots"), 0)
    collection_bots = _safe_int(identity.get("data_collection_active_bots"), 0)
    if not contract:
        next_action = "run ./scripts/ops/opsctl.sh expansion-capacity --json so the whole-system governor has a current clean-scaling contract"
    elif status == "ready":
        next_action = "allow only bounded collection-only waves sized by the clean-scaling contract"
    else:
        next_action = str(contract.get("next_action") or "clear clean-scaling blockers before expansion")
    return {
        "control_version": "clean_scaling_control_v1",
        "overall_status": status,
        "grade": str(contract.get("grade") or "missing"),
        "mode": str(contract.get("mode") or "missing"),
        "max_clean_wave_size_now": int(max_wave),
        "blocked_dimensions": [str(item) for item in blocked],
        "watch_dimensions": [str(item) for item in watch],
        "dimension_count": _safe_int(contract.get("dimension_count"), 0),
        "next_action": next_action,
        "source_surface_status": str(expansion.get("status") or "missing"),
        "source_path": str(expansion.get("path") or SURFACE_FILES.get("expansion_capacity", "")),
        "fleet_scale": {
            "total_bots": int(total_bots),
            "data_collection_active_bots": int(collection_bots),
            "collection_density": round(collection_bots / max(total_bots, 1), 4),
        },
        "invariants": list(contract.get("clean_scaling_invariants") or []),
    }


def _backlog_outcome(project_root: Path, surfaces: dict[str, dict[str, Any]], apply: bool) -> dict[str, Any]:
    storage_payload = surfaces["ingestion_storage"]["payload"]
    drainer_payload = surfaces["backpressure_drainer_fleet"]["payload"]
    super_payload = surfaces["backpressure_super_drainer"]["payload"]
    pending = max(
        [0.0, *_walk_numbers(storage_payload, ("pending_lines",)), *_walk_numbers(drainer_payload, ("pending_lines",))]
    )
    wave_count = max([0.0, *_walk_numbers(super_payload, ("wave_count", "waves_completed", "completed_waves"))])
    event = {
        "captured_at_utc": _utc_now(),
        "pending_lines_estimate": int(pending),
        "drainer_status": surfaces["backpressure_drainer_fleet"]["status"],
        "super_drainer_status": surfaces["backpressure_super_drainer"]["status"],
        "completed_wave_estimate": int(wave_count),
        "effect_verdict": "needs_followup" if pending else "no_pending_backlog_detected",
    }
    ledger_path = project_root / "governance" / "whole_system_governor" / "backlog_outcome_ledger.json"
    ledger = _load_json(ledger_path)
    events = ledger.get("events") if isinstance(ledger.get("events"), list) else []
    if apply:
        events = [*events[-49:], event]
        _write_json(ledger_path, {"ledger_version": "backlog_outcome_ledger_v1", "events": events})
    return {
        "learning_version": "backlog_outcome_learning_v1",
        "latest_event": event,
        "ledger_path": str(ledger_path.relative_to(project_root)),
        "ledger_event_count_after_apply": len(events),
        "next_measurement_contract": "compare_pending_lines_and_surface_status_before_after_each_drainer_or_organizer_action",
    }


def _self_model_upgrade(surfaces: dict[str, dict[str, Any]], identity: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [name for name, surface in surfaces.items() if not surface["exists"]]
    stale = [
        name
        for name, surface in surfaces.items()
        if surface["age_minutes"] is not None and _safe_float(surface["age_minutes"]) > 240
    ]
    return {
        "self_model_upgrade_version": "self_model_upgrade_v1",
        "registry_identity": identity,
        "surface_count": len(surfaces),
        "missing_surface_count": len(missing),
        "stale_surface_count": len(stale),
        "missing_surfaces": missing,
        "stale_surfaces": stale,
        "capability_group_count": len(groups),
        "dependency_edges": [
            ["whole_system_governor", "system_self_model"],
            ["whole_system_governor", "memory_efficiency"],
            ["whole_system_governor", "backpressure_drainer_fleet"],
            ["whole_system_governor", "quant_operational_intelligence"],
            ["whole_system_governor", "operator_cockpit"],
            ["whole_system_governor", "codex_handoff"],
        ],
        "unknowns": [
            "exact_per_bot_runtime_cost_until_runtime_metering_is_attached",
            "true_strategy_edge_until_evidence_packets_are_filled",
            "operator_preference_drift_until_decision_packets_are_acknowledged",
        ],
    }


def _operator_packet(
    pressure: dict[str, Any],
    budgets: list[dict[str, Any]],
    surfaces: dict[str, dict[str, Any]],
    clean_scaling: dict[str, Any],
) -> dict[str, Any]:
    attention: list[dict[str, Any]] = []
    if pressure["pressure_tier"] != "steady":
        attention.append(
            {
                "priority": 1,
                "title": "Pressure-aware capture downgrade active",
                "reason": f"pressure_tier={pressure['pressure_tier']} pending_lines={pressure['pending_lines_estimate']}",
                "safe_command": "./scripts/ops/opsctl.sh whole-system-governor --apply --json",
            }
        )
    if str(clean_scaling.get("overall_status") or "") != "ready":
        attention.append(
            {
                "priority": 1,
                "title": "Clean scaling gate is not ready",
                "reason": f"status={clean_scaling.get('overall_status')} blocked={','.join(clean_scaling.get('blocked_dimensions') or [])}",
                "safe_command": "./scripts/ops/opsctl.sh expansion-capacity --json",
            }
        )
    for name, surface in surfaces.items():
        if STATUS_WEIGHT.get(str(surface["status"]), 3) >= STATUS_WEIGHT["degraded"]:
            attention.append(
                {
                    "priority": 2,
                    "title": f"{name} needs review",
                    "reason": f"status={surface['status']}",
                    "safe_command": "./scripts/ops/opsctl.sh health-fast --json",
                }
            )
    high_cost = sorted(budgets, key=lambda item: (item["cost_score"], item["bot_count"]), reverse=True)[:5]
    for budget in high_cost:
        if budget["capture_tier"] in {"heartbeat", "thin_digest"}:
            attention.append(
                {
                    "priority": 3,
                    "title": f"{budget['group']} budget constrained",
                    "reason": f"capture_tier={budget['capture_tier']} cost={budget['cost_score']}",
                    "safe_command": "./scripts/ops/opsctl.sh memory-efficiency status --json",
                }
            )
    return {
        "packet_version": "operator_decision_packet_v1",
        "attention_queue": attention[:12],
        "do_not_do": [
            "do_not_enable_live_execution_from_governor_or_expansion_packs",
            "do_not_promote_collect_only_bots_without_evidence_packet",
            "do_not_raise_raw_trace_capture_under_memory_or_storage_pressure",
            "do_not_expand_when_clean_scaling_contract_is_blocked",
        ],
        "recommended_next_commands": [
            "./scripts/ops/opsctl.sh memory-efficiency status --json",
            "./scripts/ops/opsctl.sh backpressure-drainer-fleet --json",
            "./scripts/ops/opsctl.sh whole-system-governor --apply --json",
        ],
    }


def _governor_decision(
    pressure: dict[str, Any],
    identity: dict[str, Any],
    budgets: list[dict[str, Any]],
    clean_scaling: dict[str, Any],
) -> dict[str, Any]:
    mode = pressure["pressure_tier"]
    constrained_count = sum(1 for budget in budgets if budget["capture_tier"] in {"heartbeat", "thin_digest"})
    return {
        "governor_version": GOVERNOR_VERSION,
        "mode": mode,
        "authority_boundary": "advisory_budgeting_and_registry_summary_only_no_execution_no_allocation_no_halt_clearance",
        "registry_identity": identity,
        "pressure": pressure,
        "budgeted_group_count": len(budgets),
        "constrained_group_count": constrained_count,
        "clean_scaling": clean_scaling,
        "policy": {
            "new_expansion_default": "collect_only_thin_digest",
            "expansion_requires_clean_scaling_contract": True,
            "promotion_requires_evidence_court": True,
            "operator_packet_required_for_attention": True,
            "codex_communication_surface": "governance/health/codex_handoff_latest.json",
        },
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    governor = payload["governor"]
    pressure = governor["pressure"]
    packet = payload["operator_decision_packet"]
    lines = [
        "# Whole System Governor",
        "",
        f"- Version: `{payload['whole_system_governor_version']}`",
        f"- Mode: `{governor['mode']}`",
        f"- Clean scaling: `{payload['clean_scaling_control']['overall_status']}` / `{payload['clean_scaling_control']['grade']}`",
        f"- Total bots: `{governor['registry_identity']['total_bots']}`",
        f"- Active bots: `{governor['registry_identity']['active_bots']}`",
        f"- Data-collection-active bots: `{governor['registry_identity']['data_collection_active_bots']}`",
        f"- Pending lines estimate: `{pressure['pending_lines_estimate']}`",
        f"- Bad surfaces: `{pressure['bad_surface_count']}`",
        "",
        "## Attention Queue",
    ]
    if not packet["attention_queue"]:
        lines.append("- No urgent attention items detected.")
    for item in packet["attention_queue"]:
        lines.append(f"- P{item['priority']} {item['title']}: {item['reason']}")
    lines.extend(["", "## Guardrails"])
    for item in packet["do_not_do"]:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def _artifact_payloads(payload: dict[str, Any]) -> dict[Path, dict[str, Any]]:
    root = PROJECT_ROOT
    return {
        root / "governance" / "whole_system_governor" / "governor_decision_packet.json": payload["governor"],
        root / "governance" / "whole_system_governor" / "sleeve_budgets.json": {
            "generated_at_utc": payload["generated_at_utc"],
            "sleeve_budgets": payload["sleeve_budgets"],
        },
        root / "governance" / "whole_system_governor" / "evidence_court_packets.json": payload["evidence_court"],
        root / "governance" / "whole_system_governor" / "memory_triage_policy.json": payload["memory_triage_policy"],
        root / "governance" / "whole_system_governor" / "backlog_outcome_learning.json": payload["backlog_outcome_learning"],
        root / "governance" / "whole_system_governor" / "self_model_upgrade.json": payload["self_model_upgrade"],
        root / "governance" / "whole_system_governor" / "operator_decision_packet.json": payload["operator_decision_packet"],
        root / "governance" / "whole_system_governor" / "clean_scaling_contract.json": payload["clean_scaling_control"],
        CONFIG_PATH: {
            "generated_at_utc": payload["generated_at_utc"],
            "whole_system_governor_version": GOVERNOR_VERSION,
            "layers": LAYERS,
            "authority_boundary": payload["governor"]["authority_boundary"],
        },
        HEALTH_PATH: payload,
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, apply: bool = False) -> dict[str, Any]:
    now_dt = datetime.now(timezone.utc)
    generated_at = now_dt.isoformat()
    registry_path = project_root / "master_bot_registry.json"
    registry = _load_json(registry_path)
    rows = _registry_rows(registry)
    identity = _registry_identity(registry)
    surfaces = _surface_snapshot(project_root, now_dt)
    public_surfaces = {
        name: {key: value for key, value in surface.items() if key != "payload"} for name, surface in surfaces.items()
    }
    groups = _pack_groups(rows)
    pressure = _pressure_snapshot(surfaces, identity)
    budgets = _sleeve_budgets(groups, str(pressure["pressure_tier"]))
    evidence = _evidence_court(groups)
    triage = _memory_triage(pressure, budgets)
    backlog = _backlog_outcome(project_root, surfaces, apply)
    self_model = _self_model_upgrade(surfaces, identity, groups)
    clean_scaling = _clean_scaling_control(surfaces, identity)
    operator_packet = _operator_packet(pressure, budgets, surfaces, clean_scaling)
    governor = _governor_decision(pressure, identity, budgets, clean_scaling)
    payload = {
        "ok": True,
        "generated_at_utc": generated_at,
        "mode": "applied" if apply else "dry_run",
        "whole_system_governor_version": GOVERNOR_VERSION,
        "layer_count": len(LAYERS),
        "layers": LAYERS,
        "registry_path": str(registry_path.resolve()),
        "surfaces": public_surfaces,
        "governor": governor,
        "sleeve_budgets": budgets,
        "evidence_court": evidence,
        "memory_triage_policy": triage,
        "clean_scaling_control": clean_scaling,
        "backlog_outcome_learning": backlog,
        "self_model_upgrade": self_model,
        "operator_decision_packet": operator_packet,
        "recommended_apply_command": "./scripts/ops/opsctl.sh whole-system-governor --apply --json",
    }
    return payload


def apply_governor(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    payload = build_payload(project_root, apply=True)
    for path, artifact in _artifact_payloads(payload).items():
        actual = project_root / path.relative_to(PROJECT_ROOT) if path.is_absolute() else project_root / path
        _write_json(actual, artifact)

    markdown_path = project_root / MARKDOWN_PATH.relative_to(PROJECT_ROOT)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")

    registry_path = project_root / "master_bot_registry.json"
    registry = _load_json(registry_path)
    backup_path = ""
    if registry:
        backup_dir = project_root / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"master_bot_registry_before_whole_system_governor_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        shutil.copy2(registry_path, backup)
        backup_path = str(backup)
        summary = dict(registry.get("summary") or {})
        summary.update(
            {
                "whole_system_governor_version": GOVERNOR_VERSION,
                "latest_whole_system_governor": GOVERNOR_VERSION,
                "whole_system_governor_mode": payload["governor"]["mode"],
                "whole_system_governor_layer_count": len(LAYERS),
                "whole_system_governor_budgeted_group_count": len(payload["sleeve_budgets"]),
                "whole_system_governor_attention_count": len(payload["operator_decision_packet"]["attention_queue"]),
                "whole_system_governor_memory_triage_default": payload["memory_triage_policy"]["default_capture_tier"],
                "whole_system_governor_clean_scaling_status": payload["clean_scaling_control"]["overall_status"],
                "whole_system_governor_clean_scaling_grade": payload["clean_scaling_control"]["grade"],
                "whole_system_governor_applied_at_utc": payload["generated_at_utc"],
            }
        )
        registry["summary"] = summary
        registry["updated_at_utc"] = _utc_now()
        _write_json(registry_path, registry)
    payload["registry_backup_path"] = backup_path
    _write_json(project_root / HEALTH_PATH.relative_to(PROJECT_ROOT), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and apply the whole-system governor layer.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = apply_governor(project_root) if args.apply else build_payload(project_root, apply=False)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "whole_system_governor "
            f"mode={payload['mode']} governor_mode={payload['governor']['mode']} "
            f"layers={payload['layer_count']} budgets={len(payload['sleeve_budgets'])} "
            f"attention={len(payload['operator_decision_packet']['attention_queue'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
