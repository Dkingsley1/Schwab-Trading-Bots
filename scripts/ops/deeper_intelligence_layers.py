#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAYER_VERSION = "deeper_intelligence_layers_v1"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "deeper_intelligence_layers_v1.json"
DEFAULT_HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "deeper_intelligence_layers_latest.json"
DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "governance" / "system_intelligence" / "deeper_intelligence_layers_contract.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "exports" / "reports" / "operator" / "deeper_intelligence_layers_latest.md"
DEFAULT_PYCHARM_PATH = PROJECT_ROOT / "docs" / "pycharm" / "deeper_intelligence_layers_latest.md"


LAYER_DEFINITIONS: list[dict[str, Any]] = [
    {
        "layer_id": "causal_world_model",
        "title": "Causal World Model Layer",
        "purpose": "Explain why market, broker, storage, auth, memory, backlog, launcher, and sleeve states changed before the system acts.",
        "inputs": ["system_signal_bus", "process_contracts", "storage_backpressure", "runtime_pressure", "auth_lease", "watchdog"],
        "outputs": ["root_cause_graph", "causal_blocker_rank", "intervention_order"],
    },
    {
        "layer_id": "belief_ledger_confidence",
        "title": "Belief Ledger And Confidence Layer",
        "purpose": "Attach confidence, freshness, uncertainty, regime fit, and evidence age to every decision surface.",
        "inputs": ["signal_bus", "artifact_freshness", "training_quality", "model_lifecycle", "source_verification"],
        "outputs": ["belief_ledger", "confidence_floor", "abstention_reason_codes"],
    },
    {
        "layer_id": "digital_twin_replay",
        "title": "Digital Twin Replay Layer",
        "purpose": "Compare current code, last-known-good code, and proposed policies against replay and shadow evidence before promotion.",
        "inputs": ["golden_replay", "pytorch_replay_canary", "platform_brain_v6", "decision_provenance"],
        "outputs": ["twin_replay_packet", "before_after_delta", "rollback_trigger"],
    },
    {
        "layer_id": "adversarial_market_infra_simulator",
        "title": "Adversarial Market And Infrastructure Simulator",
        "purpose": "Stress market assumptions and infrastructure assumptions against bad ticks, queue floods, broker delays, route failures, and hostile liquidity.",
        "inputs": ["guard_intelligence", "storage_route", "live_runtime_separation", "global_halt", "process_fanout"],
        "outputs": ["stress_scenarios", "survival_score", "fragility_watchlist"],
    },
    {
        "layer_id": "self_scientific_method",
        "title": "Self Scientific Method Layer",
        "purpose": "Turn upgrades into hypotheses with expected benefit, evidence windows, proof artifacts, and rollback rules.",
        "inputs": ["experiment_ledger", "commands_contract", "test_results", "promotion_autopilot"],
        "outputs": ["hypothesis_packet", "proof_window", "rollback_rule"],
    },
    {
        "layer_id": "resource_economist",
        "title": "Resource Economist Layer",
        "purpose": "Allocate CPU, memory, disk, SQLite writes, training slots, paper slots, and operator attention by value and pressure.",
        "inputs": ["memory_efficiency", "runtime_throttle", "storage_quota", "paper_live_data_standard", "sleeve_budgets"],
        "outputs": ["resource_budget_curve", "earned_budget", "downgrade_or_parking_queue"],
    },
    {
        "layer_id": "promotion_court",
        "title": "Promotion Court Layer",
        "purpose": "Control collect-only to shadow to paper to live-read-only to live-eligible transitions with evidence gates.",
        "inputs": ["paper_live_data_standard", "training_quality", "promotion_quality_gate", "risk_service", "execution_lane_pipeline"],
        "outputs": ["promotion_verdict", "missing_evidence", "next_safe_lifecycle_state"],
    },
    {
        "layer_id": "living_ontology_memory_graph",
        "title": "Living Ontology And System Memory Graph",
        "purpose": "Maintain a searchable graph of bots, sleeves, launchers, reports, commands, drainers, trainers, guards, tickers, and dependencies.",
        "inputs": ["master_bot_registry", "core_bot_catalog", "system_self_model", "commands_contract", "pycharm_index"],
        "outputs": ["system_graph", "dependency_edges", "unknown_inventory"],
    },
    {
        "layer_id": "operator_dialogue",
        "title": "Operator Dialogue Layer",
        "purpose": "Summarize what changed, what is blocked, what is safe next, and what needs human approval in plain operator language.",
        "inputs": ["codex_handoff", "operator_cockpit", "documentation_reporting", "notification_ladder"],
        "outputs": ["operator_brief", "approval_queue", "daily_degradation_explainer"],
    },
    {
        "layer_id": "constitutional_risk",
        "title": "Constitutional Risk Layer",
        "purpose": "Enforce non-negotiable invariants that no model, bot, drainer, trainer, or launcher may override.",
        "inputs": ["global_halt", "live_runtime_separation", "paper_live_data_standard", "risk_service", "auth_lease"],
        "outputs": ["invariant_attestation", "hard_lockouts", "risk_constitution"],
    },
]


CORE_SURFACES: dict[str, str] = {
    "system_signal_bus": "governance/health/system_signal_bus_latest.json",
    "system_brain": "governance/health/system_brain_latest.json",
    "system_self_intelligence": "governance/health/system_self_intelligence_latest.json",
    "system_super_intelligence": "governance/health/system_super_intelligence_latest.json",
    "system_recursive_intelligence": "governance/health/system_recursive_intelligence_latest.json",
    "whole_system_governor": "governance/health/whole_system_governor_latest.json",
    "platform_brain_v6": "governance/health/platform_brain_v6_latest.json",
    "ingestion_storage": "governance/health/ingestion_storage_control_latest.json",
    "memory_efficiency": "governance/health/memory_efficiency_control_latest.json",
    "runtime_throttle": "governance/health/runtime_throttle_control_latest.json",
    "storage_quota": "governance/health/storage_quota_guard_latest.json",
    "guard_intelligence": "governance/health/guard_intelligence_latest.json",
    "process_watchdog": "governance/health/process_watchdog_latest.json",
    "process_fanout": "governance/health/process_fanout_guard_latest.json",
    "live_runtime_separation": "governance/health/live_runtime_separation_control_latest.json",
    "global_halt": "governance/health/global_killswitch_latest.json",
    "auth_lease": "governance/health/auth_lease_manager_latest.json",
    "paper_live_data_standard": "governance/health/paper_live_data_standard_latest.json",
    "training_quality": "governance/health/training_quality_control_latest.json",
    "promotion_gate": "governance/health/promotion_quality_gate_latest.json",
    "golden_replay": "governance/health/golden_replay_regression_latest.json",
    "pytorch_replay_canary": "governance/health/pytorch_replay_canary_latest.json",
    "artifact_freshness": "governance/health/artifact_freshness_slo_latest.json",
    "commands_contract": "governance/health/commands_contract_latest.json",
    "documentation_reporting": "governance/health/documentation_reporting_intelligence_latest.json",
    "codex_handoff": "governance/health/codex_handoff_latest.json",
    "pycharm_highlights": "governance/health/pycharm_active_bot_highlights_latest.json",
}


SAFE_COMMANDS: dict[str, list[str]] = {
    "refresh_deeper_layers": ["./scripts/ops/opsctl.sh", "deeper-intelligence-layers", "--apply", "--json"],
    "refresh_system_intelligence": ["./scripts/ops/opsctl.sh", "system-intelligence", "--apply", "--json"],
    "refresh_platform_brain_v6": ["./scripts/ops/opsctl.sh", "platform-brain-v6", "--apply", "--json"],
    "refresh_whole_system_governor": ["./scripts/ops/opsctl.sh", "whole-system-governor", "--apply", "--json"],
    "pressure_relief": ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"],
    "score_drainers": ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--apply", "--json"],
    "refresh_docs_reporting": ["./scripts/ops/opsctl.sh", "docs-reporting-intelligence", "--apply", "--json"],
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


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on", "active", "enabled", "ready"}


def _status(payload: dict[str, Any], default: str = "missing") -> str:
    if not payload:
        return default
    for key in ("overall_status", "status", "state"):
        value = payload.get(key)
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


def _age_minutes(payload: dict[str, Any], path: Path, now: datetime) -> float | None:
    for key in ("timestamp_utc", "updated_at_utc", "generated_at_utc", "created_at"):
        parsed = _parse_iso(payload.get(key))
        if parsed is not None:
            return round(max((now - parsed).total_seconds() / 60.0, 0.0), 3)
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None
    return round(max((now - modified).total_seconds() / 60.0, 0.0), 3)


def _registry_rows(project_root: Path) -> list[dict[str, Any]]:
    payload = _load_json(project_root / "master_bot_registry.json")
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else payload.get("bots")
    return [row for row in rows or [] if isinstance(row, dict)]


def _load_surfaces(project_root: Path, now: datetime) -> dict[str, dict[str, Any]]:
    surfaces: dict[str, dict[str, Any]] = {}
    for name, rel in CORE_SURFACES.items():
        path = project_root / rel
        payload = _load_json(path)
        surfaces[name] = {
            "name": name,
            "path": rel,
            "loaded": bool(payload),
            "status": _status(payload),
            "age_minutes": _age_minutes(payload, path, now) if payload else None,
            "payload": payload,
        }
    return surfaces


def _registry_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = [row for row in rows if _bool(row.get("active"))]
    collection = [row for row in rows if _bool(row.get("data_collection_active"))]
    paper = [row for row in rows if _bool(row.get("paper_live_data_enabled")) or _bool(row.get("paper_trading_enabled"))]
    direct = [row for row in rows if _bool(row.get("direct_execution_allowed")) or _bool(row.get("execution_enabled"))]
    live = [row for row in rows if _bool(row.get("live_trading_enabled"))]
    sleeves = {
        str(row.get("sleeve_profile") or row.get("sleeve") or row.get("profile") or "")
        for row in rows
        if str(row.get("sleeve_profile") or row.get("sleeve") or row.get("profile") or "")
    }
    return {
        "total_bots": len(rows),
        "active_bots": len(active),
        "collection_bots": len(collection),
        "paper_live_data_bots": len(paper),
        "direct_execution_allowed_bots": len(direct),
        "live_trading_enabled_bots": len(live),
        "sleeve_profile_count": len(sleeves),
    }


def _storage_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    backpressure = _as_dict(payload.get("backpressure"))
    total_pending = max(
        _safe_int(payload.get("total_pending_lines"), 0),
        _safe_int(payload.get("pending_lines_total"), 0),
        _safe_int(backpressure.get("total_pending_lines"), 0),
        _safe_int(backpressure.get("core_pending_lines"), 0)
        + _safe_int(backpressure.get("deferred_pending_lines"), 0)
        + _safe_int(backpressure.get("cold_pending_lines"), 0),
    )
    threshold = max(_safe_int(backpressure.get("pending_lines_threshold"), 15000), 1)
    pressure_index = _safe_float(payload.get("pressure_index"), _safe_float(backpressure.get("pressure_index"), 0.0))
    severity = str(payload.get("severity") or backpressure.get("severity") or "").lower()
    return {
        "status": _status(payload),
        "severity": severity,
        "pressure_index": round(pressure_index, 6),
        "total_pending_lines": int(total_pending),
        "pending_lines_threshold": int(threshold),
        "pending_ratio": round(float(total_pending) / float(threshold), 6),
        "critical": bool(severity == "critical" or pressure_index >= 3.0 or total_pending >= threshold),
    }


def _runtime_snapshot(runtime: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
    memory_snapshot = _as_dict(memory.get("memory_snapshot"))
    state = str(memory_snapshot.get("memory_pressure_state") or memory.get("memory_pressure_state") or "").lower()
    kind = str(memory_snapshot.get("memory_pressure_kind") or memory.get("memory_pressure_kind") or "").lower()
    host_score = _safe_float(runtime.get("host_saturation_score"), 0.0)
    runtime_memory = str(runtime.get("memory_pressure_level") or "").lower()
    runtime_cpu = str(runtime.get("cpu_pressure_level") or "").lower()
    high = bool(
        _status(runtime) in {"blocked", "critical", "degraded"}
        or _status(memory) in {"blocked", "critical", "degraded"}
        or state in {"yellow", "orange", "red", "critical", "warning"}
        or kind in {"swap", "compressor", "critical"}
        or runtime_memory in {"high", "critical"}
        or runtime_cpu in {"high", "critical"}
        or host_score >= 70.0
    )
    return {
        "runtime_status": _status(runtime),
        "memory_status": _status(memory),
        "memory_pressure_state": state,
        "memory_pressure_kind": kind,
        "memory_pressure_level": runtime_memory,
        "cpu_pressure_level": runtime_cpu,
        "host_saturation_score": round(host_score, 3),
        "pressure_high": high,
    }


def _paper_snapshot(payload: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    counts = _as_dict(payload.get("counts_after")) or _as_dict(payload.get("counts"))
    target = _as_dict(payload.get("paper_lane_target"))
    paper = max(_safe_int(counts.get("paper_live_data_enabled_bots"), 0), _safe_int(registry.get("paper_live_data_bots"), 0))
    direct = max(_safe_int(counts.get("direct_execution_allowed_bots"), 0), _safe_int(registry.get("direct_execution_allowed_bots"), 0))
    live = max(_safe_int(counts.get("live_trading_enabled_bots"), 0), _safe_int(registry.get("live_trading_enabled_bots"), 0))
    minimum = _safe_int(target.get("minimum"), 30)
    maximum = _safe_int(target.get("maximum"), 50)
    within = bool(target.get("within_target_band", minimum <= paper <= maximum))
    return {
        "paper_live_data_enabled_bots": paper,
        "direct_execution_allowed_bots": direct,
        "live_trading_enabled_bots": live,
        "minimum": minimum,
        "maximum": maximum,
        "within_target_band": within,
        "unsafe_trade_authority": bool(direct > 0 or live > 0),
    }


def _surface_snapshot(project_root: Path, surfaces: dict[str, dict[str, Any]], registry: dict[str, Any]) -> dict[str, Any]:
    storage = _storage_snapshot(_as_dict(surfaces["ingestion_storage"].get("payload")))
    runtime = _runtime_snapshot(
        _as_dict(surfaces["runtime_throttle"].get("payload")),
        _as_dict(surfaces["memory_efficiency"].get("payload")),
    )
    paper = _paper_snapshot(_as_dict(surfaces["paper_live_data_standard"].get("payload")), registry)
    loaded = [name for name, row in surfaces.items() if bool(row.get("loaded"))]
    missing = [name for name, row in surfaces.items() if not bool(row.get("loaded"))]
    stale = [
        name
        for name, row in surfaces.items()
        if isinstance(row.get("age_minutes"), (int, float)) and float(row.get("age_minutes")) > 360.0
    ]
    signal_bus = _as_dict(surfaces["system_signal_bus"].get("payload"))
    signal_summary = _as_dict(signal_bus.get("summary"))
    guard_payload = _as_dict(surfaces["guard_intelligence"].get("payload"))
    global_halt = _as_dict(surfaces["global_halt"].get("payload"))
    process_contracts = _as_dict(_as_dict(surfaces["system_brain"].get("payload")).get("process_contracts"))
    return {
        "loaded_surface_count": len(loaded),
        "missing_surface_count": len(missing),
        "loaded_surfaces": loaded,
        "missing_surfaces": missing,
        "stale_surface_count": len(stale),
        "stale_surfaces": stale[:12],
        "top_risk": str(signal_summary.get("top_risk") or "unknown"),
        "storage": storage,
        "runtime": runtime,
        "paper": paper,
        "guard_policy_mode": str(guard_payload.get("policy_mode") or signal_summary.get("guard_policy_mode") or ""),
        "global_halt_active": bool(global_halt.get("halt", False) or global_halt.get("global_halt_active", False)),
        "parallel_sql_writers_allowed": bool(_as_dict(process_contracts.get("global_safety_contract")).get("parallel_sql_writers_allowed", False)),
    }


def _score_from_penalties(*, base: int = 100, penalties: list[int] | None = None) -> int:
    score = base - sum(penalties or [])
    return int(max(0, min(100, score)))


def _status_from_score(score: int, *, blocked: bool = False) -> str:
    if blocked:
        return "blocked"
    if score >= 82:
        return "ready"
    if score >= 65:
        return "advisory"
    return "degraded"


def _layer_state(layer: dict[str, Any], snapshot: dict[str, Any], registry: dict[str, Any], surfaces: dict[str, dict[str, Any]]) -> dict[str, Any]:
    layer_id = str(layer["layer_id"])
    missing = set(snapshot.get("missing_surfaces", []))
    storage = _as_dict(snapshot.get("storage"))
    runtime = _as_dict(snapshot.get("runtime"))
    paper = _as_dict(snapshot.get("paper"))
    surface_penalty = min(24, _safe_int(snapshot.get("missing_surface_count"), 0) * 2)
    stale_penalty = min(12, _safe_int(snapshot.get("stale_surface_count"), 0) * 2)
    evidence: list[str] = []
    blockers: list[str] = []
    next_commands: list[list[str]] = [SAFE_COMMANDS["refresh_deeper_layers"]]
    score = 88
    decision = "observe"
    authority = "advisory_only"

    if layer_id == "causal_world_model":
        score = _score_from_penalties(base=96, penalties=[12 if "system_signal_bus" in missing else 0, 8 if "whole_system_governor" in missing else 0, surface_penalty])
        decision = "rank_root_causes_before_restart_retrain_or_expansion"
        evidence = [f"top_risk:{snapshot.get('top_risk', 'unknown')}", f"pending_lines:{storage.get('total_pending_lines', 0)}"]
        if storage.get("critical"):
            blockers.append("storage_backpressure_primary")
            next_commands.append(SAFE_COMMANDS["score_drainers"])
    elif layer_id == "belief_ledger_confidence":
        score = _score_from_penalties(base=94, penalties=[stale_penalty, surface_penalty, 10 if "artifact_freshness" in missing else 0])
        decision = "require_confidence_floor_and_freshness_age_on_all_promotions"
        evidence = [f"stale_surfaces:{snapshot.get('stale_surface_count', 0)}", f"loaded_surfaces:{snapshot.get('loaded_surface_count', 0)}"]
        if snapshot.get("stale_surface_count", 0):
            blockers.append("stale_belief_inputs")
    elif layer_id == "digital_twin_replay":
        twin_loaded = bool(surfaces["platform_brain_v6"].get("loaded") or surfaces["golden_replay"].get("loaded") or surfaces["pytorch_replay_canary"].get("loaded"))
        score = _score_from_penalties(base=90, penalties=[0 if twin_loaded else 22, 8 if "system_super_intelligence" in missing else 0])
        decision = "simulate_upgrade_and_policy_changes_against_shadow_replay_before_promotion"
        evidence = [f"platform_brain_v6_loaded:{surfaces['platform_brain_v6'].get('loaded', False)}", f"golden_replay_loaded:{surfaces['golden_replay'].get('loaded', False)}"]
        if not twin_loaded:
            blockers.append("replay_twin_surface_missing")
            next_commands.append(SAFE_COMMANDS["refresh_platform_brain_v6"])
    elif layer_id == "adversarial_market_infra_simulator":
        score = _score_from_penalties(base=92, penalties=[12 if "guard_intelligence" in missing else 0, 8 if "live_runtime_separation" in missing else 0, 12 if storage.get("critical") else 0])
        decision = "stress_bad_ticks_broker_lag_queue_refill_storage_route_failure_and_fanout_spikes"
        evidence = [f"guard_mode:{snapshot.get('guard_policy_mode', '')}", f"runtime_pressure_high:{runtime.get('pressure_high', False)}"]
        if runtime.get("pressure_high"):
            blockers.append("host_pressure_scenario_active")
    elif layer_id == "self_scientific_method":
        score = _score_from_penalties(base=91, penalties=[10 if "commands_contract" in missing else 0, 10 if "training_quality" in missing else 0])
        decision = "every_upgrade_needs_hypothesis_evidence_window_success_metric_and_rollback_rule"
        evidence = ["hypothesis_template:ready", "rollback_contract:ready"]
    elif layer_id == "resource_economist":
        pressure_penalty = 24 if storage.get("critical") else 12 if runtime.get("pressure_high") else 0
        score = _score_from_penalties(base=94, penalties=[pressure_penalty, 10 if "storage_quota" in missing else 0])
        decision = "allocate_budget_by_value_pressure_and_safety_before_new_training_or_paper_slots"
        evidence = [f"storage_pending_ratio:{storage.get('pending_ratio', 0)}", f"host_saturation:{runtime.get('host_saturation_score', 0)}"]
        if storage.get("critical") or runtime.get("pressure_high"):
            blockers.append("resource_budget_protective_mode")
            next_commands.append(SAFE_COMMANDS["pressure_relief"])
    elif layer_id == "promotion_court":
        blocked = bool(paper.get("unsafe_trade_authority", False))
        score = _score_from_penalties(base=95, penalties=[30 if blocked else 0, 12 if not paper.get("within_target_band", False) else 0, 10 if "promotion_gate" in missing else 0])
        decision = "keep_new_or_low_confidence_bots_collect_only_until_evidence_packet_passes"
        evidence = [f"paper_bots:{paper.get('paper_live_data_enabled_bots', 0)}", f"live_enabled:{paper.get('live_trading_enabled_bots', 0)}", f"direct_execution:{paper.get('direct_execution_allowed_bots', 0)}"]
        if blocked:
            blockers.append("unexpected_trade_authority_detected")
    elif layer_id == "living_ontology_memory_graph":
        score = _score_from_penalties(base=94, penalties=[18 if registry.get("total_bots", 0) == 0 else 0, 12 if "system_self_model" in missing else 0, 8 if "commands_contract" in missing else 0])
        decision = "keep_bot_sleeve_launcher_report_command_dependency_graph_current"
        evidence = [f"registered_bots:{registry.get('total_bots', 0)}", f"sleeve_profiles:{registry.get('sleeve_profile_count', 0)}"]
    elif layer_id == "operator_dialogue":
        score = _score_from_penalties(base=93, penalties=[12 if "codex_handoff" in missing else 0, 10 if "documentation_reporting" in missing else 0])
        decision = "write_operator_brief_approval_queue_and_degradation_explainer"
        evidence = [f"codex_handoff_loaded:{surfaces['codex_handoff'].get('loaded', False)}", f"docs_reporting_loaded:{surfaces['documentation_reporting'].get('loaded', False)}"]
        next_commands.append(SAFE_COMMANDS["refresh_docs_reporting"])
    elif layer_id == "constitutional_risk":
        blocked = bool(paper.get("unsafe_trade_authority", False) or snapshot.get("parallel_sql_writers_allowed", False))
        score = _score_from_penalties(base=100, penalties=[45 if blocked else 0])
        decision = "hard_invariants_override_all_model_and_bot_recommendations"
        authority = "hard_guardrail_attestation"
        evidence = [
            f"live_trading_enabled:{paper.get('live_trading_enabled_bots', 0)}",
            f"direct_execution_allowed:{paper.get('direct_execution_allowed_bots', 0)}",
            f"global_halt_active:{snapshot.get('global_halt_active', False)}",
            f"parallel_sql_writers_allowed:{snapshot.get('parallel_sql_writers_allowed', False)}",
        ]
        if paper.get("unsafe_trade_authority", False):
            blockers.append("trade_authority_invariant_violation")
        if snapshot.get("parallel_sql_writers_allowed", False):
            blockers.append("parallel_sql_writer_invariant_violation")
    else:
        score = _score_from_penalties(base=85, penalties=[surface_penalty])

    blocked = bool(
        (layer_id == "constitutional_risk" and blockers)
        or (layer_id == "promotion_court" and "unexpected_trade_authority_detected" in blockers)
    )
    status = _status_from_score(score, blocked=blocked)
    if blockers and status == "ready":
        status = "advisory"
    return {
        "layer_id": layer_id,
        "title": str(layer["title"]),
        "purpose": str(layer["purpose"]),
        "overall_status": status,
        "readiness_score": score,
        "authority": authority,
        "decision": decision,
        "inputs": list(layer["inputs"]),
        "outputs": list(layer["outputs"]),
        "evidence": evidence,
        "blockers": blockers,
        "safe_next_commands": next_commands[:4],
        "does_not_execute_trades": True,
        "does_not_start_processes_without_operator_command": True,
    }


def default_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "deeper_intelligence_layers_version": LAYER_VERSION,
        "authority_boundary": "advisory_control_plane_with_constitutional_lockout_attestation",
        "applies_to": [
            "trainers",
            "drainers",
            "data_collectors",
            "storage_routing",
            "ingestion",
            "sleeves",
            "launchers",
            "reporting",
            "operator_dialogue",
        ],
        "layers": [
            {
                "layer_id": str(layer["layer_id"]),
                "title": str(layer["title"]),
                "purpose": str(layer["purpose"]),
                "inputs": list(layer["inputs"]),
                "outputs": list(layer["outputs"]),
            }
            for layer in LAYER_DEFINITIONS
        ],
        "hard_invariants": {
            "live_trade_authority_added": False,
            "parallel_sql_writers_allowed": False,
            "models_may_override_global_halt": False,
            "new_bots_start_live_enabled": False,
            "collect_only_until_promotion_court": True,
            "operator_approval_required_for_destructive_cleanup": True,
        },
    }


def _next_actions(layers: list[dict[str, Any]], snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if _as_dict(snapshot.get("storage")).get("critical") or _as_dict(snapshot.get("runtime")).get("pressure_high"):
        actions.append({"reason": "resource_pressure_active", "command": SAFE_COMMANDS["pressure_relief"]})
    if "system_signal_bus" in snapshot.get("missing_surfaces", []):
        actions.append({"reason": "system_signal_bus_missing", "command": SAFE_COMMANDS["refresh_system_intelligence"]})
    if any("replay_twin_surface_missing" in _as_list(layer.get("blockers")) for layer in layers):
        actions.append({"reason": "digital_twin_needs_replay_surface", "command": SAFE_COMMANDS["refresh_platform_brain_v6"]})
    actions.append({"reason": "refresh_deeper_layer_packet", "command": SAFE_COMMANDS["refresh_deeper_layers"]})
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in actions:
        key = tuple(str(item) for item in _as_list(row.get("command")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped[:6]


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    project_root = Path(project_root)
    now = datetime.now(timezone.utc)
    surfaces = _load_surfaces(project_root, now)
    registry = _registry_snapshot(_registry_rows(project_root))
    snapshot = _surface_snapshot(project_root, surfaces, registry)
    layers = [_layer_state(layer, snapshot, registry, surfaces) for layer in LAYER_DEFINITIONS]
    blocked_layers = [row for row in layers if str(row.get("overall_status")) == "blocked"]
    degraded_layers = [row for row in layers if str(row.get("overall_status")) == "degraded"]
    advisory_layers = [row for row in layers if str(row.get("overall_status")) == "advisory"]
    if blocked_layers:
        overall_status = "blocked"
    elif degraded_layers:
        overall_status = "degraded"
    elif advisory_layers:
        overall_status = "advisory"
    else:
        overall_status = "ready"
    next_actions = _next_actions(layers, snapshot)
    return {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "deeper_intelligence_layers_version": LAYER_VERSION,
        "ok": overall_status != "blocked",
        "overall_status": overall_status,
        "layer_count": len(layers),
        "ready_count": sum(1 for row in layers if str(row.get("overall_status")) == "ready"),
        "advisory_count": len(advisory_layers),
        "degraded_count": len(degraded_layers),
        "blocked_count": len(blocked_layers),
        "registry_snapshot": registry,
        "surface_snapshot": {key: value for key, value in snapshot.items() if key not in {"loaded_surfaces", "missing_surfaces"}},
        "loaded_surfaces": snapshot.get("loaded_surfaces", []),
        "missing_surfaces": snapshot.get("missing_surfaces", []),
        "layers": layers,
        "layer_map": {str(row.get("layer_id")): row for row in layers},
        "next_actions": next_actions,
        "operator_dialogue_packet": {
            "summary": (
                f"{len(layers)} deeper intelligence layers are installed; "
                f"{len(blocked_layers)} blocked, {len(degraded_layers)} degraded, {len(advisory_layers)} advisory."
            ),
            "top_attention": [str(row.get("layer_id")) for row in [*blocked_layers, *degraded_layers, *advisory_layers]][:5],
            "safe_next_command": next_actions[0]["command"] if next_actions else SAFE_COMMANDS["refresh_deeper_layers"],
            "approval_required_for": [
                "live_trade_authority",
                "destructive_cleanup",
                "broad_retrain_under_pressure",
                "paper_lane_above_cap",
            ],
        },
        "contract": default_config()["hard_invariants"],
        "writes": {
            "health": str(DEFAULT_HEALTH_PATH),
            "contract": str(DEFAULT_CONTRACT_PATH),
            "config": str(DEFAULT_CONFIG_PATH),
            "markdown": str(DEFAULT_MARKDOWN_PATH),
            "pycharm": str(DEFAULT_PYCHARM_PATH),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Deeper Intelligence Layers",
        "",
        f"- Updated UTC: `{payload.get('timestamp_utc', '')}`",
        f"- Status: `{payload.get('overall_status', '')}`",
        f"- Layers: `{payload.get('layer_count', 0)}`",
        f"- Ready/Advisory/Degraded/Blocked: `{payload.get('ready_count', 0)}/{payload.get('advisory_count', 0)}/{payload.get('degraded_count', 0)}/{payload.get('blocked_count', 0)}`",
        "",
        "## Operator Dialogue Packet",
        "",
        f"- Summary: {payload.get('operator_dialogue_packet', {}).get('summary', '')}",
        f"- Safe Next Command: `{' '.join(str(item) for item in _as_list(_as_dict(payload.get('operator_dialogue_packet')).get('safe_next_command')))}`",
        "",
        "## Layer Status",
        "",
        "| Layer | Status | Score | Decision |",
        "| --- | --- | ---: | --- |",
    ]
    for row in _as_list(payload.get("layers")):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| `{row.get('layer_id', '')}` | `{row.get('overall_status', '')}` | "
            f"`{row.get('readiness_score', 0)}` | {row.get('decision', '')} |"
        )
    lines.extend(
        [
            "",
            "## Hard Invariants",
            "",
        ]
    )
    for key, value in _as_dict(payload.get("contract")).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Next Actions", ""])
    for action in _as_list(payload.get("next_actions")):
        if not isinstance(action, dict):
            continue
        lines.append(f"- `{action.get('reason', '')}`: `{' '.join(str(item) for item in _as_list(action.get('command')))}`")
    lines.extend(
        [
            "",
            "## Layer Details",
            "",
        ]
    )
    for row in _as_list(payload.get("layers")):
        if not isinstance(row, dict):
            continue
        lines.extend(
            [
                f"### {row.get('title', '')}",
                "",
                str(row.get("purpose", "")),
                "",
                f"- Authority: `{row.get('authority', '')}`",
                f"- Outputs: `{', '.join(str(item) for item in _as_list(row.get('outputs')))}`",
                f"- Blockers: `{', '.join(str(item) for item in _as_list(row.get('blockers'))) or 'none'}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    payload: dict[str, Any],
    *,
    health_path: Path = DEFAULT_HEALTH_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    markdown_path: Path = DEFAULT_MARKDOWN_PATH,
    pycharm_path: Path = DEFAULT_PYCHARM_PATH,
) -> None:
    _write_json(config_path, default_config())
    _write_json(health_path, payload)
    _write_json(
        contract_path,
        {
            "timestamp_utc": payload.get("timestamp_utc"),
            "schema_version": 1,
            "deeper_intelligence_layers_version": LAYER_VERSION,
            "layer_ids": [str(row.get("layer_id")) for row in _as_list(payload.get("layers")) if isinstance(row, dict)],
            "hard_invariants": payload.get("contract", {}),
            "authority_boundary": "advisory_control_plane_with_constitutional_lockout_attestation",
        },
    )
    markdown = render_markdown(payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    pycharm_path.parent.mkdir(parents=True, exist_ok=True)
    pycharm_path.write_text(markdown, encoding="utf-8")


def _resolve(project_root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Install and score the 10 deeper intelligence layers.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--apply", action="store_true", help="Write health, config, contract, report, and PyCharm index artifacts.")
    parser.add_argument("--json", action="store_true", help="Print JSON payload.")
    parser.add_argument("--out", default=str(DEFAULT_HEALTH_PATH))
    parser.add_argument("--config-out", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--contract-out", default=str(DEFAULT_CONTRACT_PATH))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument("--pycharm-out", default=str(DEFAULT_PYCHARM_PATH))
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root)
    if args.apply:
        write_outputs(
            payload,
            health_path=_resolve(project_root, args.out),
            config_path=_resolve(project_root, args.config_out),
            contract_path=_resolve(project_root, args.contract_out),
            markdown_path=_resolve(project_root, args.markdown_out),
            pycharm_path=_resolve(project_root, args.pycharm_out),
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "deeper_intelligence_layers "
            f"status={payload['overall_status']} layers={payload['layer_count']} "
            f"ready={payload['ready_count']} advisory={payload['advisory_count']} "
            f"degraded={payload['degraded_count']} blocked={payload['blocked_count']}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
