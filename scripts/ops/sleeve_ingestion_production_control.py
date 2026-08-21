#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "sleeve_ingestion_production_control_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.sleeve_ingestion_production_override"
SCHEMA_VERSION = 2
CONTROL_VERSION = "sleeve_ingestion_production_control_v2"
ARTIFACT_SPECS: dict[str, tuple[str, float]] = {
    "observation_rollup": ("governance/health/data_collection_observation_rollup_latest.json", 240.0),
    "paper_standard": ("governance/health/paper_live_data_standard_latest.json", 240.0),
    "health_fast": ("governance/health/health_fast_latest.json", 30.0),
    "ingestion_queue": ("governance/health/ingestion_priority_queue_latest.json", 240.0),
    "storage_autopilot": ("governance/health/storage_backpressure_autopilot_latest.json", 240.0),
    "sleeve_coverage": ("governance/health/sleeve_strategy_coverage_latest.json", 240.0),
    "collector_capabilities": (
        "governance/health/collector_capability_control_latest.json",
        30.0,
    ),
}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


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


def _grade(score: float) -> str:
    if score >= 98.0:
        return "A+"
    if score >= 94.0:
        return "A"
    if score >= 90.0:
        return "A-"
    if score >= 85.0:
        return "B+"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    return "D"


def _load_artifacts(project_root: Path) -> dict[str, dict[str, Any]]:
    return {name: load_json(project_root / rel_path) for name, (rel_path, _max_age) in ARTIFACT_SPECS.items()}


def _source_freshness_contract(project_root: Path, artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sources: dict[str, dict[str, Any]] = {}
    stale_or_missing: list[str] = []
    for name, (rel_path, max_age_minutes) in ARTIFACT_SPECS.items():
        path = project_root / rel_path
        payload = _as_dict(artifacts.get(name))
        age = payload_age_minutes(payload, path)
        has_payload = bool(payload)
        fresh = bool(has_payload and age is not None and float(age) <= float(max_age_minutes))
        if not fresh:
            stale_or_missing.append(name)
        sources[name] = {
            "path": rel_path,
            "loaded": has_payload,
            "age_minutes": round(float(age), 3) if age is not None else None,
            "max_age_minutes": float(max_age_minutes),
            "fresh": fresh,
        }
    return {
        "all_required_fresh": not stale_or_missing,
        "stale_or_missing": stale_or_missing,
        "sources": sources,
    }


def _collection_contract(rollup: dict[str, Any]) -> dict[str, Any]:
    collector_count = _safe_int(rollup.get("collector_count"), 0)
    effective_observed = _safe_int(rollup.get("effective_bots_with_observations"), _safe_int(rollup.get("bots_with_observations"), 0))
    unmanaged_zero = _safe_int(rollup.get("unmanaged_zero_observation_count"), _safe_int(rollup.get("zero_observation_count"), 0))
    data_quality_score = _safe_float(rollup.get("data_quality_score"), 0.0)
    coverage_score = _safe_float(rollup.get("collection_coverage_score"), 0.0)
    coverage_ready = bool(collector_count > 0 and effective_observed >= collector_count and unmanaged_zero <= 0)
    return {
        "status": str(rollup.get("overall_status") or "missing"),
        "collector_count": collector_count,
        "effective_bots_with_observations": effective_observed,
        "unmanaged_zero_observation_count": unmanaged_zero,
        "total_observations": _safe_int(rollup.get("total_observations"), 0),
        "training_ready_count": _safe_int(rollup.get("training_ready_count"), 0),
        "coverage_score": round(coverage_score, 3),
        "data_quality_score": round(data_quality_score, 3),
        "coverage_ready": coverage_ready,
        "repair_lane_active": bool(_as_dict(rollup.get("zero_observation_repair_lane")).get("active", False)),
    }


def _paper_standard_contract(paper: dict[str, Any]) -> dict[str, Any]:
    counts = _as_dict(paper.get("counts_after"))
    safety = _as_dict(paper.get("safety_contract"))
    non_deleted = _safe_int(counts.get("non_deleted_bots"), 0)
    data_collection_active = _safe_int(counts.get("data_collection_active_bots"), 0)
    direct_execution = _safe_int(counts.get("direct_execution_allowed_bots"), 0)
    live_trading = _safe_int(counts.get("live_trading_enabled_bots"), 0)
    live_execution_locked = bool(
        direct_execution <= 0
        and live_trading <= 0
        and str(safety.get("allow_order_execution") or "0") == "0"
        and str(safety.get("market_data_only") or "1") == "1"
        and safety.get("live_execution_allowed") is False
    )
    return {
        "status": str(paper.get("overall_status") or "missing"),
        "ok": bool(paper.get("ok", False)),
        "non_deleted_bots": non_deleted,
        "data_collection_active_bots": data_collection_active,
        "paper_live_data_enabled_bots": _safe_int(counts.get("paper_live_data_enabled_bots"), 0),
        "collection_until_standard_bots": _safe_int(counts.get("collection_until_standard_bots"), 0),
        "all_non_deleted_collecting": bool(non_deleted > 0 and data_collection_active >= non_deleted),
        "direct_execution_allowed_bots": direct_execution,
        "live_trading_enabled_bots": live_trading,
        "live_execution_locked": live_execution_locked,
        "paper_lock": str(safety.get("paper_trade_lock") or ""),
        "market_data_only": str(safety.get("market_data_only") or "1"),
        "allow_order_execution": str(safety.get("allow_order_execution") or "0"),
    }


def _runtime_contract(health_fast: dict[str, Any]) -> dict[str, Any]:
    readiness = _as_dict(health_fast.get("operational_readiness"))
    guarded = _as_dict(readiness.get("guarded_paper"))
    watchdog = _as_dict(health_fast.get("process_watchdog"))
    all_sleeves = _as_dict(watchdog.get("all_sleeves_effective_runtime"))
    return {
        "health_fast_status": str(health_fast.get("overall_status") or "missing"),
        "health_fast_ok": bool(health_fast.get("ok", False)),
        "guarded_paper_status": str(guarded.get("status") or ""),
        "guarded_blockers": list(guarded.get("blockers") or []),
        "paper_ramp_stage": str(guarded.get("paper_ramp_stage") or ""),
        "all_sleeves_status": str(all_sleeves.get("status") or ""),
        "all_sleeves_ready": bool(all_sleeves.get("ok", False) or str(all_sleeves.get("status") or "") == "ready"),
        "child_process_count": _safe_int(all_sleeves.get("child_process_count"), 0),
        "child_fanout_ok": bool(all_sleeves.get("child_fanout_ok", False)),
        "heartbeat_ok": bool(all_sleeves.get("heartbeat_ok", False)),
    }


def _queue_contract(queue: dict[str, Any]) -> dict[str, Any]:
    lanes = _as_dict(queue.get("lane_counts"))
    dispatch = _as_list(queue.get("dispatch_plan"))
    lane_ready = all(name in lanes for name in ("core", "deferred", "cold"))
    core = _as_dict(lanes.get("core"))
    deferred = _as_dict(lanes.get("deferred"))
    cold = _as_dict(lanes.get("cold"))
    return {
        "queue_depth": _safe_int(queue.get("queue_depth"), 0),
        "items_synced": _safe_int(queue.get("items_synced"), 0),
        "dispatch_count": len(dispatch),
        "lane_ready": lane_ready,
        "core_pending_lines": _safe_int(core.get("pending_lines"), 0),
        "deferred_pending_lines": _safe_int(deferred.get("pending_lines"), 0),
        "cold_pending_lines": _safe_int(cold.get("pending_lines"), 0),
        "core_adaptive_quota_share": _safe_float(core.get("adaptive_quota_share"), 0.0),
        "deferred_adaptive_quota_share": _safe_float(deferred.get("adaptive_quota_share"), 0.0),
        "cold_adaptive_quota_share": _safe_float(cold.get("adaptive_quota_share"), 0.0),
        "deterministic_dispatch_ready": bool(lane_ready and len(dispatch) > 0),
    }


def _backlog_contract(storage_autopilot: dict[str, Any]) -> dict[str, Any]:
    control = _as_dict(storage_autopilot.get("high_backlog_control"))
    production = _as_dict(control.get("production_grade_contract"))
    automation = _as_dict(control.get("automation_contract"))
    paper = _as_dict(control.get("paper_soak_boundary"))
    live = _as_dict(control.get("live_money_boundary"))
    return {
        "active": bool(control.get("active", False)),
        "class": str(control.get("class") or ""),
        "severity": str(control.get("severity") or ""),
        "state": str(production.get("state") or ""),
        "grade": str(production.get("grade") or ""),
        "score": _safe_float(production.get("score"), 0.0),
        "missing": list(production.get("missing") or []),
        "safe_to_auto_apply": bool(automation.get("safe_to_auto_apply", False)),
        "paper_allowed_with_advisory": bool(paper.get("allowed_with_advisory", False)),
        "live_money_blocked": bool(live.get("blocked", True)),
        "repair_plan_names": list(automation.get("repair_plan_names") or []),
        "next_system_action": str(control.get("next_system_action") or ""),
    }


def _coverage_contract(coverage: dict[str, Any]) -> dict[str, Any]:
    ready = bool(coverage.get("ok", False) and str(coverage.get("overall_status") or "") == "ready")
    return {
        "status": str(coverage.get("overall_status") or "missing"),
        "ok": bool(coverage.get("ok", False)),
        "coverage_ready": ready,
        "sleeve_count": _safe_int(coverage.get("sleeve_count"), 0),
        "active_runtime_sleeve_count": _safe_int(coverage.get("active_runtime_sleeve_count"), 0),
        "strategy_count": _safe_int(coverage.get("strategy_count"), 0),
        "missing_runtime_sleeves": list(coverage.get("missing_runtime_sleeves") or []),
        "strategy_covered_needs_launcher": list(coverage.get("strategy_covered_needs_launcher") or []),
    }


def _routing_contract(capabilities: dict[str, Any]) -> dict[str, Any]:
    summary = _as_dict(capabilities.get("summary"))
    routing = _as_dict(capabilities.get("ingestion_routing_contract"))
    authority = _as_dict(capabilities.get("ingestion_authority_contract"))
    assignment_count = _safe_int(summary.get("assignment_count"), 0)
    bot_binding_count = _safe_int(summary.get("bot_binding_count"), 0)
    runtime_route_count = _safe_int(routing.get("runtime_route_count"), 0)
    family_count = _safe_int(routing.get("decision_family_count"), 0)
    transport = _as_dict(routing.get("transport_contract"))
    return {
        "status": str(capabilities.get("overall_status") or "missing"),
        "structural_ready": bool(capabilities.get("ok", False)),
        "paper_soak_ready": bool(capabilities.get("paper_soak_ready", False)),
        "live_promotion_ready": bool(capabilities.get("live_promotion_ready", False)),
        "policy_id": str(routing.get("policy_id") or ""),
        "policy_receipt_sha256": str(routing.get("policy_receipt_sha256") or ""),
        "routing_artifact_receipt_sha256": str(
            routing.get("routing_artifact_receipt_sha256")
            or capabilities.get("routing_receipt_sha256")
            or ""
        ),
        "decision_policy_id": str(routing.get("decision_policy_id") or ""),
        "decision_stage": str(routing.get("decision_stage") or ""),
        "decision_family_count": family_count,
        "assignment_count": assignment_count,
        "bot_binding_count": bot_binding_count,
        "all_bots_route_bound": bool(
            assignment_count > 0 and bot_binding_count == assignment_count
        ),
        "runtime_route_count": runtime_route_count,
        "runtime_paper_ready_route_count": _safe_int(
            routing.get("runtime_paper_ready_route_count"), 0
        ),
        "runtime_live_ready_route_count": _safe_int(
            routing.get("runtime_live_ready_route_count"), 0
        ),
        "average_profile_route_quality": round(
            _safe_float(routing.get("average_profile_route_quality"), 0.0), 6
        ),
        "route_authority_safe": bool(
            authority and not any(bool(value) for value in authority.values())
        ),
        "transport_contract_complete": bool(
            transport and all(value is True for value in transport.values())
        ),
        "transport_contract": transport,
        "coverage_debt_blocks_global_collection": bool(
            routing.get("paper_data_debt_blocks_global_collection", True)
        ),
        "coverage_debt_blocks_live_promotion": bool(
            routing.get("live_data_debt_blocks_candidate_promotion", True)
        ),
    }


def _ingestion_mode(backlog: dict[str, Any], collection: dict[str, Any], paper: dict[str, Any]) -> tuple[str, float, str]:
    if not paper.get("live_execution_locked", False):
        return "blocked_live_execution_boundary", 0.0, "live execution boundary is not locked"
    if int(collection.get("unmanaged_zero_observation_count", 0) or 0) > 0:
        return "targeted_observation_repair", 0.08, "repair exact zero-observation sleeves before broad ingest"
    if backlog.get("active") and backlog.get("class") == "hot_path_backpressure":
        return "hot_path_recovery_manifest_first", 0.10, "hot core backlog gets minimal sleeve intake and core-first queueing"
    if backlog.get("active") and str(backlog.get("class") or "").startswith("managed_deferred_backlog"):
        return "production_owned_manifest_first", 0.16, "hot path remains clean while bulk deferred work drains off-hours"
    if backlog.get("active"):
        return "pressure_guarded_manifest_first", 0.20, "backlog is active, so sleeves emit through pressure-aware manifest-first intake"
    return "normal_idempotent_live_data", 0.70, "backlog is green enough for normal sleeve intake"


def _env_values(payload: dict[str, Any]) -> dict[str, str]:
    mode = _as_dict(payload.get("ingestion_mode_contract"))
    data_tiers = _as_dict(payload.get("data_tier_contract"))
    routing = _as_dict(payload.get("decision_aligned_routing_contract"))
    raw_ratio = mode.get("max_active_ratio")
    active_ratio = _safe_float(raw_ratio, 0.16) if raw_ratio is not None else 0.16
    active_ratio_text = f"{active_ratio:g}"
    return {
        "SLEEVE_INGESTION_PRODUCTION_CONTROL_ENABLED": "1",
        "SLEEVE_INGESTION_PRODUCTION_CONTROL_VERSION": CONTROL_VERSION,
        "SLEEVE_INGESTION_MODE": str(mode.get("mode") or ""),
        "SLEEVE_INGESTION_MAX_ACTIVE_RATIO": active_ratio_text,
        "SLEEVE_INGESTION_EVENT_ENVELOPE_REQUIRED": "1",
        "SLEEVE_INGESTION_IDEMPOTENCY_REQUIRED": "1",
        "SLEEVE_INGESTION_SCHEMA_VERSION_REQUIRED": "1",
        "SLEEVE_INGESTION_LANE_ROUTING_REQUIRED": "1",
        "SLEEVE_INGESTION_MANIFEST_FIRST": "1",
        "SLEEVE_INGESTION_PAYLOAD_DIGEST_REQUIRED": "1",
        "SLEEVE_INGESTION_SOURCE_TIMESTAMP_REQUIRED": "1",
        "SLEEVE_INGESTION_ROUTE_RECEIPT_REQUIRED": "1",
        "SLEEVE_INGESTION_ROUTE_ENFORCEMENT": "1",
        "SLEEVE_INGESTION_ROUTE_MAX_AGE_MINUTES": "30",
        "SLEEVE_INGESTION_ROUTING_POLICY_ID": str(routing.get("policy_id") or ""),
        "SLEEVE_INGESTION_ROUTING_RECEIPT": str(
            routing.get("routing_artifact_receipt_sha256") or ""
        ),
        "SLEEVE_INGESTION_ROUTE_SCORE_FLOOR": "0.70",
        "SLEEVE_INGESTION_CORE_PRIORITY": str(data_tiers.get("core_priority") or "1"),
        "SLEEVE_INGESTION_DEFERRED_BUDGET": str(data_tiers.get("deferred_budget") or "0"),
        "SLEEVE_INGESTION_COLD_BUDGET": str(data_tiers.get("cold_budget") or "0"),
        "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1" if active_ratio < 0.70 else "0",
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": active_ratio_text,
        "HEAVY_COLLECTORS_PAUSED_FOR_BACKLOG": "1" if bool(mode.get("pressure_limited", False)) else "0",
        "REPORT_REFRESH_PAUSED_FOR_BACKLOG": "1" if bool(mode.get("pressure_limited", False)) else "0",
        "PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS": "0",
        "PAPER_BROKER_BRIDGE_ENABLED": "1",
        "PAPER_BROKER_BRIDGE_MODE": "jsonl",
        "MARKET_DATA_ONLY": "1",
        "ALLOW_ORDER_EXECUTION": "0",
    }


def _write_override(path: Path, payload: dict[str, Any]) -> None:
    env = _env_values(payload)
    lines = [
        "# Managed by scripts/ops/sleeve_ingestion_production_control.py",
        f"# updated_at_utc={payload.get('timestamp_utc')}",
    ]
    lines.extend(f"{key}={shlex.quote(str(value))}" for key, value in sorted(env.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(project_root: Path = PROJECT_ROOT, *, apply: bool = False, override_path: Path = DEFAULT_OVERRIDE_PATH) -> dict[str, Any]:
    artifacts = _load_artifacts(project_root)
    collection = _collection_contract(artifacts["observation_rollup"])
    paper = _paper_standard_contract(artifacts["paper_standard"])
    runtime = _runtime_contract(artifacts["health_fast"])
    queue = _queue_contract(artifacts["ingestion_queue"])
    backlog = _backlog_contract(artifacts["storage_autopilot"])
    coverage = _coverage_contract(artifacts["sleeve_coverage"])
    routing = _routing_contract(artifacts["collector_capabilities"])
    freshness = _source_freshness_contract(project_root, artifacts)
    mode_name, max_active_ratio, mode_reason = _ingestion_mode(backlog, collection, paper)
    pressure_limited = bool(backlog.get("active", False) or max_active_ratio < 0.70)
    deferred_budget = "0" if backlog.get("active") and str(backlog.get("class") or "").startswith("managed_deferred_backlog") else "quota_limited"
    cold_budget = "0" if pressure_limited else "quota_limited"

    must_haves = {
        "all_non_deleted_collecting": bool(paper.get("all_non_deleted_collecting", False)),
        "observation_coverage_ready": bool(collection.get("coverage_ready", False)),
        "zero_observation_repair_clear": int(collection.get("unmanaged_zero_observation_count", 0) or 0) <= 0,
        "live_execution_locked": bool(paper.get("live_execution_locked", False)),
        "all_sleeves_runtime_ready": bool(runtime.get("all_sleeves_ready", False)),
        "guarded_paper_ready_or_not_required": str(runtime.get("guarded_paper_status") or "") in {"", "ready"},
        "ingestion_queue_lane_ready": bool(queue.get("lane_ready", False)),
        "deterministic_dispatch_ready": bool(queue.get("deterministic_dispatch_ready", False)),
        "backlog_owned_or_green": bool(
            not backlog.get("active", False)
            or (backlog.get("state") == "production_owned" and backlog.get("safe_to_auto_apply", False))
        ),
        "live_money_blocked_while_backlog_hot": bool(not backlog.get("active", False) or backlog.get("live_money_blocked", False)),
        "sleeve_coverage_ready": bool(coverage.get("coverage_ready", False)),
        "source_artifacts_fresh": bool(freshness.get("all_required_fresh", False)),
        "decision_aligned_ingestion_routing": bool(
            routing.get("policy_id") == "sleeve_ingestion_routing_v2"
            and routing.get("decision_stage") == "02_data_qualification"
            and int(routing.get("decision_family_count", 0) or 0) >= 15
        ),
        "all_bots_route_bound": bool(routing.get("all_bots_route_bound", False)),
        "runtime_sleeve_routes_defined": bool(
            int(routing.get("runtime_route_count", 0) or 0) > 0
        ),
        "routing_receipt_present": bool(
            routing.get("routing_artifact_receipt_sha256")
        ),
        "ingestion_route_authority_safe": bool(
            routing.get("route_authority_safe", False)
        ),
        "transport_contract_complete": bool(
            routing.get("transport_contract_complete", False)
        ),
        "event_envelope_required": True,
        "idempotency_required": True,
        "schema_version_required": True,
        "lane_routing_required": True,
    }
    missing = [key for key, value in must_haves.items() if not bool(value)]
    score = 100.0
    if not paper.get("live_execution_locked", False):
        score -= 50.0
    if not collection.get("coverage_ready", False):
        score -= min(24.0, max(0.0, 100.0 - float(collection.get("coverage_score") or 0.0)) * 0.4 + 10.0)
    if int(collection.get("unmanaged_zero_observation_count", 0) or 0) > 0:
        score -= 18.0
    if not runtime.get("all_sleeves_ready", False):
        score -= 16.0
    if not queue.get("lane_ready", False):
        score -= 12.0
    if not queue.get("deterministic_dispatch_ready", False):
        score -= 8.0
    if backlog.get("active") and not (backlog.get("state") == "production_owned" and backlog.get("safe_to_auto_apply", False)):
        score -= 18.0
    if backlog.get("active") and not backlog.get("live_money_blocked", False):
        score -= 35.0
    if not coverage.get("coverage_ready", False):
        score -= 8.0
    if not freshness.get("all_required_fresh", False):
        score -= min(24.0, 6.0 * len(_as_list(freshness.get("stale_or_missing"))))
    if not routing.get("structural_ready", False):
        score -= 16.0
    if not routing.get("all_bots_route_bound", False):
        score -= 12.0
    if not routing.get("route_authority_safe", False):
        score -= 40.0
    if not routing.get("transport_contract_complete", False):
        score -= 10.0
    score = round(max(min(score, 100.0), 0.0), 2)
    grade = _grade(score)
    state = "production_ready" if not missing else "production_attention_required"
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": SCHEMA_VERSION,
        "control_version": CONTROL_VERSION,
        "ok": not missing,
        "overall_status": "ready" if not missing else "blocked",
        "production_grade_contract": {
            "grade": grade,
            "score": score,
            "state": state,
            "must_haves": must_haves,
            "missing": missing,
            "policy": "grades sleeve ingestion control quality, not raw profitability or live-money readiness",
        },
        "ingestion_mode_contract": {
            "mode": mode_name,
            "max_active_ratio": round(float(max_active_ratio), 3),
            "pressure_limited": pressure_limited,
            "reason": mode_reason,
            "paper_soak_allowed": bool(backlog.get("paper_allowed_with_advisory", False) or not backlog.get("active", False)),
            "live_money_blocked": bool(backlog.get("live_money_blocked", True)),
        },
        "sleeve_event_envelope_contract": {
            "required": True,
            "required_fields": [
                "schema_version",
                "event_id",
                "idempotency_key",
                "timestamp_utc",
                "bot_id",
                "sleeve_profile",
                "broker",
                "symbol",
                "ingestion_lane",
                "source_contract",
                "payload_digest",
                "source_timestamp_utc",
                "ingestion_route_profile_id",
                "ingestion_route_receipt_sha256",
            ],
            "dedupe_policy": "idempotency_key must include bot_id, sleeve_profile, symbol, event_type, and source timestamp bucket",
            "payload_policy": "manifest-first payload references or digests are preferred while storage pressure is active",
        },
        "data_tier_contract": {
            "core_priority": "1",
            "deferred_budget": deferred_budget,
            "cold_budget": cold_budget,
            "queue_lane_source": "ingestion_priority_queue",
            "lane_policy": "core decisions and paper evidence first; deferred risk/explanation debt is quota-limited or off-hours while backlog is high",
        },
        "collection_contract": collection,
        "paper_standard_contract": paper,
        "runtime_contract": runtime,
        "ingestion_queue_contract": queue,
        "backlog_contract": backlog,
        "sleeve_coverage_contract": coverage,
        "decision_aligned_routing_contract": routing,
        "source_freshness_contract": freshness,
        "control_env_recommendations": {},
        "regression_guards": [
            "all non-deleted sleeves must collect before paper/live-data claims are ready",
            "zero-observation sleeves route to targeted repair before broad expansion",
            "every sleeve event requires schema, idempotency, lane, and payload digest fields",
            "every decision-bound event carries the exact route profile and signed route receipt used by data qualification",
            "decision-family routing replaces broad scope-only subscriptions and caps optional data fanout",
            "primary and failover providers are ranked by authority, proof, freshness, quality, coverage, error budget, and payload integrity",
            "route coverage debt can force a sleeve to collect-only but cannot stop unrelated healthy paper collection",
            "hot-core backlog forces manifest-first low duty-cycle ingestion",
            "sleeve strategy coverage artifact must be loaded and fresh before sleeve ingestion can go production-ready",
            "managed deferred backlog can be paper-safe but remains live-money blocking",
            "live/direct execution must stay disabled in this controller",
        ],
        "recommended_actions": ordered_unique(
            [
                "keep sleeve ingestion production control loaded through runtime env",
                "refresh sleeve-strategy-coverage before claiming sleeve ingestion is production-ready" if not coverage.get("coverage_ready", False) else "",
                "refresh stale or missing sleeve ingestion source artifacts before widening collection" if not freshness.get("all_required_fresh", False) else "",
                "refresh collector capability routing before trusting decision-bound route receipts" if not routing.get("structural_ready", False) else "",
                "repair exact sleeve route coverage rather than widening every bot subscription" if int(routing.get("runtime_paper_ready_route_count", 0) or 0) < int(routing.get("runtime_route_count", 0) or 0) else "",
                "keep all sleeves manifest-first and idempotent while storage pressure is active" if pressure_limited else "",
                "repair zero-observation sleeves before adding more ingestion breadth" if not collection.get("coverage_ready", False) else "",
                "let storage_backpressure_autopilot own deferred backlog while sleeves continue hot-path paper evidence" if backlog.get("active", False) else "",
                "do not use this controller as live-money promotion evidence until live canary gates independently clear",
            ]
        ),
        "apply_result": {
            "applied": bool(apply),
            "override_path": str(override_path),
        },
    }
    payload["control_env_recommendations"] = _env_values(payload)
    if apply:
        _write_override(override_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Grade and control production sleeve ingestion behavior.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_path = Path(args.out_file).expanduser()
    override_path = Path(args.override).expanduser()
    payload = build_payload(project_root, apply=bool(args.apply), override_path=override_path)
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        grade = _as_dict(payload.get("production_grade_contract")).get("grade")
        mode = _as_dict(payload.get("ingestion_mode_contract")).get("mode")
        print(f"sleeve_ingestion_production_control status={payload.get('overall_status')} grade={grade} mode={mode}")
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
