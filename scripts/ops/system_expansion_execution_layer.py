#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_expansion_execution_layer_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.system_expansion_execution_layer_override"
DEFAULT_MEMORY_PATH = PROJECT_ROOT / "governance" / "system_expansion_execution" / "operator_memory.jsonl"

LIVE_LOCK = "system_expansion_execution_layer_is_advisory_no_live_or_paper_execution_authority"

LANE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "rank": 1,
        "lane_id": "predictive_stability",
        "title": "Predictive Stability Layer",
        "contract": "forecast pressure from runtime, memory, storage, writer, drift, and halt surfaces before hard guards trip",
    },
    {
        "rank": 2,
        "lane_id": "self_healing_router",
        "title": "Self-Healing Router",
        "contract": "map blocked and degraded surfaces to safe commands, prechecks, postchecks, and fallback posture",
    },
    {
        "rank": 3,
        "lane_id": "stale_surface_autofix",
        "title": "Stale Surface Auto-Fixer",
        "contract": "identify stale required artifacts and refresh them before stale-only blockers distort the graph",
    },
    {
        "rank": 4,
        "lane_id": "schwab_indicator_feature_bridge",
        "title": "Schwab Indicator-to-Feature Bridge",
        "contract": "turn Schwab study and strategy awareness into advisory sleeve feature candidates with validation gates",
    },
    {
        "rank": 5,
        "lane_id": "collector_utility_budget",
        "title": "Collector Utility Budget",
        "contract": "score collector families by downstream value, input uniqueness, and runtime/storage cost so low-value collectors thin first",
    },
    {
        "rank": 6,
        "lane_id": "grandmaster_safe_mode",
        "title": "Grandmaster Safe Mode Per Sleeve",
        "contract": "downshift sleeves independently under pressure instead of using blunt global posture changes",
    },
    {
        "rank": 7,
        "lane_id": "training_deficiency_repair_loop",
        "title": "Training Deficiency Repair Loop",
        "contract": "convert drill and regression deficiencies into repair packets, replay checks, and measured score deltas",
    },
    {
        "rank": 8,
        "lane_id": "hot_path_storage_budget",
        "title": "Hot Path Storage Budget",
        "contract": "protect broker truth, paper decisions, fills, positions, and guards before report or research writes",
    },
    {
        "rank": 9,
        "lane_id": "capital_rotation_simulator_v2",
        "title": "Capital Rotation Simulator v2",
        "contract": "simulate paper-only sleeve money waves under regime, confidence, drawdown, and pressure states",
    },
    {
        "rank": 10,
        "lane_id": "promotion_evidence_ledger",
        "title": "Promotion Evidence Ledger",
        "contract": "track sleeve promotion evidence across paper PnL, drawdown, fill quality, regime fit, and indicator contribution",
    },
    {
        "rank": 11,
        "lane_id": "dependency_contract_hardening",
        "title": "Dependency Contract Hardening",
        "contract": "classify graph blockers as hard, soft, stale-only, advisory, or ignorable noise with clear authority boundaries",
    },
    {
        "rank": 12,
        "lane_id": "daily_operator_memory",
        "title": "Daily Operator Memory",
        "contract": "write a compact memory of what changed, what helped, what failed, and what should happen next",
    },
)


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


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _status(payload: dict[str, Any]) -> str:
    if not payload:
        return "missing"
    raw = payload.get("overall_status")
    if raw is None:
        raw = payload.get("status")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if "ok" in payload:
        return "ready" if bool(payload.get("ok")) else "blocked"
    return "unknown"


def _load_sources(project_root: Path) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    champion = project_root / "governance" / "champion_challenger"
    return {
        "self_model": load_json(health / "system_self_model_latest.json"),
        "architecture_graph": load_json(health / "system_architecture_contract_graph_latest.json"),
        "schwab_indicator": load_json(health / "schwab_indicator_intelligence_latest.json"),
        "runtime": load_json(health / "runtime_throttle_control_latest.json"),
        "memory": load_json(health / "memory_efficiency_control_latest.json"),
        "storage": load_json(health / "ingestion_storage_control_latest.json"),
        "health_fast": load_json(health / "health_fast_latest.json"),
        "paper_performance": load_json(health / "paper_performance_latest.json"),
        "paper_profitability": load_json(health / "paper_profitability_control_latest.json"),
        "capital_rotation": load_json(health / "capital_rotation_control_latest.json"),
        "capital_growth": load_json(health / "capital_growth_intelligence_latest.json"),
        "intense_drill": load_json(health / "system_intense_drill_autopilot_latest.json"),
        "adversarial_drill": load_json(health / "system_adversarial_drill_autopilot_latest.json"),
        "training_quality": load_json(health / "training_quality_control_latest.json"),
        "adaptive_regression": load_json(health / "adaptive_regression_guard_latest.json"),
        "evidence_packet": load_json(health / "evidence_packet_latest.json"),
        "command_validity": load_json(health / "command_validity_latest.json"),
        "commands_hygiene": load_json(health / "commands_hygiene_latest.json"),
        "promotion_packet": load_json(champion / "promotion_autopilot_packet_latest.json"),
        "registry": load_json(project_root / "master_bot_registry.json"),
    }


def _runtime_snapshot(sources: dict[str, Any]) -> dict[str, Any]:
    runtime = _as_dict(sources.get("runtime"))
    memory = _as_dict(sources.get("memory"))
    storage = _as_dict(sources.get("storage"))
    graph = _as_dict(sources.get("architecture_graph"))
    memory_snapshot = _as_dict(memory.get("memory_snapshot"))
    storage_snapshot = _as_dict(memory.get("storage_snapshot"))
    backpressure = _as_dict(storage.get("backpressure"))
    blocked = len(_as_list(graph.get("blocked_nodes")))
    degraded = len(_as_list(graph.get("degraded_nodes")))
    stale = len(_as_list(graph.get("stale_nodes")))
    memory_level = str(runtime.get("memory_pressure_level") or "").lower()
    compute_level = str(runtime.get("compute_pressure_level") or runtime.get("cpu_pressure_level") or "").lower()
    storage_pressure = _safe_float(storage_snapshot.get("pressure_index"), _safe_float(storage.get("pressure_index"), 0.0))
    pending_lines = _safe_int(backpressure.get("total_pending_lines"), 0)
    host_saturation = _safe_float(runtime.get("host_saturation_score"), 0.0)
    pressure_score = _clamp(
        0.20 * (1.0 if _status(runtime) in {"degraded", "needs_work", "blocked"} else 0.0)
        + 0.16 * (1.0 if _status(memory) in {"needs_work", "degraded", "blocked"} else 0.0)
        + 0.14 * (1.0 if memory_level in {"elevated", "high"} else 0.0)
        + 0.12 * (1.0 if compute_level in {"elevated", "high"} else 0.0)
        + 0.12 * _clamp(host_saturation / 100.0)
        + 0.10 * _clamp(storage_pressure)
        + 0.08 * _clamp(pending_lines / 25000.0)
        + 0.04 * _clamp(blocked / 4.0)
        + 0.04 * _clamp((degraded + stale) / 12.0)
    )
    return {
        "runtime_status": _status(runtime),
        "memory_status": _status(memory),
        "health_fast_status": _status(_as_dict(sources.get("health_fast"))),
        "storage_status": _status(storage),
        "memory_pressure_level": memory_level or "unknown",
        "compute_pressure_level": compute_level or "unknown",
        "memory_pressure_state": str(memory_snapshot.get("memory_pressure_state") or ""),
        "swap_used_gb": _safe_float(memory_snapshot.get("swap_used_gb"), 0.0),
        "compressed_store_gb": _safe_float(memory_snapshot.get("compressed_store_gb"), _safe_float(memory_snapshot.get("compressor_gb"), 0.0)),
        "host_saturation_score": round(host_saturation, 3),
        "storage_pressure_index": round(storage_pressure, 3),
        "pending_lines": pending_lines,
        "blocked_graph_nodes": blocked,
        "degraded_graph_nodes": degraded,
        "stale_graph_nodes": stale,
        "pressure_score": round(pressure_score, 4),
        "pressure_band": "high" if pressure_score >= 0.70 else "elevated" if pressure_score >= 0.45 else "watch" if pressure_score >= 0.25 else "clear",
        "guarded": pressure_score >= 0.45,
    }


def _lane_base(lane_id: str, *, status: str, priority: str, readiness: float, details: dict[str, Any]) -> dict[str, Any]:
    definition = next(item for item in LANE_DEFINITIONS if item["lane_id"] == lane_id)
    return {
        **definition,
        "status": status,
        "priority": priority,
        "readiness_score": round(_clamp(readiness), 4),
        "authority": "advisory_control_plane_no_live_or_paper_execution",
        "details": details,
    }


def _predictive_stability(sources: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    score = _safe_float(runtime.get("pressure_score"), 0.0)
    band = str(runtime.get("pressure_band") or "unknown")
    details = {
        "forecast_horizon": "next_1_to_6_hours",
        "pressure_score": round(score, 4),
        "pressure_band": band,
        "leading_signals": ordered_unique(
            [
                f"runtime={runtime.get('runtime_status')}",
                f"memory={runtime.get('memory_status')}",
                f"memory_level={runtime.get('memory_pressure_level')}",
                f"compute_level={runtime.get('compute_pressure_level')}",
                f"stale_nodes={runtime.get('stale_graph_nodes')}",
                f"blocked_nodes={runtime.get('blocked_graph_nodes')}",
            ]
        ),
        "preemptive_actions": [
            "downshift heavy MLX/reporting/training lanes when pressure_score >= 0.45",
            "prefer collector thinning before global halts when trading-path artifacts are safe",
            "refresh stale required surfaces before trusting graph-level blockers",
        ],
    }
    return _lane_base("predictive_stability", status="active_guarded" if score >= 0.45 else "ready", priority="critical", readiness=1.0 - min(score, 0.8) / 2.0, details=details)


def _self_healing_router(sources: dict[str, Any]) -> dict[str, Any]:
    graph = _as_dict(sources.get("architecture_graph"))
    commands = _as_list(graph.get("recommended_commands"))
    blocked = _as_list(graph.get("blocked_nodes"))
    degraded = _as_list(graph.get("degraded_nodes"))
    stale = set(str(item) for item in _as_list(graph.get("stale_nodes")))
    routes: list[dict[str, Any]] = []
    for idx, command in enumerate(commands):
        if not isinstance(command, list) or not command:
            continue
        target = " ".join(str(part) for part in command[1:2]) or f"route_{idx}"
        target_node = ""
        for node in list(blocked) + list(degraded):
            if str(node).replace("_", "-") in " ".join(str(part) for part in command):
                target_node = str(node)
                break
        routes.append(
            {
                "route_id": f"heal_{idx+1:02d}_{target.replace('-', '_')}",
                "target_node": target_node or target,
                "severity": "high" if target_node in blocked else "medium" if target_node else "low",
                "command": [str(part) for part in command],
                "prechecks": ["operator_stop_absent", "global_halt_status_known", "runtime_pressure_not_high_for_apply_commands"],
                "postchecks": ["rerun_architecture_contract_graph", "confirm_target_status_improved_or_mark_followup"],
                "stale_only": bool(target_node in stale),
                "authority": "safe_command_routing_no_destructive_repair",
            }
        )
    details = {
        "route_count": len(routes),
        "blocked_nodes": [str(item) for item in blocked],
        "degraded_nodes": [str(item) for item in degraded],
        "routes": routes[:20],
    }
    status = "needs_routes" if not routes and (blocked or degraded) else "ready"
    return _lane_base("self_healing_router", status=status, priority="critical", readiness=0.95 if routes else 0.35, details=details)


def _stale_surface_autofix(sources: dict[str, Any]) -> dict[str, Any]:
    graph = _as_dict(sources.get("architecture_graph"))
    nodes = [row for row in _as_list(graph.get("nodes")) if isinstance(row, dict)]
    stale_rows = [row for row in nodes if bool(row.get("artifact_stale", False))]
    rows = []
    for row in stale_rows:
        rows.append(
            {
                "node_id": str(row.get("node_id") or ""),
                "artifact": str(row.get("artifact") or ""),
                "age_minutes": _safe_float(row.get("artifact_age_minutes"), 0.0),
                "max_age_minutes": _safe_float(row.get("artifact_max_age_minutes"), 0.0),
                "commands": _as_list(row.get("commands")),
                "autofix_policy": "refresh_then_compare_last_good_hash",
            }
        )
    details = {
        "stale_count": len(rows),
        "stale_surfaces": rows,
        "suppression_policy": "stale_only_blockers_can_be_suppressed_only_after_refresh_attempt_and_dependency_chain_check",
    }
    status = "active_followup_needed" if rows else "ready"
    readiness = 1.0 - min(len(rows), 8) / 16.0
    return _lane_base("stale_surface_autofix", status=status, priority="high", readiness=readiness, details=details)


def _indicator_feature_bridge(sources: dict[str, Any]) -> dict[str, Any]:
    indicator = _as_dict(sources.get("schwab_indicator"))
    matrix = [row for row in _as_list(indicator.get("sleeve_applicability_matrix")) if isinstance(row, dict)]
    items = [row for row in _as_list(indicator.get("catalog_items")) if isinstance(row, dict)]
    item_by_name = {str(row.get("name") or ""): row for row in items}
    candidates: list[dict[str, Any]] = []
    for sleeve_row in matrix:
        sleeve = str(sleeve_row.get("sleeve") or "")
        for name in list(_as_list(sleeve_row.get("top_studies")))[:8] + list(_as_list(sleeve_row.get("top_strategies")))[:6]:
            item = item_by_name.get(str(name), {})
            candidates.append(
                {
                    "candidate_id": f"{sleeve}:{str(name)}",
                    "sleeve": sleeve,
                    "name": str(name),
                    "kind": str(item.get("kind") or "unknown"),
                    "families": _as_list(item.get("families")),
                    "required_inputs": _as_list(item.get("required_inputs")),
                    "validation_gate": "walk_forward_then_paper_evidence_before_weight",
                    "authority": "feature_candidate_no_execution",
                }
            )
    coverage = _as_dict(indicator.get("coverage"))
    details = {
        "catalog_status": _status(indicator),
        "catalog_item_count": _safe_int(coverage.get("catalog_item_count"), len(items)),
        "study_count": _safe_int(coverage.get("study_count"), 0),
        "strategy_count": _safe_int(coverage.get("strategy_count"), 0),
        "feature_candidate_count": len(candidates),
        "candidate_sample": candidates[:120],
        "bridge_policy": "study_strategy_features_are_candidates_until_input_quality_walk_forward_and_paper_evidence_clear",
    }
    status = "ready" if candidates else "needs_indicator_catalog"
    return _lane_base("schwab_indicator_feature_bridge", status=status, priority="high", readiness=0.95 if candidates else 0.2, details=details)


def _collector_utility_budget(sources: dict[str, Any]) -> dict[str, Any]:
    registry = _as_dict(sources.get("registry"))
    rows = [row for row in _as_list(registry.get("sub_bots")) if isinstance(row, dict)]
    collector_counts: Counter[str] = Counter()
    sleeve_counts: Counter[str] = Counter()
    for row in rows:
        if not bool(row.get("active", False)):
            continue
        sleeve = str(row.get("sleeve_profile") or row.get("sleeve_family") or "unknown")
        sleeve_counts[sleeve] += 1
        for key in ("schwab_direct_inputs", "data_intake_collections", "proxy_data_sources"):
            for raw in _as_list(row.get(key)):
                collector_counts[str(raw)] += 1
    if not collector_counts:
        collector_counts.update({"quotes": 1, "chains": 1, "market_hours": 1, "fundamentals": 1, "corporate_actions": 1})
    budget_rows = []
    max_count = max(collector_counts.values() or [1])
    for name, count in collector_counts.most_common(40):
        count_norm = count / max_count
        required = name in {"quotes", "orders", "positions", "market_hours", "listed_equity_etf_quotes", "listed_option_chains"}
        utility = _clamp(0.50 * count_norm + (0.30 if required else 0.0) + (0.20 if "schwab" in name or "quote" in name or "chain" in name else 0.0))
        budget_rows.append(
            {
                "collector": name,
                "observed_active_bot_refs": int(count),
                "utility_score": round(utility, 4),
                "tier": "hot_keep" if required or utility >= 0.65 else "warm_thin_last" if utility >= 0.35 else "cold_thin_first",
            }
        )
    details = {
        "collector_count": len(collector_counts),
        "sleeve_count": len(sleeve_counts),
        "budget_rows": budget_rows,
        "pressure_policy": "thin_cold_collectors_before_hot_trading_path_inputs_when_runtime_guarded",
    }
    return _lane_base("collector_utility_budget", status="ready", priority="high", readiness=0.9, details=details)


def _grandmaster_safe_mode(sources: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    indicator = _as_dict(sources.get("schwab_indicator"))
    matrix = [row for row in _as_list(indicator.get("sleeve_applicability_matrix")) if isinstance(row, dict)]
    pressure = _safe_float(runtime.get("pressure_score"), 0.0)
    rows = []
    for row in matrix[:30]:
        sleeve = str(row.get("sleeve") or "unknown")
        mapped = _safe_int(row.get("mapped_item_count"), 0)
        if pressure >= 0.70:
            mode = "observe_only"
        elif pressure >= 0.45 and any(token in sleeve for token in ("intraday", "day_trading", "microstructure")):
            mode = "thin_collect_only"
        elif pressure >= 0.45:
            mode = "guarded_collect"
        else:
            mode = "normal_paper_observation"
        rows.append(
            {
                "sleeve": sleeve,
                "safe_mode": mode,
                "mapped_indicator_count": mapped,
                "paper_execution_authority": False,
                "live_execution_authority": False,
                "reason": "pressure_guarded" if pressure >= 0.45 else "pressure_clear_or_watch",
            }
        )
    details = {
        "pressure_score": round(pressure, 4),
        "sleeve_mode_count": len(rows),
        "sleeve_modes": rows,
        "grandmaster_policy": "per_sleeve_downshift_before_global_blunt_controls",
    }
    return _lane_base("grandmaster_safe_mode", status="active_guarded" if pressure >= 0.45 else "ready", priority="critical", readiness=0.9 if rows else 0.4, details=details)


def _training_deficiency_repair_loop(sources: dict[str, Any]) -> dict[str, Any]:
    graph = _as_dict(sources.get("architecture_graph"))
    intense = _as_dict(sources.get("intense_drill"))
    adversarial = _as_dict(sources.get("adversarial_drill"))
    training = _as_dict(sources.get("training_quality"))
    adaptive = _as_dict(sources.get("adaptive_regression"))
    deficient = ordered_unique(
        [str(item) for item in _as_list(graph.get("blocked_nodes"))]
        + [str(item) for item in _as_list(graph.get("degraded_nodes"))]
        + [str(item) for item in _as_list(intense.get("deficient_families"))]
        + [str(item) for item in _as_list(adversarial.get("weak_points"))]
    )
    repair_packets = []
    for idx, item in enumerate(deficient[:24], start=1):
        repair_packets.append(
            {
                "repair_id": f"repair_{idx:02d}_{item}",
                "target": item,
                "source_status": {
                    "training_quality": _status(training),
                    "adaptive_regression": _status(adaptive),
                    "intense_drill": _status(intense),
                    "adversarial_drill": _status(adversarial),
                },
                "sequence": ["isolate_failure_cluster", "apply_smallest_safe_patch", "run_replay_or_guard", "record_score_delta"],
            }
        )
    details = {
        "deficiency_count": len(deficient),
        "repair_packets": repair_packets,
        "loop_contract": "drill_result_to_repair_to_replay_to_score_delta_memory",
    }
    status = "active_followup_needed" if repair_packets else "ready"
    return _lane_base("training_deficiency_repair_loop", status=status, priority="critical", readiness=0.8 if repair_packets else 0.95, details=details)


def _hot_path_storage_budget(sources: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    pressure = _safe_float(runtime.get("pressure_score"), 0.0)
    rows = [
        {"tier": "hot", "surfaces": ["broker_truth", "paper_decisions", "fills", "positions", "halt_flags", "decision_provenance"], "degrade_order": "never_first", "minimum_retention": "protect_latest_and_recent"},
        {"tier": "warm", "surfaces": ["training_diagnostics", "indicator_features", "capital_rotation", "promotion_evidence"], "degrade_order": "thin_after_cold", "minimum_retention": "latest_plus_recent_rollup"},
        {"tier": "cold", "surfaces": ["large_reports", "verbose_explainers", "old_pdf_exports", "research_scratch"], "degrade_order": "thin_first_under_pressure", "minimum_retention": "latest_or_digest_only"},
    ]
    details = {
        "pressure_score": round(pressure, 4),
        "storage_policy_rows": rows,
        "active_mode": "cold_thin_first" if pressure >= 0.45 else "normal_retention",
    }
    return _lane_base("hot_path_storage_budget", status="ready", priority="high", readiness=0.9, details=details)


def _capital_rotation_simulator_v2(sources: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    rotation = _as_dict(sources.get("capital_rotation"))
    rows = [row for row in _as_list(rotation.get("sleeve_rotation_plan")) if isinstance(row, dict)]
    scenarios = []
    for row in rows[:24]:
        profile = str(row.get("profile") or "unknown")
        signed = _safe_float(row.get("signed_rotation_pressure_norm"), 0.0)
        for scenario, multiplier in (("risk_on", 1.15), ("baseline", 1.0), ("risk_off", 0.55)):
            delta = signed * multiplier
            if bool(runtime.get("guarded")) and delta > 0:
                action = "hold_inflow_under_pressure"
                delta = 0.0
            else:
                action = "paper_expand_candidate" if delta >= 0.30 else "reduce_or_quarantine" if delta <= -0.25 else "hold"
            scenarios.append(
                {
                    "profile": profile,
                    "scenario": scenario,
                    "simulated_rotation_pressure": round(delta, 4),
                    "action": action,
                    "live_money_delta": 0.0,
                }
            )
    details = {
        "scenario_count": len(scenarios),
        "source_status": _status(rotation),
        "scenarios": scenarios,
        "authority": "paper_simulation_only_no_live_money_movement",
    }
    return _lane_base("capital_rotation_simulator_v2", status="ready" if rows else "needs_capital_rotation_source", priority="high", readiness=0.9 if rows else 0.35, details=details)


def _promotion_evidence_ledger(sources: dict[str, Any]) -> dict[str, Any]:
    paper = _as_dict(sources.get("paper_performance"))
    rotation = _as_dict(sources.get("capital_rotation"))
    rows = [row for row in _as_list(paper.get("sleeve_latest")) if isinstance(row, dict)]
    if not rows:
        rows = [row for row in _as_list(rotation.get("sleeve_rotation_plan")) if isinstance(row, dict)]
    ledger = []
    for row in rows[:50]:
        profile = str(row.get("profile") or row.get("sleeve") or "unknown")
        ledger.append(
            {
                "profile": profile,
                "evidence_status": "incomplete_until_30_60_90_day_packet",
                "paper_pnl": _safe_float(row.get("net_pnl"), _safe_float(row.get("ending_net_pnl_total"), 0.0)),
                "win_rate": row.get("win_rate"),
                "required_evidence": ["30_day_repeatability", "60_day_regime_coverage", "90_day_drawdown_profile", "fill_quality", "indicator_contribution", "failure_modes"],
                "promotion_authority": False,
            }
        )
    details = {
        "ledger_row_count": len(ledger),
        "ledger_rows": ledger,
        "promotion_policy": "evidence_required_before_micro_live_discussion",
    }
    return _lane_base("promotion_evidence_ledger", status="ready" if ledger else "needs_paper_evidence", priority="high", readiness=0.8 if ledger else 0.25, details=details)


def _dependency_contract_hardening(sources: dict[str, Any]) -> dict[str, Any]:
    graph = _as_dict(sources.get("architecture_graph"))
    nodes = [row for row in _as_list(graph.get("nodes")) if isinstance(row, dict)]
    classifications = []
    counts: Counter[str] = Counter()
    for row in nodes:
        status = str(row.get("status") or "")
        required = bool(row.get("required", False))
        stale = bool(row.get("artifact_stale", False))
        if status == "blocked" and required:
            cls = "hard_blocker"
        elif status == "blocked":
            cls = "soft_blocker"
        elif stale and required:
            cls = "stale_required"
        elif status not in {"ready"} and required:
            cls = "degraded_required"
        elif status not in {"ready"}:
            cls = "advisory_noise_or_followup"
        else:
            cls = "healthy_contract"
        counts[cls] += 1
        classifications.append({"node_id": row.get("node_id"), "status": status, "required": required, "stale": stale, "contract_class": cls})
    details = {
        "contract_class_counts": dict(sorted(counts.items())),
        "classifications": classifications,
        "hardening_policy": "hard_blockers_stop_widening_soft_and_stale_blockers_route_to_refresh_or_observe",
    }
    status = "active_followup_needed" if counts.get("hard_blocker") or counts.get("stale_required") else "ready"
    return _lane_base("dependency_contract_hardening", status=status, priority="critical", readiness=0.85, details=details)


def _daily_operator_memory(sources: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    self_model = _as_dict(sources.get("self_model"))
    graph = _as_dict(sources.get("architecture_graph"))
    summary = str(self_model.get("self_summary") or "").strip()
    details = {
        "memory_entry": {
            "timestamp_utc": iso_now(),
            "summary": summary[:1200],
            "pressure_band": runtime.get("pressure_band"),
            "blocked_nodes": _as_list(graph.get("blocked_nodes")),
            "degraded_nodes": _as_list(graph.get("degraded_nodes"))[:12],
            "next_focus": "self_healing_router_then_stale_surface_autofix_then_indicator_feature_bridge",
        },
        "memory_path": str(DEFAULT_MEMORY_PATH),
        "write_policy": "append_on_apply_only",
    }
    return _lane_base("daily_operator_memory", status="ready", priority="medium", readiness=0.9, details=details)


def _build_lanes(sources: dict[str, Any], runtime: dict[str, Any]) -> list[dict[str, Any]]:
    lanes = [
        _predictive_stability(sources, runtime),
        _self_healing_router(sources),
        _stale_surface_autofix(sources),
        _indicator_feature_bridge(sources),
        _collector_utility_budget(sources),
        _grandmaster_safe_mode(sources, runtime),
        _training_deficiency_repair_loop(sources),
        _hot_path_storage_budget(sources, runtime),
        _capital_rotation_simulator_v2(sources, runtime),
        _promotion_evidence_ledger(sources),
        _dependency_contract_hardening(sources),
        _daily_operator_memory(sources, runtime),
    ]
    return sorted(lanes, key=lambda item: _safe_int(item.get("rank"), 999))


def _rollup(lanes: list[dict[str, Any]]) -> dict[str, Any]:
    active = [lane for lane in lanes if str(lane.get("status")) not in {"ready"}]
    critical = [lane for lane in lanes if str(lane.get("priority")) == "critical"]
    return {
        "lane_count": len(lanes),
        "ready_lane_count": len(lanes) - len(active),
        "followup_lane_count": len(active),
        "critical_lane_count": len(critical),
        "average_readiness_score": round(sum(_safe_float(lane.get("readiness_score"), 0.0) for lane in lanes) / max(len(lanes), 1), 4),
        "active_followup_lanes": [str(lane.get("lane_id")) for lane in active],
    }


def _write_override(path: Path, payload: dict[str, Any]) -> None:
    rollup = _as_dict(payload.get("rollup"))
    lines = [
        "# Generated by system_expansion_execution_layer.py",
        f"SYSTEM_EXPANSION_EXECUTION_READY={1 if payload.get('ok') else 0}",
        f"SYSTEM_EXPANSION_LANE_COUNT={rollup.get('lane_count', 0)}",
        f"SYSTEM_EXPANSION_READY_LANE_COUNT={rollup.get('ready_lane_count', 0)}",
        f"SYSTEM_EXPANSION_FOLLOWUP_LANE_COUNT={rollup.get('followup_lane_count', 0)}",
        f"SYSTEM_EXPANSION_AVERAGE_READINESS={rollup.get('average_readiness_score', 0)}",
        "SYSTEM_EXPANSION_LIVE_EXECUTION_AUTHORITY=0",
        "SYSTEM_EXPANSION_PAPER_EXECUTION_AUTHORITY=0",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_memory(path: Path, payload: dict[str, Any]) -> None:
    lane = next((row for row in _as_list(payload.get("lanes")) if isinstance(row, dict) and row.get("lane_id") == "daily_operator_memory"), {})
    entry = _as_dict(_as_dict(lane.get("details")).get("memory_entry"))
    if not entry:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")


def build_payload(project_root: Path = PROJECT_ROOT, *, apply: bool = False, override_path: Path = DEFAULT_OVERRIDE_PATH, memory_path: Path = DEFAULT_MEMORY_PATH) -> dict[str, Any]:
    sources = _load_sources(project_root)
    runtime = _runtime_snapshot(sources)
    lanes = _build_lanes(sources, runtime)
    rollup = _rollup(lanes)
    ok = len(lanes) == len(LANE_DEFINITIONS)
    status = "system_expansion_execution_ready_guarded" if runtime.get("guarded") else "system_expansion_execution_ready"
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": status if ok else "system_expansion_execution_incomplete",
        "authority_boundary": LIVE_LOCK,
        "runtime_snapshot": runtime,
        "rollup": rollup,
        "lanes": lanes,
        "recommended_sequence": [
            "self_healing_router",
            "stale_surface_autofix",
            "predictive_stability",
            "collector_utility_budget",
            "schwab_indicator_feature_bridge",
            "grandmaster_safe_mode",
            "training_deficiency_repair_loop",
            "hot_path_storage_budget",
            "capital_rotation_simulator_v2",
            "promotion_evidence_ledger",
            "dependency_contract_hardening",
            "daily_operator_memory",
        ],
        "recommended_commands": {
            "refresh_expansion_layer": ["./scripts/ops/opsctl.sh", "system-expansion-execution", "--json"],
            "apply_expansion_layer": ["./scripts/ops/opsctl.sh", "system-expansion-execution", "--apply", "--json"],
            "refresh_architecture_graph": ["./scripts/ops/opsctl.sh", "system-architecture-contract-graph", "--apply", "--json"],
            "refresh_self_model": ["./scripts/ops/opsctl.sh", "big-platform-brain", "--json"],
        },
        "source_status": {name: _status(_as_dict(value)) for name, value in sources.items() if name != "registry"},
    }
    if apply:
        _write_override(override_path, payload)
        _append_memory(memory_path, payload)
        payload["write_result"] = {"applied": True, "override_path": str(override_path), "memory_path": str(memory_path)}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the 12-lane system expansion execution layer.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override-path", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--memory-path", default=str(DEFAULT_MEMORY_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(
        project_root,
        apply=bool(args.apply),
        override_path=Path(args.override_path).expanduser(),
        memory_path=Path(args.memory_path).expanduser(),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        rollup = _as_dict(payload.get("rollup"))
        print(
            "system_expansion_execution_layer "
            f"status={payload.get('overall_status')} "
            f"lanes={rollup.get('lane_count', 0)} "
            f"ready={rollup.get('ready_lane_count', 0)} "
            f"followup={rollup.get('followup_lane_count', 0)} "
            f"avg_readiness={rollup.get('average_readiness_score', 0)}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
