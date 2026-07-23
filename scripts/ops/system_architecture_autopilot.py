#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops import system_architecture_contract_graph
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from . import system_architecture_contract_graph
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_architecture_autopilot_latest.json"
DEFAULT_PLAN_PATH = PROJECT_ROOT / "governance" / "architecture_contracts" / "system_architecture_autopilot_plan_latest.json"
DEFAULT_BENEFIT_PATH = PROJECT_ROOT / "governance" / "architecture_contracts" / "system_architecture_benefit_backlog_latest.json"

Runner = Callable[[list[str], Path, int], dict[str, Any]]
GraphBuilder = Callable[[Path, bool], dict[str, Any]]

PHASE_BY_CLASS = {
    "readiness": 0,
    "runtime": 0,
    "storage": 0,
    "infrastructure": 1,
    "paper_execution": 1,
    "sleeve_runtime": 1,
    "governance_registry": 2,
    "governance": 2,
    "adaptive_governance": 2,
    "architecture": 3,
    "self_model": 4,
}
PHASE_NAMES = {
    0: "foundation_truth",
    1: "runtime_and_paper_fanout",
    2: "governance_and_drift",
    3: "architecture_contracts",
    4: "self_model_and_operator_view",
}
BLOCKED_STATUSES = {"blocked", "critical", "missing"}
READYISH_STATUSES = {"ready", "ok", "stable", "advisory", "guarded_ready", "guarded_relief", "clear_ready", "armed"}
LIVE_COMMAND_PATTERNS = ("start-live", "clear-all-halts", "operator-release")
PAPER_RUNTIME_NODES = {"paper_ramp", "all_sleeves_launcher", "runtime_paper_guard"}
GOVERNANCE_NODES = {"grade_regression_guard", "section_grade_guard", "adaptive_regression_guard", "system_drift_guard"}
BENEFIT_BLUEPRINTS: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "paper_runtime_load_shedder",
        "title": "Paper Runtime Load Shedder",
        "phase": 1,
        "signals": ("guarded_paper_blocked_runtime", "paper_execution_hot", "runtime_degraded"),
        "architecture_delta": "add a paper-only load-shed contract that downshifts paper execution consumers before they block guarded paper",
        "benefit": "keeps all sleeves collecting while preventing hot paper execution CPU from holding the paper gate closed",
        "safe_commands": [
            ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--max-renice-processes", "30", "--json"],
            ["./scripts/ops/opsctl.sh", "runtime-paper-regression-guard", "--json"],
        ],
        "guards": ["does_not_stop_all_sleeves_launcher", "does_not_enable_live_execution", "paper_only_runtime_boundary"],
        "acceptance_criteria": ["guarded_paper_ready", "paper_execution_cpu_below_hot_band", "runtime_status_ready_or_advisory"],
    },
    {
        "candidate_id": "single_writer_pressure_arbitration",
        "title": "Single Writer Pressure Arbitration",
        "phase": 1,
        "signals": ("storage_writer_hot", "storage_backpressure_apply_failed", "runtime_degraded"),
        "architecture_delta": "centralize storage writer, drain, retention, and paper execution pressure arbitration into one single-writer decision lane",
        "benefit": "prevents storage cleanup, SQL writer catch-up, and paper execution from competing for the same CPU/write budget",
        "safe_commands": [
            ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--quick-bounded", "--json"],
        ],
        "guards": ["single_writer_only", "bounded_storage_apply", "no_duplicate_sqlite_writers"],
        "acceptance_criteria": ["storage_writer_hot_false", "raw_live_backlog_within_gate", "storage_backpressure_apply_ready"],
    },
    {
        "candidate_id": "governance_closeout_lane",
        "title": "Governance Closeout Lane",
        "phase": 2,
        "signals": ("adaptive_memory_blocked", "incident_closeout_blocked", "live_canary_blocked"),
        "architecture_delta": "give incident closeout and live canary their own first-class recovery lane upstream of section-grade and drift blockers",
        "benefit": "turns persistent governance red rows into the next explicit closeout workflow instead of repeated generic guard refreshes",
        "safe_commands": [
            ["./scripts/ops/opsctl.sh", "incident-closeout", "--json"],
            ["./scripts/ops/opsctl.sh", "live-canary-control", "--json"],
            ["./scripts/ops/opsctl.sh", "section-grade-autopilot", "--apply", "--json"],
        ],
        "guards": ["no_operator_release", "no_live_promotion", "evidence_first"],
        "acceptance_criteria": ["open_incident_count_zero_or_waived", "live_canary_preclearance_ready", "section_grades_at_floor"],
    },
    {
        "candidate_id": "artifact_freshness_slo_mesh",
        "title": "Artifact Freshness SLO Mesh",
        "phase": 2,
        "signals": ("stale_nodes", "adaptive_degraded_stale_surfaces", "system_drift_blocked"),
        "architecture_delta": "publish per-surface freshness SLOs and refresh ownership so stale-only regressions stop masquerading as system risk",
        "benefit": "keeps dashboards, one-number guards, and storage guards from becoming noisy blockers after the runtime has already recovered",
        "safe_commands": [
            ["./scripts/ops/opsctl.sh", "system-drift-registry", "--json"],
            ["./scripts/ops/opsctl.sh", "system-drift-guard", "--json"],
        ],
        "guards": ["read_only_refresh_first", "no_destructive_cleanup", "surface_owner_required"],
        "acceptance_criteria": ["stale_node_count_zero", "stale_degraded_surfaces_below_threshold"],
    },
    {
        "candidate_id": "operator_phone_decision_card",
        "title": "Operator Phone Decision Card",
        "phase": 4,
        "signals": ("operator_visibility_needed", "architecture_blocked", "guarded_paper_blocked_runtime"),
        "architecture_delta": "emit a tiny phone-safe decision card with paper, live, runtime, governance, and top safe command in one object",
        "benefit": "makes mobile monitoring actionable without tailing large JSON artifacts",
        "safe_commands": [["./scripts/ops/opsctl.sh", "system-architecture-autopilot", "--apply", "--json"]],
        "guards": ["read_only_summary", "no_secrets", "no_live_authority"],
        "acceptance_criteria": ["single_screen_summary", "top_safe_command_present", "paper_live_boundary_visible"],
    },
    {
        "candidate_id": "preflight_repair_simulator",
        "title": "Preflight Repair Simulator",
        "phase": 3,
        "signals": ("repair_plan_present", "safe_repairs_present", "architecture_blocked"),
        "architecture_delta": "simulate repair attempts against dependencies, timeouts, and authority before running any apply command",
        "benefit": "reduces repeated repair churn and avoids spending maintenance windows on commands that cannot clear the root blocker",
        "safe_commands": [["./scripts/ops/opsctl.sh", "system-architecture-autopilot", "--apply", "--json"]],
        "guards": ["simulation_before_apply", "bounded_timeout", "authority_filter_required"],
        "acceptance_criteria": ["preflight_result_for_each_step", "root_blocker_not_retried_without_changed_evidence"],
    },
    {
        "candidate_id": "capability_budget_router",
        "title": "Capability Budget Router",
        "phase": 3,
        "signals": ("runtime_degraded", "bot_owned_pressure_dominant", "active_regression_count_high"),
        "architecture_delta": "allocate CPU, storage writer, paper execution, collector, training, and reporting budgets by current value and pressure",
        "benefit": "lets useful sleeves keep running while low-value support work yields under pressure",
        "safe_commands": [
            ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "memory-pressure-intelligence", "--apply", "--json"],
        ],
        "guards": ["paper_live_boundary_preserved", "training_yields_first", "reporting_yields_before_collectors"],
        "acceptance_criteria": ["runtime_ready_or_advisory", "collector_health_ready", "paper_execution_not_hot"],
    },
)


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    if isinstance(raw, tuple):
        return list(raw)
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


def _parse_json_output(text: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(text or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _tail_text(text: str, *, max_lines: int = 12, max_chars: int = 4000) -> str:
    tail = "\n".join(str(text or "").splitlines()[-max_lines:])
    if len(tail) <= max_chars:
        return tail
    return "...truncated...\n" + tail[-max_chars:]


def _run(cmd: list[str], project_root: Path, timeout_sec: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        return {
            "cmd": list(cmd),
            "rc": int(proc.returncode),
            "payload": _parse_json_output(proc.stdout or ""),
            "stdout_tail": _tail_text(proc.stdout or ""),
            "stderr_tail": _tail_text(proc.stderr or ""),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return {
            "cmd": list(cmd),
            "rc": 124,
            "payload": _parse_json_output(stdout),
            "stdout_tail": _tail_text(stdout),
            "stderr_tail": _tail_text(stderr) or "timeout",
        }


def _build_graph(project_root: Path, apply: bool) -> dict[str, Any]:
    return system_architecture_contract_graph.build_payload(project_root, apply=apply)


def _dependency_depth(node_id: str, nodes: dict[str, dict[str, Any]], visiting: set[str] | None = None) -> int:
    if node_id not in nodes:
        return 0
    active = set(visiting or set())
    if node_id in active:
        return 0
    active.add(node_id)
    deps = [str(dep) for dep in _as_list(nodes[node_id].get("depends_on")) if str(dep) in nodes]
    if not deps:
        return 0
    return 1 + max(_dependency_depth(dep, nodes, active) for dep in deps)


def _blocked_dependency_count(node: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> int:
    count = 0
    for dep_id in _as_list(node.get("depends_on")):
        dep = nodes.get(str(dep_id))
        if dep and str(dep.get("status") or "") in BLOCKED_STATUSES:
            count += 1
    return count


def _safe_repair_allowed(node: dict[str, Any]) -> bool:
    if _as_list(node.get("authority_violations")):
        return False
    authority = str(node.get("authority") or "")
    if "live" in authority and "paper" not in authority and "observation" not in authority:
        return False
    for cmd in _as_list(node.get("commands")):
        joined = " ".join(str(part) for part in _as_list(cmd))
        if "start-live" in joined or "clear-all-halts" in joined or "operator-release" in joined:
            return False
    return True


def _plan_from_graph(graph: dict[str, Any], *, max_steps: int) -> list[dict[str, Any]]:
    nodes = {str(row.get("node_id") or ""): row for row in _as_list(graph.get("nodes")) if isinstance(row, dict)}
    candidates = [
        node
        for node in nodes.values()
        if str(node.get("status") or "") != "ready" or bool(node.get("artifact_stale", False))
    ]
    ranked = sorted(
        candidates,
        key=lambda node: (
            PHASE_BY_CLASS.get(str(node.get("class") or ""), 9),
            _dependency_depth(str(node.get("node_id") or ""), nodes),
            _blocked_dependency_count(node, nodes),
            0 if str(node.get("status") or "") in BLOCKED_STATUSES else 1,
            str(node.get("node_id") or ""),
        ),
    )

    plan: list[dict[str, Any]] = []
    seen_commands: set[str] = set()
    for node in ranked:
        commands = [cmd for cmd in _as_list(node.get("commands")) if isinstance(cmd, list) and cmd]
        command = [str(part) for part in commands[0]] if commands else []
        key = " ".join(command)
        if not key or key in seen_commands:
            continue
        seen_commands.add(key)
        phase = PHASE_BY_CLASS.get(str(node.get("class") or ""), 9)
        plan.append(
            {
                "node_id": str(node.get("node_id") or ""),
                "phase": phase,
                "phase_name": PHASE_NAMES.get(phase, "other"),
                "class": str(node.get("class") or ""),
                "status": str(node.get("status") or ""),
                "artifact_stale": bool(node.get("artifact_stale", False)),
                "blocked_dependency_count": _blocked_dependency_count(node, nodes),
                "safe_to_execute": _safe_repair_allowed(node),
                "cmd": command,
                "timeout_sec": 300 if phase <= 2 else 600,
                "reason": "blocked" if str(node.get("status") or "") in BLOCKED_STATUSES else "stale" if bool(node.get("artifact_stale", False)) else "degraded",
            }
        )
        if len(plan) >= max(int(max_steps), 1):
            break
    return plan


def _phase_rollup(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phases: dict[int, dict[str, Any]] = {}
    for step in plan:
        phase = _safe_int(step.get("phase"), 9)
        row = phases.setdefault(
            phase,
            {
                "phase": phase,
                "phase_name": str(step.get("phase_name") or PHASE_NAMES.get(phase, "other")),
                "step_count": 0,
                "safe_step_count": 0,
                "nodes": [],
            },
        )
        row["step_count"] += 1
        if bool(step.get("safe_to_execute", False)):
            row["safe_step_count"] += 1
        row["nodes"].append(str(step.get("node_id") or ""))
    return [phases[key] for key in sorted(phases)]


def _command_key(command: Any) -> str:
    return " ".join(str(part) for part in _as_list(command))


def _nodes_by_id(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("node_id") or ""): row for row in _as_list(graph.get("nodes")) if isinstance(row, dict)}


def _readyish(status: Any) -> bool:
    return str(status or "").strip().lower() in READYISH_STATUSES


def _non_ready_dependency_ids(node: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> list[str]:
    deps: list[str] = []
    for dep_id in _as_list(node.get("depends_on")):
        dep = nodes.get(str(dep_id))
        if dep and not _readyish(dep.get("status")):
            deps.append(str(dep_id))
    return deps


def _downstream_ids(node_id: str, nodes: dict[str, dict[str, Any]]) -> list[str]:
    downstream: list[str] = []
    queue = [node_id]
    seen = {node_id}
    while queue:
        current = queue.pop(0)
        for candidate_id, candidate in nodes.items():
            if candidate_id in seen:
                continue
            if current in {str(dep) for dep in _as_list(candidate.get("depends_on"))}:
                downstream.append(candidate_id)
                queue.append(candidate_id)
                seen.add(candidate_id)
    return downstream


def _dependency_explainability(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = _nodes_by_id(graph)
    blocked_or_degraded = [
        node
        for node in nodes.values()
        if not _readyish(node.get("status")) or bool(node.get("artifact_stale", False))
    ]
    root_rows: list[dict[str, Any]] = []
    for node in blocked_or_degraded:
        non_ready_deps = _non_ready_dependency_ids(node, nodes)
        if non_ready_deps:
            continue
        node_id = str(node.get("node_id") or "")
        root_rows.append(
            {
                "node_id": node_id,
                "class": str(node.get("class") or ""),
                "status": str(node.get("status") or ""),
                "artifact_stale": bool(node.get("artifact_stale", False)),
                "downstream_impacted_nodes": _downstream_ids(node_id, nodes),
                "summary": str(node.get("summary") or ""),
            }
        )
    return {
        "node_count": _safe_int(graph.get("node_count"), len(nodes)),
        "edge_count": _safe_int(graph.get("edge_count"), len(_as_list(graph.get("edges")))),
        "root_non_ready_nodes": root_rows,
        "blocked_edges": _as_list(graph.get("blocked_edges"))[:10],
        "layer_rollup": _as_dict(graph.get("layers")),
        "contract": "nodes declare freshness, authority, dependencies, and one bounded refresh command",
    }


def _repair_order_contract(repair_plan: list[dict[str, Any]]) -> dict[str, Any]:
    rollup = _phase_rollup(repair_plan)
    return {
        "phase_count": len(rollup),
        "phases": rollup,
        "first_active_phase": rollup[0]["phase_name"] if rollup else "none",
        "dependency_ordering_policy": "lower phases repair first; governance blockers precede architecture and self-model widening",
        "safe_command_count": sum(1 for step in repair_plan if bool(step.get("safe_to_execute", False))),
        "blocked_command_count": sum(1 for step in repair_plan if not bool(step.get("safe_to_execute", False))),
    }


def _live_execution_firewall(graph: dict[str, Any], repair_plan: list[dict[str, Any]]) -> dict[str, Any]:
    authority_violations = _as_list(graph.get("authority_violations"))
    unsafe_steps = [step for step in repair_plan if not bool(step.get("safe_to_execute", False))]
    live_like_commands = [
        list(step.get("cmd") or [])
        for step in repair_plan
        if any(pattern in _command_key(step.get("cmd")) for pattern in LIVE_COMMAND_PATTERNS)
    ]
    status = "ready"
    if authority_violations:
        status = "blocked"
    elif unsafe_steps or live_like_commands:
        status = "guarded"
    return {
        "status": status,
        "live_execution_authority": False,
        "authority_violation_count": _safe_int(graph.get("authority_violation_count"), len(authority_violations)),
        "authority_violations": authority_violations,
        "unsafe_repair_step_count": len(unsafe_steps),
        "unsafe_nodes": [str(step.get("node_id") or "") for step in unsafe_steps],
        "live_like_commands_filtered": live_like_commands,
        "blocked_patterns": list(LIVE_COMMAND_PATTERNS),
        "contract": "safe repairs never start live execution, clear halts, or release operator stops",
    }


def _paper_governance_split(project_root: Path, graph: dict[str, Any]) -> dict[str, Any]:
    nodes = _nodes_by_id(graph)
    health = load_json(project_root / "governance" / "health" / "health_fast_latest.json")
    operational = _as_dict(health.get("operational_readiness"))
    guarded_paper = _as_dict(operational.get("guarded_paper"))
    live_execution = _as_dict(operational.get("live_execution"))
    all_sleeves = _as_dict(_as_dict(health.get("process_watchdog")).get("all_sleeves_effective_runtime"))
    paper_statuses = {
        node_id: str(nodes.get(node_id, {}).get("status") or "missing")
        for node_id in sorted(PAPER_RUNTIME_NODES)
    }
    graph_paper_ready = bool(paper_statuses) and all(_readyish(status) for status in paper_statuses.values())
    guarded_status = str(guarded_paper.get("status") or "").strip().lower()
    health_paper_status_present = bool(guarded_status)
    health_paper_ready = guarded_status == "ready"
    guarded_paper_ready = health_paper_ready if health_paper_status_present else graph_paper_ready
    strict_governance_ready = str(graph.get("overall_status") or "").strip().lower() == "ready"
    return {
        "guarded_paper_status": str(guarded_paper.get("status") or ("ready" if graph_paper_ready else "unknown")),
        "guarded_paper_ready": guarded_paper_ready,
        "guarded_paper_blockers": _as_list(guarded_paper.get("blockers")),
        "all_sleeves_status": str(all_sleeves.get("status") or paper_statuses.get("all_sleeves_launcher") or "unknown"),
        "all_sleeves_child_process_count": _safe_int(all_sleeves.get("child_process_count"), 0),
        "live_execution_status": str(live_execution.get("status") or "blocked_read_only"),
        "paper_graph_statuses": paper_statuses,
        "paper_contract_ready": graph_paper_ready,
        "strict_governance_status": str(graph.get("overall_status") or "unknown"),
        "strict_governance_ready": strict_governance_ready,
        "paper_governance_split_active": bool(guarded_paper_ready and not strict_governance_ready),
        "contract": "guarded paper may keep running when strict live/governance gates remain blocked",
    }


def _adaptive_memory_pressure(project_root: Path) -> dict[str, Any]:
    artifact_path = project_root / "governance" / "health" / "adaptive_regression_guard_latest.json"
    payload = load_json(artifact_path)
    surfaces = [row for row in _as_list(payload.get("surfaces")) if isinstance(row, dict)]
    critical = [
        {
            "surface": str(row.get("surface") or row.get("surface_id") or ""),
            "state": str(row.get("state") or row.get("status") or ""),
            "adaptive_severity": str(row.get("adaptive_severity") or row.get("base_severity") or ""),
            "summary": str(row.get("summary") or ""),
            "consecutive_non_ready_count": _safe_int(_as_dict(row.get("memory")).get("consecutive_non_ready_count"), 0),
            "consecutive_blocked_count": _safe_int(_as_dict(row.get("memory")).get("consecutive_blocked_count"), 0),
        }
        for row in surfaces
        if str(row.get("state") or row.get("status") or "").strip().lower() == "blocked"
        or str(row.get("adaptive_severity") or "").strip().lower() == "critical"
        or bool(row.get("repeated_blocked_regression", False))
    ]
    degraded = [
        {
            "surface": str(row.get("surface") or row.get("surface_id") or ""),
            "state": str(row.get("state") or row.get("status") or ""),
            "summary": str(row.get("summary") or ""),
            "consecutive_non_ready_count": _safe_int(_as_dict(row.get("memory")).get("consecutive_non_ready_count"), 0),
        }
        for row in surfaces
        if str(row.get("state") or row.get("status") or "").strip().lower() == "degraded"
    ]
    status = "missing" if not payload else ("blocked" if critical else "tracking" if degraded else "ready")
    return {
        "status": status,
        "artifact_path": str(artifact_path),
        "overall_status": str(payload.get("overall_status") or "missing"),
        "active_regression_count": _safe_int(payload.get("active_regression_count"), 0),
        "persistent_regression_count": _safe_int(payload.get("persistent_regression_count"), 0),
        "critical_regression_count": _safe_int(payload.get("critical_regression_count"), len(critical)),
        "critical_surfaces": critical[:10],
        "degraded_surfaces": degraded[:10],
        "pressure_context": _as_dict(payload.get("pressure_context")),
        "contract": "repeated non-ready surfaces escalate by memory, not by one noisy run",
    }


def _operator_visibility(project_root: Path, repair_plan: list[dict[str, Any]]) -> dict[str, Any]:
    top_commands = [list(step.get("cmd") or []) for step in repair_plan[:8] if step.get("cmd")]
    return {
        "artifact_paths": {
            "latest": str(DEFAULT_OUT_PATH if project_root == PROJECT_ROOT else project_root / "governance" / "health" / "system_architecture_autopilot_latest.json"),
            "plan": str(DEFAULT_PLAN_PATH if project_root == PROJECT_ROOT else project_root / "governance" / "architecture_contracts" / "system_architecture_autopilot_plan_latest.json"),
            "contract_graph": str(project_root / "governance" / "health" / "system_architecture_contract_graph_latest.json"),
        },
        "operator_commands": {
            "plan_only": ["./scripts/ops/opsctl.sh", "system-architecture-autopilot", "--apply", "--json"],
            "safe_refresh": ["./scripts/ops/opsctl.sh", "system-architecture-autopilot", "--apply", "--execute-safe-repairs", "--max-step-timeout-seconds", "180"],
            "graph_status": ["./scripts/ops/opsctl.sh", "system-architecture-contract-graph", "--apply", "--json"],
            "phone_summary": [
                "jq",
                "{overall_status, repair_step_count, safe_repair_step_count, phase_rollup, paper_governance_split, live_execution_firewall}",
                "governance/health/system_architecture_autopilot_latest.json",
            ],
        },
        "next_repair_commands": top_commands,
        "contract": "one artifact gives the operator the plan, safety boundaries, paper status, and next commands",
    }


def _architecture_expansion_layers(
    project_root: Path,
    final_graph: dict[str, Any],
    repair_plan: list[dict[str, Any]],
) -> dict[str, Any]:
    dependency_map = _dependency_explainability(final_graph)
    repair_order = _repair_order_contract(repair_plan)
    live_firewall = _live_execution_firewall(final_graph, repair_plan)
    paper_split = _paper_governance_split(project_root, final_graph)
    adaptive_memory = _adaptive_memory_pressure(project_root)
    operator_visibility = _operator_visibility(project_root, repair_plan)
    degradation = {
        "overall_status": str(final_graph.get("overall_status") or ""),
        "blocked_nodes": _as_list(final_graph.get("blocked_nodes")),
        "degraded_nodes": _as_list(final_graph.get("degraded_nodes")),
        "stale_nodes": _as_list(final_graph.get("stale_nodes")),
        "root_non_ready_nodes": dependency_map["root_non_ready_nodes"],
        "contract": "blocked/degraded status names the surface, upstream dependency, and first safe command",
    }
    layers = [
        {
            "point": 1,
            "layer_id": "dependency_map",
            "title": "System Dependency Map",
            "status": "ready" if dependency_map["node_count"] > 0 else "missing",
            "what_it_adds": "turns standalone health artifacts into a graph of dependencies and authority boundaries",
            "evidence": dependency_map,
            "operator_command": ["./scripts/ops/opsctl.sh", "system-architecture-contract-graph", "--apply", "--json"],
        },
        {
            "point": 2,
            "layer_id": "repair_ordering",
            "title": "Dependency-Ordered Repair Phases",
            "status": "ready" if repair_order["phase_count"] > 0 else "idle",
            "what_it_adds": "keeps repairs in foundation, runtime, governance, architecture, then self-model order",
            "evidence": repair_order,
            "operator_command": ["./scripts/ops/opsctl.sh", "system-architecture-autopilot", "--apply", "--json"],
        },
        {
            "point": 3,
            "layer_id": "live_execution_firewall",
            "title": "Live Execution Boundary Firewall",
            "status": live_firewall["status"],
            "what_it_adds": "lets safe architecture refreshes run without creating live-order authority",
            "evidence": live_firewall,
            "operator_command": ["./scripts/ops/opsctl.sh", "system-architecture-autopilot", "--apply", "--json"],
        },
        {
            "point": 4,
            "layer_id": "paper_governance_split",
            "title": "Guarded Paper And Strict Governance Split",
            "status": "ready" if paper_split["guarded_paper_ready"] else "blocked",
            "what_it_adds": "keeps paper trading continuity independent from stricter live-readiness gates",
            "evidence": paper_split,
            "operator_command": ["./scripts/ops/opsctl.sh", "health-fast", "--json"],
        },
        {
            "point": 5,
            "layer_id": "degradation_explainability",
            "title": "Explainable Degradation",
            "status": "ready" if degradation["root_non_ready_nodes"] or final_graph else "missing",
            "what_it_adds": "names root non-ready surfaces and downstream impact instead of reporting a generic degraded state",
            "evidence": degradation,
            "operator_command": ["./scripts/ops/opsctl.sh", "system-architecture-contract-graph", "--apply", "--json"],
        },
        {
            "point": 6,
            "layer_id": "adaptive_memory_pressure",
            "title": "Adaptive Regression Memory",
            "status": adaptive_memory["status"],
            "what_it_adds": "separates one-off noise from persistent blocked surfaces using run-to-run memory",
            "evidence": adaptive_memory,
            "operator_command": ["./scripts/ops/opsctl.sh", "adaptive-regression-guard", "--apply", "--json"],
        },
        {
            "point": 7,
            "layer_id": "operator_visibility",
            "title": "Operator Visibility And Phone-Friendly Commands",
            "status": "ready",
            "what_it_adds": "publishes the exact artifacts and compact commands needed to inspect or refresh the architecture layer",
            "evidence": operator_visibility,
            "operator_command": operator_visibility["operator_commands"]["phone_summary"],
        },
    ]
    return {
        "layer_count": len(layers),
        "ready_layer_count": sum(1 for layer in layers if layer["status"] in {"ready", "tracking", "guarded", "idle"}),
        "blocked_layer_count": sum(1 for layer in layers if layer["status"] == "blocked"),
        "layers": layers,
        "dependency_map": dependency_map,
        "repair_order_contract": repair_order,
        "live_execution_firewall": live_firewall,
        "paper_governance_split": paper_split,
        "degradation_explainability": degradation,
        "adaptive_memory_pressure": adaptive_memory,
        "operator_visibility": operator_visibility,
    }


def _runtime_pressure_evidence(project_root: Path) -> dict[str, Any]:
    runtime = load_json(project_root / "governance" / "health" / "runtime_throttle_control_latest.json")
    soft_cap = _as_dict(runtime.get("soft_cap_advisory_reclassification"))
    measurements = _as_dict(soft_cap.get("measurements"))
    if not measurements:
        measurements = _as_dict(runtime.get("measurements"))
    return {
        "overall_status": str(runtime.get("overall_status") or "missing"),
        "host_saturation_score": _safe_float(runtime.get("host_saturation_score"), 0.0),
        "compute_pressure_level": str(runtime.get("compute_pressure_level") or ""),
        "memory_pressure_level": str(runtime.get("memory_pressure_level") or ""),
        "bot_owned_cpu_percent": _safe_float(measurements.get("bot_owned_cpu_percent"), 0.0),
        "storage_writer_cpu_percent": _safe_float(measurements.get("storage_writer_cpu_percent"), 0.0),
        "paper_execution_cpu_percent": _safe_float(measurements.get("paper_execution_cpu_percent"), 0.0),
        "paper_execution_hot": bool(measurements.get("paper_execution_hot", False)),
        "storage_writer_hot": bool(measurements.get("storage_writer_hot", False)),
        "paper_execution_paused": bool(measurements.get("paper_execution_paused", False)),
        "paper_ramp_armed": bool(measurements.get("paper_ramp_armed", False)),
        "bot_owned_pressure_dominant": bool(measurements.get("bot_owned_pressure_dominant", False)),
    }


def _signal_context(
    project_root: Path,
    final_graph: dict[str, Any],
    expansion: dict[str, Any],
    repair_plan: list[dict[str, Any]],
) -> dict[str, Any]:
    runtime = _runtime_pressure_evidence(project_root)
    storage_autopilot = load_json(project_root / "governance" / "health" / "storage_backpressure_autopilot_latest.json")
    paper_split = _as_dict(expansion.get("paper_governance_split"))
    adaptive = _as_dict(expansion.get("adaptive_memory_pressure"))
    critical_surfaces = {str(row.get("surface") or "") for row in _as_list(adaptive.get("critical_surfaces")) if isinstance(row, dict)}
    degraded_surfaces = [row for row in _as_list(adaptive.get("degraded_surfaces")) if isinstance(row, dict)]
    signals = {
        "architecture_blocked": str(final_graph.get("overall_status") or "") == "blocked",
        "repair_plan_present": bool(repair_plan),
        "safe_repairs_present": any(bool(step.get("safe_to_execute", False)) for step in repair_plan),
        "guarded_paper_blocked_runtime": (
            str(paper_split.get("guarded_paper_status") or "") == "blocked"
            and "runtime_status=degraded" in {str(item) for item in _as_list(paper_split.get("guarded_paper_blockers"))}
        ),
        "runtime_degraded": str(runtime.get("overall_status") or "") == "degraded",
        "paper_execution_hot": bool(runtime.get("paper_execution_hot", False)),
        "storage_writer_hot": bool(runtime.get("storage_writer_hot", False)),
        "storage_backpressure_apply_failed": str(storage_autopilot.get("overall_status") or "") == "apply_failed",
        "adaptive_memory_blocked": str(adaptive.get("status") or "") == "blocked",
        "incident_closeout_blocked": "incident_closeout" in critical_surfaces,
        "live_canary_blocked": "live_canary" in critical_surfaces,
        "stale_nodes": bool(_as_list(final_graph.get("stale_nodes"))),
        "adaptive_degraded_stale_surfaces": any("stale" in str(row.get("summary") or "").lower() for row in degraded_surfaces),
        "system_drift_blocked": "system_drift_guard" in {str(item) for item in _as_list(final_graph.get("blocked_nodes"))},
        "operator_visibility_needed": True,
        "bot_owned_pressure_dominant": bool(runtime.get("bot_owned_pressure_dominant", False))
        or _safe_float(runtime.get("bot_owned_cpu_percent"), 0.0) >= 200.0,
        "active_regression_count_high": _safe_int(adaptive.get("active_regression_count"), 0) >= 5,
    }
    return {
        "signals": signals,
        "runtime": runtime,
        "storage_backpressure_autopilot": {
            "overall_status": str(storage_autopilot.get("overall_status") or "missing"),
            "ok": bool(storage_autopilot.get("ok", False)),
        },
        "adaptive": {
            "status": str(adaptive.get("status") or ""),
            "active_regression_count": _safe_int(adaptive.get("active_regression_count"), 0),
            "persistent_regression_count": _safe_int(adaptive.get("persistent_regression_count"), 0),
            "critical_regression_count": _safe_int(adaptive.get("critical_regression_count"), 0),
            "critical_surfaces": sorted(critical_surfaces),
        },
        "paper": {
            "guarded_paper_status": str(paper_split.get("guarded_paper_status") or ""),
            "guarded_paper_ready": bool(paper_split.get("guarded_paper_ready", False)),
            "paper_contract_ready": bool(paper_split.get("paper_contract_ready", False)),
            "all_sleeves_status": str(paper_split.get("all_sleeves_status") or ""),
            "all_sleeves_child_process_count": _safe_int(paper_split.get("all_sleeves_child_process_count"), 0),
        },
    }


def _score_blueprint(blueprint: dict[str, Any], signal_context: dict[str, Any]) -> dict[str, Any]:
    signals = _as_dict(signal_context.get("signals"))
    trigger_names = [str(item) for item in _as_list(blueprint.get("signals"))]
    matched = [name for name in trigger_names if bool(signals.get(name, False))]
    score = 10 * len(matched)
    if matched and str(blueprint.get("candidate_id") or "") == "paper_runtime_load_shedder":
        runtime = _as_dict(signal_context.get("runtime"))
        score += min(int(_safe_float(runtime.get("paper_execution_cpu_percent"), 0.0) // 10), 10)
    if matched and str(blueprint.get("candidate_id") or "") == "governance_closeout_lane":
        adaptive = _as_dict(signal_context.get("adaptive"))
        score += min(_safe_int(adaptive.get("critical_regression_count"), 0) * 2, 12)
    if str(blueprint.get("candidate_id") or "") == "operator_phone_decision_card":
        score = max(score, 8)
    status = "active" if score >= 20 else "recommended" if score > 0 else "candidate"
    return {
        "candidate_id": str(blueprint.get("candidate_id") or ""),
        "title": str(blueprint.get("title") or ""),
        "phase": _safe_int(blueprint.get("phase"), 9),
        "status": status,
        "score": score,
        "matched_signals": matched,
        "missing_signals": [name for name in trigger_names if name not in matched],
        "architecture_delta": str(blueprint.get("architecture_delta") or ""),
        "benefit": str(blueprint.get("benefit") or ""),
        "safe_commands": [list(cmd) for cmd in _as_list(blueprint.get("safe_commands")) if isinstance(cmd, list)],
        "guards": [str(item) for item in _as_list(blueprint.get("guards"))],
        "acceptance_criteria": [str(item) for item in _as_list(blueprint.get("acceptance_criteria"))],
        "authority_contract": {
            "does_not_enable_live_execution": True,
            "requires_apply_for_mutation": True,
            "operator_release_allowed": False,
        },
    }


def _architecture_benefit_backlog(
    project_root: Path,
    final_graph: dict[str, Any],
    expansion: dict[str, Any],
    repair_plan: list[dict[str, Any]],
) -> dict[str, Any]:
    signal_context = _signal_context(project_root, final_graph, expansion, repair_plan)
    candidates = [_score_blueprint(blueprint, signal_context) for blueprint in BENEFIT_BLUEPRINTS]
    ranked = sorted(candidates, key=lambda row: (-_safe_int(row.get("score"), 0), _safe_int(row.get("phase"), 9), str(row.get("candidate_id") or "")))
    active = [row for row in ranked if str(row.get("status") or "") == "active"]
    recommended = [row for row in ranked if str(row.get("status") or "") == "recommended"]
    top = ranked[0] if ranked else {}
    return {
        "generation": "system_architecture_benefit_backlog_v1",
        "candidate_count": len(ranked),
        "active_candidate_count": len(active),
        "recommended_candidate_count": len(recommended),
        "top_candidate_id": str(top.get("candidate_id") or ""),
        "top_candidate_score": _safe_int(top.get("score"), 0),
        "top_candidate_title": str(top.get("title") or ""),
        "signal_context": signal_context,
        "candidates": ranked,
        "active_candidates": active,
        "recommended_candidates": recommended,
        "new_architecture_contract": {
            "benefit_ranked": True,
            "live_safe": True,
            "paper_runtime_pressure_aware": True,
            "single_writer_aware": True,
            "governance_closeout_aware": True,
            "phone_operator_aware": True,
        },
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    execute_safe_repairs: bool = False,
    max_steps: int = 10,
    max_step_timeout_sec: int = 300,
    runner: Runner | None = None,
    graph_builder: GraphBuilder | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    build_graph = graph_builder or _build_graph
    run_step = runner or _run

    initial_graph = build_graph(project_root, bool(apply))
    repair_plan = _plan_from_graph(initial_graph, max_steps=max_steps)
    executable_plan = [step for step in repair_plan if bool(step.get("safe_to_execute", False))]

    attempts: list[dict[str, Any]] = []
    if apply and execute_safe_repairs:
        for step in executable_plan:
            timeout_sec = min(_safe_int(step.get("timeout_sec"), max_step_timeout_sec), max(int(max_step_timeout_sec), 1))
            result = run_step(list(step["cmd"]), project_root, timeout_sec)
            payload = _as_dict(result.get("payload"))
            attempts.append(
                {
                    "node_id": step["node_id"],
                    "phase": step["phase"],
                    "cmd": list(result.get("cmd") or []),
                    "rc": int(result.get("rc", 1)),
                    "timeout_sec": timeout_sec,
                    "payload_summary": {
                        key: payload.get(key)
                        for key in ("overall_status", "ok", "blocked_node_count", "blocked_surface_count", "below_floor_count")
                        if key in payload
                    },
                    "stdout_tail": str(result.get("stdout_tail") or ""),
                    "stderr_tail": str(result.get("stderr_tail") or ""),
                }
            )

    final_graph = build_graph(project_root, bool(apply))
    final_status = str(final_graph.get("overall_status") or "")
    recommended_commands = [list(step["cmd"]) for step in repair_plan if step.get("cmd")]
    expansion = _architecture_expansion_layers(project_root, final_graph, repair_plan)
    benefit_backlog = _architecture_benefit_backlog(project_root, final_graph, expansion, repair_plan)
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": final_status == "ready",
        "overall_status": final_status,
        "apply": bool(apply),
        "execute_safe_repairs": bool(execute_safe_repairs),
        "max_steps": int(max_steps),
        "max_step_timeout_sec": int(max_step_timeout_sec),
        "repair_step_count": len(repair_plan),
        "safe_repair_step_count": len(executable_plan),
        "attempt_count": len(attempts),
        "phase_count": len(_phase_rollup(repair_plan)),
        "initial_graph": {
            "overall_status": str(initial_graph.get("overall_status") or ""),
            "blocked_node_count": _safe_int(initial_graph.get("blocked_node_count"), 0),
            "degraded_node_count": _safe_int(initial_graph.get("degraded_node_count"), 0),
            "stale_node_count": _safe_int(initial_graph.get("stale_node_count"), 0),
            "blocked_edge_count": _safe_int(initial_graph.get("blocked_edge_count"), 0),
        },
        "final_graph": {
            "overall_status": final_status,
            "blocked_node_count": _safe_int(final_graph.get("blocked_node_count"), 0),
            "degraded_node_count": _safe_int(final_graph.get("degraded_node_count"), 0),
            "stale_node_count": _safe_int(final_graph.get("stale_node_count"), 0),
            "blocked_edge_count": _safe_int(final_graph.get("blocked_edge_count"), 0),
        },
        "repair_plan": repair_plan,
        "phase_rollup": _phase_rollup(repair_plan),
        "architecture_expansion_summary": {
            "generation": "architecture_expansion_points_v1",
            "layer_count": expansion["layer_count"],
            "ready_layer_count": expansion["ready_layer_count"],
            "blocked_layer_count": expansion["blocked_layer_count"],
            "paper_governance_split_active": bool(expansion["paper_governance_split"].get("paper_governance_split_active", False)),
            "live_execution_firewall_status": str(expansion["live_execution_firewall"].get("status") or ""),
            "adaptive_memory_status": str(expansion["adaptive_memory_pressure"].get("status") or ""),
        },
        "architecture_benefit_summary": {
            "generation": str(benefit_backlog.get("generation") or ""),
            "candidate_count": _safe_int(benefit_backlog.get("candidate_count"), 0),
            "active_candidate_count": _safe_int(benefit_backlog.get("active_candidate_count"), 0),
            "recommended_candidate_count": _safe_int(benefit_backlog.get("recommended_candidate_count"), 0),
            "top_candidate_id": str(benefit_backlog.get("top_candidate_id") or ""),
            "top_candidate_score": _safe_int(benefit_backlog.get("top_candidate_score"), 0),
            "top_candidate_title": str(benefit_backlog.get("top_candidate_title") or ""),
        },
        "architecture_expansion_layers": expansion["layers"],
        "architecture_benefit_backlog": benefit_backlog,
        "dependency_map": expansion["dependency_map"],
        "repair_order_contract": expansion["repair_order_contract"],
        "live_execution_firewall": expansion["live_execution_firewall"],
        "paper_governance_split": expansion["paper_governance_split"],
        "degradation_explainability": expansion["degradation_explainability"],
        "adaptive_memory_pressure": expansion["adaptive_memory_pressure"],
        "operator_visibility": expansion["operator_visibility"],
        "attempts": attempts,
        "architecture_autopilot_contract": {
            "generation": "system_architecture_autopilot_v1",
            "plans_from_contract_graph": True,
            "dependency_ordered_phases": True,
            "does_not_enable_live_execution": True,
            "execute_repairs_requires_explicit_flag": True,
            "safe_repairs_are_bounded_refreshes": True,
        },
        "recommended_commands": recommended_commands,
        "recommended_actions": ordered_unique(
            [
                "run with --execute-safe-repairs only when you want the autopilot to refresh safe architecture surfaces"
                if repair_plan and not execute_safe_repairs
                else "",
                "fix lower-numbered architecture phases before widening later architecture layers"
                if repair_plan
                else "",
                "guarded paper is separated from strict governance, so paper can stay ready while live/governance gates remain blocked"
                if bool(expansion["paper_governance_split"].get("paper_governance_split_active", False))
                else "",
                "live execution firewall is active; safe architecture repairs do not clear halts or start live order paths",
                "inspect architecture_expansion_layers for the seven point-by-point system benefits",
                f"next beneficial architecture: {benefit_backlog.get('top_candidate_id')} ({benefit_backlog.get('top_candidate_title')})"
                if benefit_backlog.get("top_candidate_id")
                else "",
            ]
            + [f"phase {row['phase']}: {row['phase_name']} has {row['step_count']} step(s)" for row in _phase_rollup(repair_plan)]
        ),
    }

    if apply:
        plan_path = DEFAULT_PLAN_PATH if project_root == PROJECT_ROOT else project_root / "governance" / "architecture_contracts" / "system_architecture_autopilot_plan_latest.json"
        write_payload(plan_path, payload)
        benefit_path = DEFAULT_BENEFIT_PATH if project_root == PROJECT_ROOT else project_root / "governance" / "architecture_contracts" / "system_architecture_benefit_backlog_latest.json"
        write_payload(benefit_path, benefit_backlog)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan dependency-ordered architecture repairs from the system architecture contract graph.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--execute-safe-repairs", action="store_true")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-step-timeout-seconds", type=int, default=300)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).expanduser().resolve(),
        apply=bool(args.apply),
        execute_safe_repairs=bool(args.execute_safe_repairs),
        max_steps=int(args.max_steps),
        max_step_timeout_sec=int(args.max_step_timeout_seconds),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "system_architecture_autopilot "
            f"overall_status={payload.get('overall_status', '')} "
            f"repair_step_count={payload.get('repair_step_count', 0)} "
            f"attempt_count={payload.get('attempt_count', 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
