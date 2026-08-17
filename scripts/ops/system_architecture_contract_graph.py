#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_architecture_contract_graph_latest.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "system_architecture_contract_graph_v1.json"
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "governance" / "architecture_contracts" / "system_architecture_contract_graph_latest.json"

READY_STATUSES = {
    "ready",
    "ok",
    "stable",
    "watch",
    "advisory",
    "guarded_ready",
    "guarded_relief",
    "clear_ready",
    "armed",
    "schwab_indicator_intelligence_ready",
    "schwab_indicator_intelligence_ready_cached",
    "system_expansion_execution_ready",
    "system_expansion_execution_ready_guarded",
}
DEGRADED_STATUSES = {"degraded", "needs_attention", "needs_work", "thin", "warning", "warn", "stale"}
BLOCKED_STATUSES = {"blocked", "critical", "failed", "fatal", "missing"}
LIVE_ENABLE_FLAGS = {
    "ALLOW_ORDER_EXECUTION",
    "EXECUTION_LANE_LIVE_ENABLED",
    "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR",
    "TOP_BOT_ENABLE_LIVE_EXECUTION",
}
SELF_REFERENCE_DRIFT_SURFACES = {
    "adaptive_regression_guard",
    "system_architecture_contract_graph",
    "system_architecture_autopilot",
    "system_drift_guard",
    "master_infrastructure_supervisor",
    "infrastructure_autofix",
}

CONTRACT_NODES: tuple[dict[str, Any], ...] = (
    {
        "node_id": "health_fast",
        "title": "Fast Health",
        "class": "readiness",
        "artifact": "governance/health/health_fast_latest.json",
        "max_age_minutes": 10,
        "required": True,
        "authority": "read_only_readiness",
        "commands": [["./scripts/ops/opsctl.sh", "health-fast", "--json"]],
        "depends_on": [],
    },
    {
        "node_id": "runtime_throttle",
        "title": "Runtime Throttle",
        "class": "runtime",
        "artifact": "governance/health/runtime_throttle_control_latest.json",
        "max_age_minutes": 15,
        "required": True,
        "authority": "resource_control_plane",
        "commands": [["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"]],
        "depends_on": ["health_fast"],
    },
    {
        "node_id": "process_watchdog",
        "title": "Process Watchdog",
        "class": "infrastructure",
        "artifact": "governance/health/process_watchdog_latest.json",
        "max_age_minutes": 15,
        "required": False,
        "authority": "process_observation_and_repair",
        "commands": [["./scripts/ops/opsctl.sh", "process-watchdog", "--json"]],
        "depends_on": ["runtime_throttle"],
    },
    {
        "node_id": "storage_control",
        "title": "Storage Control",
        "class": "storage",
        "artifact": "governance/health/ingestion_storage_control_latest.json",
        "max_age_minutes": 20,
        "required": True,
        "authority": "single_writer_backpressure_control",
        "commands": [["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"]],
        "depends_on": ["runtime_throttle"],
    },
    {
        "node_id": "paper_ramp",
        "title": "Paper Ramp",
        "class": "paper_execution",
        "artifact": "governance/health/paper_400_ramp_latest.json",
        "max_age_minutes": 30,
        "required": True,
        "authority": "guarded_paper_capacity_only",
        "commands": [["./scripts/ops/opsctl.sh", "paper-400-ramp", "--json"]],
        "depends_on": ["health_fast", "runtime_throttle", "storage_control"],
    },
    {
        "node_id": "all_sleeves_launcher",
        "title": "All Sleeves Launcher",
        "class": "sleeve_runtime",
        "artifact": "governance/health/all_sleeves_launcher_latest.json",
        "max_age_minutes": 10,
        "required": True,
        "authority": "paper_collection_launcher",
        "commands": [["./scripts/ops/opsctl.sh", "health-fast", "--json"]],
        "depends_on": ["paper_ramp", "process_watchdog", "storage_control"],
    },
    {
        "node_id": "grade_regression_guard",
        "title": "Grade Regression Guard",
        "class": "governance",
        "artifact": "governance/health/grade_regression_guard_latest.json",
        "max_age_minutes": 45,
        "required": True,
        "authority": "grade_regression_observation",
        "commands": [["./scripts/ops/opsctl.sh", "grade-regression-guard", "--json"]],
        "depends_on": ["storage_control"],
    },
    {
        "node_id": "section_grade_guard",
        "title": "Section Grade Guard",
        "class": "governance",
        "artifact": "governance/health/section_grade_guard_latest.json",
        "max_age_minutes": 45,
        "required": True,
        "authority": "section_floor_observation",
        "commands": [["./scripts/ops/opsctl.sh", "section-grade-guard", "--json"]],
        "depends_on": ["grade_regression_guard"],
    },
    {
        "node_id": "runtime_paper_guard",
        "title": "Runtime Paper Guard",
        "class": "governance",
        "artifact": "governance/health/runtime_paper_regression_guard_latest.json",
        "max_age_minutes": 45,
        "required": True,
        "authority": "paper_runtime_regression_observation",
        "commands": [["./scripts/ops/opsctl.sh", "runtime-paper-regression-guard", "--json"]],
        "depends_on": ["paper_ramp", "runtime_throttle"],
    },
    {
        "node_id": "adaptive_regression_guard",
        "title": "Adaptive Regression Guard",
        "class": "adaptive_governance",
        "artifact": "governance/health/adaptive_regression_guard_latest.json",
        "max_age_minutes": 10,
        "required": True,
        "authority": "adaptive_memory_observation_only",
        "commands": [["./scripts/ops/opsctl.sh", "adaptive-regression-guard", "--apply", "--json"]],
        "depends_on": ["grade_regression_guard", "section_grade_guard", "runtime_paper_guard", "system_drift_registry"],
    },
    {
        "node_id": "system_drift_registry",
        "title": "System Drift Registry",
        "class": "governance_registry",
        "artifact": "governance/health/system_drift_registry_latest.json",
        "max_age_minutes": 240,
        "required": False,
        "authority": "surface_registry",
        "commands": [["./scripts/ops/opsctl.sh", "system-drift-registry", "--json"]],
        "depends_on": [],
    },
    {
        "node_id": "system_drift_guard",
        "title": "System Drift Guard",
        "class": "governance",
        "artifact": "governance/health/system_drift_guard_latest.json",
        "max_age_minutes": 45,
        "required": False,
        "authority": "artifact_drift_observation",
        "commands": [["./scripts/ops/opsctl.sh", "system-drift-guard", "--json"]],
        "depends_on": ["system_drift_registry", "adaptive_regression_guard"],
    },
    {
        "node_id": "schwab_indicator_intelligence",
        "title": "Schwab Indicator Intelligence",
        "class": "intelligence_layer",
        "artifact": "governance/health/schwab_indicator_intelligence_latest.json",
        "max_age_minutes": 720,
        "required": False,
        "authority": "advisory_study_strategy_catalog_no_execution",
        "commands": [["./scripts/ops/opsctl.sh", "schwab-indicator-intelligence", "--json"]],
        "depends_on": ["health_fast", "runtime_throttle", "system_drift_registry"],
    },
    {
        "node_id": "system_expansion_execution",
        "title": "System Expansion Execution",
        "class": "intelligence_layer",
        "artifact": "governance/health/system_expansion_execution_layer_latest.json",
        "max_age_minutes": 720,
        "required": False,
        "authority": "advisory_12_lane_expansion_control_no_execution",
        "commands": [["./scripts/ops/opsctl.sh", "system-expansion-execution", "--json"]],
        "depends_on": ["schwab_indicator_intelligence", "runtime_throttle", "system_drift_registry"],
    },
    {
        "node_id": "distributed_cell_architecture",
        "title": "Distributed Cell Architecture",
        "class": "architecture",
        "artifact": "governance/health/distributed_cell_architecture_latest.json",
        "max_age_minutes": 360,
        "required": False,
        "authority": "cell_federation_contract",
        "commands": [["./scripts/ops/opsctl.sh", "distributed-cell-architecture", "--apply", "--json"]],
        "depends_on": ["storage_control", "runtime_throttle", "adaptive_regression_guard"],
    },
    {
        "node_id": "architecture_hardening",
        "title": "Architecture Hardening",
        "class": "architecture",
        "artifact": "governance/health/system_architecture_hardening_latest.json",
        "max_age_minutes": 360,
        "required": False,
        "authority": "cross_layer_hardening_contract",
        "commands": [["./scripts/ops/opsctl.sh", "system-architecture-hardening", "--apply", "--json"]],
        "depends_on": ["distributed_cell_architecture", "adaptive_regression_guard", "health_fast"],
    },
    {
        "node_id": "system_self_model",
        "title": "System Self Model",
        "class": "self_model",
        "artifact": "governance/health/system_self_model_latest.json",
        "max_age_minutes": 360,
        "required": False,
        "authority": "operator_self_model_report",
        "commands": [["./scripts/ops/opsctl.sh", "big-platform-brain", "--json"]],
        "depends_on": ["architecture_hardening", "system_drift_guard", "adaptive_regression_guard"],
    },
)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _normalize_status(payload: dict[str, Any]) -> str:
    if not payload:
        return "missing"
    raw = str(payload.get("overall_status") or payload.get("status") or payload.get("state") or "").strip().lower()
    if not raw and "ok" in payload:
        raw = "ready" if bool(payload.get("ok")) else "blocked"
    if raw in READY_STATUSES:
        return "ready"
    if raw in DEGRADED_STATUSES:
        return "degraded"
    if raw in BLOCKED_STATUSES:
        return "blocked"
    return raw or "missing"


def _scan_truthy_flags(value: Any, *, source: str, path: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, raw in value.items():
            key_text = str(key)
            next_path = (*path, key_text)
            if key_text in LIVE_ENABLE_FLAGS and str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}:
                rows.append({"node_id": source, "path": ".".join(next_path), "value": raw})
            rows.extend(_scan_truthy_flags(raw, source=source, path=next_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_scan_truthy_flags(item, source=source, path=(*path, str(index))))
    return rows


def _all_sleeves_health_fast_reconciliation(project_root: Path) -> dict[str, Any]:
    health_fast = load_json(project_root / "governance" / "health" / "health_fast_latest.json")
    operational = health_fast.get("operational_readiness") if isinstance(health_fast.get("operational_readiness"), dict) else {}
    guarded_paper = operational.get("guarded_paper") if isinstance(operational.get("guarded_paper"), dict) else {}
    process_watchdog = health_fast.get("process_watchdog") if isinstance(health_fast.get("process_watchdog"), dict) else {}
    effective_runtime = (
        process_watchdog.get("all_sleeves_effective_runtime")
        if isinstance(process_watchdog.get("all_sleeves_effective_runtime"), dict)
        else {}
    )
    active = bool(
        bool(guarded_paper.get("ok", False))
        and str(guarded_paper.get("status") or "").strip().lower() in {"ready", "armed", "guarded_ready"}
        and bool(effective_runtime.get("ok", False))
        and str(effective_runtime.get("status") or "").strip().lower() in {"ready", "guarded_ready"}
        and bool(effective_runtime.get("launcher_live", False))
        and bool(effective_runtime.get("child_process_live", False))
        and bool(effective_runtime.get("child_fanout_ok", False))
        and bool(effective_runtime.get("heartbeat_ok", False))
        and _safe_float(effective_runtime.get("child_process_count"), 0.0) >= 4.0
    )
    return {
        "active": active,
        "source": "health_fast.process_watchdog.all_sleeves_effective_runtime",
        "guarded_paper_status": str(guarded_paper.get("status") or ""),
        "launcher_artifact_reason": str(effective_runtime.get("launcher_artifact_reason") or ""),
        "child_process_count": int(_safe_float(effective_runtime.get("child_process_count"), 0.0)),
        "policy": "fresh health-fast process fanout can certify all-sleeves runtime when the raw launcher artifact is stale or self-blocked",
    }


def _guarded_paper_strict_clear(project_root: Path) -> bool:
    health_fast = load_json(project_root / "governance" / "health" / "health_fast_latest.json")
    operational = _as_dict(health_fast.get("operational_readiness"))
    guarded_paper = _as_dict(operational.get("guarded_paper"))
    live_execution = _as_dict(operational.get("live_execution"))
    guarded_ready = bool(guarded_paper.get("ok", False)) and str(guarded_paper.get("status") or "").strip().lower() in {
        "ready",
        "armed",
        "guarded_ready",
    }
    live_locked = str(live_execution.get("status") or "").strip().lower() in {
        "blocked_read_only",
        "locked",
        "read_only",
        "disabled",
    }
    operational_health_ready = bool(
        health_fast.get("strict_all_clear", False)
        or (
            bool(health_fast.get("ok", False))
            and str(health_fast.get("overall_status") or "").strip().lower() in {"ready", "guarded_ready"}
        )
    )
    return bool(operational_health_ready and guarded_ready and live_locked)


def _drift_guard_self_reference_reconciliation(project_root: Path) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "health" / "system_drift_guard_latest.json")
    surfaces = _as_list(payload.get("surfaces"))
    metrics = _as_dict(payload.get("metrics"))
    non_ready_surfaces: set[str] = set()
    blocked_surfaces: set[str] = set()
    for row in surfaces:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        status = _normalize_status({"overall_status": row.get("status"), "ok": row.get("ok")})
        if status == "blocked":
            blocked_surfaces.add(name)
            non_ready_surfaces.add(name)
        elif status == "degraded":
            non_ready_surfaces.add(name)
    active = bool(
        _guarded_paper_strict_clear(project_root)
        and _normalize_status(payload) == "degraded"
        and _safe_float(metrics.get("blocked_surface_count"), 0.0) == 0.0
        and not blocked_surfaces
        and non_ready_surfaces
        and non_ready_surfaces <= SELF_REFERENCE_DRIFT_SURFACES
    )
    return {
        "active": active,
        "source": "system_drift_guard.surfaces",
        "managed_surfaces": sorted(non_ready_surfaces),
        "blocked_surfaces": sorted(blocked_surfaces),
        "guarded_paper_strict_clear": _guarded_paper_strict_clear(project_root),
        "reason": "guarded_paper_architecture_self_reference_debt",
        "policy": (
            "fresh guarded-paper strict-clear evidence can break the architecture/drift/supervisor loop "
            "when all remaining drift surfaces are self-reference governance debt"
        ),
    }


def _node_from_contract(project_root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    rel_artifact = Path(str(contract["artifact"]))
    artifact_path = project_root / rel_artifact
    payload = load_json(artifact_path)
    status = _normalize_status(payload)
    age_minutes = payload_age_minutes(payload, artifact_path) if payload else None
    max_age = _safe_float(contract.get("max_age_minutes"), 0.0)
    stale = bool(age_minutes is not None and max_age > 0 and age_minutes > max_age)
    required = bool(contract.get("required", False))
    normalized_status = status
    if stale and normalized_status == "ready":
        normalized_status = "degraded"
    if not payload and not required:
        normalized_status = "degraded"
    reconciliations: list[dict[str, Any]] = []
    if str(contract["node_id"]) == "all_sleeves_launcher" and normalized_status == "blocked":
        reconciliation = _all_sleeves_health_fast_reconciliation(project_root)
        if bool(reconciliation.get("active", False)):
            normalized_status = "ready"
            reconciliations.append(reconciliation)
    if str(contract["node_id"]) == "system_drift_guard" and normalized_status == "degraded":
        reconciliation = _drift_guard_self_reference_reconciliation(project_root)
        if bool(reconciliation.get("active", False)):
            normalized_status = "ready"
            reconciliations.append(reconciliation)
    authority_violations = _scan_truthy_flags(payload, source=str(contract["node_id"])) if payload else []
    if authority_violations:
        normalized_status = "blocked"
    return {
        "node_id": str(contract["node_id"]),
        "title": str(contract["title"]),
        "class": str(contract["class"]),
        "artifact": str(rel_artifact),
        "artifact_present": bool(payload),
        "artifact_age_minutes": None if age_minutes is None else round(float(age_minutes), 2),
        "artifact_max_age_minutes": max_age,
        "artifact_stale": stale,
        "status": normalized_status,
        "raw_status": status,
        "ok": normalized_status == "ready",
        "required": required,
        "authority": str(contract.get("authority") or ""),
        "authority_violations": authority_violations,
        "reconciliations": reconciliations,
        "commands": [list(cmd) for cmd in _as_list(contract.get("commands")) if isinstance(cmd, list)],
        "depends_on": [str(item) for item in _as_list(contract.get("depends_on"))],
        "summary": str(payload.get("summary") or payload.get("reason") or "") if payload else "artifact_missing",
    }


def _dependency_edges(nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for node_id, node in nodes.items():
        for dep_id in _as_list(node.get("depends_on")):
            dep = nodes.get(str(dep_id))
            if dep is None:
                status = "blocked"
                reason = "dependency_not_declared"
                dep_required = True
            else:
                dep_required = bool(dep.get("required", False))
                dep_status = str(dep.get("status") or "")
                if dep_status == "ready":
                    status = "ready"
                    reason = "dependency_ready"
                elif dep_status == "blocked" and dep_required:
                    status = "blocked"
                    reason = "required_dependency_blocked"
                elif dep_status == "missing" and dep_required:
                    status = "blocked"
                    reason = "required_dependency_missing"
                else:
                    status = "degraded"
                    reason = f"dependency_{dep_status or 'unknown'}"
            edges.append(
                {
                    "from": str(dep_id),
                    "to": node_id,
                    "status": status,
                    "reason": reason,
                    "required_dependency": dep_required,
                }
            )
    return edges


def _layer_rollup(nodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rollup: dict[str, dict[str, Any]] = {}
    for node in nodes.values():
        layer = str(node.get("class") or "unknown")
        row = rollup.setdefault(layer, {"node_count": 0, "ready_count": 0, "degraded_count": 0, "blocked_count": 0, "stale_count": 0})
        row["node_count"] += 1
        status = str(node.get("status") or "")
        if status == "ready":
            row["ready_count"] += 1
        elif status == "blocked":
            row["blocked_count"] += 1
        else:
            row["degraded_count"] += 1
        if bool(node.get("artifact_stale", False)):
            row["stale_count"] += 1
    return rollup


def build_payload(project_root: Path = PROJECT_ROOT, *, apply: bool = False) -> dict[str, Any]:
    project_root = project_root.resolve()
    node_rows = [_node_from_contract(project_root, contract) for contract in CONTRACT_NODES]
    nodes = {row["node_id"]: row for row in node_rows}
    edges = _dependency_edges(nodes)
    blocked_nodes = [row for row in node_rows if row["status"] == "blocked"]
    degraded_nodes = [row for row in node_rows if row["status"] not in {"ready", "blocked"}]
    stale_nodes = [row for row in node_rows if bool(row.get("artifact_stale", False))]
    blocked_edges = [row for row in edges if row["status"] == "blocked"]
    authority_violations = [violation for row in node_rows for violation in _as_list(row.get("authority_violations"))]

    overall_status = "ready"
    if blocked_nodes or blocked_edges or authority_violations:
        overall_status = "blocked"
    elif degraded_nodes or stale_nodes:
        overall_status = "degraded"

    recommended_commands: list[list[str]] = []
    for row in blocked_nodes + degraded_nodes + stale_nodes:
        for cmd in _as_list(row.get("commands")):
            if isinstance(cmd, list) and cmd:
                recommended_commands.append([str(part) for part in cmd])
    deduped_commands: list[list[str]] = []
    seen: set[str] = set()
    for cmd in recommended_commands:
        key = " ".join(cmd)
        if key in seen:
            continue
        seen.add(key)
        deduped_commands.append(cmd)

    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply": bool(apply),
        "node_count": len(node_rows),
        "edge_count": len(edges),
        "blocked_node_count": len(blocked_nodes),
        "degraded_node_count": len(degraded_nodes),
        "stale_node_count": len(stale_nodes),
        "blocked_edge_count": len(blocked_edges),
        "authority_violation_count": len(authority_violations),
        "layers": _layer_rollup(nodes),
        "nodes": node_rows,
        "edges": edges,
        "blocked_nodes": [row["node_id"] for row in blocked_nodes],
        "degraded_nodes": [row["node_id"] for row in degraded_nodes],
        "stale_nodes": [row["node_id"] for row in stale_nodes],
        "blocked_edges": blocked_edges,
        "authority_violations": authority_violations,
        "architecture_contract_graph": {
            "generation": "system_architecture_contract_graph_v1",
            "single_writer_authority": "storage_control",
            "live_execution_authority": False,
            "paper_execution_authority": "paper_ramp",
            "adaptive_governance_memory": "adaptive_regression_guard",
            "drift_registry_source": "system_drift_registry",
            "indicator_intelligence_source": "schwab_indicator_intelligence",
            "expansion_execution_source": "system_expansion_execution",
            "state_mutation_requires_apply": True,
        },
        "recommended_commands": deduped_commands,
        "recommended_actions": ordered_unique(
            [
                "refresh blocked contract graph nodes before widening architecture" if blocked_nodes else "",
                "refresh stale architecture artifacts so dependency edges stop inheriting old evidence" if stale_nodes else "",
                "investigate live-execution authority violations before any promotion" if authority_violations else "",
            ]
            + [f"{edge['to']} depends on {edge['from']}: {edge['reason']}" for edge in blocked_edges[:8]]
        ),
    }

    if apply:
        write_payload(DEFAULT_CONFIG_PATH if project_root == PROJECT_ROOT else project_root / "config" / "system_architecture_contract_graph_v1.json", {"nodes": list(CONTRACT_NODES)})
        graph_path = DEFAULT_GRAPH_PATH if project_root == PROJECT_ROOT else project_root / "governance" / "architecture_contracts" / "system_architecture_contract_graph_latest.json"
        write_payload(graph_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the architecture contract graph linking cells, guards, artifacts, freshness, and authority boundaries.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root, apply=bool(args.apply))
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "system_architecture_contract_graph "
            f"overall_status={payload.get('overall_status', '')} "
            f"nodes={payload.get('node_count', 0)} "
            f"blocked={payload.get('blocked_node_count', 0)} "
            f"stale={payload.get('stale_node_count', 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
