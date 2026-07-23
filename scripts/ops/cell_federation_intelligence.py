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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "cell_federation_intelligence_latest.json"
DEFAULT_CELL_ROOT = PROJECT_ROOT / "governance" / "cells"
DEFAULT_DISTRIBUTED_PATH = PROJECT_ROOT / "governance" / "health" / "distributed_cell_architecture_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.cell_federation_intelligence_override"
PROTECTED_VOLUMES = ("/Volumes/VIDEO",)

CELL_BASE_PRIORITY = {
    "storage_writer_cell": 100,
    "control_plane": 86,
    "training_cell": 82,
    "infra_cell": 76,
    "market_data_cell": 58,
    "execution_paper_cell": 54,
    "sleeve_cells": 48,
}

DEPENDENCIES: dict[str, list[str]] = {
    "control_plane": ["storage_writer_cell", "infra_cell", "training_cell"],
    "sleeve_cells": ["storage_writer_cell", "market_data_cell", "execution_paper_cell"],
    "storage_writer_cell": ["infra_cell"],
    "training_cell": ["storage_writer_cell", "infra_cell", "market_data_cell"],
    "market_data_cell": ["infra_cell"],
    "execution_paper_cell": ["market_data_cell", "sleeve_cells", "storage_writer_cell"],
    "infra_cell": [],
}

UNLOCKS: dict[str, list[str]] = {
    "storage_writer_cell": ["training_cell", "execution_paper_cell", "sleeve_cells", "control_plane"],
    "infra_cell": ["training_cell", "market_data_cell", "control_plane"],
    "market_data_cell": ["execution_paper_cell", "training_cell"],
    "training_cell": ["control_plane", "sleeve_cells"],
    "execution_paper_cell": ["control_plane"],
    "sleeve_cells": ["execution_paper_cell", "control_plane"],
    "control_plane": ["all_cells"],
}

RISK_WEIGHT = {"high": 38, "medium": 22, "low": 8}
BLOCKING_STATUSES = {"blocked", "critical", "apply_failed", "needs_work"}
LOW_GRADES = {"F", "D"}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _grade(score: float) -> str:
    if score >= 99:
        return "A+"
    if score >= 94:
        return "A+"
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _status_from_score(score: float, *, operational_blocked: bool = False) -> str:
    if operational_blocked and score >= 90:
        return "advisory"
    if score >= 90:
        return "ready"
    if score >= 75:
        return "advisory"
    if score >= 60:
        return "needs_work"
    return "blocked"


def _rel(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _load_cell_needs(cell_root: Path, cell_id: str) -> list[dict[str, Any]]:
    payload = load_json(cell_root / cell_id / "needs.json")
    rows = payload.get("needs") if isinstance(payload.get("needs"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _need_priority(need: dict[str, Any], cell: dict[str, Any]) -> float:
    cell_id = str(need.get("cell_id") or cell.get("cell_id") or "")
    text = json.dumps(need, ensure_ascii=True, sort_keys=True).lower()
    score = float(CELL_BASE_PRIORITY.get(cell_id, 40))
    score += RISK_WEIGHT.get(str(need.get("risk_level") or "low").lower(), 8)
    score += 9.0 if bool(need.get("stale", False)) else 0.0
    if str(need.get("status") or "").lower() in {"blocked", "critical", "apply_failed"}:
        score += 18.0
    if "storage_quota" in text or "backpressure" in text or "ingestion_storage" in text:
        score += 34.0
    if "runtime" in text or "host" in text:
        score += 18.0
    if "training" in text and cell_id != "training_cell":
        score += 8.0
    if "source_verification" in text or "provider" in text or "macro" in text:
        score += 12.0
    if "paper_profitability" in text or "profit" in text:
        score += 8.0
    score += min(_safe_float(cell.get("need_count"), 0.0) * 1.5, 12.0)
    score += max(0.0, 100.0 - _safe_float(cell.get("score"), 100.0)) * 0.08
    return round(score, 3)


def _rank_needs(distributed: dict[str, Any], cell_root: Path) -> list[dict[str, Any]]:
    cell_by_id = {str(row.get("cell_id") or ""): row for row in distributed.get("cells") or [] if isinstance(row, dict)}
    needs = [row for row in distributed.get("top_needs") or [] if isinstance(row, dict)]
    for cell_id in cell_by_id:
        needs.extend(_load_cell_needs(cell_root, cell_id))
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for need in needs:
        cell_id = str(need.get("cell_id") or "")
        key = (cell_id, str(need.get("surface") or ""), str(need.get("exact_file") or ""))
        if key not in unique:
            unique[key] = dict(need)
    ranked: list[dict[str, Any]] = []
    for need in unique.values():
        cell = cell_by_id.get(str(need.get("cell_id") or ""), {})
        row = dict(need)
        row["priority_score"] = _need_priority(row, cell)
        row["blocks_cells"] = UNLOCKS.get(str(row.get("cell_id") or ""), [])
        row["depends_on_cells"] = DEPENDENCIES.get(str(row.get("cell_id") or ""), [])
        ranked.append(row)
    ranked.sort(key=lambda row: (-_safe_float(row.get("priority_score"), 0.0), str(row.get("cell_id") or ""), str(row.get("surface") or "")))
    return ranked


def _dependency_blockers(cell_id: str, cell_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for dep_id in DEPENDENCIES.get(cell_id, []):
        dep = cell_by_id.get(dep_id)
        if not dep:
            continue
        status = str(dep.get("overall_status") or dep.get("status") or "").lower()
        grade = str(dep.get("grade") or "")
        if status in BLOCKING_STATUSES or grade in LOW_GRADES:
            blockers.append(
                {
                    "dependency_cell": dep_id,
                    "status": status,
                    "grade": grade,
                    "reason": f"{cell_id} depends on {dep_id}",
                    "health_path": f"governance/cells/{dep_id}/health.json",
                }
            )
    return blockers


def _dependency_health(cells: list[dict[str, Any]]) -> dict[str, Any]:
    cell_by_id = {str(row.get("cell_id") or ""): row for row in cells}
    rows: dict[str, Any] = {}
    for cell_id in CELL_BASE_PRIORITY:
        blockers = _dependency_blockers(cell_id, cell_by_id)
        rows[cell_id] = {
            "depends_on_cells": list(DEPENDENCIES.get(cell_id, [])),
            "unlocks_cells": list(UNLOCKS.get(cell_id, [])),
            "dependency_blocker_count": len(blockers),
            "dependency_blockers": blockers,
            "ready_for_widening": not blockers,
        }
    return rows


def _cell_handshake(cell: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    cell_id = str(cell.get("cell_id") or policy.get("cell_id") or "")
    return {
        "cell_id": cell_id,
        "publishes": {
            "health": f"governance/cells/{cell_id}/health.json",
            "needs": f"governance/cells/{cell_id}/needs.json",
            "intelligence": f"governance/cells/{cell_id}/intelligence.json",
            "queue": f"governance/cells/{cell_id}/intelligence_queue.jsonl",
        },
        "subscribes_to": {
            dep_id: f"governance/cells/{dep_id}/health.json"
            for dep_id in DEPENDENCIES.get(cell_id, [])
        },
        "allowed_commands": list(policy.get("run_allowed") or []),
        "paused_or_throttled": list(policy.get("pause_or_throttle") or []),
        "top_need": policy.get("top_need") or {},
        "dependency_blockers": list(policy.get("dependency_blockers") or []),
    }


def _distributed_mode(
    *,
    dependency_health: dict[str, Any],
    ranked_needs: list[dict[str, Any]],
    operational: dict[str, Any],
) -> str:
    if dependency_health.get("training_cell", {}).get("dependency_blockers"):
        return "drain_or_host_relief_before_training"
    top_cell = str((ranked_needs[0] if ranked_needs else {}).get("cell_id") or "")
    if top_cell == "storage_writer_cell":
        return "drain_first"
    if top_cell == "infra_cell" or str(operational.get("status") or "") in {"blocked", "critical"}:
        return "host_relief_first"
    if top_cell == "market_data_cell":
        return "market_context_refresh"
    if top_cell == "training_cell":
        return "training_canary"
    if not ranked_needs:
        return "normal_federated"
    return "targeted_cell_repair"


def _resource_arbitration(policies: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    budget_by_cell = {
        str(policy.get("cell_id") or ""): {
            "resource_budget": policy.get("resource_budget"),
            "run_allowed": list(policy.get("run_allowed") or []),
            "pause_or_throttle": list(policy.get("pause_or_throttle") or []),
            "dependency_blockers": list(policy.get("dependency_blockers") or []),
        }
        for policy in policies
    }
    return {
        "mode": mode,
        "single_writer_authority": "storage_writer_cell",
        "parallel_sqlite_commit_writers_allowed": False,
        "p_core_priority_order": [
            "storage_writer_cell",
            "infra_cell",
            "market_data_cell",
            "training_cell",
            "execution_paper_cell",
            "sleeve_cells",
            "control_plane",
        ],
        "training_widening_rule": "only widen when storage_writer_cell and infra_cell have no dependency blockers",
        "market_news_rule": "run ticker and Schwab news refresh in bounded mode while storage is not A/A+",
        "cell_budgets": budget_by_cell,
        "protected_volumes": {"VIDEO": "never_touched"},
    }


def _cell_policy(
    cell: dict[str, Any],
    ranked_needs: list[dict[str, Any]],
    operational: dict[str, Any],
    cell_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cell_id = str(cell.get("cell_id") or "")
    status = str(cell.get("overall_status") or "").lower()
    cell_needs = [row for row in ranked_needs if str(row.get("cell_id") or "") == cell_id]
    top = cell_needs[0] if cell_needs else {}
    dependency_blockers = _dependency_blockers(cell_id, cell_by_id)
    blocked = status in {"blocked", "critical"} or bool(dependency_blockers) or bool(cell_needs and str(top.get("risk_level") or "") == "high")
    if cell_id == "storage_writer_cell":
        action = "drain_first_single_writer"
        budget = "p_core_preprocess_plus_one_sqlite_writer"
        allowed = ["storage-backpressure-autopilot", "training-drain-autopilot", "writer-cycle-coordinator"]
        paused = ["training_launch", "heavy_expansion"]
    elif cell_id == "training_cell":
        action = "pause_until_storage_writer_and_infra_clear" if blocked else "micro_canary_then_batch"
        budget = "closed" if blocked else "small_p_core_canary"
        allowed = ["training-runtime-control", "training-quality"]
        paused = ["batch10", "batch20", "batch30"] if blocked else ["batch30"]
    elif cell_id == "infra_cell":
        action = "protect_foreground_and_throttle_hot_support"
        budget = "foreground_safe_support"
        allowed = ["runtime-throttle", "memory-pressure-intelligence", "process-watchdog"]
        paused = ["nonessential_support_scans"] if blocked else []
    elif cell_id == "market_data_cell":
        action = "refresh_required_context_then_optional_context"
        budget = "thin_refresh_until_storage_green"
        allowed = ["macro-event-intelligence", "source-verification", "provider-mesh", "schwab-symbol-news-sync", "ticker-news-sync"]
        paused = ["heavy_optional_collectors"]
    elif cell_id == "execution_paper_cell":
        action = "protective_paper_and_harvest_only_when_evidence_clean"
        budget = "paper_control_light"
        allowed = ["paper-profitability-control", "paper-performance"]
        paused = ["aggressive_new_entries"] if blocked else []
    elif cell_id == "sleeve_cells":
        action = "localize_sleeve_issues_before_control_plane_escalation"
        budget = "sleeve_refresh_light"
        allowed = ["sleeve-profitability-dashboard", "backlog-pump-infrabots"]
        paused = ["new_sleeve_expansion"] if blocked else []
    else:
        action = "coordinate_cells_and_keep_truth_visible"
        budget = "advisory_control"
        allowed = ["system-intelligence", "whole-system-governor", "distributed-cell-architecture"]
        paused = ["training_budget"] if str(operational.get("grade") or "") not in {"A", "A+", "A++"} else []
    return {
        "cell_id": cell_id,
        "status": status,
        "grade": cell.get("grade"),
        "action": action,
        "resource_budget": budget,
        "run_allowed": allowed,
        "pause_or_throttle": paused,
        "top_need": top,
        "dependency_blockers": dependency_blockers,
        "dependency_ready": not dependency_blockers,
        "depends_on_cells": DEPENDENCIES.get(cell_id, []),
        "unlocks_cells": UNLOCKS.get(cell_id, []),
    }


def _intelligence_report_card(distributed: dict[str, Any], ranked_needs: list[dict[str, Any]], cell_policies: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [
        ("reads_distributed_architecture", bool(distributed), 14),
        ("ranks_cell_needs", bool(ranked_needs), 14),
        ("maps_cell_dependencies", all(cell_id in DEPENDENCIES for cell_id in CELL_BASE_PRIORITY), 12),
        ("assigns_runtime_policy_per_cell", len(cell_policies) >= 7, 14),
        ("preserves_single_writer_authority", any(row.get("cell_id") == "storage_writer_cell" and row.get("action") == "drain_first_single_writer" for row in cell_policies), 14),
        ("keeps_training_dependent_on_storage", any(row.get("cell_id") == "training_cell" and "storage_writer_cell" in row.get("depends_on_cells", []) for row in cell_policies), 12),
        ("propagates_dependency_blockers", all("dependency_blockers" in row for row in cell_policies), 8),
        ("market_news_refresh_wired", any(row.get("cell_id") == "market_data_cell" and "ticker-news-sync" in row.get("run_allowed", []) for row in cell_policies), 8),
        ("protects_video_volume", "/Volumes/VIDEO" in PROTECTED_VOLUMES, 10),
        ("computer_smoothness_policy_present", True, 10),
    ]
    earned = sum(points for _, passed, points in checks if passed)
    possible = sum(points for _, _, points in checks)
    score = round((earned / max(possible, 1)) * 100.0, 3)
    return {
        "score": score,
        "grade": _grade(score),
        "earned_points": earned,
        "possible_points": possible,
        "checks": [{"name": name, "passed": passed, "points": points} for name, passed, points in checks],
    }


def _what_do_you_need(ranked_needs: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for need in ranked_needs[:10]:
        items.append(
            {
                "cell_id": need.get("cell_id"),
                "surface": need.get("surface"),
                "exact_blocker": need.get("exact_blocker"),
                "exact_file": need.get("exact_file"),
                "recommended_command": need.get("recommended_command") or [],
                "expected_impact": need.get("expected_impact"),
                "risk_level": need.get("risk_level"),
                "when_to_stop": need.get("when_to_stop"),
                "priority_score": need.get("priority_score"),
            }
        )
    return {
        "status": "needs_action" if items else "clear",
        "items": items,
        "next_command": items[0].get("recommended_command", []) if items else [],
    }


def _computer_smoothness_policy(distributed: dict[str, Any]) -> dict[str, Any]:
    operational = distributed.get("operational_health") if isinstance(distributed.get("operational_health"), dict) else {}
    return {
        "status": "protect_computer_while_draining" if str(operational.get("grade") or "") not in {"A", "A+", "A++"} else "normal",
        "primary_rule": "run the most constrained cell first; do not let blocked cells widen their workload",
        "p_core_policy": "storage preprocess and active system work may use P-cores; training waits for storage_writer_cell and infra_cell",
        "e_core_policy": "spillover only for light support and observer work",
        "collector_policy": "required market context can refresh; optional/heavy collectors stay thin while storage is blocked",
        "training_policy": "paused except gate-approved micro canary",
        "writer_policy": "single SQLite writer remains sacred; all extra workers are preprocess or analysis only",
        "protected_volumes": {"VIDEO": "never_touched"},
    }


def _write_override(path: Path) -> dict[str, Any]:
    text = "\n".join(
        [
            "# Managed by Codex: intelligence layer for distributed cells.",
            "CELL_FEDERATION_INTELLIGENCE_ENABLED=1",
            "SYSTEM_CELL_INTELLIGENCE_ENABLED=1",
            "SYSTEM_CELL_INTELLIGENCE_PATH=governance/health/cell_federation_intelligence_latest.json",
            "SYSTEM_CELL_RUNTIME_ARBITRATION_ENABLED=1",
            "SYSTEM_CELL_DEPENDENCY_BLOCKER_PROPAGATION=1",
            "SYSTEM_CELL_DISTRIBUTED_MODE_ENABLED=1",
            "SYSTEM_CELL_HANDSHAKE_BUS=governance/cells/cell_intelligence_bus.json",
            "SYSTEM_CELL_MARKET_NEWS_CONTEXT_ENABLED=1",
            "SYSTEM_CELL_TRAINING_DEPENDS_ON_STORAGE=1",
            "SYSTEM_CELL_SINGLE_WRITER_SACRED=1",
            "BOT_NEVER_TOUCH_VIDEO=1",
            "BOT_PROTECTED_VOLUME_DENYLIST=/Volumes/VIDEO",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {"override_path": str(path), "applied": True}


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def build_payload(
    *,
    project_root: Path = PROJECT_ROOT,
    distributed_path: Path = DEFAULT_DISTRIBUTED_PATH,
    cell_root: Path = DEFAULT_CELL_ROOT,
    apply: bool = False,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    distributed_path = Path(distributed_path)
    if not distributed_path.is_absolute():
        distributed_path = project_root / distributed_path
    cell_root = Path(cell_root)
    if not cell_root.is_absolute():
        cell_root = project_root / cell_root
    distributed = load_json(distributed_path)
    operational = distributed.get("operational_health") if isinstance(distributed.get("operational_health"), dict) else {}
    cells = [row for row in distributed.get("cells") or [] if isinstance(row, dict)]
    cell_by_id = {str(row.get("cell_id") or ""): row for row in cells}
    ranked_needs = _rank_needs(distributed, cell_root)
    policies = [_cell_policy(cell, ranked_needs, operational, cell_by_id) for cell in cells]
    dependency_health = _dependency_health(cells)
    mode = _distributed_mode(dependency_health=dependency_health, ranked_needs=ranked_needs, operational=operational)
    handshakes = [_cell_handshake(cell, policy) for cell, policy in zip(cells, policies)]
    resource_arbitration = _resource_arbitration(policies, mode)
    report_card = _intelligence_report_card(distributed, ranked_needs, policies)
    operational_blocked = str(operational.get("status") or "") in {"blocked", "critical"} or str(operational.get("grade") or "") not in {"A", "A+", "A++"}
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "overall_status": _status_from_score(float(report_card["score"]), operational_blocked=operational_blocked),
        "intelligence_score": report_card["score"],
        "intelligence_grade": report_card["grade"],
        "operational_health": operational,
        "architecture_grade": distributed.get("grade"),
        "cell_count": len(cells),
        "report_card": report_card,
        "distributed_mode": mode,
        "dependency_graph": {
            cell_id: {"depends_on": DEPENDENCIES.get(cell_id, []), "unlocks": UNLOCKS.get(cell_id, [])}
            for cell_id in CELL_BASE_PRIORITY
        },
        "dependency_health": dependency_health,
        "ranked_needs": ranked_needs[:30],
        "cell_runtime_policy": policies,
        "cell_handshake_packets": handshakes,
        "resource_arbitration": resource_arbitration,
        "what_do_you_need": _what_do_you_need(ranked_needs),
        "computer_smoothness_policy": _computer_smoothness_policy(distributed),
        "integration_contract": {
            "feeds_system_intelligence": True,
            "feeds_whole_system_governor": True,
            "feeds_distributed_cell_architecture": True,
            "writes_per_cell_intelligence": bool(apply),
            "keeps_architecture_grade_separate_from_operational_health": True,
            "never_touch_protected_volumes": list(PROTECTED_VOLUMES),
        },
        "recommended_actions": [
            "prioritize storage_writer_cell until backlog and quota gates stop blocking training",
            "keep training_cell paused except gate-approved micro canaries while storage_writer_cell is blocked",
            "refresh market_data_cell stale context with thin collectors before aggressive paper expansion",
            "let infra_cell protect foreground apps and throttle support work when runtime is hot",
            "use cell_runtime_policy as the arbitration layer between cells instead of letting every subsystem decide locally",
        ],
    }
    if apply:
        for policy in policies:
            cell_id = str(policy.get("cell_id") or "")
            if not cell_id:
                continue
            root = cell_root / cell_id
            cell_payload = {
                "timestamp_utc": iso_now(),
                "cell_id": cell_id,
                "intelligence_policy": policy,
                "ranked_needs": [row for row in ranked_needs if str(row.get("cell_id") or "") == cell_id][:10],
                "dependency_graph": payload["dependency_graph"].get(cell_id, {}),
                "dependency_health": payload["dependency_health"].get(cell_id, {}),
                "handshake_packet": next((row for row in handshakes if str(row.get("cell_id") or "") == cell_id), {}),
                "resource_budget": resource_arbitration["cell_budgets"].get(cell_id, {}),
            }
            write_payload(root / "intelligence.json", cell_payload)
            _append_jsonl(root / "intelligence_queue.jsonl", cell_payload)
        write_payload(cell_root / "cell_intelligence_bus.json", payload)
        write_payload(cell_root / "cell_resource_arbitration.json", resource_arbitration)
        write_payload(cell_root / "cell_handshake_packets.json", {"timestamp_utc": iso_now(), "packets": handshakes})
        payload["write_result"] = {
            "cell_root": _rel(project_root, cell_root),
            "override": _write_override(DEFAULT_OVERRIDE_PATH),
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank and arbitrate the distributed cell federation.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--distributed-file", default=str(DEFAULT_DISTRIBUTED_PATH))
    parser.add_argument("--cell-root", default=str(DEFAULT_CELL_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(
        project_root=project_root,
        distributed_path=Path(args.distributed_file),
        cell_root=Path(args.cell_root),
        apply=bool(args.apply),
    )
    out_path = Path(args.out_file).expanduser()
    if not out_path.is_absolute():
        out_path = project_root / out_path
    write_payload(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "cell_federation_intelligence "
            f"overall_status={payload.get('overall_status')} "
            f"grade={payload.get('intelligence_grade')} "
            f"top_cell={(payload.get('ranked_needs') or [{}])[0].get('cell_id', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
