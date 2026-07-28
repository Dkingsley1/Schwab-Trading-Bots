#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


SCHEMA_VERSION = 1
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "infrabot_library_self_awareness_v1.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "infrabot_library_self_awareness_control_latest.json"
DEFAULT_EXTERNAL_CONTEXT_PATH = PROJECT_ROOT / "exports" / "external_context" / "infrabot_library_self_awareness_control_latest.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "exports" / "reports" / "operator" / "infrabot_library_self_awareness_control_latest.md"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.infrabot_library_self_awareness_override"

READY_STATUSES = {"ready", "ok", "active", "guarded", "advisory", "watch", "needs_action"}
BAD_STATUSES = {"blocked", "critical", "failed", "error", "missing"}
SINGLE_WRITER_HINTS = ("storage", "writer", "sql", "backpressure", "stateful", "quota")


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _string_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


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
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "ready", "ok", "active", "guarded"}


def _path_value(payload: Any, dotted: str, default: Any = None) -> Any:
    current = payload
    for part in str(dotted or "").split("."):
        if not part:
            continue
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
        return default
    return current


def _path_exists(payload: Any, dotted: str) -> bool:
    sentinel = object()
    return _path_value(payload, dotted, sentinel) is not sentinel


def _extract_status(payload: dict[str, Any], status_path: str = "") -> str:
    if status_path:
        status = str(_path_value(payload, status_path, "") or "").strip().lower()
        if status:
            return status
    for key in ("overall_status", "status", "state"):
        status = str(payload.get(key) or "").strip().lower()
        if status:
            return status
    if "ok" in payload:
        return "ready" if bool(payload.get("ok")) else "blocked"
    return "present" if payload else "missing"


def _project_path(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else project_root / path


def _artifact_row(project_root: Path, requirement: dict[str, Any]) -> dict[str, Any]:
    name = str(requirement.get("name") or "").strip()
    path = _project_path(project_root, requirement.get("path"))
    required = bool(requirement.get("required", False))
    exists = path.exists()
    payload = load_json(path) if exists else {}
    status = _extract_status(payload, str(requirement.get("status_path") or "")) if exists else "missing"
    ready_statuses = set(_string_list(requirement.get("ready_statuses")) or READY_STATUSES)
    blockers: list[str] = []
    if required and not exists:
        blockers.append("artifact_missing")
    elif exists and status not in ready_statuses:
        blockers.append(f"status_not_ready:{status}")

    selected: dict[str, Any] = {}
    for dotted in _string_list(requirement.get("truthy_paths")):
        value = _path_value(payload, dotted)
        selected[dotted] = value
        if not _bool(value):
            blockers.append(f"truthy_path_failed:{dotted}")
    for dotted in _string_list(requirement.get("falsey_paths")):
        value = _path_value(payload, dotted)
        selected[dotted] = value
        if _bool(value):
            blockers.append(f"falsey_path_failed:{dotted}")
    for dotted in _string_list(requirement.get("zero_count_paths")):
        value = _path_value(payload, dotted)
        selected[dotted] = value
        ok = len(value) == 0 if isinstance(value, list) else _safe_float(value, 999999.0) == 0.0
        if not ok:
            blockers.append(f"zero_count_failed:{dotted}")

    max_values = requirement.get("max_value_by_path") if isinstance(requirement.get("max_value_by_path"), dict) else {}
    for dotted, ceiling in max_values.items():
        value = _path_value(payload, str(dotted), 0)
        selected[str(dotted)] = value
        if _safe_float(value, 0.0) > _safe_float(ceiling):
            blockers.append(f"max_value_failed:{dotted}")
    min_values = requirement.get("min_value_by_path") if isinstance(requirement.get("min_value_by_path"), dict) else {}
    for dotted, floor in min_values.items():
        value = _path_value(payload, str(dotted))
        selected[str(dotted)] = value
        if _safe_float(value, -999999.0) < _safe_float(floor):
            blockers.append(f"min_value_failed:{dotted}")
    expected = requirement.get("expected_values") if isinstance(requirement.get("expected_values"), dict) else {}
    for dotted, expected_value in expected.items():
        value = _path_value(payload, str(dotted))
        selected[str(dotted)] = value
        if str(value) != str(expected_value):
            blockers.append(f"expected_value_failed:{dotted}")

    blockers = ordered_unique(blockers)
    return {
        "name": name,
        "path": str(path),
        "required": required,
        "exists": exists,
        "status": status,
        "ready": not blockers,
        "blocking": bool(required and blockers),
        "blockers": blockers,
        "selected_values": selected,
        "summary_keys": sorted(payload.keys())[:24] if payload else [],
    }


def _health(project_root: Path, filename: str) -> dict[str, Any]:
    return load_json(project_root / "governance" / "health" / filename)


def _command_key(command: list[Any]) -> tuple[str, ...]:
    return tuple(str(part) for part in command)


def _command_text(command: list[Any]) -> str:
    return " ".join(str(part) for part in command)


def _command_lane(command: list[Any], configured_lanes: list[dict[str, Any]]) -> str:
    text = _command_text(command).lower()
    for lane in configured_lanes:
        lane_name = str(lane.get("lane") or "")
        if lane_name and lane_name.replace("_", "-") in text:
            return lane_name
        for owner in _string_list(lane.get("owner_controls")):
            if owner.replace("_", "-") in text or owner in text:
                return lane_name
    if "source-verification" in text or "collector-contract" in text:
        return "source_truth"
    if "auth" in text or "production-readiness" in text:
        return "auth_live_lock"
    if "runtime" in text or "memory" in text or "mlx" in text or "library" in text:
        return "runtime_memory"
    if "regression" in text or "drift" in text or "grade" in text:
        return "governance_regression"
    if any(hint in text for hint in SINGLE_WRITER_HINTS):
        return "storage_writer"
    return "general_infrabot"


def _lane_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for raw in _as_list(config.get("infrabot_efficiency_lanes")):
        if not isinstance(raw, dict):
            continue
        lane = str(raw.get("lane") or "").strip()
        if lane:
            specs[lane] = raw
    return specs


def _collect_default_lane_commands(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane in _as_list(config.get("infrabot_efficiency_lanes")):
        if not isinstance(lane, dict):
            continue
        for command in _as_list(lane.get("default_commands")):
            if isinstance(command, list) and command:
                rows.append({"source": "config_default", "reason": "default_lane_command", "command": [str(part) for part in command]})
    return rows


def _collect_payload_commands(name: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in (
        "repair_plan",
        "advisory_repair_plan",
        "pre_apply_repair_plan",
        "pre_apply_advisory_repair_plan",
        "recommended_commands",
        "ordered_repair_commands",
    ):
        for raw in _as_list(payload.get(field)):
            command = raw.get("cmd") if isinstance(raw, dict) else raw
            if isinstance(command, list) and command:
                rows.append(
                    {
                        "source": name,
                        "reason": str(raw.get("reason") or field) if isinstance(raw, dict) else field,
                        "command": [str(part) for part in command],
                    }
                )
    for check in _as_list(payload.get("checks")):
        if not isinstance(check, dict):
            continue
        for command in _as_list(check.get("repair_commands")):
            if isinstance(command, list) and command:
                rows.append(
                    {
                        "source": name,
                        "reason": f"check:{check.get('name', '')}",
                        "command": [str(part) for part in command],
                    }
                )
    return rows


def _infrabot_efficiency_plan(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    lane_spec_list = [raw for raw in _as_list(config.get("infrabot_efficiency_lanes")) if isinstance(raw, dict)]
    specs = _lane_specs(config)
    rows = _collect_default_lane_commands(config)
    for filename, name in (
        ("infrastructure_autofix_bot_latest.json", "infrastructure_autofix_bot"),
        ("master_infrastructure_supervisor_latest.json", "master_infrastructure_supervisor"),
        ("system_drift_autopilot_latest.json", "system_drift_autopilot"),
        ("infrabot_adaptive_governor_latest.json", "infrabot_adaptive_governor"),
    ):
        rows.extend(_collect_payload_commands(name, _health(project_root, filename)))

    seen: set[tuple[str, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        command = [str(part) for part in _as_list(row.get("command"))]
        if not command:
            continue
        key = _command_key(command)
        if key in seen:
            continue
        seen.add(key)
        lane = _command_lane(command, lane_spec_list)
        spec = specs.get(lane, {})
        max_parallel = max(1, _safe_int(spec.get("max_parallel"), 1 if lane == "storage_writer" else 2))
        single_writer = bool(max_parallel == 1 or lane == "storage_writer" or any(hint in _command_text(command).lower() for hint in SINGLE_WRITER_HINTS))
        deduped.append(
            {
                "lane": lane,
                "source": str(row.get("source") or ""),
                "reason": str(row.get("reason") or ""),
                "command": command,
                "max_parallel": 1 if single_writer else max_parallel,
                "single_writer_or_pressure_sensitive": single_writer,
                "pressure_policy": str(spec.get("pressure_policy") or "run_with_default_infrabot_rate_limits"),
                "stop_condition": str(spec.get("stop_condition") or "owning health artifact returns ready or advisory-only"),
                "authority_boundary": "safe_repair_or_read_only_no_live_execution_no_dependency_mutation",
            }
        )
    lane_counts = Counter(str(row.get("lane") or "general_infrabot") for row in deduped)
    return {
        "mode": "dedupe_group_and_rate_limit_infrabot_commands",
        "command_count": len(deduped),
        "lane_counts": dict(sorted(lane_counts.items())),
        "single_writer_command_count": sum(1 for row in deduped if row.get("single_writer_or_pressure_sensitive")),
        "max_parallel_writer_commands": 1,
        "commands": deduped,
        "efficiency_contract": {
            "deduplicates_repair_commands": True,
            "storage_and_writer_repairs_single_parallel": True,
            "uses_existing_owner_controls": True,
            "does_not_run_repairs_directly": True,
        },
    }


def _first_command(plan: dict[str, Any], lane: str) -> list[str]:
    for row in _as_list(plan.get("commands")):
        if isinstance(row, dict) and str(row.get("lane") or "") == lane and isinstance(row.get("command"), list):
            return [str(part) for part in row["command"]]
    return []


def _managed_dashboard_needs(project_root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    dashboard = _health(project_root, "runtime_gate_dashboard_latest.json")
    overall = _as_dict(dashboard.get("overall"))
    rows: list[dict[str, Any]] = []
    for raw in _as_list(overall.get("managed_controls")):
        if not isinstance(raw, dict):
            continue
        attention = str(raw.get("attention") or "managed_attention").strip()
        lane = "storage_writer" if "storage" in attention else "governance_regression"
        if "coordination" in attention:
            lane = "auth_live_lock"
        if "infrastructure" in attention:
            lane = "governance_regression"
        rows.append(
            {
                "need_id": attention,
                "status": "managed_soak_advisory",
                "owner": str(raw.get("managed_by") or "unattended_soak_readiness"),
                "symptom": attention,
                "why_it_matters": "This is visible production telemetry, but the soak manager has verified paper/auth/storage/livefeed contracts are still clean.",
                "safe_next_command": _first_command(plan, lane),
                "stop_condition": str(raw.get("when_to_unmanage") or "item returns ready or stops being managed by soak readiness"),
                "authority_boundary": "visibility_and_safe_repair_only_no_live_execution",
                "urgency": "watch",
                "soak_impact": "does_not_block_guarded_paper_soak_while_soak_management_context_remains_ready",
                "evidence": raw,
            }
        )
    return rows


def _artifact_needs(artifact_rows: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in artifact_rows:
        if row.get("ready"):
            continue
        lane = "runtime_memory"
        name = str(row.get("name") or "")
        if "library" in name or "mlx" in name:
            lane = "runtime_memory"
        elif "storage" in name or "writer" in name:
            lane = "storage_writer"
        elif "auth" in name or "production" in name:
            lane = "auth_live_lock"
        elif "source" in name:
            lane = "source_truth"
        elif "regression" in name or "drift" in name or "infra" in name:
            lane = "governance_regression"
        rows.append(
            {
                "need_id": name,
                "status": "hard_blocker" if row.get("blocking") else "owned_advisory",
                "owner": lane,
                "symptom": ",".join(_string_list(row.get("blockers"))) or f"{name}_not_ready",
                "why_it_matters": "Required control artifacts must stay current and truthful so the soak does not hide paper trading, storage, auth, or routing faults.",
                "safe_next_command": _first_command(plan, lane),
                "stop_condition": "artifact row reports ready and all required paths pass",
                "authority_boundary": "safe_repair_or_read_only_no_live_execution_no_dependency_mutation",
                "urgency": "critical" if row.get("blocking") else "watch",
                "soak_impact": "blocks_soak_confidence" if row.get("blocking") else "visible_but_managed",
                "evidence": row,
            }
        )
    return rows


def _library_upgrade_scope(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    library_utilization = _health(project_root, "library_utilization_router_latest.json")
    library_upgrade = _health(project_root, "library_upgrade_route_control_latest.json")
    mlx_router = _health(project_root, "mlx_intelligence_router_latest.json")
    coverage = _as_dict(library_utilization.get("coverage"))
    candidate_matrix = _as_dict(library_utilization.get("candidate_library_matrix"))
    upgrade_plan = _as_dict(library_upgrade.get("upgrade_plan"))
    route_matrix = _as_dict(library_upgrade.get("route_matrix"))
    runtime_family_counts = _as_dict(candidate_matrix.get("runtime_family_counts"))
    return {
        "mode": str(upgrade_plan.get("mode") or "route_now_plan_upgrades_without_mutating_dependencies"),
        "dependency_mutation_allowed_during_soak": bool(upgrade_plan.get("soak_dependency_mutation_allowed", False)),
        "existing_python_packages_managed": _safe_int(coverage.get("managed_non_mlx_package_count"), 0),
        "existing_python_coverage_ratio": _safe_float(coverage.get("coverage_ratio"), 0.0),
        "existing_library_route_count": _safe_int(route_matrix.get("route_count"), 0),
        "existing_library_blocked_routes": _safe_int(route_matrix.get("blocked_route_count"), 0),
        "existing_library_degraded_routes": _safe_int(route_matrix.get("degraded_route_count"), 0),
        "hard_upgrade_blockers": _safe_int(upgrade_plan.get("hard_blocker_count"), 0),
        "actionable_upgrade_packages": _safe_int(upgrade_plan.get("actionable_package_count"), 0),
        "candidate_package_count": _safe_int(candidate_matrix.get("candidate_package_count"), 0),
        "candidate_route_coverage_ratio": _safe_float(candidate_matrix.get("mapped_candidate_ratio"), 0.0),
        "candidate_python_packages": _safe_int(runtime_family_counts.get("python"), 0),
        "candidate_mlx_packages": _safe_int(runtime_family_counts.get("mlx"), 0),
        "configured_candidate_additions": _string_list(config.get("candidate_library_additions")),
        "mlx_status": str(mlx_router.get("overall_status") or ""),
        "mlx_runtime_caps": _as_dict(mlx_router.get("runtime_caps")),
        "upgrade_safety_contract": {
            "route_existing_python_and_mlx_libraries": True,
            "stage_new_candidates_without_installing": True,
            "pip_install_or_lock_rewrite_ran": False,
            "maintenance_required_for_dependency_mutation": True,
        },
    }


def _quality_checks(
    artifact_rows: list[dict[str, Any]],
    library_scope: dict[str, Any],
    infrabot_plan: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, bool]:
    contract = _as_dict(config.get("control_contract"))
    comms = _as_dict(config.get("communication_contract"))
    required_fields = _string_list(comms.get("required_fields"))
    required_ready = all(not row.get("blocking") for row in artifact_rows)
    single_writer_ok = bool(_safe_int(infrabot_plan.get("max_parallel_writer_commands"), 99) <= 1)
    return {
        "required_artifacts_ready": required_ready,
        "existing_python_libraries_fully_routed": _safe_float(library_scope.get("existing_python_coverage_ratio"), 0.0) >= 1.0,
        "library_upgrade_hard_blockers_zero": _safe_int(library_scope.get("hard_upgrade_blockers"), 0) == 0,
        "library_routes_not_blocked_or_degraded": _safe_int(library_scope.get("existing_library_blocked_routes"), 0) == 0
        and _safe_int(library_scope.get("existing_library_degraded_routes"), 0) == 0,
        "dependency_mutation_disabled": not bool(library_scope.get("dependency_mutation_allowed_during_soak")),
        "live_execution_authority_false": not bool(contract.get("live_execution_authority", True)),
        "paper_soak_safe_contract": bool(contract.get("paper_soak_safe", False)),
        "infrabot_command_dedupe_present": _safe_int(infrabot_plan.get("command_count"), 0) > 0,
        "single_writer_storage_cap": single_writer_ok,
        "communication_contract_complete": all(field in required_fields for field in ("owner", "symptom", "safe_next_command", "stop_condition", "authority_boundary", "soak_impact")),
    }


def _grade(ready_count: int, total_count: int) -> str:
    if total_count <= 0:
        return "F"
    if ready_count == total_count:
        return "A+"
    ratio = ready_count / total_count
    if ratio >= 0.9:
        return "A"
    if ratio >= 0.75:
        return "B"
    if ratio >= 0.5:
        return "C"
    if ratio >= 0.25:
        return "D"
    return "F"


def _write_env_override(path: Path, payload: dict[str, Any]) -> bool:
    env = {
        "INFRABOT_LIBRARY_SELF_AWARENESS_ENABLED": "1",
        "INFRABOT_REPAIR_COMMAND_DEDUPE": "1",
        "INFRABOT_STORAGE_WRITER_MAX_PARALLEL": "1",
        "INFRABOT_NEEDS_COMMUNICATION_DEPTH": "owner_command_stop_condition_authority",
        "LIBRARY_UPGRADE_ROUTE_DEPENDENCY_MUTATION_ALLOWED": "0",
        "PRIMARY_ML_RUNTIME_BACKEND": "mlx",
    }
    lines = ["# Auto-managed by scripts/ops/infrabot_library_self_awareness_control.py"]
    for key, value in sorted(env.items()):
        safe_value = str(value).replace("'", "'\"'\"'")
        lines.append(f"{key}='{safe_value}'")
    lines.append(f"INFRABOT_LIBRARY_SELF_AWARENESS_GRADE='{payload.get('grade', '')}'")
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def render_markdown(payload: dict[str, Any]) -> str:
    library_scope = _as_dict(payload.get("library_upgrade_scope"))
    plan = _as_dict(payload.get("infrabot_efficiency_plan"))
    lines = [
        "# Infrabot Library Self Awareness Control",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc', '')}`",
        f"Status: `{payload.get('overall_status', '')}`",
        f"Grade: `{payload.get('grade', '')}`",
        "",
        "## Library Routing",
        "",
        f"- Existing Python packages managed: `{library_scope.get('existing_python_packages_managed', 0)}`",
        f"- Existing Python coverage ratio: `{library_scope.get('existing_python_coverage_ratio', 0.0)}`",
        f"- Existing library routes: `{library_scope.get('existing_library_route_count', 0)}`",
        f"- Hard upgrade blockers: `{library_scope.get('hard_upgrade_blockers', 0)}`",
        f"- Dependency mutation during soak: `{library_scope.get('dependency_mutation_allowed_during_soak', False)}`",
        f"- Candidate packages staged: `{library_scope.get('candidate_package_count', 0)}`",
        f"- Candidate MLX packages staged: `{library_scope.get('candidate_mlx_packages', 0)}`",
        "",
        "## Infrabot Efficiency",
        "",
        f"- Deduped commands: `{plan.get('command_count', 0)}`",
        f"- Single-writer or pressure-sensitive commands: `{plan.get('single_writer_command_count', 0)}`",
        f"- Lane counts: `{json.dumps(plan.get('lane_counts') or {}, sort_keys=True)}`",
        "",
        "## Needs Brief",
        "",
    ]
    for need in _as_list(payload.get("self_awareness_need_brief"))[:20]:
        if not isinstance(need, dict):
            continue
        command = " ".join(str(part) for part in _as_list(need.get("safe_next_command"))) or "none"
        lines.append(f"- `{need.get('status', '')}` `{need.get('need_id', '')}` owner=`{need.get('owner', '')}` urgency=`{need.get('urgency', '')}`")
        lines.append(f"  command: `{command}`")
        lines.append(f"  stop: {need.get('stop_condition', '')}")
    lines.extend(["", "## Control Contract", "", "- Live execution authority: `false`", "- Dependency mutation during soak: `false`", "- Raw profitability truth remains external and evidence-based."])
    return "\n".join(lines).rstrip() + "\n"


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path = DEFAULT_CONFIG,
    apply: bool = False,
) -> dict[str, Any]:
    config = load_json(config_path)
    artifact_rows = [
        _artifact_row(project_root, raw)
        for raw in _as_list(config.get("artifact_requirements"))
        if isinstance(raw, dict)
    ]
    infrabot_plan = _infrabot_efficiency_plan(project_root, config)
    library_scope = _library_upgrade_scope(project_root, config)
    need_rows = _managed_dashboard_needs(project_root, infrabot_plan) + _artifact_needs(artifact_rows, infrabot_plan)
    need_rows = sorted(
        need_rows,
        key=lambda row: (
            0 if str(row.get("status") or "") == "hard_blocker" else 1,
            str(row.get("urgency") or ""),
            str(row.get("need_id") or ""),
        ),
    )
    quality_checks = _quality_checks(artifact_rows, library_scope, infrabot_plan, config)
    ready_quality_count = sum(1 for value in quality_checks.values() if value)
    blockers = ordered_unique(
        [
            f"{row.get('name')}:{blocker}"
            for row in artifact_rows
            if row.get("blocking")
            for blocker in _string_list(row.get("blockers"))
        ]
        + [f"quality:{name}" for name, ok in quality_checks.items() if not ok]
    )
    overall_ready = bool(not blockers and ready_quality_count == len(quality_checks))
    contract = _as_dict(config.get("control_contract"))
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": SCHEMA_VERSION,
        "source": "infrabot_library_self_awareness_control",
        "ok": overall_ready,
        "overall_status": "ready" if overall_ready else "needs_work",
        "grade": _grade(ready_quality_count, len(quality_checks)),
        "target_grade": str(config.get("target_grade") or "A+"),
        "quality_checks": quality_checks,
        "ready_quality_count": ready_quality_count,
        "quality_check_count": len(quality_checks),
        "blockers": blockers,
        "artifact_rows": artifact_rows,
        "library_upgrade_scope": library_scope,
        "infrabot_efficiency_plan": infrabot_plan,
        "self_awareness_need_brief": need_rows,
        "communications_contract": _as_dict(config.get("communication_contract")),
        "control_contract": {
            "live_execution_authority": False,
            "live_orders_must_remain_disabled": True,
            "paper_soak_safe": bool(contract.get("paper_soak_safe", True)),
            "dependency_mutation_allowed_during_soak": False,
            "pip_install_or_lock_rewrite_ran": False,
            "raw_profitability_truth_must_remain_visible": bool(contract.get("raw_profitability_truth_must_remain_visible", True)),
            "managed_dashboard_attention_does_not_fake_green": True,
        },
        "recommended_actions": ordered_unique(
            [
                "./scripts/ops/opsctl.sh library-utilization-router --apply --json",
                "./scripts/ops/opsctl.sh library-upgrade-route --apply --json",
                "./scripts/ops/opsctl.sh mlx-intelligence-router --apply --json",
                "./scripts/ops/opsctl.sh infrabot-library-self-awareness --apply --json",
                "keep dependency installs in a maintenance window; this control only routes and stages during soak",
                "run the first safe_next_command for any hard_blocker need, then refresh this control",
            ]
        ),
        "apply_result": {
            "applied": bool(apply),
            "dependency_mutation_ran": False,
            "live_execution_authority_changed": False,
        },
        "artifact_paths": {
            "json": str(DEFAULT_OUT_PATH),
            "external_context": str(DEFAULT_EXTERNAL_CONTEXT_PATH),
            "markdown": str(DEFAULT_MARKDOWN_PATH),
            "config": str(config_path),
            "env_override": str(DEFAULT_OVERRIDE_PATH),
        },
    }
    return payload


def write_outputs(
    payload: dict[str, Any],
    *,
    out_path: Path = DEFAULT_OUT_PATH,
    external_context_path: Path = DEFAULT_EXTERNAL_CONTEXT_PATH,
    markdown_path: Path = DEFAULT_MARKDOWN_PATH,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
    apply: bool = False,
) -> None:
    if apply:
        payload["apply_result"]["env_override_changed"] = _write_env_override(override_path, payload)
    write_payload(out_path, payload)
    write_payload(external_context_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Coordinate infrabot efficiency, library upgrade routing, and deep needs communication.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--external-context-file", type=Path, default=DEFAULT_EXTERNAL_CONTEXT_PATH)
    parser.add_argument("--markdown-file", type=Path, default=DEFAULT_MARKDOWN_PATH)
    parser.add_argument("--override-file", type=Path, default=DEFAULT_OVERRIDE_PATH)
    parser.add_argument("--apply", action="store_true", help="Write health, external context, markdown, and env override artifacts.")
    parser.add_argument("--check", action="store_true", help="Exit nonzero unless the control is ready.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(args.project_root.resolve(), config_path=args.config.resolve(), apply=bool(args.apply))
    write_outputs(
        payload,
        out_path=args.out_file.expanduser(),
        external_context_path=args.external_context_file.expanduser(),
        markdown_path=args.markdown_file.expanduser(),
        override_path=args.override_file.expanduser(),
        apply=bool(args.apply),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "infrabot_library_self_awareness_control "
            f"status={payload.get('overall_status', '')} "
            f"grade={payload.get('grade', '')} "
            f"needs={len(payload.get('self_awareness_need_brief') or [])}"
        )
    return 2 if args.check and not payload.get("ok") else 0


if __name__ == "__main__":
    raise SystemExit(main())
