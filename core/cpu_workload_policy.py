from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "cpu_workload_policy_v1.json"
TASKPOLICY_CANDIDATES = (
    "/usr/sbin/taskpolicy",
    "/usr/bin/taskpolicy",
)


class CPUWorkloadPolicyError(ValueError):
    pass


def taskpolicy_executable() -> str:
    discovered = shutil.which("taskpolicy")
    for candidate in (discovered, *TASKPOLICY_CANDIDATES):
        if candidate and os.access(candidate, os.X_OK):
            return str(candidate)
    return ""


def _bounded_nice(value: Any, default: int = 0) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        parsed = int(default)
    return min(max(parsed, 0), 20)


def _positive_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        return max(int(default), 0)
    return parsed if parsed > 0 else max(int(default), 0)


def load_cpu_workload_policy(path: Path | str | None = None) -> dict[str, Any]:
    policy_path = Path(path) if path is not None else DEFAULT_POLICY_PATH
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CPUWorkloadPolicyError(f"cpu_workload_policy_unreadable:{policy_path}") from exc
    if not isinstance(payload, dict):
        raise CPUWorkloadPolicyError("cpu_workload_policy_must_be_object")
    if int(payload.get("schema_version", 0) or 0) != 1:
        raise CPUWorkloadPolicyError("cpu_workload_policy_schema_version_invalid")
    if not bool(payload.get("policy_locked", False)):
        raise CPUWorkloadPolicyError("cpu_workload_policy_must_be_locked")

    classes = payload.get("workload_classes")
    if not isinstance(classes, dict) or not classes:
        raise CPUWorkloadPolicyError("cpu_workload_classes_missing")
    for name, contract in classes.items():
        if not str(name).strip() or not isinstance(contract, dict):
            raise CPUWorkloadPolicyError("cpu_workload_class_invalid")
        has_floor = "nice_floor" in contract
        has_ceiling = "nice_ceiling" in contract
        if has_floor == has_ceiling:
            raise CPUWorkloadPolicyError(f"cpu_workload_class_nice_boundary_invalid:{name}")
        boundary = contract.get("nice_floor") if has_floor else contract.get("nice_ceiling")
        if isinstance(boundary, bool) or not isinstance(boundary, int) or not 0 <= boundary <= 20:
            raise CPUWorkloadPolicyError(f"cpu_workload_class_nice_value_invalid:{name}")

    authority = payload.get("authority")
    if not isinstance(authority, dict) or any(bool(value) for value in authority.values()):
        raise CPUWorkloadPolicyError("cpu_workload_policy_authority_must_remain_zero")
    return payload


def workload_class_contract(policy: Mapping[str, Any], workload_class: str) -> dict[str, Any]:
    classes = policy.get("workload_classes") if isinstance(policy.get("workload_classes"), Mapping) else {}
    name = str(workload_class or "").strip().lower()
    contract = classes.get(name)
    if not isinstance(contract, Mapping):
        raise CPUWorkloadPolicyError(f"unknown_cpu_workload_class:{name or 'empty'}")
    return {"workload_class": name, **dict(contract)}


def resolve_shadow_workload_class(
    policy: Mapping[str, Any],
    *,
    explicit: str = "",
    profile: str = "",
    lifecycle_state: str = "",
    environment: Mapping[str, str] | None = None,
) -> str:
    classes = policy.get("workload_classes") if isinstance(policy.get("workload_classes"), Mapping) else {}
    requested = str(explicit or "").strip().lower()
    if requested:
        if requested not in classes:
            raise CPUWorkloadPolicyError(f"unknown_cpu_workload_class:{requested}")
        return requested

    resolution = policy.get("shadow_resolution") if isinstance(policy.get("shadow_resolution"), Mapping) else {}
    collect_states = {
        str(value or "").strip().lower()
        for value in resolution.get("collect_only_lifecycle_states", [])
        if str(value or "").strip()
    }
    if str(lifecycle_state or "").strip().lower() in collect_states:
        return "data_collection"

    env = environment if isinstance(environment, Mapping) else {}
    research_flags = resolution.get("research_only_environment_flags", [])
    if any(str(env.get(str(flag), "0")).strip().lower() in {"1", "true", "yes", "on"} for flag in research_flags):
        return "research_training"

    profile_name = str(profile or "").strip().lower()
    if profile_name.endswith("_research") or profile_name.startswith("research_"):
        return "research_training"
    default_class = str(resolution.get("default_class") or "market_decision").strip().lower()
    return default_class if default_class in classes else "market_decision"


def nice_target_for_class(
    policy: Mapping[str, Any],
    workload_class: str,
    requested_nice: Any,
) -> int:
    contract = workload_class_contract(policy, workload_class)
    requested = _bounded_nice(requested_nice, 0)
    if "nice_ceiling" in contract:
        return min(requested, _bounded_nice(contract.get("nice_ceiling"), 0))
    return max(requested, _bounded_nice(contract.get("nice_floor"), 15))


def runtime_priority_decision(
    policy: Mapping[str, Any],
    *,
    workload_class: str,
    requested_nice: Any,
    current_nice: Any,
) -> dict[str, Any]:
    contract = workload_class_contract(policy, workload_class)
    current = _bounded_nice(current_nice, 0)
    target = nice_target_for_class(policy, workload_class, requested_nice)
    has_ceiling = "nice_ceiling" in contract
    self_deprioritize_delta = 0 if has_ceiling else max(target - current, 0)
    managed_restart_required = bool(has_ceiling and current > target)
    priority_compliant = bool(current <= target) if has_ceiling else bool(current >= target)
    return {
        **contract,
        "current_nice": current,
        "target_nice": target,
        "self_deprioritize_delta": self_deprioritize_delta,
        "managed_restart_required": managed_restart_required,
        "priority_compliant": priority_compliant,
        "hard_affinity_claimed": False,
    }


def resource_partition(
    policy: Mapping[str, Any],
    *,
    performance_core_count: int,
    efficiency_core_count: int,
) -> dict[str, int | bool]:
    config = policy.get("resource_partition") if isinstance(policy.get("resource_partition"), Mapping) else {}
    p_cores = max(int(performance_core_count), 1)
    e_cores = max(int(efficiency_core_count), 0)
    reserve = min(_positive_int(config.get("foreground_performance_core_reserve"), 1), max(p_cores - 1, 0))
    max_critical = _positive_int(config.get("maximum_critical_shared_workers"), p_cores)
    max_service = _positive_int(config.get("maximum_efficiency_service_workers"), e_cores)
    return {
        "performance_core_count": p_cores,
        "foreground_performance_core_reserve": reserve,
        "critical_shared_performance_workers": max(min(p_cores - reserve, max_critical), 1),
        "efficiency_core_count": e_cores,
        "efficiency_service_workers": min(e_cores, max_service),
        "critical_efficiency_spillover_allowed": bool(config.get("critical_efficiency_spillover_allowed", False)),
        "support_efficiency_spillover_allowed": bool(config.get("support_efficiency_spillover_allowed", True)),
    }


def cpu_environment_contract(
    policy: Mapping[str, Any],
    *,
    performance_core_count: int,
    efficiency_core_count: int,
) -> dict[str, str]:
    partition = resource_partition(
        policy,
        performance_core_count=performance_core_count,
        efficiency_core_count=efficiency_core_count,
    )
    classes = policy.get("workload_classes") if isinstance(policy.get("workload_classes"), Mapping) else {}

    def ceiling(name: str, default: int) -> str:
        row = classes.get(name) if isinstance(classes.get(name), Mapping) else {}
        return str(_bounded_nice(row.get("nice_ceiling"), default))

    def floor(name: str, default: int) -> str:
        row = classes.get(name) if isinstance(classes.get(name), Mapping) else {}
        return str(_bounded_nice(row.get("nice_floor"), default))

    return {
        "BOT_CPU_WORKLOAD_POLICY_ID": str(policy.get("policy_id") or "cpu_workload_policy_v1"),
        "BOT_CPU_WORKLOAD_POLICY_LOCKED": "1",
        "BOT_CPU_HARD_AFFINITY_SUPPORTED": "0",
        "BOT_CPU_TASKPOLICY_SELF_HEAL": "1",
        "BOT_CPU_CRITICAL_SUPERVISOR_MAX_NICE": ceiling("critical_supervisor", 0),
        "BOT_CPU_LIVE_EXECUTION_MAX_NICE": ceiling("live_execution", 0),
        "BOT_CPU_PAPER_EXECUTION_MAX_NICE": ceiling("paper_execution", 0),
        "BOT_CPU_MARKET_DECISION_MAX_NICE": ceiling("market_decision", 4),
        "BOT_CPU_DATA_COLLECTION_MIN_NICE": floor("data_collection", 12),
        "BOT_CPU_RESEARCH_TRAINING_MIN_NICE": floor("research_training", 15),
        "BOT_CPU_STORAGE_MAINTENANCE_MIN_NICE": floor("storage_maintenance", 15),
        "BOT_CPU_CRITICAL_SHARED_WORKERS": str(partition["critical_shared_performance_workers"]),
        "BOT_CPU_FOREGROUND_APP_RESERVE": str(partition["foreground_performance_core_reserve"]),
        "BOT_CPU_SUPPORT_SPILLOVER_WORKERS": str(partition["efficiency_service_workers"]),
        "BOT_CPU_CRITICAL_EFFICIENCY_SPILLOVER": "0",
    }
