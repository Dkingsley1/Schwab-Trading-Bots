#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops import library_utilization_router, mlx_library_upgrade
    from scripts.ops.long_runtime_common import iso_now, ordered_unique, write_payload
else:
    from . import library_utilization_router, mlx_library_upgrade
    from .long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload


DEFAULT_LOCK = PROJECT_ROOT / "config" / "requirements.lock.txt"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv314" / "bin" / "python"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "library_upgrade_route_control_latest.json"
DEFAULT_EXTERNAL_CONTEXT_PATH = PROJECT_ROOT / "exports" / "external_context" / "library_upgrade_route_control_latest.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "exports" / "reports" / "operator" / "library_upgrade_route_control_latest.md"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.library_upgrade_route_control_override"

MLX_LANE = "mlx_acceleration"
OPTIONAL_MLX_COMPATIBILITY_EXCLUDED = {
    "mlx-cluster": "native extension is not compatible with the latest MLX Metal device API",
    "mlx-data": "no compatible distribution is available for the active Python 3.14 runtime",
    "mlx-graphs": "requires mlx-cluster and older shared dependency pins under latest MLX",
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _norm_package(name: str) -> str:
    return library_utilization_router._norm_package(name)


def _load_installed_versions(installed_versions: dict[str, str] | None) -> dict[str, str]:
    raw = installed_versions if isinstance(installed_versions, dict) else library_utilization_router._load_installed_versions()
    return {
        _norm_package(name): str(version)
        for name, version in raw.items()
        if _norm_package(name)
    }


def _package_status(package: str, lock_versions: dict[str, str], installed_versions: dict[str, str]) -> str:
    return library_utilization_router._package_status(package, lock_versions, installed_versions)


def _mlx_package_inventory(lock_versions: dict[str, str], installed_versions: dict[str, str]) -> list[dict[str, Any]]:
    packages = sorted(
        package
        for package in set(lock_versions) | set(installed_versions)
        if library_utilization_router._is_mlx_routed(package)
    )
    rows: list[dict[str, Any]] = []
    for package in packages:
        status = _package_status(package, lock_versions, installed_versions)
        if status == "missing_runtime" and package in OPTIONAL_MLX_COMPATIBILITY_EXCLUDED:
            status = "compatibility_excluded_optional"
        rows.append(
            {
                "package": package,
                "lane": MLX_LANE,
                "locked_version": lock_versions.get(package),
                "installed_version": installed_versions.get(package),
                "status": status,
                "source": "locked" if package in lock_versions else "runtime_only",
                "fallback_packages": [],
                "available_fallback_packages": [],
                "compatibility_note": OPTIONAL_MLX_COMPATIBILITY_EXCLUDED.get(package, ""),
            }
        )
    return rows


def _route_status(rows: list[dict[str, Any]], lane: str) -> str:
    statuses = {str(row.get("status") or "") for row in rows}
    if "missing_runtime" in statuses:
        return "blocked"
    if lane == MLX_LANE and statuses.intersection({"runtime_behind_lock", "version_mismatch"}):
        return "degraded"
    if library_utilization_router._lane_drift_is_critical(lane, statuses):
        return "degraded"
    if any(
        item in statuses
        for item in (
            "runtime_behind_lock",
            "version_mismatch",
            "runtime_ahead_of_lock",
            "optional_fallback_active",
            "compatibility_excluded_optional",
            "runtime_only",
        )
    ):
        return "advisory"
    return "ready" if rows else "thin"


def _mlx_route(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "lane": MLX_LANE,
        "status": _route_status(rows, MLX_LANE),
        "workload_family": "apple_silicon_mlx_inference_embeddings_audio_graph_and_quant_research",
        "priority": "primary_ml_backend_guarded_by_runtime_caps",
        "target_surfaces": ["mlx_intelligence_router", "quant_model_control", "model_lifecycle", "feature_embedding_lanes"],
        "package_count": len(rows),
        "locked_package_count": sum(1 for row in rows if str(row.get("source") or "") == "locked"),
        "runtime_only_package_count": sum(1 for row in rows if str(row.get("source") or "") == "runtime_only"),
        "packages": [str(row.get("package") or "") for row in rows],
        "blocked_packages": [str(row.get("package") or "") for row in rows if str(row.get("status") or "") == "missing_runtime"],
        "version_mismatch_packages": [
            str(row.get("package") or "")
            for row in rows
            if str(row.get("status") or "") in {"version_mismatch", "runtime_behind_lock"}
        ],
        "runtime_ahead_packages": [
            str(row.get("package") or "")
            for row in rows
            if str(row.get("status") or "") == "runtime_ahead_of_lock"
        ],
        "compatibility_excluded_packages": [
            str(row.get("package") or "")
            for row in rows
            if str(row.get("status") or "") == "compatibility_excluded_optional"
        ],
    }


def _route_matrix(routes: list[dict[str, Any]], package_rows: list[dict[str, Any]]) -> dict[str, Any]:
    package_to_lane = {
        str(row.get("package") or ""): str(row.get("lane") or "")
        for row in package_rows
        if str(row.get("package") or "")
    }
    lane_to_packages = {
        str(route.get("lane") or ""): list(route.get("packages") or [])
        for route in routes
    }
    unmapped = [package for package, lane in package_to_lane.items() if not lane]
    return {
        "package_count": len(package_to_lane),
        "mapped_package_count": len(package_to_lane) - len(unmapped),
        "mapped_package_ratio": round((len(package_to_lane) - len(unmapped)) / max(len(package_to_lane), 1), 4),
        "unmapped_packages": unmapped,
        "package_to_lane": package_to_lane,
        "lane_to_packages": lane_to_packages,
        "route_count": len(routes),
        "active_route_count": sum(1 for route in routes if str(route.get("status") or "") in {"ready", "advisory"}),
        "blocked_route_count": sum(1 for route in routes if str(route.get("status") or "") == "blocked"),
        "degraded_route_count": sum(1 for route in routes if str(route.get("status") or "") == "degraded"),
    }


def _upgrade_action(row: dict[str, Any], python_bin: Path) -> dict[str, Any]:
    package = str(row.get("package") or "")
    locked = row.get("locked_version")
    installed = row.get("installed_version")
    status = str(row.get("status") or "")
    command: list[str] = []
    action = "keep_current_route"
    maintenance_required = False
    soak_safe = True
    if status == "missing_runtime":
        action = "install_locked_package_before_enabling_lane"
        maintenance_required = True
        soak_safe = False
    elif status in {"runtime_behind_lock", "version_mismatch"}:
        action = "upgrade_runtime_to_locked_version_in_maintenance_window"
        maintenance_required = True
        soak_safe = False
    elif status == "runtime_ahead_of_lock":
        action = "adopt_runtime_version_into_lock_after_canary_evidence"
    elif status == "optional_fallback_active":
        action = "keep_fallback_route_active_and_install_pinned_optional_package_later"
        maintenance_required = True
    elif status == "compatibility_excluded_optional":
        action = "keep_optional_mlx_satellite_deferred_until_compatible_distribution_exists"
    elif status == "runtime_only":
        action = "decide_pin_or_remove_runtime_only_package_after_soak"
    if locked and action in {
        "install_locked_package_before_enabling_lane",
        "upgrade_runtime_to_locked_version_in_maintenance_window",
        "keep_fallback_route_active_and_install_pinned_optional_package_later",
    }:
        command = [str(python_bin), "-m", "pip", "install", "-U", f"{package}=={locked}"]
    return {
        "package": package,
        "lane": str(row.get("lane") or ""),
        "status": status,
        "locked_version": locked,
        "installed_version": installed,
        "action": action,
        "soak_safe_now": soak_safe,
        "maintenance_required": maintenance_required,
        "command": command,
        "available_fallback_packages": list(row.get("available_fallback_packages") or []),
        "compatibility_note": str(row.get("compatibility_note") or ""),
    }


def _upgrade_plan(package_rows: list[dict[str, Any]], python_bin: Path) -> dict[str, Any]:
    rows = [_upgrade_action(row, python_bin) for row in package_rows]
    actionable = [
        row
        for row in rows
        if str(row.get("action") or "") != "keep_current_route"
    ]
    hard = [
        row
        for row in rows
        if str(row.get("status") or "") == "missing_runtime"
        or (
            str(row.get("status") or "") in {"runtime_behind_lock", "version_mismatch"}
            and (
                str(row.get("lane") or "") == MLX_LANE
                or str(row.get("lane") or "") in library_utilization_router.CRITICAL_RUNTIME_LANES
            )
        )
    ]
    by_status = Counter(str(row.get("status") or "") for row in rows)
    by_lane = Counter(str(row.get("lane") or "") for row in actionable)
    return {
        "mode": "route_now_plan_upgrades_without_mutating_dependencies",
        "soak_dependency_mutation_allowed": False,
        "package_count": len(rows),
        "actionable_package_count": len(actionable),
        "hard_blocker_count": len(hard),
        "status_counts": dict(sorted(by_status.items())),
        "actionable_by_lane": dict(sorted(by_lane.items())),
        "hard_blockers": hard,
        "actions": actionable,
    }


def _recommended_env(router_payload: dict[str, Any]) -> dict[str, str]:
    router_env = router_payload.get("recommended_runtime_env") if isinstance(router_payload.get("recommended_runtime_env"), dict) else {}
    env = {
        "LIBRARY_UPGRADE_ROUTE_CONTROL_ENABLED": "1",
        "LIBRARY_UPGRADE_ROUTE_MODE": "soak_safe_route_now_upgrade_later",
        "LIBRARY_UPGRADE_ROUTE_DEPENDENCY_MUTATION_ALLOWED": "0",
        "LIBRARY_UPGRADE_ROUTE_LOCK_RECONCILIATION_REQUIRED": "1",
        "LIBRARY_ROUTE_DEFAULT_ML_BACKEND": "mlx",
        "PRIMARY_ML_RUNTIME_BACKEND": "mlx",
        "PORTABLE_MODEL_REPLAY_POLICY": "canary_or_off_hours_only",
        "OPTIONAL_LIBRARY_FALLBACKS_ENABLED": "1",
    }
    for key in (
        "LIBRARY_UTILIZATION_PROFILE",
        "LIBRARY_ASYNC_REQUEST_CONCURRENCY_CAP",
        "LIBRARY_SQL_WRITER_WORKER_CAP",
        "LIBRARY_DATAFRAME_WORKER_CAP",
        "LIBRARY_PORTABLE_MODEL_REPLAY_JOBS",
        "LIBRARY_REPORT_RENDER_JOBS",
        "LIBRARY_RESPECT_MLX_JOB_CAP",
    ):
        if key in router_env:
            env[key] = str(router_env[key])
    return env


def _write_env_override(path: Path, env: dict[str, str]) -> bool:
    lines = ["# Auto-managed by scripts/ops/library_upgrade_route_control.py"]
    for key, value in sorted(env.items()):
        safe_value = str(value).replace("'", "'\"'\"'")
        lines.append(f"{key}='{safe_value}'")
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _recommended_actions(plan: dict[str, Any], router_payload: dict[str, Any]) -> list[str]:
    coverage = router_payload.get("coverage") if isinstance(router_payload.get("coverage"), dict) else {}
    status_counts = plan.get("status_counts") if isinstance(plan.get("status_counts"), dict) else {}
    return ordered_unique(
        [
            "./scripts/ops/opsctl.sh library-upgrade-route --apply --json",
            "./scripts/ops/opsctl.sh library-utilization-router --apply --json",
            "./scripts/ops/opsctl.sh mlx-intelligence-router --apply --json",
            "keep dependency installs and lock rewrites out of the unattended soak unless hard_blocker_count is nonzero",
            "reconcile newer runtime versions into requirements.lock.txt after canary evidence"
            if _safe_int(coverage.get("runtime_ahead_of_lock_count"), 0)
            else "",
            "install missing locked packages during a maintenance window before enabling their owner lane"
            if _safe_int(coverage.get("missing_runtime_count"), 0)
            else "",
            "keep optional package fallback routes active while collecting and paper trading"
            if _safe_int(coverage.get("optional_fallback_active_count"), 0)
            else "",
            "defer compatibility-excluded MLX satellites until their pins support the active runtime"
            if _safe_int(status_counts.get("compatibility_excluded_optional"), 0)
            else "",
            "run runtime-throttle after route changes so caps consume the latest library contract",
        ]
    )


def _overall_status(plan: dict[str, Any], route_matrix: dict[str, Any]) -> str:
    if _safe_int(plan.get("hard_blocker_count"), 0):
        return "blocked"
    if _safe_int(route_matrix.get("degraded_route_count"), 0):
        return "degraded"
    if _safe_int(plan.get("actionable_package_count"), 0):
        return "advisory"
    return "ready"


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    lock_file: Path | None = None,
    python_bin: Path | None = None,
    installed_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    lock_path = lock_file or project_root / "config" / "requirements.lock.txt"
    python = python_bin or project_root / ".venv314" / "bin" / "python"
    installed = _load_installed_versions(installed_versions)
    lock_versions = library_utilization_router._load_lock_versions(lock_path)
    router_payload = library_utilization_router.build_payload(project_root, lock_file=lock_path, installed_versions=installed)
    mlx_plan = mlx_library_upgrade.build_payload(lock_path=lock_path, python_bin=python)
    non_mlx_rows = list(router_payload.get("package_inventory") or [])
    mlx_rows = _mlx_package_inventory(lock_versions, installed)
    routes = [_mlx_route(mlx_rows)] + list(router_payload.get("workload_routes") or [])
    package_rows = mlx_rows + non_mlx_rows
    route_matrix = _route_matrix(routes, package_rows)
    plan = _upgrade_plan(package_rows, python)
    status = _overall_status(plan, route_matrix)
    env = _recommended_env(router_payload)
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": status in {"ready", "advisory"},
        "overall_status": status,
        "lock_file": str(lock_path),
        "python_bin": str(python),
        "route_matrix": route_matrix,
        "workload_routes": routes,
        "upgrade_plan": plan,
        "library_router_summary": {
            "overall_status": router_payload.get("overall_status"),
            "ok": bool(router_payload.get("ok", False)),
            "coverage": router_payload.get("coverage") if isinstance(router_payload.get("coverage"), dict) else {},
            "runtime_caps": router_payload.get("runtime_caps") if isinstance(router_payload.get("runtime_caps"), dict) else {},
        },
        "mlx_upgrade_summary": {
            "ok": bool(mlx_plan.get("ok", False)),
            "package_count": len(mlx_plan.get("packages") or []),
            "install_command": mlx_plan.get("install_command") or [],
            "recommended_after_apply": mlx_plan.get("recommended_after_apply") or [],
        },
        "recommended_runtime_env": env,
        "control_contract": {
            "routes_all_known_libraries": bool(route_matrix.get("mapped_package_ratio") == 1.0),
            "dependency_mutation_during_soak": "disabled",
            "upgrade_policy": "plan_and_route_first_apply_package_changes_only_in_maintenance_window",
            "default_ml_backend": "mlx",
            "optional_fallback_policy": "declared_fallback_libraries_keep_routes_available_when_optional_pins_are_missing",
            "lock_reconciliation_policy": "newer_runtime_versions_require canary_then_lock_update_not_soak_failure",
        },
        "recommended_actions": _recommended_actions(plan, router_payload),
        "artifact_paths": {
            "json": str(DEFAULT_OUT_PATH),
            "external_context": str(DEFAULT_EXTERNAL_CONTEXT_PATH),
            "markdown": str(DEFAULT_MARKDOWN_PATH),
            "env_override": str(DEFAULT_OVERRIDE_PATH),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    matrix = payload.get("route_matrix") if isinstance(payload.get("route_matrix"), dict) else {}
    plan = payload.get("upgrade_plan") if isinstance(payload.get("upgrade_plan"), dict) else {}
    lines = [
        "# Library Upgrade Route Control",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc', '')}`",
        f"Overall status: `{payload.get('overall_status', '')}`",
        "",
        "## Route Matrix",
        "",
        f"- Packages: `{matrix.get('package_count', 0)}`",
        f"- Routes: `{matrix.get('route_count', 0)}`",
        f"- Mapped package ratio: `{matrix.get('mapped_package_ratio', 0.0)}`",
        f"- Blocked routes: `{matrix.get('blocked_route_count', 0)}`",
        f"- Degraded routes: `{matrix.get('degraded_route_count', 0)}`",
        "",
        "## Upgrade Plan",
        "",
        f"- Mode: `{plan.get('mode', '')}`",
        f"- Actionable packages: `{plan.get('actionable_package_count', 0)}`",
        f"- Hard blockers: `{plan.get('hard_blocker_count', 0)}`",
        f"- Dependency mutation during soak: `{plan.get('soak_dependency_mutation_allowed', False)}`",
        "",
        "## Recommended Actions",
        "",
    ]
    for action in payload.get("recommended_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict[str, Any],
    *,
    out_path: Path = DEFAULT_OUT_PATH,
    external_context_path: Path = DEFAULT_EXTERNAL_CONTEXT_PATH,
    markdown_path: Path = DEFAULT_MARKDOWN_PATH,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
    apply: bool = False,
) -> dict[str, Any]:
    apply_result = {"applied": False, "override_path": str(override_path), "override_changed": False}
    if apply:
        env = payload.get("recommended_runtime_env") if isinstance(payload.get("recommended_runtime_env"), dict) else {}
        apply_result = {
            "applied": True,
            "override_path": str(override_path),
            "override_changed": _write_env_override(override_path, {str(key): str(value) for key, value in env.items()}),
            "env_override_count": len(env),
            "dependency_mutation_ran": False,
        }
    payload["apply_result"] = apply_result
    write_payload(out_path, payload)
    write_payload(external_context_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return apply_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Route installed libraries and produce a soak-safe upgrade/reconciliation plan.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK))
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--external-context-file", default=str(DEFAULT_EXTERNAL_CONTEXT_PATH))
    parser.add_argument("--markdown-file", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(
        project_root,
        lock_file=Path(args.lock_file).expanduser(),
        python_bin=Path(args.python_bin).expanduser(),
    )
    write_outputs(
        payload,
        out_path=Path(args.out_file).expanduser(),
        external_context_path=Path(args.external_context_file).expanduser(),
        markdown_path=Path(args.markdown_file).expanduser(),
        override_path=Path(args.override_file).expanduser(),
        apply=args.apply,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        matrix = payload.get("route_matrix") if isinstance(payload.get("route_matrix"), dict) else {}
        plan = payload.get("upgrade_plan") if isinstance(payload.get("upgrade_plan"), dict) else {}
        print(
            "library_upgrade_route_control "
            f"status={payload.get('overall_status', '')} "
            f"routes={_safe_int(matrix.get('route_count'), 0)} "
            f"actionable={_safe_int(plan.get('actionable_package_count'), 0)} "
            f"hard_blockers={_safe_int(plan.get('hard_blocker_count'), 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
