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
    from scripts.ops.dependency_activation_smoke import build_payload as build_dependency_activation_smoke
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, status_rank, write_payload
    from scripts.ops.production_readiness_control import build_payload as build_production_readiness
else:
    from .dependency_activation_smoke import build_payload as build_dependency_activation_smoke
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, status_rank, write_payload
    from .production_readiness_control import build_payload as build_production_readiness


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "production_soak_enhancement_v1.json"
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "production_soak_enhancement_latest.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "exports" / "reports" / "operator" / "production_soak_enhancement_latest.md"

BLOCKING_STATUSES = {"blocked", "critical", "failed", "error"}
GUARDED_STATUSES = {"guarded", "ready_guarded", "pending_install", "advisory", "thin", "missing", "degraded", "needs_work"}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _project_path(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else project_root / path


def _domain_by_name(readiness: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("name") or ""): row
        for row in readiness.get("domains") or []
        if isinstance(row, dict) and str(row.get("name") or "")
    }


def _domain_status(domains: dict[str, dict[str, Any]], name: str) -> str:
    row = domains.get(name) if isinstance(domains.get(name), dict) else {}
    return str(row.get("status") or "missing").strip().lower()


def _soak_status(*statuses: str) -> str:
    normalized = [str(status or "missing").strip().lower() for status in statuses]
    if any(status in BLOCKING_STATUSES for status in normalized):
        return "blocked"
    if any(status in GUARDED_STATUSES for status in normalized):
        return "guarded"
    return "ready"


def _item(
    *,
    control_number: int,
    item_id: str,
    title: str,
    status: str,
    source_domain: str,
    soak_enhancement: str,
    evidence: dict[str, Any] | None = None,
    next_actions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "control_number": int(control_number),
        "id": item_id,
        "title": title,
        "status": str(status or "guarded").strip().lower(),
        "source_domain": source_domain,
        "soak_enhancement": soak_enhancement,
        "evidence": evidence or {},
        "next_actions": ordered_unique(next_actions or []),
    }


def _configured_soak_text(config: dict[str, Any], item_id: str) -> tuple[int, str, str]:
    for row in config.get("soak_enhancements") or []:
        if not isinstance(row, dict) or str(row.get("id") or "") != item_id:
            continue
        return (
            int(row.get("control_number") or 0),
            str(row.get("source_domain") or ""),
            str(row.get("soak_enhancement") or ""),
        )
    return 0, "", ""


def _release_command(project_root: Path, config: dict[str, Any], dependency_batch: str) -> str:
    command = str(config.get("release_command") or "").strip()
    if command:
        return command
    return (
        f"./scripts/ops/opsctl.sh production-soak-enhancement --dependency-batch {dependency_batch} --apply --json --exit-zero "
        "&& ./scripts/ops/opsctl.sh production-readiness --apply --json --exit-zero "
        "&& ./.venv314/bin/python -m pytest -q tests/test_dependency_activation_smoke.py "
        "tests/test_production_readiness_control.py tests/test_production_soak_enhancement.py "
        "tests/test_library_utilization_router.py tests/test_mlx_intelligence_router.py"
    )


def _safe_profile_plan(project_root: Path, dependency_smoke: dict[str, Any], dependency_batch: str) -> dict[str, Any]:
    rows = [row for row in dependency_smoke.get("candidate_smoke_rows") or [] if isinstance(row, dict)]
    pending = [row for row in rows if str(row.get("status") or "") == "pending_install"]
    packages = [
        {
            "package": str(row.get("package") or ""),
            "module": str(row.get("module") or ""),
            "status": str(row.get("status") or ""),
            "activation_profiles": _string_list(row.get("activation_profiles")),
            "initial_activation_batch": str(row.get("initial_activation_batch") or ""),
        }
        for row in rows
    ]
    package_names = [row["package"] for row in packages if row["package"]]
    install_command = (
        "./.venv314/bin/python -m pip install --upgrade --upgrade-strategy only-if-needed "
        + " ".join(package_names)
        if package_names
        else ""
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "overall_status": "pending_install" if pending else "ready",
        "batch": dependency_batch,
        "package_count": len(packages),
        "pending_install_count": len(pending),
        "packages": packages,
        "target_lock_files": [
            str(project_root / "config" / "runtime_profiles" / "live.lock.txt"),
            str(project_root / "config" / "runtime_profiles" / "ops.lock.txt"),
            str(project_root / "config" / "requirements.lock.txt"),
        ],
        "sandbox_install_command": install_command,
        "promotion_commands": [
            f"./scripts/ops/opsctl.sh dependency-activation-smoke --batch {dependency_batch} --import-smoke --json",
            "./scripts/ops/opsctl.sh library-utilization-router --apply --json",
        ],
        "live_runtime_mutated": False,
        "policy": "stage_profile_lock_changes_only_after_sandbox_install_and_import_smoke_pass",
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path | None = None,
    dependency_batch: str = "",
) -> dict[str, Any]:
    config_path = config_path or DEFAULT_CONFIG
    config = load_json(config_path)
    dependency_batch = dependency_batch or str(config.get("default_dependency_batch") or "production_core_safe")
    readiness = build_production_readiness(project_root, dependency_batch=dependency_batch)
    dependency_smoke = build_dependency_activation_smoke(project_root, batch=dependency_batch)
    domains = _domain_by_name(readiness)
    profile_plan = _safe_profile_plan(project_root, dependency_smoke, dependency_batch)

    dep_status = _domain_status(domains, "dependency_activation_smoke_runner")
    firewall = domains.get("live_execution_risk_firewall") or {}
    firewall_evidence = firewall.get("evidence") if isinstance(firewall.get("evidence"), dict) else {}
    replay = domains.get("deterministic_replay_harness") or {}
    replay_evidence = replay.get("evidence") if isinstance(replay.get("evidence"), dict) else {}
    release_status = _domain_status(domains, "release_gates")
    redaction = domains.get("observability_redaction") or {}
    redaction_evidence = redaction.get("evidence") if isinstance(redaction.get("evidence"), dict) else {}
    rollback = domains.get("incident_and_rollback_system") or {}
    rollback_evidence = rollback.get("evidence") if isinstance(rollback.get("evidence"), dict) else {}
    cockpit_source = project_root / "scripts" / "ops" / "operator_cockpit.py"
    cockpit_status = "ready" if cockpit_source.exists() and "production_readiness_control_latest.json" in cockpit_source.read_text(encoding="utf-8", errors="ignore") else "guarded"

    item_defs = {
        "sandbox_dependency_activation": {
            "title": "Sandbox Dependency Activation",
            "status": _soak_status(dep_status),
            "evidence": {
                "dependency_batch": dependency_batch,
                "selected_candidate_count": int((dependency_smoke.get("summary") or {}).get("selected_candidate_count", 0) or 0),
                "pending_install_count": int((dependency_smoke.get("summary") or {}).get("pending_install_count", 0) or 0),
                "import_failed_count": int((dependency_smoke.get("summary") or {}).get("import_failed_count", 0) or 0),
                "does_not_install_packages": bool((dependency_smoke.get("control_contract") or {}).get("does_not_install_packages", False)),
            },
            "next_actions": dependency_smoke.get("recommended_actions") or [],
        },
        "live_submit_firewall": {
            "title": "Live Submit Firewall",
            "status": "ready" if bool(firewall_evidence.get("live_order_allowed")) else _soak_status(str(firewall.get("status") or "guarded")),
            "evidence": {
                "base_trader_hook": "core/base_trader.py:_live_place_order",
                "firewall_helper": "core/live_execution_controls.py:production_order_firewall_check",
                "execution_armed": bool(firewall_evidence.get("execution_armed", False)),
                "market_data_only": bool(firewall_evidence.get("market_data_only", True)),
                "live_order_allowed": bool(firewall_evidence.get("live_order_allowed", False)),
                "blockers": _string_list(firewall.get("blockers")),
            },
            "next_actions": firewall.get("recommended_actions") or [],
        },
        "deterministic_replay_baseline": {
            "title": "Deterministic Replay Baseline",
            "status": _soak_status(str(replay.get("status") or "missing")),
            "evidence": {
                "fingerprint_count": int(replay_evidence.get("fingerprint_count", 0) or 0),
                "fingerprint_hash": str(replay_evidence.get("fingerprint_hash") or ""),
                "baseline_present": bool(replay_evidence.get("baseline_present", False)),
                "baseline_path": str(replay_evidence.get("baseline_path") or ""),
            },
            "next_actions": replay.get("recommended_actions") or [],
        },
        "production_release_command": {
            "title": "Production Release Command",
            "status": _soak_status(release_status),
            "evidence": {
                "command": _release_command(project_root, config, dependency_batch),
                "release_gate_status": release_status,
            },
            "next_actions": domains.get("release_gates", {}).get("recommended_actions") or [],
        },
        "telemetry_redaction_canary": {
            "title": "Telemetry Redaction Canary",
            "status": _soak_status(str(redaction.get("status") or "missing")),
            "evidence": {
                "sample_count": len(redaction_evidence.get("sample_rows") or []),
                "pattern_count": int(redaction_evidence.get("pattern_count", 0) or 0),
                "sample_rows": redaction_evidence.get("sample_rows") or [],
                "enabled_by_default": bool(redaction_evidence.get("enabled_by_default", False)),
            },
            "next_actions": redaction.get("recommended_actions") or [],
        },
        "safe_profile_promotion_plan": {
            "title": "Safe Profile Promotion Plan",
            "status": _soak_status(str(profile_plan.get("overall_status") or "guarded")),
            "evidence": {
                "package_count": int(profile_plan.get("package_count", 0) or 0),
                "pending_install_count": int(profile_plan.get("pending_install_count", 0) or 0),
                "target_lock_files": profile_plan.get("target_lock_files") or [],
                "live_runtime_mutated": False,
            },
            "next_actions": profile_plan.get("promotion_commands") or [],
        },
        "rollback_drill": {
            "title": "Rollback Drill",
            "status": _soak_status(str(rollback.get("status") or "missing")),
            "evidence": {
                "rollback_manifest_path": str(rollback_evidence.get("rollback_manifest_path") or ""),
                "missing_snapshot_count": int(rollback_evidence.get("missing_snapshot_count", 0) or 0),
                "snapshot_manifest_hash": str(((rollback_evidence.get("manifest") or {}).get("snapshot_manifest_hash")) or ""),
                "rollback_commands": (rollback_evidence.get("manifest") or {}).get("rollback_commands") or [],
            },
            "next_actions": rollback.get("recommended_actions") or [],
        },
        "cockpit_operator_surface": {
            "title": "Cockpit Operator Surface",
            "status": cockpit_status,
            "evidence": {
                "operator_cockpit_source": str(cockpit_source),
                "production_readiness_surface_expected": True,
                "production_soak_surface_expected": True,
            },
            "next_actions": ["run ./scripts/ops/opsctl.sh operator-cockpit --json after soak enhancement refresh"] if cockpit_status != "ready" else [],
        },
    }

    items: list[dict[str, Any]] = []
    for item_id, row in item_defs.items():
        number, source_domain, soak_text = _configured_soak_text(config, item_id)
        items.append(
            _item(
                control_number=number or len(items) + 1,
                item_id=item_id,
                title=str(row["title"]),
                status=str(row["status"]),
                source_domain=source_domain or str(row.get("source_domain") or ""),
                soak_enhancement=soak_text,
                evidence=row.get("evidence") if isinstance(row.get("evidence"), dict) else {},
                next_actions=row.get("next_actions") if isinstance(row.get("next_actions"), list) else [],
            )
        )
    items = sorted(items, key=lambda row: int(row.get("control_number", 0) or 0))
    statuses = [str(row.get("status") or "guarded") for row in items]
    overall_status = _soak_status(*statuses)
    blockers = ordered_unique(f"{row.get('id')}:{row.get('status')}" for row in items if str(row.get("status") or "") in BLOCKING_STATUSES)
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status != "blocked",
        "overall_status": overall_status,
        "dependency_batch": dependency_batch,
        "control_count": len(items),
        "ready_control_count": sum(1 for row in items if row.get("status") == "ready"),
        "guarded_control_count": sum(1 for row in items if row.get("status") == "guarded"),
        "blocked_control_count": sum(1 for row in items if row.get("status") in BLOCKING_STATUSES),
        "controls": items,
        "blockers": blockers,
        "recommended_actions": ordered_unique(
            [
                action
                for row in items
                for action in _string_list(row.get("next_actions"))
            ]
            + [
                "keep live execution disabled during soak; use the firewall evidence as the promotion gate",
                "install production_core_safe only in a sandbox, then rerun import smoke before lock promotion",
            ]
        ),
        "production_readiness": {
            "overall_status": str(readiness.get("overall_status") or ""),
            "live_runtime_promotion_allowed": bool(readiness.get("live_runtime_promotion_allowed", False)),
            "domain_count": int(readiness.get("domain_count", 0) or 0),
            "blocked_domain_count": int(readiness.get("blocked_domain_count", 0) or 0),
            "guarded_domain_count": int(readiness.get("guarded_domain_count", 0) or 0),
        },
        "profile_promotion_plan": profile_plan,
        "control_contract": {
            "covers_controls_1_through_8": len(items) == 8,
            "live_runtime_mutated": False,
            "dependency_installation_is_separate_from_activation": True,
            "soak_can_run_with_live_orders_disabled": True,
            "status_rank": status_rank(overall_status),
        },
        "artifact_paths": {
            "json": str(DEFAULT_OUT),
            "markdown": str(DEFAULT_MARKDOWN),
            "config": str(config_path),
            **(config.get("artifact_paths") if isinstance(config.get("artifact_paths"), dict) else {}),
        },
    }


def _write_auxiliary_artifacts(project_root: Path, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    artifact_paths = config.get("artifact_paths") if isinstance(config.get("artifact_paths"), dict) else {}
    written: dict[str, str] = {}
    controls = {str(row.get("id") or ""): row for row in payload.get("controls") or [] if isinstance(row, dict)}

    replay_evidence = controls.get("deterministic_replay_baseline", {}).get("evidence") or {}
    baseline_path = Path(str(replay_evidence.get("baseline_path") or ""))
    if not str(baseline_path):
        baseline_path = _project_path(project_root, artifact_paths.get("replay_baseline") or "governance/health/production_readiness_replay_baseline.json")
    baseline = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "source": "production_soak_enhancement",
        "fingerprint_hash": str(replay_evidence.get("fingerprint_hash") or ""),
        "fingerprint_count": int(replay_evidence.get("fingerprint_count", 0) or 0),
        "policy": "baseline_is_soak_evidence_not_live_promotion",
    }
    write_payload(baseline_path, baseline)
    written["replay_baseline"] = str(baseline_path)

    telemetry_path = _project_path(project_root, artifact_paths.get("telemetry_canary") or "governance/health/telemetry_redaction_canary_latest.json")
    telemetry_evidence = controls.get("telemetry_redaction_canary", {}).get("evidence") or {}
    telemetry = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "overall_status": controls.get("telemetry_redaction_canary", {}).get("status", "guarded"),
        "sample_rows": telemetry_evidence.get("sample_rows") or [],
        "sample_count": int(telemetry_evidence.get("sample_count", 0) or 0),
        "enabled_by_default": bool(telemetry_evidence.get("enabled_by_default", False)),
        "policy": "telemetry_exporters_remain_off_or_local_until_canary_passes",
    }
    write_payload(telemetry_path, telemetry)
    written["telemetry_canary"] = str(telemetry_path)

    profile_path = _project_path(project_root, artifact_paths.get("profile_promotion_plan") or "governance/health/safe_profile_promotion_plan_latest.json")
    write_payload(profile_path, payload.get("profile_promotion_plan") if isinstance(payload.get("profile_promotion_plan"), dict) else {})
    written["profile_promotion_plan"] = str(profile_path)

    rollback_path = _project_path(project_root, artifact_paths.get("rollback_drill") or "governance/rollback/production_rollback_drill_latest.json")
    rollback_evidence = controls.get("rollback_drill", {}).get("evidence") or {}
    rollback_drill = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "overall_status": controls.get("rollback_drill", {}).get("status", "guarded"),
        "rollback_manifest_path": str(rollback_evidence.get("rollback_manifest_path") or ""),
        "snapshot_manifest_hash": str(rollback_evidence.get("snapshot_manifest_hash") or ""),
        "rollback_commands": rollback_evidence.get("rollback_commands") or [],
        "executed_commands": [],
        "dry_run_only": True,
        "policy": "rollback_drill_verifies_inputs_without_executing_recovery_commands",
    }
    write_payload(rollback_path, rollback_drill)
    written["rollback_drill"] = str(rollback_path)
    return written


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Production Soak Enhancement",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc', '')}`",
        f"Overall status: `{payload.get('overall_status', '')}`",
        f"Dependency batch: `{payload.get('dependency_batch', '')}`",
        "",
        "## Controls",
        "",
    ]
    for row in payload.get("controls") or []:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('control_number')}. {row.get('id')}`: `{row.get('status')}` - "
            f"{row.get('soak_enhancement')}"
        )
    lines.extend(["", "## Recommended Actions", ""])
    for action in payload.get("recommended_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict[str, Any],
    *,
    project_root: Path,
    config: dict[str, Any],
    out_path: Path = DEFAULT_OUT,
    markdown_path: Path = DEFAULT_MARKDOWN,
    apply: bool = False,
) -> dict[str, Any]:
    result = {"json": str(out_path), "markdown": str(markdown_path), "auxiliary_artifacts_written": {}}
    if apply:
        result["auxiliary_artifacts_written"] = _write_auxiliary_artifacts(project_root, payload, config)
    payload["write_result"] = result
    write_payload(out_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize the eight production-soak enhancers without enabling live order execution.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dependency-batch", default="")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--markdown-file", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--exit-zero", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_json(config_path)
    payload = build_payload(PROJECT_ROOT, config_path=config_path, dependency_batch=args.dependency_batch)
    write_outputs(
        payload,
        project_root=PROJECT_ROOT,
        config=config,
        out_path=Path(args.out_file),
        markdown_path=Path(args.markdown_file),
        apply=args.apply,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "production_soak_enhancement "
            f"status={payload.get('overall_status')} "
            f"controls={payload.get('control_count', 0)} "
            f"blocked={payload.get('blocked_control_count', 0)}"
        )
    return 0 if args.exit_zero or bool(payload.get("ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
