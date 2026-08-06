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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload
    from scripts.ops import infrabot_adaptive_governor, production_excellence_control, production_quality_control, production_quality_slo_guard
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload
    from . import infrabot_adaptive_governor, production_excellence_control, production_quality_control, production_quality_slo_guard


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "production_hardening_watch_latest.json"
SCHEMA_VERSION = 1


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _status_from(*, quality: dict[str, Any], slo: dict[str, Any], executed: bool) -> str:
    if _safe_int(slo.get("breach_count"), 0) > 0:
        return "repairing" if executed else "blocked"
    if _safe_int(slo.get("warning_count"), 0) > 0:
        return "repairing" if executed else "degraded"
    if _safe_int(slo.get("active_lane_count"), 0) > 0:
        return "repairing" if executed else "watch"
    if str(quality.get("overall_status") or "").strip().lower() == "ready":
        return "ready"
    return "coordinating"


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    execute_safe_repairs: bool = False,
    execute_on_watch: bool = False,
    max_actions: int = 8,
    max_execute_actions: int = 2,
    command_timeout_seconds: int = 240,
) -> dict[str, Any]:
    quality = production_quality_control.build_payload(
        project_root,
        refresh_contract=False,
        apply=apply,
        execute_safe_repairs=False,
        max_actions=max_actions,
        max_execute_actions=max_execute_actions,
        command_timeout_seconds=command_timeout_seconds,
    )
    slo = production_quality_slo_guard.build_payload(project_root, refresh_quality=False, apply=apply)
    excellence = production_excellence_control.build_payload(project_root)
    evidence_refresh = load_json(project_root / "governance" / "health" / "readiness_evidence_refresh_latest.json")
    evidence_accrual = load_json(project_root / "governance" / "health" / "readiness_evidence_accrual_latest.json")
    blocker_rollup = load_json(project_root / "governance" / "health" / "readiness_blocker_rollup_latest.json")
    if apply:
        write_payload(
            project_root / "governance" / "health" / "production_excellence_control_latest.json",
            excellence,
        )
    warning_count = _safe_int(slo.get("warning_count"), 0)
    breach_count = _safe_int(slo.get("breach_count"), 0)
    active_lane_count = _safe_int(slo.get("active_lane_count"), 0)
    execute_due = bool(execute_safe_repairs and (execute_on_watch or warning_count > 0 or breach_count > 0))
    governor = infrabot_adaptive_governor.build_payload(
        project_root,
        apply=apply,
        refresh_needs=False,
        max_actions=max_actions,
        execute_safe_repairs=execute_due,
        max_execute_actions=max_execute_actions,
        command_timeout_seconds=command_timeout_seconds,
    )
    execution = _as_dict(_as_dict(governor.get("apply_result")).get("safe_repair_execution"))
    executed_count = _safe_int(execution.get("executed_count"), 0)
    overall_status = _status_from(quality=quality, slo=slo, executed=executed_count > 0)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "overall_status": overall_status,
        "ok": overall_status == "ready",
        "source": "production_hardening_watch",
        "live_execution_authority": False,
        "safe_apply_only": True,
        "quality": {
            "overall_status": quality.get("overall_status"),
            "active_lane_count": quality.get("active_lane_count", 0),
            "active_lane_ids": [str(row.get("lane_id") or "") for row in _as_list(quality.get("active_lanes")) if isinstance(row, dict)],
        },
        "slo": {
            "overall_status": slo.get("overall_status"),
            "active_lane_count": active_lane_count,
            "warning_count": warning_count,
            "breach_count": breach_count,
            "warning_lane_ids": [str(row.get("lane_id") or "") for row in _as_list(slo.get("warning_lanes")) if isinstance(row, dict)],
            "breached_lane_ids": [str(row.get("lane_id") or "") for row in _as_list(slo.get("breached_lanes")) if isinstance(row, dict)],
        },
        "production_excellence": {
            "overall_status": excellence.get("overall_status"),
            "overall_grade": excellence.get("overall_grade"),
            "overall_score": excellence.get("overall_score"),
            "ready_pillar_count": excellence.get("ready_pillar_count", 0),
            "pillar_count": excellence.get("pillar_count", 10),
            "blocked_pillars": excellence.get("blocked_pillars", []),
            "candidate": excellence.get("candidate", {}),
            "live_money_consideration_ready": excellence.get("live_money_consideration_ready", False),
            "paper_runtime_impact": "none",
        },
        "readiness_evidence": {
            "refresh_status": evidence_refresh.get("overall_status"),
            "refresh_operational_failures": evidence_refresh.get("operational_failures", []),
            "accrual_status": evidence_accrual.get("overall_status"),
            "stalled_metric_ids": evidence_accrual.get("stalled_metric_ids", []),
            "unique_root_cause_count": blocker_rollup.get("unique_root_cause_count", 0),
            "root_causes": blocker_rollup.get("root_causes", []),
        },
        "governor": {
            "overall_status": governor.get("overall_status"),
            "action_counts": _as_dict(_as_dict(governor.get("adaptive_policy_router")).get("action_counts")),
            "recommended_commands": _as_list(_as_dict(governor.get("adaptive_policy_router")).get("recommended_commands")),
            "safety_guard": _as_dict(governor.get("safety_guard")),
        },
        "execution_policy": {
            "execute_safe_repairs_requested": bool(execute_safe_repairs),
            "execute_on_watch": bool(execute_on_watch),
            "execute_due": execute_due,
            "execute_trigger": "watch" if execute_due and execute_on_watch else "breach" if breach_count > 0 else "warning" if warning_count > 0 else "none",
            "max_actions": max(int(max_actions), 1),
            "max_execute_actions": max(int(max_execute_actions), 0),
            "command_timeout_seconds": max(int(command_timeout_seconds), 30),
            "safe_repairs_use_governor_exact_allowlist": True,
            "governor_refresh_needs": False,
            "quality_refresh_contract": False,
        },
        "execution_result": execution,
        "repair_execution_triggered": execute_due,
        "repair_execution_attempted_count": executed_count,
        "control_contract": {
            "scheduled_watch_safe_for_launchd": True,
            "default_mode_publish_only_until_warning_or_breach": True,
            "no_live_execution_authority": True,
            "no_training_launch_authority": True,
            "no_competing_sqlite_writer_authority": True,
            "no_source_registry_refresh": True,
            "uses_published_contracts_for_scheduled_watch": True,
            "bounded_serialized_evidence_refresh_precedes_scheduled_watch": True,
            "repeated_failures_backed_off_by_governor_self_healing": True,
        },
        "recommended_actions": [
            "install production hardening watch under launchd for unattended soak coverage",
            "keep default execution gated until warning or breach unless operator chooses execute-on-watch",
            "use production-quality-slo state to distinguish fresh issues from recurring production blockers",
            "keep live orders disabled until live-canary readiness and production-quality SLO both clear",
            "treat incomplete ten-pillar evidence as live-money debt without degrading a healthy paper soak",
        ],
    }
    if apply:
        write_payload(project_root / "governance" / "health" / DEFAULT_OUT_PATH.name, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Continuous production-hardening watch for unattended soak quality.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--apply", action="store_true", help="Publish production_hardening_watch_latest.json and supporting contracts.")
    parser.add_argument("--execute-safe-repairs", action="store_true", help="Delegate safe repairs to the infrabot governor when SLO warning/breach gates allow it.")
    parser.add_argument("--execute-on-watch", action="store_true", help="Allow safe repair delegation even before SLO warning/breach.")
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--max-execute-actions", type=int, default=2)
    parser.add_argument("--command-timeout-seconds", type=int, default=240)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.execute_on_watch and not args.execute_safe_repairs:
        parser.error("--execute-on-watch requires --execute-safe-repairs")

    payload = build_payload(
        args.project_root.resolve(),
        apply=args.apply,
        execute_safe_repairs=args.execute_safe_repairs,
        execute_on_watch=args.execute_on_watch,
        max_actions=args.max_actions,
        max_execute_actions=args.max_execute_actions,
        command_timeout_seconds=args.command_timeout_seconds,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "production_hardening_watch "
            f"status={payload['overall_status']} "
            f"active_lanes={payload['slo']['active_lane_count']} "
            f"warnings={payload['slo']['warning_count']} "
            f"breaches={payload['slo']['breach_count']} "
            f"executed={payload['repair_execution_attempted_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
