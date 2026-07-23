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
    from scripts.ops import infrabot_adaptive_governor, live_canary_readiness_contract
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
    from . import infrabot_adaptive_governor, live_canary_readiness_contract


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "production_quality_control_latest.json"
LIVE_READINESS_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "live_canary_readiness_contract_latest.json"
SCHEMA_VERSION = 1


def _cmd(*parts: str) -> list[str]:
    return ["./scripts/ops/opsctl.sh", *parts]


LANE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "lane_id": "raw_profitability_recovery",
        "title": "Raw profitability recovery",
        "gate_id": "raw_profitability_posture",
        "severity": "critical",
        "owner_capabilities": ["paper_performance_refresh", "paper_profitability_control"],
        "commands": [
            _cmd("paper-performance", "--json"),
            _cmd("paper-profitability-control", "--apply", "--json"),
        ],
        "stop_when": "raw_profitability_posture gate is ready and raw profitability grade is A or better.",
        "expected_impact": "Refreshes paper-performance evidence and reapplies weak-profile containment before any promotion or live-money decision.",
    },
    {
        "lane_id": "paper_trading_continuity",
        "title": "Sleeve paper-trading continuity",
        "gate_id": "sleeve_paper_trading_continuity",
        "severity": "critical",
        "owner_capabilities": [
            "paper_execution_truth_layer",
            "runtime_paper_regression_guard",
            "paper_ramp_guard",
            "global_halt_refresh",
        ],
        "commands": [
            _cmd("paper-truth", "--json"),
            _cmd("runtime-paper-regression-guard", "--json"),
            _cmd("global-halt-refresh", "--json"),
            _cmd("paper-400-ramp", "--apply", "--json"),
        ],
        "stop_when": "paper truth, runtime paper guard, and paper ramp are ready with no unexplained sleeve dropouts.",
        "expected_impact": "Reconciles paper execution truth, halt state, and ramp arming so eligible sleeves stay paper trading.",
    },
    {
        "lane_id": "auth_token_continuity",
        "title": "Auth and token continuity",
        "gate_id": "auth_token_continuity",
        "severity": "critical",
        "owner_capabilities": ["broker_auth_supervisor", "global_halt_refresh", "paper_ramp_guard"],
        "commands": [
            _cmd("schwab-auth-supervisor", "--apply", "--json"),
            _cmd("auth-lease", "--json"),
            _cmd("global-halt-refresh", "--json"),
            _cmd("paper-400-ramp", "--apply", "--json"),
        ],
        "stop_when": "broker readiness is true, Schwab/auth lease are ready, and token lease is above the proactive floor.",
        "expected_impact": "Refreshes auth evidence before token drift can pause paper trading during the soak.",
    },
    {
        "lane_id": "storage_pressure_clean",
        "title": "Storage pressure clearance",
        "gate_id": "storage_pressure_clean",
        "severity": "critical",
        "owner_capabilities": [
            "writer_cycle_coordinator",
            "storage_backpressure_autopilot",
            "external_backlog_drain_handoff",
        ],
        "commands": [
            _cmd("writer-cycle-coordinator", "--apply", "--handoff-only", "--json"),
            _cmd("storage-backpressure-autopilot", "--apply", "--quick-bounded", "--json"),
            _cmd("external-backlog-drain", "--apply", "--follow-through", "--poll-seconds", "5", "--wait-timeout-seconds", "45", "--json"),
            _cmd("health-gates", "--json"),
        ],
        "stop_when": "ingestion storage control is ready, pressure index is below policy, and pending lines are below policy.",
        "expected_impact": "Coordinates single-writer handoff and bounded backlog drainage instead of launching competing cleanup work.",
    },
    {
        "lane_id": "promotion_paper_freshness",
        "title": "Promotion and paper gate freshness",
        "gate_id": "promotion_paper_gate_freshness",
        "severity": "high",
        "owner_capabilities": [
            "daily_verify_auto_remediation",
            "promotion_quality_gate",
            "paper_performance_refresh",
            "runtime_paper_regression_guard",
        ],
        "commands": [
            _cmd("health-gates", "--json"),
            _cmd("promotion-quality-gate", "--json"),
            _cmd("promotion-autopilot", "--json"),
            _cmd("paper-performance", "--json"),
            _cmd("runtime-paper-regression-guard", "--json"),
            _cmd("daily-verify-remediation", "--apply", "--json"),
        ],
        "stop_when": "health, promotion, paper-performance, and runtime-paper artifacts are fresh and ready.",
        "expected_impact": "Turns stale latest-artifact warnings into a concrete refresh path before they block unattended soak evidence.",
    },
    {
        "lane_id": "source_and_ci_integrity",
        "title": "Source and CI integrity",
        "gate_ids": ["runtime_source_mutation_guard", "ci_production_guardrails"],
        "severity": "critical",
        "owner_capabilities": ["source_mutation_guard", "production_flow_smoke"],
        "commands": [
            _cmd("source-mutation-guard", "--check-clean", "--json"),
            _cmd("production-flow-smoke", "--json"),
        ],
        "stop_when": "source mutation guard and production-flow smoke are clean.",
        "expected_impact": "Prevents runtime or generated artifacts from mutating canonical source while CI enforces the same contract.",
    },
)


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _unique_commands(commands: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for command in commands:
        key = tuple(str(item) for item in command)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(key))
    return out


def _read_or_build_live_readiness(project_root: Path, *, refresh_contract: bool, apply: bool) -> dict[str, Any]:
    out_path = project_root / "governance" / "health" / LIVE_READINESS_OUT_PATH.name
    if refresh_contract:
        payload = live_canary_readiness_contract.build_payload(project_root, out_path=out_path)
        if apply:
            write_payload(out_path, payload)
        return payload
    payload = load_json(out_path)
    if payload:
        return payload
    return live_canary_readiness_contract.build_payload(project_root, out_path=out_path)


def _gate_index(readiness: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for gate in _as_list(readiness.get("gates")):
        if isinstance(gate, dict):
            gate_id = str(gate.get("gate_id") or "").strip()
            if gate_id:
                rows[gate_id] = gate
    return rows


def _lane_blocking_reasons(lane: dict[str, Any], readiness: dict[str, Any], gates: dict[str, dict[str, Any]]) -> list[str]:
    top_blockers = [str(item) for item in _as_list(readiness.get("blockers")) if str(item or "").strip()]
    gate_ids = [str(lane.get("gate_id") or "").strip(), *[str(item).strip() for item in _as_list(lane.get("gate_ids"))]]
    reasons: list[str] = []
    for gate_id in [item for item in gate_ids if item]:
        gate = gates.get(gate_id, {})
        if f"{gate_id}_blocked" in top_blockers or (gate and not bool(gate.get("ready", False))):
            reasons.append(f"{gate_id}_blocked")
        reasons.extend(str(item) for item in _as_list(gate.get("blockers")) if str(item or "").strip())
    return ordered_unique(reasons)


def _active_lanes(readiness: dict[str, Any]) -> list[dict[str, Any]]:
    gates = _gate_index(readiness)
    lanes: list[dict[str, Any]] = []
    for definition in LANE_DEFINITIONS:
        reasons = _lane_blocking_reasons(definition, readiness, gates)
        if not reasons:
            continue
        lane = {
            "lane_id": definition["lane_id"],
            "title": definition["title"],
            "gate_ids": ordered_unique(
                [
                    str(definition.get("gate_id") or ""),
                    *[str(item) for item in _as_list(definition.get("gate_ids"))],
                ]
            ),
            "severity": definition["severity"],
            "blocking_reasons": reasons,
            "owner_capabilities": list(definition["owner_capabilities"]),
            "commands": list(definition["commands"]),
            "safe_under_pressure": True,
            "live_execution_authority": False,
            "stop_when": definition["stop_when"],
            "expected_impact": definition["expected_impact"],
        }
        lanes.append(lane)
    return lanes


def _status_for(readiness: dict[str, Any], lanes: list[dict[str, Any]]) -> str:
    if bool(readiness.get("live_canary_money_ready", False)):
        return "ready"
    if any(str(lane.get("severity")) == "critical" for lane in lanes):
        return "blocked"
    if lanes:
        return "coordinating"
    return "waiting_for_sustained_window"


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    refresh_contract: bool = False,
    apply: bool = False,
    execute_safe_repairs: bool = False,
    max_actions: int = 8,
    max_execute_actions: int = 3,
    command_timeout_seconds: int = 300,
) -> dict[str, Any]:
    readiness = _read_or_build_live_readiness(project_root, refresh_contract=refresh_contract, apply=apply)
    lanes = _active_lanes(readiness)
    ordered_commands = _unique_commands([command for lane in lanes for command in _as_list(lane.get("commands")) if isinstance(command, list)])
    governor_command = _cmd(
        "infrabot-adaptive-governor",
        "--apply",
        "--execute-safe-repairs",
        "--max-actions",
        str(max_actions),
        "--max-execute-actions",
        str(max_execute_actions),
        "--command-timeout-seconds",
        str(command_timeout_seconds),
        "--json",
    )
    overall_status = _status_for(readiness, lanes)
    checks = {
        "live_execution_authority_false": True,
        "safe_apply_only": True,
        "live_canary_contract_present": bool(readiness),
        "all_active_lanes_have_commands": all(bool(lane.get("commands")) for lane in lanes),
        "all_active_lanes_have_stop_conditions": all(bool(str(lane.get("stop_when") or "").strip()) for lane in lanes),
        "governor_safe_execution_path_declared": bool(governor_command),
    }

    execution_result: dict[str, Any] = {}
    if execute_safe_repairs:
        execution_payload = infrabot_adaptive_governor.build_payload(
            project_root,
            apply=True,
            max_actions=max_actions,
            execute_safe_repairs=True,
            max_execute_actions=max_execute_actions,
            command_timeout_seconds=command_timeout_seconds,
        )
        execution_result = _as_dict(_as_dict(execution_payload.get("apply_result")).get("safe_repair_execution"))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "overall_status": overall_status,
        "ok": overall_status == "ready",
        "source": "production_quality_control",
        "live_execution_authority": False,
        "safe_apply_only": True,
        "live_orders_must_remain_disabled": not bool(readiness.get("live_canary_money_ready", False)),
        "live_canary_readiness": {
            "overall_status": readiness.get("overall_status"),
            "live_canary_money_ready": bool(readiness.get("live_canary_money_ready", False)),
            "ready_gate_count": readiness.get("ready_gate_count", 0),
            "gate_count": readiness.get("gate_count", 0),
            "blockers": _as_list(readiness.get("blockers")),
        },
        "active_lane_count": len(lanes),
        "active_lanes": lanes,
        "ordered_repair_commands": ordered_commands,
        "governor_safe_execution_command": governor_command,
        "quality_checks": checks,
        "production_contract": {
            "operator_goal": "production_level_quality_before_live_canary_money",
            "repair_bias": "deterministic_safe_repairs_before_broad_fanout",
            "runtime_source_mutation_allowed": False,
            "manual_live_execution_override_allowed": False,
            "repeat_blockers_require_cooldown_and_feedback": True,
        },
        "execution_result": execution_result,
        "recommended_actions": ordered_unique(
            [
                "publish production-quality-control after each live-canary readiness refresh",
                "execute safe repairs only through infrabot-adaptive-governor exact allowlist",
                "keep live orders disabled while any active production-quality lane remains",
                "rerun live-canary-readiness after safe repairs complete",
            ]
        ),
    }
    if apply:
        write_payload(project_root / "governance" / "health" / DEFAULT_OUT_PATH.name, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish the production-quality control plane for live-canary blockers.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--apply", action="store_true", help="Write production_quality_control_latest.json.")
    parser.add_argument("--refresh-contract", action="store_true", help="Rebuild live_canary_readiness_contract_latest.json before planning.")
    parser.add_argument("--execute-safe-repairs", action="store_true", help="Delegate exact allowlisted safe repairs to infrabot-adaptive-governor.")
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--max-execute-actions", type=int, default=3)
    parser.add_argument("--command-timeout-seconds", type=int, default=300)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.execute_safe_repairs and not args.apply:
        parser.error("--execute-safe-repairs requires --apply")

    payload = build_payload(
        args.project_root.resolve(),
        refresh_contract=args.refresh_contract,
        apply=args.apply,
        execute_safe_repairs=args.execute_safe_repairs,
        max_actions=args.max_actions,
        max_execute_actions=args.max_execute_actions,
        command_timeout_seconds=args.command_timeout_seconds,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "production_quality_control "
            f"status={payload['overall_status']} "
            f"active_lanes={payload['active_lane_count']} "
            f"live_ready={int(not payload['live_orders_must_remain_disabled'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
