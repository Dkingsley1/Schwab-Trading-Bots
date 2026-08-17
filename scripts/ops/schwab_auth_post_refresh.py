#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.provider_access_guard import mark_provider_recovered
    from scripts.ops.long_runtime_common import iso_now, write_payload
else:
    from core.provider_access_guard import mark_provider_recovered
    from .long_runtime_common import PROJECT_ROOT, iso_now, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "schwab_auth_post_refresh_latest.json"
PAPER_ONLY_ENV = {
    "MARKET_DATA_ONLY": "1",
    "ALLOW_ORDER_EXECUTION": "0",
    "TOP_BOT_ENABLE_LIVE_EXECUTION": "0",
    "EXECUTION_LANE_LIVE_ENABLED": "0",
    "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR": "0",
    "BOT_LIVE_MONEY_LOCKED_DURING_SOAK": "1",
}


def _dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _parse_payload(stdout: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(stdout or "").splitlines() if line.strip()]):
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _run_step(name: str, cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(PAPER_ONLY_ENV)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = int(proc.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        rc = 124
        timed_out = True
    return {
        "name": name,
        "cmd": cmd,
        "rc": rc,
        "timed_out": timed_out,
        "payload": _parse_payload(stdout),
        "stdout_tail": "\n".join(stdout.splitlines()[-6:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-6:]),
    }


def _step_contract(name: str, payload: dict[str, Any], rc: int, timed_out: bool) -> tuple[bool, str]:
    if timed_out:
        return False, "timeout"
    if name == "token_guard":
        broker = _dict(payload.get("broker_readiness"))
        ok = bool(payload.get("ok", False) and broker.get("ready_for_open", False))
        return ok, "token_and_broker_ready" if ok else "token_or_broker_not_ready"
    if name == "auth_lease":
        status = str(payload.get("overall_status") or "").strip().lower()
        state = str(payload.get("lease_state") or "").strip().lower()
        ok = status in {"ready", "degraded"} and state not in {"critical", "blocked", ""}
        return ok, f"auth_lease_{state or status or 'missing'}"
    if name == "account_snapshot":
        ok = bool(payload.get("ok", False) and payload.get("broker_truth_ok", False))
        return ok, "broker_account_truth_ready" if ok else "broker_account_truth_not_ready"
    if name == "covered_call_watch":
        ok = bool(payload) and int(rc) == 0
        return ok, "covered_call_context_refreshed" if ok else "covered_call_context_refresh_failed"
    if name == "account_position_study":
        ok = bool(payload.get("ok", False))
        return ok, "account_position_awareness_ready" if ok else "account_position_awareness_not_ready"
    if name == "position_opportunity_watch":
        ok = bool(payload.get("ok", False))
        return ok, "position_opportunity_watch_ready" if ok else "position_opportunity_watch_not_ready"
    if name == "sleeve_allocator":
        ok = bool(payload.get("target_weights")) and int(rc) == 0
        return ok, "sleeve_allocator_refreshed" if ok else "sleeve_allocator_refresh_failed"
    if name == "portfolio_risk_ledger":
        ok = bool(payload.get("limits")) and int(rc) == 0
        return ok, "portfolio_risk_refreshed" if ok else "portfolio_risk_refresh_failed"
    if name == "portfolio_capacity_curves":
        status = str(payload.get("overall_status") or "").strip().lower()
        ok = bool(payload) and int(rc) == 0 and status in {"ready", "degraded"}
        return ok, "portfolio_capacity_curves_refreshed" if ok else "portfolio_capacity_curves_refresh_failed"
    if name == "portfolio_allocator":
        ok = bool(payload.get("ok", False)) and int(rc) == 0
        return ok, "portfolio_allocator_refreshed" if ok else "portfolio_allocator_refresh_failed"
    if name == "account_buildout_plan":
        status = str(payload.get("overall_status") or "").strip().lower()
        ok = bool(payload.get("ok", False)) and status in {"ready", "watch"}
        return ok, f"account_buildout_plan_{status or 'missing'}"
    if name == "paper_truth":
        status = str(payload.get("overall_status") or "").strip().lower()
        ok = bool(payload) and int(rc) in {0, 2}
        return ok, f"paper_truth_{status or 'missing'}"
    return int(rc) == 0, "command_completed" if int(rc) == 0 else "command_failed"


def _step_summary(step: dict[str, Any]) -> dict[str, Any]:
    payload = _dict(step.get("payload"))
    name = str(step.get("name") or "")
    summary: dict[str, Any] = {
        "name": name,
        "rc": int(step.get("rc", 1) or 0),
        "timed_out": bool(step.get("timed_out", False)),
        "contract_ok": bool(step.get("contract_ok", False)),
        "contract_reason": str(step.get("contract_reason") or ""),
    }
    if name == "token_guard":
        summary.update(
            token_ready=bool(payload.get("token_ready_after", False)),
            broker_ready=bool(_dict(payload.get("broker_readiness")).get("ready_for_open", False)),
        )
    elif name == "auth_lease":
        summary.update(overall_status=payload.get("overall_status"), lease_state=payload.get("lease_state"))
    elif name == "account_snapshot":
        summary.update(
            account_count=int(payload.get("account_count", 0) or 0),
            position_rows=int(payload.get("position_rows", 0) or 0),
            broker_truth_ok=bool(payload.get("broker_truth_ok", False)),
            broker_truth_grade=str(payload.get("broker_truth_v2_grade") or ""),
        )
    elif name == "account_position_study":
        summary.update(
            account_count=int(payload.get("account_count", 0) or 0),
            position_count=int(payload.get("position_count", 0) or 0),
            underlying_count=int(payload.get("underlying_count", 0) or 0),
        )
    elif name == "position_opportunity_watch":
        summary.update(
            observed_underlyings=int(payload.get("observed_underlying_count", 0) or 0),
            candidate_count=int(payload.get("candidate_count", 0) or 0),
            abstention_count=int(payload.get("abstention_count", 0) or 0),
        )
    elif name == "sleeve_allocator":
        summary.update(
            broker=str(payload.get("broker") or ""),
            gross_risk_budget=float(payload.get("gross_risk_budget", 0.0) or 0.0),
            sleeve_count=len(_dict(payload.get("target_weights"))),
        )
    elif name == "portfolio_risk_ledger":
        limits = _dict(payload.get("limits"))
        summary.update(
            risk_level=str(payload.get("risk_level") or ""),
            risk_score=float(payload.get("risk_score", 0.0) or 0.0),
            gross_exposure_cap=float(limits.get("gross_exposure_cap", 0.0) or 0.0),
            max_single_symbol_share=float(limits.get("max_single_symbol_share", 0.0) or 0.0),
        )
    elif name == "portfolio_capacity_curves":
        curve_summary = _dict(payload.get("summary"))
        summary.update(
            overall_status=str(payload.get("overall_status") or ""),
            curve_count=int(curve_summary.get("curve_count", 0) or 0),
            allocator_ready=bool(curve_summary.get("allocator_ready", False)),
        )
    elif name == "portfolio_allocator":
        allocator_summary = _dict(payload.get("summary"))
        summary.update(
            overall_status=str(payload.get("overall_status") or ""),
            approved_intent_count=int(allocator_summary.get("approved_intent_count", 0) or 0),
            gross_budget=float(allocator_summary.get("gross_budget", 0.0) or 0.0),
        )
    elif name == "account_buildout_plan":
        summary.update(
            overall_status=str(payload.get("overall_status") or ""),
            plan_state=str(payload.get("plan_state") or ""),
            buildout_ready=bool(payload.get("buildout_ready", False)),
            account_count=int(payload.get("account_count", 0) or 0),
            action_count=int(payload.get("action_count", 0) or 0),
            review_count=int(payload.get("review_count", 0) or 0),
        )
    elif name == "paper_truth":
        summary.update(
            overall_status=str(payload.get("overall_status") or ""),
            grade=str(payload.get("grade") or ""),
            score=float(payload.get("score", 0.0) or 0.0),
            raw_metric_score=float(payload.get("raw_metric_score", 0.0) or 0.0),
            operational_conformance_complete=bool(payload.get("operational_conformance_complete", False)),
            promotion_ready=bool(payload.get("promotion_ready", False)),
            failed_checks=list(payload.get("failed_checks") or []),
        )
    return summary


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    runner: Callable[..., dict[str, Any]] = _run_step,
) -> dict[str, Any]:
    project_root = Path(project_root).expanduser().resolve()
    opsctl = project_root / "scripts" / "ops" / "opsctl.sh"
    commands = [
        ("token_guard", [str(opsctl), "token-refresh", "--json"], 120, True),
        ("auth_lease", [str(opsctl), "auth-lease", "--json"], 60, True),
        ("account_snapshot", [str(opsctl), "schwab-account-snapshot-refresh", "--skip-derived", "--json"], 180, True),
        ("covered_call_watch", [str(opsctl), "covered-call-roll-watch", "--json"], 60, True),
        ("account_position_study", [str(opsctl), "account-position-study", "--json"], 120, True),
        ("position_opportunity_watch", [str(opsctl), "position-opportunity-watch", "--json"], 120, True),
        ("sleeve_allocator", [str(opsctl), "sleeve-allocator", "--json"], 120, True),
        ("portfolio_risk_ledger", [str(opsctl), "portfolio-risk-ledger", "--json"], 120, True),
        ("portfolio_capacity_curves", [str(opsctl), "portfolio-capacity-curves", "--json"], 120, True),
        ("portfolio_allocator", [str(opsctl), "portfolio-allocator", "--json"], 120, True),
        ("account_buildout_plan", [str(opsctl), "account-buildout-plan", "--json"], 120, True),
        (
            "paper_truth",
            [sys.executable, str(project_root / "scripts" / "ops" / "paper_execution_truth_layer.py"), "--json"],
            180,
            False,
        ),
    ]

    steps: list[dict[str, Any]] = []
    hard_failure = ""
    provider_recovery: dict[str, Any] = {}
    for name, cmd, timeout_sec, required in commands:
        step = runner(name, cmd, cwd=project_root, timeout_sec=timeout_sec)
        payload = _dict(step.get("payload"))
        contract_ok, contract_reason = _step_contract(
            name,
            payload,
            int(step.get("rc", 1) or 0),
            bool(step.get("timed_out", False)),
        )
        step["contract_ok"] = contract_ok
        step["contract_reason"] = contract_reason
        steps.append(step)
        if name == "account_snapshot" and contract_ok:
            provider_recovery = mark_provider_recovered(
                project_root,
                "schwab",
                evidence="verified_post_auth_account_snapshot_and_broker_truth",
                force=True,
            )
        if required and not contract_ok:
            hard_failure = f"{name}:{contract_reason}"
            break

    paper_payload = _dict(steps[-1].get("payload")) if steps and steps[-1].get("name") == "paper_truth" else {}
    paper_ready = bool(
        paper_payload.get("ok", False)
        and str(paper_payload.get("overall_status") or "").strip().lower() == "ready"
    )
    refresh_completed = len(steps) == len(commands) and not hard_failure
    paper_step = steps[-1] if steps and steps[-1].get("name") == "paper_truth" else {}
    paper_refresh_ok = bool(paper_step.get("contract_ok", False))
    if hard_failure:
        status = "blocked"
    elif not refresh_completed or not paper_refresh_ok:
        status = "degraded"
    else:
        status = "ready"

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": status == "ready",
        "overall_status": status,
        "refresh_completed": bool(refresh_completed),
        "hard_failure": hard_failure,
        "paper_truth_ready": bool(paper_ready),
        "downstream_attention": (
            []
            if paper_ready
            else [
                f"paper_truth_{str(paper_payload.get('overall_status') or 'not_ready').strip().lower()}",
                *[str(item) for item in (paper_payload.get("failed_checks") or []) if str(item)],
                *[str(item) for item in (paper_payload.get("grade_blocking_warnings") or []) if str(item)],
            ]
        ),
        "provider_recovery": {
            "state": provider_recovery.get("state"),
            "active": bool(provider_recovery.get("active", False)),
            "recovery_evidence": provider_recovery.get("recovery_evidence"),
            "forced_recovery_from_verified_request": bool(
                provider_recovery.get("forced_recovery_from_verified_request", False)
            ),
        },
        "steps": [_step_summary(step) for step in steps],
        "safety_contract": {
            "market_data_only": True,
            "paper_execution_allowed": True,
            "live_execution_allowed": False,
            "opens_browser": False,
            "interactive_oauth_allowed": False,
            "account_snapshot_is_read_only": True,
            "position_opportunities_do_not_publish_execution_intents": True,
            "account_buildout_is_advisory_paper_only": True,
            "account_buildout_does_not_publish_execution_intents": True,
        },
        "regression_contract": {
            "refresh_dependency_order": [name for name, _, _, _ in commands],
            "stop_before_account_access_when_auth_not_ready": True,
            "paper_truth_rebuilt_after_account_and_position_truth": True,
            "paper_truth_evaluator_is_non_recursive": True,
            "stale_account_and_broker_truth_have_bounded_repair_owner": True,
            "downstream_paper_watch_does_not_misclassify_auth_recovery": True,
            "safe_environment": dict(PAPER_ONLY_ENV),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh Schwab auth-dependent paper truth in dependency order.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(Path(args.project_root))
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "schwab_auth_post_refresh "
            f"status={payload.get('overall_status')} "
            f"refresh_completed={int(bool(payload.get('refresh_completed')))} "
            f"paper_truth_ready={int(bool(payload.get('paper_truth_ready')))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
