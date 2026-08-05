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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "sleeve_isolation_guard_latest.json"
NON_RUNTIME_DAILY_VERIFY_CHECKS = {
    "promotion_quality_gate",
    "retrain_schema_compatibility_guard",
}
RESOLVABLE_DAILY_VERIFY_ARTIFACTS = {
    "bot_support_owner_guard": "bot_support_owner_guard_latest.json",
    "data_source_divergence_bot": "data_source_divergence_latest.json",
    "execution_queue_stress_bot": "execution_queue_stress_latest.json",
    "snapshot_coverage_sentinel": "snapshot_coverage_latest.json",
    "new_bot_admission_guard": "new_bot_admission_guard_latest.json",
    "replay_hash_registry_guard": "replay_hash_registry_guard_latest.json",
}
SESSION_PAUSE_REASONS = {
    "weekend",
    "post_window",
    "pre_window",
    "market_closed",
    "session_gate",
    "outside_session",
}
SUSTAINED_INGESTION_ERROR_RATE = 0.80
SUSTAINED_INGESTION_MIN_REQUESTS = 12


def _daily_verify_check_resolved(project_root: Path, daily_verify: dict[str, Any], name: str) -> bool:
    key = str(name or "").strip()
    if key in NON_RUNTIME_DAILY_VERIFY_CHECKS:
        return True
    if key == "incomplete_run_recovered":
        note = str(daily_verify.get("note", "") or "").lower()
        return bool(
            "recovered_stale_progress" in note
            or (
                daily_verify.get("running") is False
                and int(daily_verify.get("completed_checks", 0) or 0) > 0
            )
        )
    artifact_name = RESOLVABLE_DAILY_VERIFY_ARTIFACTS.get(key, "")
    if not artifact_name:
        return False
    artifact = load_json(project_root / "governance" / "health" / artifact_name)
    return artifact.get("ok") is True


def _unresolved_daily_verify_checks(project_root: Path, daily_verify: dict[str, Any]) -> list[str]:
    raw_failed_checks = daily_verify.get("failed_checks") if isinstance(daily_verify.get("failed_checks"), list) else []
    unresolved: list[str] = []
    for item in raw_failed_checks:
        name = str(item or "").strip()
        if not name:
            continue
        if _daily_verify_check_resolved(project_root, daily_verify, name):
            continue
        unresolved.append(name)
    return unresolved


def _is_session_pause(row: dict[str, Any]) -> bool:
    loop_state = str(row.get("loop_state") or "").strip().lower()
    pause_reason = str(row.get("pause_reason") or "").strip().lower()
    return loop_state == "paused_session_gate" or pause_reason in SESSION_PAUSE_REASONS


def _is_isolated_pause(row: dict[str, Any]) -> bool:
    loop_state = str(row.get("loop_state") or "").strip().lower()
    pause_reason = str(row.get("pause_reason") or "").strip().lower()
    if not ("paused" in loop_state or "killswitch" in loop_state or "quarantine" in loop_state):
        return False
    if _is_session_pause(row):
        return False
    return bool(
        "killswitch" in loop_state
        or "quarantine" in loop_state
        or "anomaly" in loop_state
        or pause_reason in {"data_anomaly", "anomaly", "quarantine", "killswitch", "risk_halt"}
    )


def _is_failed_ingestion(row: dict[str, Any]) -> bool:
    loop_state = str(row.get("loop_state") or "").strip().lower()
    if loop_state != "running":
        return False
    iter_requests = int(row.get("iter_total_requests", 0) or 0)
    total_requests = int(row.get("total_request_count", 0) or 0)
    iter_error_rate = float(row.get("iter_error_rate", 0.0) or 0.0)
    total_error_rate = float(row.get("total_error_rate", 0.0) or 0.0)
    return bool(
        iter_requests > 0
        and total_requests >= SUSTAINED_INGESTION_MIN_REQUESTS
        and iter_error_rate >= SUSTAINED_INGESTION_ERROR_RATE
        and total_error_rate >= SUSTAINED_INGESTION_ERROR_RATE
    )


def _ingress_rows(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((project_root / "governance" / "health").glob("data_ingress_latest_*.json")):
        payload = load_json(path)
        if not payload:
            continue
        total_counts = payload.get("total_counts") if isinstance(payload.get("total_counts"), dict) else {}
        total_errors = int(total_counts.get("api_error", 0) or 0)
        total_successes = sum(int(total_counts.get(key, 0) or 0) for key in ("api_ok", "cache_ok", "simulate_ok"))
        total_requests = total_errors + total_successes
        rows.append(
            {
                "artifact": path.name,
                "profile": str(payload.get("profile") or ""),
                "domain": str(payload.get("domain") or ""),
                "broker": str(payload.get("broker") or ""),
                "loop_state": str(payload.get("loop_state") or ""),
                "pause_reason": str(payload.get("pause_reason") or ""),
                "iter_error_rate": float(payload.get("iter_error_rate", 0.0) or 0.0),
                "iter_total_requests": int(payload.get("iter_total_requests", 0) or 0),
                "total_request_count": total_requests,
                "total_error_count": total_errors,
                "total_error_rate": round(total_errors / total_requests, 6) if total_requests > 0 else 0.0,
            }
        )
    return rows


def build_payload(project_root: Path = PROJECT_ROOT, *, max_quarantine_events: int = 120) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    quarantine_pressure = load_json(health_root / "quarantine_pressure_latest.json")
    daily_verify = load_json(health_root / "daily_auto_verify_latest.json")
    lane_thaw = load_json(health_root / "lane_thaw_controller_latest.json")
    ingress_rows = _ingress_rows(project_root)

    isolated_lanes = [row for row in ingress_rows if _is_isolated_pause(row)]
    failed_ingestion_lanes = [row for row in ingress_rows if _is_failed_ingestion(row)]
    session_paused_lanes = [
        row
        for row in ingress_rows
        if _is_session_pause(row) and not _is_isolated_pause(row)
    ]
    running_lanes = [row for row in ingress_rows if str(row.get("loop_state") or "") == "running"]
    healthy_running_lanes = [row for row in running_lanes if row not in failed_ingestion_lanes]
    quarantine_events = int(quarantine_pressure.get("quarantine_events", 0) or 0)
    unresolved_checks = _unresolved_daily_verify_checks(project_root, daily_verify)

    overall_status = "ready"
    unhealthy_lane_count = len(isolated_lanes) + len(failed_ingestion_lanes)
    if unhealthy_lane_count >= 2 or quarantine_events > int(max_quarantine_events):
        overall_status = "blocked"
    elif unhealthy_lane_count > 0 or quarantine_events > 0:
        overall_status = "degraded"
    isolated_lane_count = len(isolated_lanes)
    running_lane_count = len(running_lanes)
    total_lane_count = max(isolated_lane_count + running_lane_count, 1)
    blast_radius_score = round((len(healthy_running_lanes) / total_lane_count) * 100.0, 2)
    thaw_candidates = lane_thaw.get("candidates") if isinstance(lane_thaw.get("candidates"), list) else []
    thaw_blocked = lane_thaw.get("blocked") if isinstance(lane_thaw.get("blocked"), list) else []
    thaw_rows = lane_thaw.get("lanes") if isinstance(lane_thaw.get("lanes"), list) else []
    systemic_guardrails = lane_thaw.get("systemic_guardrails") if isinstance(lane_thaw.get("systemic_guardrails"), dict) else {}
    release_ready_candidates = [
        row
        for row in thaw_rows
        if isinstance(row, dict)
        and str(row.get("thaw_state") or "").strip().lower() == "candidate"
        and bool(((row.get("thaw_contract") or {}).get("release_ready")))
    ]
    supervised_candidates = [
        row
        for row in release_ready_candidates
        if str(((row.get("thaw_contract") or {}).get("stage") or "")).strip().lower() == "supervised_canary"
    ]
    micro_probe_candidates = [
        row
        for row in release_ready_candidates
        if str(((row.get("thaw_contract") or {}).get("stage") or "")).strip().lower() == "micro_probe"
    ]
    operator_review_rows = [
        row
        for row in thaw_rows
        if str(((row.get("thaw_contract") or {}).get("stage") or "")).strip().lower() == "operator_review"
    ]
    systemic_halt_active = bool(systemic_guardrails.get("global_killswitch_active", False))
    systemic_risk_halts = int(systemic_guardrails.get("risk_halt_events", 0) or 0)
    systemic_snapshot_failures = int(systemic_guardrails.get("account_snapshot_failure_count", 0) or 0)
    systemic_write_failures = int(systemic_guardrails.get("write_failure_count", 0) or 0)
    systemic_queue_depth = int(systemic_guardrails.get("queue_depth", 0) or 0)
    repeatable_thaw_ready = bool(
        isolated_lane_count > 0
        and len(release_ready_candidates) > 0
        and not unresolved_checks
        and not systemic_halt_active
        and systemic_risk_halts <= 0
        and systemic_snapshot_failures <= 0
        and systemic_write_failures <= 0
        and systemic_queue_depth < 10000
    )

    recommended_actions = ordered_unique(
        [
            "keep healthy sleeves running while anomaly-killed lanes stay quarantined" if isolated_lanes else "",
            "route sustained ingestion failures to the provider fallback before lane kill switches accumulate" if failed_ingestion_lanes else "",
            "route investigation to the paused sleeves instead of draining the entire runtime" if len(isolated_lanes) >= 1 else "",
            "reduce quarantine churn before expanding sleeve count again" if quarantine_events > int(max_quarantine_events) else "",
            "clear unresolved daily verify blockers before reenabling isolated sleeves" if unresolved_checks else "",
            "run supervised single-lane canaries before widening thaw scope for repeat or high-caution sleeves" if supervised_candidates else "",
            "only allow micro-probe thaw on isolated sleeves when the runtime-level halt, data-plane, and cooldown guardrails are green" if micro_probe_candidates else "",
            "keep operator-review sleeves quarantined until their chronic or repeat-trip incident review is closed" if operator_review_rows else "",
            "do not thaw isolated sleeves while the global halt, incident risk halt, or data-plane recovery pressure is still active" if systemic_halt_active or systemic_risk_halts > 0 or systemic_snapshot_failures > 0 or systemic_write_failures > 0 or systemic_queue_depth >= 10000 else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "blast_radius_score": blast_radius_score,
        "quarantine_pressure": {
            "events": quarantine_events,
            "max_quarantine_events": int(max_quarantine_events),
            "top_symbols": quarantine_pressure.get("top_symbols") if isinstance(quarantine_pressure.get("top_symbols"), list) else [],
        },
        "sleeve_matrix": {
            "isolated_lanes": isolated_lanes,
            "isolated_lane_count": isolated_lane_count,
            "session_paused_lanes": session_paused_lanes[:12],
            "session_paused_lane_count": len(session_paused_lanes),
            "running_lane_count": running_lane_count,
            "healthy_running_lane_count": len(healthy_running_lanes),
            "running_examples": running_lanes[:6],
            "failed_ingestion_lanes": failed_ingestion_lanes[:12],
            "failed_ingestion_lane_count": len(failed_ingestion_lanes),
        },
        "gates": {
            "unresolved_daily_verify_checks": unresolved_checks,
            "isolation_required": bool(isolated_lanes or failed_ingestion_lanes),
        },
        "repeatable_thaw_contract": {
            "ready": repeatable_thaw_ready,
            "candidate_count": len(release_ready_candidates),
            "supervised_candidate_count": len(supervised_candidates),
            "micro_probe_candidate_count": len(micro_probe_candidates),
            "operator_review_count": len(operator_review_rows),
            "blocked_count": len(thaw_blocked),
            "candidate_examples": release_ready_candidates[:4],
            "blocked_examples": thaw_blocked[:4],
        },
        "systemic_guardrails": {
            "global_killswitch_active": systemic_halt_active,
            "risk_halt_events": systemic_risk_halts,
            "account_snapshot_failure_count": systemic_snapshot_failures,
            "write_failure_count": systemic_write_failures,
            "queue_depth": systemic_queue_depth,
        },
        "infra_bots": ["sleeve_isolation_guard", "quarantine_pressure_bot", "data_ingress_latest_*"],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Track sleeve quarantine and isolation so one failing lane does not poison the runtime.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--max-quarantine-events", type=int, default=120)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), max_quarantine_events=int(args.max_quarantine_events))
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "sleeve_isolation_guard "
            f"overall_status={payload.get('overall_status', '')} "
            f"isolated_lane_count={int(((payload.get('sleeve_matrix') or {}).get('isolated_lane_count', 0) or 0))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
