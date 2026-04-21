#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "operator_cockpit_latest.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "exports" / "reports" / "operator" / "operator_cockpit_latest.md"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Operator Cockpit",
        "",
        f"- Timestamp UTC: `{payload.get('timestamp_utc', '')}`",
        f"- Overall Status: `{payload.get('overall_status', '')}`",
        "",
        "## Immediate Focus",
        "",
    ]
    for item in payload.get("recommended_actions") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Upgrade Lanes", ""])
    for key, row in (payload.get("upgrade_lanes") or {}).items():
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{key}`: `{row.get('status', '')}`"
            + (f" ({row.get('summary', '')})" if str(row.get("summary") or "").strip() else "")
        )
    lines.extend(["", "## Long-Run Lanes", ""])
    for key, row in (payload.get("long_run_lanes") or {}).items():
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{key}`: `{row.get('status', '')}`"
            + (f" ({row.get('summary', '')})" if str(row.get("summary") or "").strip() else "")
        )
    lines.extend(["", "## Key Surfaces", ""])
    for key, row in (payload.get("surfaces") or {}).items():
        if not isinstance(row, dict):
            continue
        lines.append(f"- `{key}`: `{row.get('status', '')}`")
    return "\n".join(lines) + "\n"


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    runtime = _load_json(health_root / "runtime_gate_dashboard_latest.json")
    platform = _load_json(health_root / "platform_control_plane_latest.json")
    training = _load_json(health_root / "training_report_latest.json")
    training_quality = _load_json(health_root / "training_quality_control_latest.json")
    storage = _load_json(health_root / "ingestion_storage_control_latest.json")
    governor = _load_json(health_root / "ingestion_storage_governor_latest.json")
    backlog_drain = _load_json(health_root / "external_backlog_drain_latest.json")
    backlog_retry_bot = _load_json(health_root / "external_backlog_retry_bot_latest.json")
    queue = _load_json(health_root / "ingestion_priority_queue_latest.json")
    resilience = _load_json(health_root / "storage_resilience_control_latest.json")
    split_brain = _load_json(health_root / "storage_split_brain_reconciler_latest.json")
    requalification = _load_json(health_root / "training_requalification_latest.json")
    coverage_seed = _load_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json")
    calibration = _load_json(health_root / "calibration_abstention_control_latest.json")
    paper_calibration = _load_json(health_root / "paper_execution_calibration_latest.json")
    remediation = _load_json(health_root / "daily_verify_auto_remediation_bot_latest.json")
    storage_tier = _load_json(health_root / "storage_tier_policy_latest.json")
    training_runtime = _load_json(health_root / "training_runtime_control_latest.json")
    regime_control = _load_json(health_root / "regime_control_plane_latest.json")
    supportability_control = _load_json(health_root / "supportability_control_latest.json")
    provider_mesh = _load_json(health_root / "provider_mesh_latest.json")
    service_control_plane = _load_json(health_root / "service_control_plane_latest.json")
    teacher_quality = _load_json(project_root / "governance" / "distillation" / "teacher_quality_latest.json")
    bot_quality_autopilot = _load_json(health_root / "bot_quality_autopilot_latest.json")
    infrastructure_autofix = _load_json(health_root / "infrastructure_autofix_bot_latest.json")
    live_runtime_separation = _load_json(health_root / "live_runtime_separation_control_latest.json")
    rolling_restart = _load_json(health_root / "rolling_restart_controller_latest.json")
    auth_lease = _load_json(health_root / "auth_lease_manager_latest.json")
    blackstart_recovery = _load_json(health_root / "blackstart_recovery_latest.json")
    sleeve_isolation = _load_json(health_root / "sleeve_isolation_guard_latest.json")
    artifact_freshness = _load_json(health_root / "artifact_freshness_slo_latest.json")
    snapshot_cache = _load_json(health_root / "runtime_snapshot_cache_control_latest.json")
    remote_alert = _load_json(health_root / "remote_alert_control_latest.json")
    storage_quota = _load_json(health_root / "storage_quota_guard_latest.json")
    release_freeze = _load_json(health_root / "release_freeze_guard_latest.json")
    roster_expansion = _load_json(health_root / "roster_expansion_slots_latest.json")
    roster_resilience = _load_json(health_root / "roster_resilience_planner_latest.json")
    chaos_drills = _load_json(health_root / "chaos_drill_coordinator_latest.json")

    attention = runtime.get("overall", {}).get("attention") if isinstance(runtime.get("overall"), dict) else []
    recommended_actions = _ordered_unique(
        list(attention or [])
        + list((storage_tier.get("upgrade_plan") or {}).get("recommended_actions") or [])
        + list((training_runtime.get("recommended_actions") or [])[:3])
        + list((provider_mesh.get("recommended_actions") or [])[:3])
        + list((service_control_plane.get("recommended_actions") or [])[:3])
        + list((regime_control.get("recommended_actions") or [])[:3])
        + list((roster_expansion.get("recommended_actions") or [])[:2])
        + list((supportability_control.get("recommended_actions") or [])[:3])
        + list((teacher_quality.get("recommended_actions") or [])[:2])
        + list((bot_quality_autopilot.get("recommended_actions") or [])[:2])
        + list((infrastructure_autofix.get("recommended_actions") or [])[:2])
        + list(storage.get("top_actions") or [])
        + list((governor.get("top_actions") or [])[:2])
        + list((backlog_drain.get("top_actions") or [])[:2])
        + list((backlog_retry_bot.get("recommended_actions") or [])[:2])
        + list((queue.get("top_actions") or [])[:2])
        + list((resilience.get("top_actions") or [])[:2])
        + list((requalification.get("recommended_actions") or [])[:2])
        + list((coverage_seed.get("recommended_actions") or [])[:2])
        + list((calibration.get("top_actions") or [])[:2])
        + list((paper_calibration.get("top_actions") or [])[:2])
        + list((remediation.get("recommended_actions") or [])[:2])
        + list((live_runtime_separation.get("recommended_actions") or [])[:2])
        + list((rolling_restart.get("recommended_actions") or [])[:2])
        + list((auth_lease.get("recommended_actions") or [])[:2])
        + list((blackstart_recovery.get("recommended_actions") or [])[:2])
        + list((sleeve_isolation.get("recommended_actions") or [])[:2])
        + list((artifact_freshness.get("recommended_actions") or [])[:2])
        + list((snapshot_cache.get("recommended_actions") or [])[:2])
        + list((remote_alert.get("recommended_actions") or [])[:2])
        + list((storage_quota.get("recommended_actions") or [])[:2])
        + list((release_freeze.get("recommended_actions") or [])[:2])
        + list((roster_resilience.get("recommended_actions") or [])[:2])
        + list((chaos_drills.get("recommended_actions") or [])[:2])
    )[:14]

    overall_status = "ready"
    if bool(runtime.get("overall", {}).get("ok", True)) is False or str(storage.get("overall_status") or "") == "blocked":
        overall_status = "degraded"
    if int(((split_brain.get("summary") or {}).get("unresolved_conflicts", 0) or 0) > 0):
        overall_status = "degraded"
    if bool(((governor.get("sql_primary_db") or {}).get("route_drift", False))):
        overall_status = "degraded"
    if str(training_runtime.get("overall_status") or "") == "blocked":
        overall_status = "degraded"
    if str(supportability_control.get("overall_status") or "") == "blocked":
        overall_status = "degraded"
    if str(bot_quality_autopilot.get("overall_status") or "") in {"blocked", "degraded"}:
        overall_status = "degraded"
    if str(infrastructure_autofix.get("overall_status") or "") in {"blocked", "degraded"}:
        overall_status = "degraded"
    if str(storage_tier.get("overall_status") or "") == "blocked":
        overall_status = "degraded"
    if str(regime_control.get("overall_status") or "") == "degraded":
        overall_status = "degraded"
    if str(provider_mesh.get("overall_status") or "") in {"blocked", "degraded"}:
        overall_status = "degraded"
    if str(service_control_plane.get("overall_status") or "") in {"blocked", "degraded"}:
        overall_status = "degraded"
    for row in (
        live_runtime_separation,
        rolling_restart,
        auth_lease,
        blackstart_recovery,
        sleeve_isolation,
        artifact_freshness,
        snapshot_cache,
        remote_alert,
        storage_quota,
        release_freeze,
        roster_expansion,
        roster_resilience,
        chaos_drills,
    ):
        status = str((row or {}).get("overall_status") or "")
        if status in {"blocked", "degraded"}:
            overall_status = "degraded"

    upgrade_lanes = {
        "storage_split": {
            "status": str(storage_tier.get("overall_status") or "missing"),
            "summary": (
                f"hot_path_over_budget_bytes={int(((storage_tier.get('pressure') or {}).get('hot_path_over_budget_bytes', 0) or 0))}"
                if storage_tier
                else ""
            ),
        },
        "training_runtime": {
            "status": str(training_runtime.get("overall_status") or "missing"),
            "summary": (
                f"snapshot_ready={int(bool(training_runtime.get('snapshot_ready', False)))} "
                f"precompute_targets={len(training_runtime.get('precompute_targets') or [])}"
                if training_runtime
                else ""
            ),
        },
        "coverage_seeding": {
            "status": str(coverage_seed.get("overall_status") or ("ready" if coverage_seed else "missing")),
            "summary": (
                f"coverage_shortfall_bots={int(coverage_seed.get('coverage_shortfall_bots', 0) or 0)} "
                f"seed_queue={len(coverage_seed.get('seed_queue') or [])}"
                if coverage_seed
                else ""
            ),
        },
        "regime_engine": {
            "status": str(regime_control.get("overall_status") or "missing"),
            "summary": (
                f"{str(regime_control.get('regime_state') or '')} {str(regime_control.get('stance_label') or '')}".strip()
                if regime_control
                else ""
            ),
        },
        "lifecycle_teaching": {
            "status": str(supportability_control.get("overall_status") or "missing"),
            "summary": (
                f"supportability={float(((supportability_control.get('supportability') or {}).get('active_supportability_score', 0.0) or 0.0)):.3f} "
                f"students_without_teachers={int(((supportability_control.get('teacher_student') or {}).get('students_without_teachers', 0) or 0))}"
                if supportability_control
                else ""
            ),
        },
        "roster_expansion": {
            "status": str(roster_expansion.get("overall_status") or "missing"),
            "summary": (
                f"registered_slots={int(((roster_expansion.get('summary') or {}).get('registered_slot_count', 0) or 0))} "
                f"missing_slots={int(((roster_expansion.get('summary') or {}).get('missing_slot_count', 0) or 0))}"
                if roster_expansion
                else ""
            ),
        },
        "teacher_quality": {
            "status": str(teacher_quality.get("overall_status") or "missing"),
            "summary": (
                f"qualified_teachers={int(((teacher_quality.get('summary') or {}).get('qualified_teacher_count', 0) or 0))} "
                f"elite_teachers={int(((teacher_quality.get('summary') or {}).get('elite_teacher_count', 0) or 0))}"
                if teacher_quality
                else ""
            ),
        },
        "bot_quality_autopilot": {
            "status": str(bot_quality_autopilot.get("overall_status") or "missing"),
            "summary": (
                f"quality_queue={len(bot_quality_autopilot.get('quality_upgrade_queue') or [])}"
                if bot_quality_autopilot
                else ""
            ),
        },
        "execution_realism": {
            "status": str(paper_calibration.get("overall_status") or ("ready" if paper_calibration else "missing")),
            "summary": (
                f"mae_bps={float(((paper_calibration.get('metrics') or {}).get('mae_bps', 0.0) or 0.0)):.3f}"
                if paper_calibration
                else ""
            ),
        },
        "operator_cockpit": {
            "status": overall_status,
            "summary": "unified control plane",
        },
    }
    service_upgrade_lanes = service_control_plane.get("upgrade_lanes") if isinstance(service_control_plane.get("upgrade_lanes"), dict) else {}
    for key in (
        "control_plane",
        "provider_mesh",
        "execution_gateway",
        "retrain_pipeline",
        "event_history",
        "runtime_separation",
        "operator_cockpit_contract",
    ):
        if isinstance(service_upgrade_lanes.get(key), dict):
            row = service_upgrade_lanes.get(key) or {}
            upgrade_lanes[key] = {
                "status": str(row.get("status") or "missing"),
                "summary": str(row.get("summary") or ""),
            }
    long_run_lanes = {
        "live_runtime_separation": {
            "status": str(live_runtime_separation.get("overall_status") or "missing"),
            "summary": (
                f"contention_score={int(((live_runtime_separation.get('shared_host_pressure') or {}).get('contention_score', 0) or 0))}"
                if live_runtime_separation
                else ""
            ),
        },
        "rolling_restart": {
            "status": str(rolling_restart.get("overall_status") or "missing"),
            "summary": (
                f"restart_due={int(bool(rolling_restart.get('restart_due', False)))} "
                f"scope={str(rolling_restart.get('recommended_scope') or '')}"
                if rolling_restart
                else ""
            ),
        },
        "auth_lease": {
            "status": str(auth_lease.get("overall_status") or "missing"),
            "summary": (
                f"lease_state={str(auth_lease.get('lease_state') or '')} "
                f"expires_in_seconds={float(((auth_lease.get('lease_budget') or {}).get('expires_in_seconds', 0.0) or 0.0)):.1f}"
                if auth_lease
                else ""
            ),
        },
        "blackstart_recovery": {
            "status": str(blackstart_recovery.get("overall_status") or "missing"),
            "summary": f"stages={len(blackstart_recovery.get('stages') or [])}" if blackstart_recovery else "",
        },
        "sleeve_isolation": {
            "status": str(sleeve_isolation.get("overall_status") or "missing"),
            "summary": (
                f"isolated_lanes={int(((sleeve_isolation.get('sleeve_matrix') or {}).get('isolated_lane_count', 0) or 0))}"
                if sleeve_isolation
                else ""
            ),
        },
        "artifact_freshness_slo": {
            "status": str(artifact_freshness.get("overall_status") or "missing"),
            "summary": (
                f"stale_required={int(((artifact_freshness.get('sla_summary') or {}).get('stale_required', 0) or 0))}"
                if artifact_freshness
                else ""
            ),
        },
        "runtime_snapshot_cache": {
            "status": str(snapshot_cache.get("overall_status") or "missing"),
            "summary": (
                f"snapshot_ready={int(bool(((snapshot_cache.get('cache_health') or {}).get('snapshot_ready', False))))}"
                if snapshot_cache
                else ""
            ),
        },
        "remote_alert_control": {
            "status": str(remote_alert.get("overall_status") or "missing"),
            "summary": (
                f"unacked_critical={int(((remote_alert.get('critical_backlog') or {}).get('unacked_count', 0) or 0))}"
                if remote_alert
                else ""
            ),
        },
        "storage_quota_guard": {
            "status": str(storage_quota.get("overall_status") or "missing"),
            "summary": (
                f"hard_breaches={int(((storage_quota.get('quota_summary') or {}).get('hard_breaches', 0) or 0))}"
                if storage_quota
                else ""
            ),
        },
        "release_freeze": {
            "status": str(release_freeze.get("overall_status") or "missing"),
            "summary": (
                f"active={int(bool(((release_freeze.get('window') or {}).get('active', False))))}"
                if release_freeze
                else ""
            ),
        },
        "infrastructure_autofix": {
            "status": str(infrastructure_autofix.get("overall_status") or "missing"),
            "summary": (
                f"repair_plan={int(infrastructure_autofix.get('applyable_repair_count', 0) or 0)}"
                if infrastructure_autofix
                else ""
            ),
        },
        "roster_resilience": {
            "status": str(roster_resilience.get("overall_status") or "missing"),
            "summary": (
                f"bench_depth={int(((roster_resilience.get('bench') or {}).get('bench_depth', 0) or 0))}"
                if roster_resilience
                else ""
            ),
        },
        "chaos_drills": {
            "status": str(chaos_drills.get("overall_status") or "missing"),
            "summary": f"overdue={len(chaos_drills.get('overdue_drills') or [])}" if chaos_drills else "",
        },
    }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 4,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "recommended_actions": recommended_actions,
        "upgrade_lanes": upgrade_lanes,
        "long_run_lanes": long_run_lanes,
        "surfaces": {
            "runtime_gate_dashboard": {"status": str(runtime.get("overall", {}).get("status") or "")},
            "platform_control_plane": {"status": str((platform.get("institutional_readiness") or {}).get("overall_status") or "")},
            "provider_mesh": {"status": str(provider_mesh.get("overall_status") or "missing") if provider_mesh else "missing"},
            "service_control_plane": {"status": str(service_control_plane.get("overall_status") or "missing") if service_control_plane else "missing"},
            "training_report": {"status": str(training.get("overall_status") or "")},
            "training_quality_control": {"status": str(training_quality.get("overall_status") or "")},
            "ingestion_storage_control": {"status": str(storage.get("overall_status") or "")},
            "ingestion_storage_governor": {"status": str(governor.get("profile") or "missing") if governor else "missing"},
            "storage_tier_policy": {"status": str(storage_tier.get("overall_status") or "missing") if storage_tier else "missing"},
            "training_runtime_control": {"status": str(training_runtime.get("overall_status") or "missing") if training_runtime else "missing"},
            "external_backlog_drain": {"status": str(backlog_drain.get("overall_status") or "")},
            "external_backlog_retry_bot": {"status": str(backlog_retry_bot.get("overall_status") or "")},
            "ingestion_priority_queue": {"status": "ready" if queue else "missing"},
            "storage_resilience_control": {"status": str(resilience.get("overall_status") or "")},
            "storage_split_brain_reconciler": {"status": "needs_review" if int(((split_brain.get("summary") or {}).get("unresolved_conflicts", 0) or 0) > 0) else "ready"},
            "training_requalification_lane": {"status": "ready" if requalification else "missing"},
            "walk_forward_coverage_seed": {"status": str(coverage_seed.get("overall_status") or "missing") if coverage_seed else "missing"},
            "regime_control_plane": {"status": str(regime_control.get("overall_status") or "missing") if regime_control else "missing"},
            "supportability_control": {"status": str(supportability_control.get("overall_status") or "missing") if supportability_control else "missing"},
            "teacher_quality_guard": {"status": str(teacher_quality.get("overall_status") or "missing") if teacher_quality else "missing"},
            "bot_quality_autopilot": {"status": str(bot_quality_autopilot.get("overall_status") or "missing") if bot_quality_autopilot else "missing"},
            "infrastructure_autofix_bot": {"status": str(infrastructure_autofix.get("overall_status") or "missing") if infrastructure_autofix else "missing"},
            "live_runtime_separation_control": {"status": str(live_runtime_separation.get("overall_status") or "missing") if live_runtime_separation else "missing"},
            "rolling_restart_controller": {"status": str(rolling_restart.get("overall_status") or "missing") if rolling_restart else "missing"},
            "auth_lease_manager": {"status": str(auth_lease.get("overall_status") or "missing") if auth_lease else "missing"},
            "blackstart_recovery": {"status": str(blackstart_recovery.get("overall_status") or "missing") if blackstart_recovery else "missing"},
            "sleeve_isolation_guard": {"status": str(sleeve_isolation.get("overall_status") or "missing") if sleeve_isolation else "missing"},
            "artifact_freshness_slo": {"status": str(artifact_freshness.get("overall_status") or "missing") if artifact_freshness else "missing"},
            "runtime_snapshot_cache_control": {"status": str(snapshot_cache.get("overall_status") or "missing") if snapshot_cache else "missing"},
            "remote_alert_control": {"status": str(remote_alert.get("overall_status") or "missing") if remote_alert else "missing"},
            "storage_quota_guard": {"status": str(storage_quota.get("overall_status") or "missing") if storage_quota else "missing"},
            "release_freeze_guard": {"status": str(release_freeze.get("overall_status") or "missing") if release_freeze else "missing"},
            "roster_expansion_slots": {"status": str(roster_expansion.get("overall_status") or "missing") if roster_expansion else "missing"},
            "roster_resilience_planner": {"status": str(roster_resilience.get("overall_status") or "missing") if roster_resilience else "missing"},
            "chaos_drill_coordinator": {"status": str(chaos_drills.get("overall_status") or "missing") if chaos_drills else "missing"},
            "calibration_abstention_control": {"status": str(calibration.get("overall_status") or "")},
            "paper_execution_calibration": {"status": str(paper_calibration.get("overall_status") or "missing") if paper_calibration else "missing"},
            "daily_verify_auto_remediation_bot": {"status": str(remediation.get("overall_status") or "")},
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a single operator cockpit across runtime, storage, training, and remediation surfaces.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    markdown_path = Path(args.markdown_out).expanduser()
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "operator_cockpit "
            f"overall_status={payload.get('overall_status', '')} "
            f"recommended_actions={len(payload.get('recommended_actions') or [])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
