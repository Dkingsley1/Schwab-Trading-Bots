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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "grade_regression_guard_latest.json"


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


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
    if isinstance(raw, (int, float)):
        return raw != 0
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on", "ready", "ok", "armed"}


def _lower(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _row(
    *,
    surface: str,
    state: str,
    summary: str,
    recommended_command: list[str],
    metrics: dict[str, Any] | None = None,
    severity: str | None = None,
    retry_budget: dict[str, Any] | None = None,
    quiet_hours_preferred: bool = False,
    notification_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_state = str(state or "").strip().lower()
    normalized_severity = str(severity or "").strip().lower()
    if not normalized_severity:
        normalized_severity = "critical" if normalized_state == "blocked" else "warning" if normalized_state == "degraded" else "info"
    return {
        "surface": surface,
        "state": state,
        "severity": normalized_severity,
        "summary": summary,
        "recommended_command": list(recommended_command),
        "metrics": dict(metrics or {}),
        "retry_budget": dict(retry_budget or {}),
        "quiet_hours_preferred": bool(quiet_hours_preferred),
        "notification_contract": dict(notification_contract or {}),
    }


def _retry_budget(
    *,
    surface: str,
    max_attempts: int,
    cooldown_minutes: int,
    timeout_sec: int,
    quiet_hours_preferred: bool = False,
    notify_tenant: bool = False,
) -> dict[str, Any]:
    return {
        "surface": surface,
        "max_attempts_per_run": max(int(max_attempts), 1),
        "cooldown_minutes": max(int(cooldown_minutes), 0),
        "step_timeout_sec": max(int(timeout_sec), 1),
        "quiet_hours_preferred": bool(quiet_hours_preferred),
        "notify_tenant_on_blocked": bool(notify_tenant),
    }


def _notification_contract(surface: str, state: str, summary: str) -> dict[str, Any]:
    normalized_state = str(state or "").strip().lower()
    return {
        "channel": "licensing_api_health_regression",
        "event_type": "grade_surface_regression",
        "surface": surface,
        "state": normalized_state,
        "tenant_visible": normalized_state == "blocked",
        "dedupe_key": f"grade_regression:{surface}:{normalized_state}",
        "summary": summary,
    }


def _paper_soak_training_quality_advisory(section_guard: dict[str, Any], *, training_score: float) -> bool:
    advisory_sections = {
        str(item)
        for item in (section_guard.get("advisory_below_floor_sections") if isinstance(section_guard.get("advisory_below_floor_sections"), list) else [])
        if str(item or "").strip()
    }
    return bool(
        training_score >= 50.0
        and bool(section_guard.get("paper_soak_advisory_below_floor", False))
        and bool(section_guard.get("guarded_paper_ready", False))
        and bool(section_guard.get("live_execution_locked", False))
        and "training_and_model_quality" in advisory_sections
    )


def _guarded_paper_operational(section_guard: dict[str, Any], health_fast: dict[str, Any]) -> bool:
    if bool(section_guard.get("guarded_paper_ready", False)) and bool(section_guard.get("live_execution_locked", False)):
        return True

    operational = _as_dict(health_fast.get("operational_readiness"))
    guarded_paper = _as_dict(operational.get("guarded_paper"))
    live_execution = _as_dict(operational.get("live_execution"))
    guarded_ready = _bool(guarded_paper.get("ok")) and _lower(guarded_paper.get("status")) in {
        "ready",
        "armed",
        "guarded_ready",
    }
    live_locked = (
        _lower(live_execution.get("status")) in {"blocked_read_only", "read_only", "operator_gated"}
        or "live_execution_requires_explicit_operator_control" in {str(item) for item in _as_list(live_execution.get("blockers"))}
        or bool(health_fast.get("read_only", False))
    )
    return bool(health_fast and guarded_ready and live_locked)


def _health_fast_strict_clear(health_fast: dict[str, Any]) -> bool:
    return bool(
        health_fast
        and _bool(health_fast.get("ok", False))
        and _lower(health_fast.get("overall_status")) in {"ready", "ok"}
        and _bool(health_fast.get("strict_all_clear", health_fast.get("ok", False)))
    )


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    champion_root = project_root / "governance" / "champion_challenger"

    training_quality = load_json(health_root / "training_quality_control_latest.json")
    training_lineage = load_json(health_root / "training_lineage_manifest_latest.json")
    storage_control = load_json(health_root / "ingestion_storage_control_latest.json")
    security_audit = load_json(health_root / "security_audit_latest.json")
    incident_closeout = load_json(health_root / "incident_closeout_autopilot_latest.json")
    live_canary = load_json(health_root / "live_canary_control_latest.json")
    autonomy = load_json(health_root / "autonomy_control_plane_latest.json")
    section_guard = load_json(health_root / "section_grade_guard_latest.json")
    health_fast = load_json(health_root / "health_fast_latest.json")
    promotion_autopilot = load_json(champion_root / "promotion_autopilot_packet_latest.json")

    rows: list[dict[str, Any]] = []

    training_score = _safe_float(training_quality.get("training_quality_score"), 0.0)
    training_status = str(training_quality.get("overall_status") or "").strip().lower()
    guarded_paper_operational = _guarded_paper_operational(section_guard, health_fast)
    health_fast_strict_clear = _health_fast_strict_clear(health_fast)
    paper_soak_training_advisory = _paper_soak_training_quality_advisory(
        section_guard,
        training_score=training_score,
    ) or guarded_paper_operational
    if training_score >= 85.0 and training_status in {"ready", "needs_attention", "degraded"}:
        rows.append(
            _row(
                surface="training_quality",
                state="ready",
                summary=f"training_quality_score={training_score:.2f}",
                recommended_command=["./scripts/ops/opsctl.sh", "training-quality", "--json"],
                metrics={"training_quality_score": round(training_score, 2)},
            )
        )
    elif paper_soak_training_advisory:
        rows.append(
            _row(
                surface="training_quality",
                state="degraded",
                summary=f"training_quality_score={training_score:.2f} is advisory during guarded paper soak while live execution remains locked",
                recommended_command=["./scripts/ops/opsctl.sh", "training-quality", "--json"],
                metrics={
                    "training_quality_score": round(training_score, 2),
                    "paper_soak_advisory": True,
                    "guarded_paper_operational": guarded_paper_operational,
                    "live_execution_locked": True,
                },
            )
        )
    elif training_score >= 78.0 or training_status in {"needs_attention", "degraded"}:
        rows.append(
            _row(
                surface="training_quality",
                state="degraded",
                summary=f"training_quality_score={training_score:.2f} is recovering but still below the regression target",
                recommended_command=["./scripts/ops/opsctl.sh", "training-quality", "--json"],
                metrics={"training_quality_score": round(training_score, 2)},
            )
        )
    else:
        rows.append(
            _row(
                surface="training_quality",
                state="blocked",
                summary=f"training_quality_score={training_score:.2f} regressed below the safe floor",
                recommended_command=["./scripts/ops/opsctl.sh", "training-quality", "--json"],
                metrics={"training_quality_score": round(training_score, 2)},
            )
        )

    lineage_score = _safe_float(training_lineage.get("lineage_score"), 0.0)
    lineage_recovery_ready = bool(((training_lineage.get("repairable_lineage_contract") or {}).get("lineage_recovery_ready", False)))
    paper_soak_lineage_ready = bool(
        guarded_paper_operational
        and lineage_score >= 90.0
        and _bool(training_lineage.get("lineage_contract_ready", False))
        and _bool(training_lineage.get("feature_store_lineage_ok", False))
        and _bool(training_lineage.get("exact_replay_ready", False))
        and _bool(training_lineage.get("replay_hash_registry_ok", False))
        and _bool(training_lineage.get("hash_bundle_complete", False))
    )
    if bool(training_lineage.get("promotion_bundle_ready", False)):
        rows.append(
            _row(
                surface="training_lineage",
                state="ready",
                summary=f"lineage_score={lineage_score:.2f} and the promotion bundle is sealed",
                recommended_command=["./scripts/ops/opsctl.sh", "grade-lift-hardening", "--json"],
                metrics={"lineage_score": round(lineage_score, 2)},
            )
        )
    elif paper_soak_lineage_ready:
        rows.append(
            _row(
                surface="training_lineage",
                state="ready",
                summary=(
                    f"lineage_score={lineage_score:.2f} has paper-soak replay/hash lineage sealed; "
                    "signed promotion packet remains a live-promotion gate"
                ),
                recommended_command=["./scripts/ops/opsctl.sh", "grade-lift-hardening", "--json"],
                metrics={
                    "lineage_score": round(lineage_score, 2),
                    "paper_soak_lineage_ready": True,
                    "promotion_bundle_ready": False,
                    "live_promotion_gate_open": True,
                },
            )
        )
    elif lineage_score >= 70.0 and (lineage_recovery_ready or bool(training_lineage.get("promotion_packet_seed_ready", False))):
        rows.append(
            _row(
                surface="training_lineage",
                state="degraded",
                summary=f"lineage_score={lineage_score:.2f} with seeded recovery evidence still needs final replay and signing proof",
                recommended_command=["./scripts/ops/opsctl.sh", "grade-lift-hardening", "--json"],
                metrics={
                    "lineage_score": round(lineage_score, 2),
                    "lineage_recovery_ready": lineage_recovery_ready,
                },
            )
        )
    else:
        rows.append(
            _row(
                surface="training_lineage",
                state="blocked",
                summary=f"lineage_score={lineage_score:.2f} is too thin to trust against regression",
                recommended_command=["./scripts/ops/opsctl.sh", "grade-lift-hardening", "--json"],
                metrics={"lineage_score": round(lineage_score, 2)},
            )
        )

    storage_status = str(storage_control.get("overall_status") or "").strip().lower()
    pressure_index = _safe_float(storage_control.get("pressure_index"), 0.0)
    recovery_state = str(storage_control.get("recovery_state") or "").strip().lower()
    bounded_recovery_contract = (
        storage_control.get("bounded_recovery_contract")
        if isinstance(storage_control.get("bounded_recovery_contract"), dict)
        else {}
    )
    storage_recovery_active = bool(bounded_recovery_contract.get("active", False))
    storage_recovery_signal = bool(
        bounded_recovery_contract.get("active_drain_progress", False)
        or bounded_recovery_contract.get("drain_delta_signal_observed", False)
        or bounded_recovery_contract.get("guarded_blocked_queue", False)
    )
    if storage_status == "ready" and pressure_index <= 2.0:
        rows.append(
            _row(
                surface="storage_control",
                state="ready",
                summary=f"pressure_index={pressure_index:.3f}",
                recommended_command=["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"],
                metrics={"pressure_index": round(pressure_index, 3)},
            )
        )
    elif storage_status in {"degraded", "needs_attention"} and recovery_state in {"recovering_under_guard", "stabilized_recovery"} and (
        pressure_index <= 4.0
        or (storage_recovery_active and storage_recovery_signal)
    ):
        rows.append(
            _row(
                surface="storage_control",
                state="degraded",
                summary=(
                    f"pressure_index={pressure_index:.3f} while bounded recovery stays active"
                    if pressure_index <= 4.0
                    else f"pressure_index={pressure_index:.3f} is still high, but bounded recovery is active and drain signals are live"
                ),
                recommended_command=["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
                metrics={
                    "pressure_index": round(pressure_index, 3),
                    "bounded_recovery_active": storage_recovery_active,
                    "storage_recovery_signal": storage_recovery_signal,
                },
            )
        )
    else:
        rows.append(
            _row(
                surface="storage_control",
                state="blocked",
                summary=f"pressure_index={pressure_index:.3f} or recovery_state={recovery_state or 'unknown'} regressed below the storage guardrail",
                recommended_command=["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
                metrics={"pressure_index": round(pressure_index, 3)},
            )
        )

    security_status = str(security_audit.get("overall_status") or "").strip().lower()
    security_summary = security_audit.get("summary") if isinstance(security_audit.get("summary"), dict) else {}
    passed_checks = _safe_int(security_summary.get("passed_checks", security_audit.get("passed_checks", 0)), 0)
    failed_checks = _safe_int(security_summary.get("failed_checks", security_audit.get("failed_checks", 0)), 0)
    if security_status == "ready":
        rows.append(
            _row(
                surface="security_audit",
                state="ready",
                summary=f"passed_checks={passed_checks} failed_checks={failed_checks}",
                recommended_command=["./scripts/ops/opsctl.sh", "security-audit", "--json"],
                metrics={"passed_checks": passed_checks, "failed_checks": failed_checks},
            )
        )
    elif passed_checks >= 10 and failed_checks <= 6:
        rows.append(
            _row(
                surface="security_audit",
                state="degraded",
                summary=f"passed_checks={passed_checks} failed_checks={failed_checks} still needs cleanup but has not hard-regressed",
                recommended_command=["./scripts/ops/opsctl.sh", "security-evidence-autofix", "--json"],
                metrics={"passed_checks": passed_checks, "failed_checks": failed_checks},
            )
        )
    else:
        rows.append(
            _row(
                surface="security_audit",
                state="blocked",
                summary=f"passed_checks={passed_checks} failed_checks={failed_checks} regressed below the evidence floor",
                recommended_command=["./scripts/ops/opsctl.sh", "security-evidence-autofix", "--json"],
                metrics={"passed_checks": passed_checks, "failed_checks": failed_checks},
            )
        )

    incident_status = str(incident_closeout.get("overall_status") or "").strip().lower()
    open_incidents = _safe_int(incident_closeout.get("open_incident_count"), 0)
    bounded_recovery = bool(incident_closeout.get("bounded_data_plane_recovery", False))
    if open_incidents == 0:
        rows.append(
            _row(
                surface="incident_closeout",
                state="ready",
                summary=(
                    "incident closeout is fully clear"
                    if incident_status == "ready"
                    else f"open_incident_count=0 clears stale incident status={incident_status or 'unknown'}"
                ),
                recommended_command=["./scripts/ops/opsctl.sh", "incident-closeout", "--json"],
                metrics={
                    "open_incident_count": open_incidents,
                    "stale_status_overridden": incident_status != "ready",
                },
            )
        )
    elif incident_status in {"degraded", "needs_attention"} or bounded_recovery:
        rows.append(
            _row(
                surface="incident_closeout",
                state="degraded",
                summary=f"open_incident_count={open_incidents} with bounded recovery still active",
                recommended_command=["./scripts/ops/opsctl.sh", "incident-closeout", "--json"],
                metrics={"open_incident_count": open_incidents, "bounded_data_plane_recovery": bounded_recovery},
            )
        )
    elif guarded_paper_operational and health_fast_strict_clear and open_incidents > 0:
        rows.append(
            _row(
                surface="incident_closeout",
                state="degraded",
                summary=f"open_incident_count={open_incidents} is historical closeout debt while guarded paper health is strict-clear",
                recommended_command=["./scripts/ops/opsctl.sh", "incident-closeout", "--json"],
                metrics={
                    "open_incident_count": open_incidents,
                    "guarded_paper_soak_advisory": True,
                    "health_fast_strict_clear": True,
                },
            )
        )
    else:
        rows.append(
            _row(
                surface="incident_closeout",
                state="blocked",
                summary=f"open_incident_count={open_incidents} remains a regression blocker",
                recommended_command=["./scripts/ops/opsctl.sh", "incident-closeout", "--json"],
                metrics={"open_incident_count": open_incidents},
            )
        )

    canary_status = str(live_canary.get("overall_status") or "").strip().lower()
    if bool(live_canary.get("supervised_canary_ready", False)):
        rows.append(
            _row(
                surface="live_canary",
                state="ready",
                summary="supervised canary is ready",
                recommended_command=["./scripts/ops/opsctl.sh", "live-canary-control", "--json"],
                metrics={"recommended_mode": str(live_canary.get("recommended_mode") or "")},
            )
        )
    elif (
        bool(live_canary.get("staged_preclearance_ready", False))
        or canary_status in {"degraded", "needs_attention"}
        or guarded_paper_operational
    ):
        rows.append(
            _row(
                surface="live_canary",
                state="degraded",
                summary=(
                    f"recommended_mode={str(live_canary.get('recommended_mode') or '') or 'unknown'} is validate-only while guarded paper is ready and live execution remains locked"
                    if guarded_paper_operational
                    else f"recommended_mode={str(live_canary.get('recommended_mode') or '') or 'unknown'} is still staged, not supervised"
                ),
                recommended_command=["./scripts/ops/opsctl.sh", "live-canary-control", "--json"],
                metrics={
                    "recommended_mode": str(live_canary.get("recommended_mode") or ""),
                    "guarded_paper_soak_advisory": guarded_paper_operational,
                    "live_execution_locked": guarded_paper_operational,
                },
            )
        )
    else:
        rows.append(
            _row(
                surface="live_canary",
                state="blocked",
                summary="live canary fell below staged preclearance",
                recommended_command=["./scripts/ops/opsctl.sh", "live-canary-control", "--json"],
                metrics={"recommended_mode": str(live_canary.get("recommended_mode") or "")},
            )
        )

    autonomy_score = _safe_float(autonomy.get("autonomy_score"), 0.0)
    autonomy_status = str(autonomy.get("overall_status") or "").strip().lower()
    if autonomy_score >= 70.0 and autonomy_status == "ready":
        rows.append(
            _row(
                surface="autonomy_control",
                state="ready",
                summary=f"autonomy_score={autonomy_score:.2f}",
                recommended_command=["./scripts/ops/opsctl.sh", "autonomy-control", "--json"],
                metrics={"autonomy_score": round(autonomy_score, 2)},
            )
        )
    elif autonomy_score >= 50.0 or autonomy_status in {"degraded", "needs_attention"}:
        rows.append(
            _row(
                surface="autonomy_control",
                state="degraded",
                summary=f"autonomy_score={autonomy_score:.2f} is stable enough to protect gains but not yet self-clearing",
                recommended_command=["./scripts/ops/opsctl.sh", "autonomy-control", "--json"],
                metrics={"autonomy_score": round(autonomy_score, 2)},
            )
        )
    else:
        rows.append(
            _row(
                surface="autonomy_control",
                state="blocked",
                summary=f"autonomy_score={autonomy_score:.2f} regressed below the prevention floor",
                recommended_command=["./scripts/ops/opsctl.sh", "autonomy-control", "--json"],
                metrics={"autonomy_score": round(autonomy_score, 2)},
            )
        )

    promotion_status = str(promotion_autopilot.get("overall_status") or "").strip().lower()
    packet_score = _safe_float(promotion_autopilot.get("packet_completeness_score"), 0.0)
    if bool(promotion_autopilot.get("promotion_ready", False)):
        rows.append(
            _row(
                surface="promotion_autopilot",
                state="ready",
                summary=f"packet_completeness_score={packet_score:.2f}",
                recommended_command=["./scripts/ops/opsctl.sh", "promotion-autopilot", "--json"],
                metrics={"packet_completeness_score": round(packet_score, 2)},
            )
        )
    elif promotion_status in {"degraded", "needs_attention"} or packet_score >= 25.0:
        rows.append(
            _row(
                surface="promotion_autopilot",
                state="degraded",
                summary=f"packet_completeness_score={packet_score:.2f} with repairable gates still open",
                recommended_command=["./scripts/ops/opsctl.sh", "promotion-autopilot", "--json"],
                metrics={"packet_completeness_score": round(packet_score, 2)},
            )
        )
    else:
        rows.append(
            _row(
                surface="promotion_autopilot",
                state="blocked",
                summary=f"packet_completeness_score={packet_score:.2f} fell below the promotion recovery floor",
                recommended_command=["./scripts/ops/opsctl.sh", "promotion-autopilot", "--json"],
                metrics={"packet_completeness_score": round(packet_score, 2)},
            )
        )

    retry_policies = {
        "training_quality": _retry_budget(surface="training_quality", max_attempts=2, cooldown_minutes=30, timeout_sec=180, notify_tenant=True),
        "training_lineage": _retry_budget(surface="training_lineage", max_attempts=2, cooldown_minutes=45, timeout_sec=180, notify_tenant=True),
        "storage_control": _retry_budget(surface="storage_control", max_attempts=1, cooldown_minutes=60, timeout_sec=900, quiet_hours_preferred=True),
        "security_audit": _retry_budget(surface="security_audit", max_attempts=1, cooldown_minutes=120, timeout_sec=300, notify_tenant=True),
        "incident_closeout": _retry_budget(surface="incident_closeout", max_attempts=2, cooldown_minutes=20, timeout_sec=180, notify_tenant=True),
        "live_canary": _retry_budget(surface="live_canary", max_attempts=2, cooldown_minutes=20, timeout_sec=180, notify_tenant=True),
        "autonomy_control": _retry_budget(surface="autonomy_control", max_attempts=2, cooldown_minutes=30, timeout_sec=180),
        "promotion_autopilot": _retry_budget(surface="promotion_autopilot", max_attempts=2, cooldown_minutes=30, timeout_sec=180, notify_tenant=True),
    }
    for row in rows:
        surface = str(row.get("surface") or "")
        state = str(row.get("state") or "")
        row["retry_budget"] = retry_policies.get(
            surface,
            _retry_budget(surface=surface or "unknown", max_attempts=1, cooldown_minutes=30, timeout_sec=180),
        )
        row["quiet_hours_preferred"] = bool((row.get("retry_budget") or {}).get("quiet_hours_preferred", False))
        row["notification_contract"] = _notification_contract(surface, state, str(row.get("summary") or ""))

    blocked_count = sum(1 for row in rows if row["state"] == "blocked")
    degraded_count = sum(1 for row in rows if row["state"] == "degraded")
    overall_status = "ready"
    if blocked_count:
        overall_status = "blocked"
    elif degraded_count:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            str(row.get("summary") or "")
            for row in rows
            if str(row.get("state") or "") in {"blocked", "degraded"}
        ]
        + [
            "keep the regression autopilot active so training, promotion, storage, and incident surfaces are republished before score drift compounds"
            if rows
            else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "blocked_surface_count": blocked_count,
        "degraded_surface_count": degraded_count,
        "surface_count": len(rows),
        "surfaces": rows,
        "regression_guardrail_contract": {
            "generation": "grade_regression_guard_v3",
            "per_surface_retry_budgets": True,
            "quiet_hours_aware": True,
            "tenant_notification_contract": True,
            "guarded_paper_soak_advisory_gates": True,
            "blocked_surfaces_notify_tenant": [
                str(row.get("surface") or "")
                for row in rows
                if str(row.get("state") or "") == "blocked"
                and bool(((row.get("notification_contract") or {}).get("tenant_visible", False)))
            ],
        },
        "upgrade_track": {
            "family": "infrabots",
            "upgradeable": True,
            "current_generation": "grade_regression_guard_v3",
            "co_managed_with": [
                "grade_lift_hardening",
                "runtime_artifact_refresh",
                "autonomy_control_plane",
            ],
            "future_upgrade_paths": [
                "automatic threshold tuning from recent live grades",
                "tenant-specific score baselines and notification routing for the licensing API",
                "launchd escalation hooks that page only on blocked regressions after retry budget exhaustion",
            ],
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch the highest-value grade surfaces and flag regression before it turns into a larger system downgrade.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "grade_regression_guard "
            f"overall_status={payload.get('overall_status', '')} "
            f"blocked_surface_count={payload.get('blocked_surface_count', 0)} "
            f"degraded_surface_count={payload.get('degraded_surface_count', 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
