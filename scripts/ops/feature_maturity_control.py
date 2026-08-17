#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "feature_maturity_control_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.feature_maturity_control_override"
TARGET_LEVEL = 4

MATURITY_LABELS = {
    0: "missing",
    1: "implemented",
    2: "instrumented",
    3: "guarded",
    4: "automated_research_grade",
    5: "production_proven",
}


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


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on", "ready", "active", "ok", "armed"}


def _status_clear(status: Any, *, allow_watch: bool = True) -> bool:
    normalized = str(status or "").strip().lower()
    allowed = {"ready", "ok", "active", "stable", "armed", "complete", "constrained", "advisory"}
    if allow_watch:
        allowed.update({"watch", "needs_attention"})
    return normalized in allowed


def _health(project_root: Path, filename: str) -> tuple[Path, dict[str, Any]]:
    path = project_root / "governance" / "health" / filename
    return path, load_json(path)


def _artifact(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    age = payload_age_minutes(payload, path) if payload else None
    return {
        "path": str(path),
        "present": bool(payload),
        "age_minutes": round(float(age), 3) if age is not None else None,
    }


def _criterion(id_: str, ok: bool, evidence: str, repair: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": id_,
        "ok": bool(ok),
        "evidence": evidence,
        "repair_command": repair or [],
    }


def _row(
    *,
    slug: str,
    title: str,
    target_level: int,
    criteria: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    safety_mode: str = "normal",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    passed = sum(1 for item in criteria if bool(item.get("ok", False)))
    level = max(0, min(int(passed), 5))
    gap = max(int(target_level) - level, 0)
    if level >= target_level:
        status = "ready"
    elif level >= max(target_level - 1, 0):
        status = "watch"
    elif level >= 2:
        status = "needs_work"
    else:
        status = "blocked"
    blockers = [str(item.get("id")) for item in criteria if not bool(item.get("ok", False))]
    repairs = [item.get("repair_command") for item in criteria if not bool(item.get("ok", False)) and item.get("repair_command")]
    return {
        "slug": slug,
        "title": title,
        "overall_status": status,
        "target_level": int(target_level),
        "maturity_level": level,
        "maturity_label": MATURITY_LABELS.get(level, "unknown"),
        "target_label": MATURITY_LABELS.get(int(target_level), "unknown"),
        "maturity_gap": gap,
        "safety_mode": safety_mode,
        "criteria_passed": passed,
        "criteria_total": len(criteria),
        "criteria": criteria,
        "blockers": blockers,
        "recommended_commands": repairs[:5],
        "artifacts": artifacts,
        "notes": notes or [],
    }


def _paper_trading(project_root: Path) -> dict[str, Any]:
    runtime_path, runtime = _health(project_root, "runtime_throttle_control_latest.json")
    performance_path, performance = _health(project_root, "paper_performance_latest.json")
    paper_policy = _as_dict(runtime.get("paper_execution_policy"))
    capacity = _as_dict(runtime.get("paper_capacity_contract"))
    release = _as_dict(runtime.get("release_contract"))
    runtime_policy = _as_dict(capacity.get("runtime_policy"))
    criteria = [
        _criterion("paper_policy_artifact_present", bool(paper_policy), "runtime paper policy present"),
        _criterion("paper_execution_allowed", _bool(paper_policy.get("paper_execution_allowed")), str(paper_policy.get("reason") or "")),
        _criterion("paper_ramp_armed", _bool(paper_policy.get("armed")) and not _bool(paper_policy.get("pause_paper_execution")), str(paper_policy.get("stage") or "")),
        _criterion("paper_capacity_ready", _bool(capacity.get("ready_for_700_bot_paper")), f"active_bots={capacity.get('active_bot_count')} paper_tagged={capacity.get('paper_tagged_count')}"),
        _criterion("live_execution_locked", _bool(release.get("live_lane_should_be_read_only")) or _bool(runtime_policy.get("live_execution_blocked")), "paper/live separation enforced"),
    ]
    return _row(
        slug="paper_trading_execution",
        title="Paper Trading Execution",
        target_level=TARGET_LEVEL,
        criteria=criteria,
        artifacts=[_artifact(runtime_path, runtime), _artifact(performance_path, performance)],
        safety_mode="paper_only_live_locked",
    )


def _ingestion_sql(project_root: Path) -> dict[str, Any]:
    storage_path, storage = _health(project_root, "ingestion_storage_control_latest.json")
    sql_path, sql = _health(project_root, "sql_link_service_progress_latest.json")
    backpressure = _as_dict(storage.get("backpressure"))
    smart = _as_dict(_as_dict(sql.get("shard_writer_lane_contract")).get("smart_shard_parallelism"))
    pending_total = max(_safe_int(backpressure.get("total_pending_lines"), 0), _safe_int(storage.get("total_pending_lines"), 0))
    threshold = max(_safe_int(backpressure.get("pending_lines_threshold"), 15000), 1)
    oldest = max(_safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0), _safe_float(storage.get("oldest_pending_age_seconds"), 0.0))
    criteria = [
        _criterion("storage_artifact_present", bool(storage), "ingestion storage health present"),
        _criterion("storage_pressure_stable", pending_total < threshold and oldest < 240.0 and _safe_float(storage.get("pressure_index"), 0.0) < 1.0, f"pending={pending_total}/{threshold} oldest={oldest:.1f}s"),
        _criterion("sql_writer_complete", str(sql.get("current_step") or "").lower() == "complete" and _status_clear(sql.get("status")), str(sql.get("status") or "")),
        _criterion("sql_shards_clean", _safe_int(sql.get("timed_out_shard_count"), 0) == 0, f"timed_out={sql.get('timed_out_shard_count')} completed={sql.get('completed_shard_count')}/{sql.get('planned_shard_count')}"),
        _criterion("smart_single_writer_parallelism", _bool(smart.get("enabled")) and _bool(smart.get("enforced_single_primary_merge_writer")), str(smart.get("policy") or "")),
    ]
    return _row(
        slug="data_ingestion_sql_telemetry",
        title="Data Intake, Ingestion, And SQL Telemetry",
        target_level=TARGET_LEVEL,
        criteria=criteria,
        artifacts=[_artifact(storage_path, storage), _artifact(sql_path, sql)],
    )


def _runtime_governance(project_root: Path) -> dict[str, Any]:
    runtime_path, runtime = _health(project_root, "runtime_throttle_control_latest.json")
    infra_path, infra = _health(project_root, "infrabot_adaptive_governor_latest.json")
    stabilization_path, stabilization = _health(project_root, "platform_stabilization_quality_latest.json")
    controller = _as_dict(runtime.get("controller_contract"))
    attribution = _as_dict(runtime.get("host_pressure_attribution"))
    criteria = [
        _criterion("runtime_governor_present", bool(runtime), "runtime throttle artifact present"),
        _criterion("runtime_not_degraded", _status_clear(runtime.get("overall_status")), f"status={runtime.get('overall_status')} saturation={runtime.get('host_saturation_score')}"),
        _criterion("controller_apply_capable", str(controller.get("mode") or "") in {"apply_capable", "applied"} and _bool(controller.get("safe_while_live")), str(controller.get("mode") or "")),
        _criterion("support_pressure_identified", "support_trim_required" in attribution or "support_jobs_hot" in attribution, "host pressure attribution separates support jobs"),
        _criterion("infra_or_stabilizer_present", bool(infra or stabilization), "adaptive governor or stabilization artifact present"),
    ]
    return _row(
        slug="runtime_governance_self_healing",
        title="Runtime Governance And Self-Healing",
        target_level=TARGET_LEVEL,
        criteria=criteria,
        artifacts=[_artifact(runtime_path, runtime), _artifact(infra_path, infra), _artifact(stabilization_path, stabilization)],
    )


def _training_ml(project_root: Path) -> dict[str, Any]:
    training_path, training = _health(project_root, "training_runtime_control_latest.json")
    backend = _as_dict(training.get("runtime_backend_parity"))
    launch = _as_dict(training.get("training_launch_contract"))
    quality = _as_dict(training.get("training_quality"))
    criteria = [
        _criterion("training_artifact_present", bool(training), "training runtime control present"),
        _criterion("snapshot_ready", _bool(training.get("snapshot_ready")), f"snapshot_age={training.get('snapshot_age_minutes')}m"),
        _criterion("runtime_backend_ready", str(backend.get("parity_state") or "") == "ready" and bool(backend.get("runtime_matches_current", False)), str(backend.get("runtime_python_version") or "")),
        _criterion("prep_allowed", _bool(launch.get("prep_allowed")) and not _as_list(launch.get("prep_blockers")), f"mode={launch.get('mode')}"),
        _criterion("quality_not_blocked", str(quality.get("overall_status") or "").lower() not in {"blocked", "critical"} and _safe_float(quality.get("training_quality_score"), 0.0) >= 90.0, f"score={quality.get('training_quality_score')} status={quality.get('overall_status')}"),
    ]
    notes = []
    if not _bool(launch.get("launch_allowed")):
        notes.append("full training launch is intentionally gated until host headroom and autonomic budget reopen")
    return _row(
        slug="ml_training_pipeline",
        title="ML Training Pipeline",
        target_level=TARGET_LEVEL,
        criteria=criteria,
        artifacts=[_artifact(training_path, training)],
        safety_mode="prep_only_until_headroom_reopens",
        notes=notes,
    )


def _labeling_quality(project_root: Path) -> dict[str, Any]:
    quality_path, quality = _health(project_root, "training_quality_control_latest.json")
    bot_quality_path, bot_quality = _health(project_root, "bot_quality_autopilot_latest.json")
    needs_path, needs = _health(project_root, "bot_needs_intelligence_latest.json")
    criteria = [
        _criterion("training_quality_present", bool(quality), "training quality artifact present"),
        _criterion("training_quality_score_high", _safe_float(quality.get("training_quality_score"), 0.0) >= 90.0, f"score={quality.get('training_quality_score')}"),
        _criterion("not_quality_blocked", str(quality.get("overall_status") or "").lower() not in {"blocked", "critical"}, str(quality.get("overall_status") or "")),
        _criterion("bot_quality_control_present", bool(bot_quality), "bot quality autopilot present"),
        _criterion("needs_queue_visible", bool(needs) and isinstance(needs.get("need_counts"), dict), "bot needs queue visible"),
    ]
    return _row(
        slug="data_labeling_quality",
        title="Data Labeling And Training Quality",
        target_level=TARGET_LEVEL,
        criteria=criteria,
        artifacts=[_artifact(quality_path, quality), _artifact(bot_quality_path, bot_quality), _artifact(needs_path, needs)],
    )


def _drift_anomaly(project_root: Path) -> dict[str, Any]:
    drift_path, drift = _health(project_root, "system_drift_guard_latest.json")
    registry_path, registry = _health(project_root, "system_drift_registry_latest.json")
    watchdog_path, watchdog = _health(project_root, "watchdog_intelligence_latest.json")
    memory_path, memory = _health(project_root, "memory_pressure_intelligence_latest.json")
    criteria = [
        _criterion("drift_guard_present", bool(drift), "system drift guard present", ["./scripts/ops/opsctl.sh", "system-drift-guard", "--json"]),
        _criterion("drift_registry_present", bool(registry), "system drift registry present", ["./scripts/ops/opsctl.sh", "system-drift-registry", "--json"]),
        _criterion("watchdog_or_memory_present", bool(watchdog or memory), "watchdog/memory anomaly surface present", ["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--json"]),
        _criterion(
            "no_critical_drift",
            str(drift.get("overall_status") or "").lower() not in {"critical", "blocked"},
            str(drift.get("overall_status") or ""),
            ["./scripts/ops/opsctl.sh", "system-drift-autopilot", "--apply", "--max-steps", "3", "--json"],
        ),
        _criterion(
            "automatic_repair_route_visible",
            bool(
                _as_list(drift.get("recommended_commands"))
                or _as_list(drift.get("recommended_actions"))
                or _as_list(watchdog.get("recommended_commands"))
                or memory
            ),
            "repair route exists",
            ["./scripts/ops/opsctl.sh", "system-drift-autopilot", "--apply", "--max-steps", "3", "--json"],
        ),
    ]
    return _row(
        slug="drift_anomaly_monitoring",
        title="Drift And Anomaly Monitoring",
        target_level=TARGET_LEVEL,
        criteria=criteria,
        artifacts=[_artifact(drift_path, drift), _artifact(registry_path, registry), _artifact(watchdog_path, watchdog), _artifact(memory_path, memory)],
    )


def _explainability_governance(project_root: Path) -> dict[str, Any]:
    provenance_path, provenance = _health(project_root, "decision_provenance_cards_latest.json")
    telemetry_path, telemetry = _health(project_root, "governance_telemetry_compactor_latest.json")
    packet_path, packet = _health(project_root, "evidence_packet_latest.json")
    criteria = [
        _criterion("decision_provenance_present", bool(provenance), "decision provenance surface present", ["./scripts/ops/opsctl.sh", "decision-provenance", "--json"]),
        _criterion("provenance_not_blocked", str(provenance.get("overall_status") or "").lower() not in {"blocked", "critical"}, str(provenance.get("overall_status") or ""), ["./scripts/ops/opsctl.sh", "decision-provenance", "--json"]),
        _criterion("governance_telemetry_present", bool(telemetry), "governance telemetry compactor present", ["./scripts/ops/opsctl.sh", "governance-telemetry-compactor", "--json"]),
        _criterion("evidence_packet_present", bool(packet), "evidence packet surface present", ["./scripts/ops/opsctl.sh", "evidence-packet", "--json"]),
        _criterion("cards_or_evidence_counted", _safe_int(provenance.get("card_count"), 0) > 0 or bool(packet), f"cards={provenance.get('card_count')}", ["./scripts/ops/opsctl.sh", "evidence-packet", "--json"]),
    ]
    return _row(
        slug="decision_explainability_governance",
        title="Decision Explainability And Governance",
        target_level=TARGET_LEVEL,
        criteria=criteria,
        artifacts=[_artifact(provenance_path, provenance), _artifact(telemetry_path, telemetry), _artifact(packet_path, packet)],
    )


def _live_safety(project_root: Path) -> dict[str, Any]:
    runtime_path, runtime = _health(project_root, "runtime_throttle_control_latest.json")
    live_path, live = _health(project_root, "live_runtime_separation_control_latest.json")
    broker_path, broker = _health(project_root, "broker_readiness_latest.json")
    release = _as_dict(runtime.get("release_contract"))
    paper_policy = _as_dict(runtime.get("paper_execution_policy"))
    criteria = [
        _criterion("runtime_release_contract_present", bool(release), "release contract present"),
        _criterion("live_read_only_locked", _bool(release.get("live_lane_should_be_read_only")) or _bool(release.get("release_live_lane_should_be_read_only")), str(release.get("effective_live_read_only_reason") or "")),
        _criterion("paper_trade_lock_active", _bool(release.get("paper_trade_lock_active")) or _bool(paper_policy.get("paper_execution_allowed")), "paper-trade lock or paper policy active"),
        _criterion("live_separation_surface_present", bool(live), "live runtime separation control present"),
        _criterion("broker_readiness_observed", bool(broker), "broker readiness artifact present"),
    ]
    return _row(
        slug="live_execution_safety",
        title="Live Execution Safety",
        target_level=TARGET_LEVEL,
        criteria=criteria,
        artifacts=[_artifact(runtime_path, runtime), _artifact(live_path, live), _artifact(broker_path, broker)],
        safety_mode="mature_means_locked_until_release",
        notes=["do not score this feature by whether live orders are enabled; score it by separation, broker truth, and release controls"],
    )


def _feature_store_flow(project_root: Path) -> dict[str, Any]:
    source_path, source = _health(project_root, "source_verification_latest.json")
    rollup_path, rollup = _health(project_root, "data_collection_observation_rollup_latest.json")
    point_path, point = _health(project_root, "point_in_time_event_store_latest.json")
    sql_path, sql = _health(project_root, "sql_link_service_progress_latest.json")
    criteria = [
        _criterion("source_verification_present", bool(source), "source verification present"),
        _criterion("source_verification_not_blocked", str(source.get("overall_status") or "").lower() not in {"blocked", "critical"}, str(source.get("overall_status") or "")),
        _criterion("observation_rollup_present", bool(rollup), "data collection observation rollup present"),
        _criterion("point_in_time_or_sql_present", bool(point or sql), "point-in-time store or SQL telemetry present"),
        _criterion("feature_flow_has_observations", _safe_int(rollup.get("bots_with_observations"), 0) > 0 or _safe_int(sql.get("merged_rows_this_cycle"), 0) > 0, f"observed={rollup.get('bots_with_observations')} merged={sql.get('merged_rows_this_cycle')}"),
    ]
    return _row(
        slug="feature_store_research_data_flow",
        title="Feature Store And Research Data Flow",
        target_level=TARGET_LEVEL,
        criteria=criteria,
        artifacts=[_artifact(source_path, source), _artifact(rollup_path, rollup), _artifact(point_path, point), _artifact(sql_path, sql)],
    )


def _rows(project_root: Path) -> list[dict[str, Any]]:
    return [
        _paper_trading(project_root),
        _ingestion_sql(project_root),
        _runtime_governance(project_root),
        _training_ml(project_root),
        _labeling_quality(project_root),
        _drift_anomaly(project_root),
        _explainability_governance(project_root),
        _live_safety(project_root),
        _feature_store_flow(project_root),
    ]


def _overall_status(rows: list[dict[str, Any]]) -> str:
    if any(str(row.get("overall_status")) == "blocked" for row in rows):
        return "blocked"
    if any(str(row.get("overall_status")) == "needs_work" for row in rows):
        return "needs_work"
    if any(str(row.get("overall_status")) == "watch" for row in rows):
        return "advisory"
    return "ready"


def _recommended_env(payload: dict[str, Any]) -> dict[str, str]:
    below = [row for row in _as_list(payload.get("features")) if _safe_int(_as_dict(row).get("maturity_gap"), 0) > 0]
    return {
        "FEATURE_MATURITY_CONTROL_ENABLED": "1",
        "FEATURE_MATURITY_TARGET_LEVEL": str(TARGET_LEVEL),
        "FEATURE_MATURITY_BELOW_TARGET_COUNT": str(len(below)),
        "FEATURE_MATURITY_LOWEST_LEVEL": str(payload.get("lowest_maturity_level")),
        "FEATURE_MATURITY_LIVE_EXECUTION_TARGET": "locked_release_gated",
        "FEATURE_MATURITY_AUTOMATION_TARGET": "guarded_self_healing_research_grade",
        "PAPER_TRADE_LOCK": "1",
        "ALLOW_ORDER_EXECUTION": "0",
        "MARKET_DATA_ONLY": "1",
        "SQL_LINK_SERVICE_SMART_SHARD_PARALLELISM": "1",
        "TRAINING_READY_ONLY_MICROBATCH_ENABLED": "1",
        "TRAINING_REQUIRE_BACKPRESSURE_CLEAR": "1",
        "DATA_LABELING_AUTO_REPAIR_ENABLED": "1",
        "FEATURE_STORE_HEALTH_GATE_ENABLED": "1",
        "DRIFT_MONITORING_REQUIRED": "1",
        "DECISION_EXPLAINABILITY_REQUIRED": "1",
        "PROVIDER_SOURCE_VERIFICATION_REQUIRED": "1",
        "SELF_HEALING_INFRA_LINKS_REQUIRED": "1",
        "LIVE_EXECUTION_RELEASE_GATE_REQUIRED": "1",
    }


def _write_env_override(path: Path, env: dict[str, str]) -> bool:
    lines = ["# Auto-managed by scripts/ops/feature_maturity_control.py"]
    for key in sorted(env):
        lines.append(f"{key}={shlex.quote(str(env[key]))}")
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    rows = _rows(project_root)
    below = [row for row in rows if _safe_int(row.get("maturity_gap"), 0) > 0]
    sorted_below = sorted(below, key=lambda row: (-_safe_int(row.get("maturity_gap"), 0), _safe_int(row.get("maturity_level"), 0), str(row.get("slug"))))
    next_commands: list[list[str]] = []
    seen: set[str] = set()
    for row in sorted_below:
        for command in _as_list(row.get("recommended_commands")):
            if not isinstance(command, list):
                continue
            key = " ".join(str(part) for part in command)
            if key in seen:
                continue
            seen.add(key)
            next_commands.append([str(part) for part in command])
            if len(next_commands) >= 8:
                break
        if len(next_commands) >= 8:
            break
    lowest = min([_safe_int(row.get("maturity_level"), 0) for row in rows] or [0])
    average = sum(_safe_float(row.get("maturity_level"), 0.0) for row in rows) / float(len(rows) or 1)
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "target_level": TARGET_LEVEL,
        "target_label": MATURITY_LABELS[TARGET_LEVEL],
        "max_level": 5,
        "overall_status": _overall_status(rows),
        "ok": not below,
        "feature_count": len(rows),
        "features_at_or_above_target": len(rows) - len(below),
        "features_below_target": len(below),
        "lowest_maturity_level": lowest,
        "average_maturity_level": round(average, 3),
        "below_target_slugs": [str(row.get("slug")) for row in sorted_below],
        "features": rows,
        "next_best_commands": next_commands,
        "policy": "level_all_features_to_guarded_automated_research_grade_without_enabling_live_order_execution",
    }
    payload["recommended_env_overrides"] = _recommended_env(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Uniform maturity control plane for major platform features.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override-path", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser()
    payload = build_payload(project_root)
    out_path = Path(args.out).expanduser()
    write_payload(out_path, payload)
    payload["out_path"] = str(out_path)
    if args.apply:
        override_path = Path(args.override_path).expanduser()
        changed = _write_env_override(override_path, _as_dict(payload.get("recommended_env_overrides")))
        payload["apply_result"] = {
            "applied": True,
            "override_path": str(override_path),
            "override_changed": bool(changed),
        }
        write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "feature_maturity "
            f"status={payload['overall_status']} "
            f"target={payload['target_label']} "
            f"at_target={payload['features_at_or_above_target']}/{payload['feature_count']} "
            f"lowest={payload['lowest_maturity_level']}"
        )
    return 0 if payload["overall_status"] in {"ready", "advisory", "needs_work"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
