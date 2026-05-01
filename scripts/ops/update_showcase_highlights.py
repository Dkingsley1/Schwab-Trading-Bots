#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import html
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = PROJECT_ROOT / "docs" / "showcase"
GENERATED_ROOT = DOCS_ROOT / "generated"
README_PATH = PROJECT_ROOT / "README.md"
HIGHLIGHTS_JSON = GENERATED_ROOT / "highlights_latest.json"
HIGHLIGHTS_MD = GENERATED_ROOT / "highlights_latest.md"
SPECIAL_FEATURES_HTML = GENERATED_ROOT / "special_features_latest.html"
README_START = "<!-- SHOWCASE_HIGHLIGHTS_START -->"
README_END = "<!-- SHOWCASE_HIGHLIGHTS_END -->"


def _safe_load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _fmt_pct(raw: float | None) -> str:
    if raw is None:
        return "n/a"
    return f"{raw * 100.0:.1f}%"


def _fmt_ratio_pct(raw: float | None) -> str:
    if raw is None:
        return "n/a"
    return f"{raw * 100.0:.2f}%"


def _fmt_compact_timestamp(raw: Any) -> str:
    if not raw:
        return "unknown"
    text = str(raw).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return text
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _payload_age_hours(payload: Mapping[str, Any] | None) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    raw = payload.get("timestamp_utc")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max((datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600.0, 0.0)


def _fmt_age_hours(raw: float | None) -> str:
    if raw is None:
        return "unknown age"
    if raw < 1.0:
        return f"{raw * 60.0:.0f}m old"
    if raw < 48.0:
        return f"{raw:.1f}h old"
    return f"{raw / 24.0:.1f}d old"


def _active_bot_summary() -> dict[str, Any]:
    registry = _safe_load_json(PROJECT_ROOT / "master_bot_registry.json", default={})
    sub_bots = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    active = [row for row in sub_bots if isinstance(row, Mapping) and bool(row.get("active"))]
    roles = Counter(str(row.get("bot_role") or "unknown") for row in active)
    protected_lanes = registry.get("master_policy", {}).get("protected_collection_lane_floors", {}) if isinstance(registry.get("master_policy"), Mapping) else {}
    top_active = []
    for row in sorted(active, key=lambda item: _safe_float(item.get("test_accuracy"), -1.0), reverse=True)[:5]:
        top_active.append(
            {
                "bot_id": row.get("bot_id"),
                "bot_role": row.get("bot_role"),
                "test_accuracy": _safe_float(row.get("test_accuracy"), 0.0),
                "quality_score": _safe_float(row.get("quality_score"), 0.0),
                "reason": row.get("reason"),
            }
        )
    return {
        "total_registered": len(sub_bots),
        "active_count": len(active),
        "active_roles": dict(roles),
        "protected_collection_lane_floors": protected_lanes,
        "top_active_bots": top_active,
    }


def _live_lane_summary() -> dict[str, Any]:
    health_dir = PROJECT_ROOT / "governance" / "health"
    lane_files = sorted(health_dir.glob("data_ingress_latest_*.json"))
    lanes = []
    running = 0
    for path in lane_files:
        payload = _safe_load_json(path, default={})
        if not isinstance(payload, Mapping):
            continue
        loop_state = str(payload.get("loop_state") or "").strip().lower()
        lane_name = path.stem.replace("data_ingress_latest_", "")
        lanes.append(
            {
                "lane": lane_name,
                "loop_state": loop_state,
                "iter": int(_safe_float(payload.get("iter"), 0.0)),
                "api_error": int(_safe_float(payload.get("api_error"), 0.0)),
            }
        )
        if loop_state == "running":
            running += 1
    return {
        "lane_count": len(lanes),
        "running_count": running,
        "lanes": lanes,
    }


def _artifact_snapshot() -> dict[str, Any]:
    health_dir = PROJECT_ROOT / "governance" / "health"
    reports_dir = PROJECT_ROOT / "exports" / "reports"
    champion_dir = PROJECT_ROOT / "governance" / "champion_challenger"
    crypto_ctx = _safe_load_json(health_dir / "crypto_market_context_sync_latest.json", default={})
    divergence = _safe_load_json(health_dir / "data_source_divergence_latest.json", default={})
    correlation = _safe_load_json(health_dir / "market_crypto_correlation_sync_latest.json", default={})
    training = _safe_load_json(health_dir / "training_success_latest.json", default={})
    watchdog = _safe_load_json(health_dir / "shadow_watchdog_tripwire_latest.json", default={})
    process_watchdog = _safe_load_json(health_dir / "process_watchdog_latest.json", default={})
    live_readiness = _safe_load_json(health_dir / "live_readiness_smoke_latest.json", default={})
    runtime_separation = _safe_load_json(health_dir / "live_runtime_separation_control_latest.json", default={})
    platform_control = _safe_load_json(health_dir / "platform_control_plane_latest.json", default={})
    pytorch_replay = _safe_load_json(health_dir / "pytorch_replay_canary_latest.json", default={})
    autonomy_control = _safe_load_json(health_dir / "autonomy_control_plane_latest.json", default={})
    incident_timeline = _safe_load_json(health_dir / "incident_timeline_latest.json", default={})
    portable_brain = _safe_load_json(health_dir / "portable_brain_contract_latest.json", default={})
    switchboard = _safe_load_json(health_dir / "mode_switchboard_mission_control_latest.json", default={})
    provenance = _safe_load_json(health_dir / "decision_provenance_cards_latest.json", default={})
    notifications = _safe_load_json(health_dir / "notification_escalation_ladder_latest.json", default={})
    incident_review = _safe_load_json(health_dir / "incident_review_packet_latest.json", default={})
    architecture_upgrades = _safe_load_json(health_dir / "architecture_upgrade_scoreboard_latest.json", default={})
    chaos_drills = _safe_load_json(health_dir / "chaos_drill_coordinator_latest.json", default={})
    promotion_autopilot = _safe_load_json(champion_dir / "promotion_autopilot_packet_latest.json", default={})
    daily_ops = _safe_load_json(reports_dir / "daily_ops_report_latest.json", default={})
    macro_event = _safe_load_json(health_dir / "macro_event_intelligence_latest.json", default={})
    return {
        "crypto_context": crypto_ctx if isinstance(crypto_ctx, Mapping) else {},
        "divergence": divergence if isinstance(divergence, Mapping) else {},
        "correlation": correlation if isinstance(correlation, Mapping) else {},
        "training": training if isinstance(training, Mapping) else {},
        "watchdog": watchdog if isinstance(watchdog, Mapping) else {},
        "process_watchdog": process_watchdog if isinstance(process_watchdog, Mapping) else {},
        "live_readiness": live_readiness if isinstance(live_readiness, Mapping) else {},
        "runtime_separation": runtime_separation if isinstance(runtime_separation, Mapping) else {},
        "platform_control": platform_control if isinstance(platform_control, Mapping) else {},
        "pytorch_replay": pytorch_replay if isinstance(pytorch_replay, Mapping) else {},
        "autonomy_control": autonomy_control if isinstance(autonomy_control, Mapping) else {},
        "incident_timeline": incident_timeline if isinstance(incident_timeline, Mapping) else {},
        "portable_brain": portable_brain if isinstance(portable_brain, Mapping) else {},
        "switchboard": switchboard if isinstance(switchboard, Mapping) else {},
        "provenance": provenance if isinstance(provenance, Mapping) else {},
        "notifications": notifications if isinstance(notifications, Mapping) else {},
        "incident_review": incident_review if isinstance(incident_review, Mapping) else {},
        "architecture_upgrades": architecture_upgrades if isinstance(architecture_upgrades, Mapping) else {},
        "chaos_drills": chaos_drills if isinstance(chaos_drills, Mapping) else {},
        "promotion_autopilot": promotion_autopilot if isinstance(promotion_autopilot, Mapping) else {},
        "daily_ops": daily_ops if isinstance(daily_ops, Mapping) else {},
        "macro_event": macro_event if isinstance(macro_event, Mapping) else {},
    }


def _special_features_map(artifacts: Mapping[str, Any]) -> dict[str, str]:
    architecture_upgrades = artifacts.get("architecture_upgrades") if isinstance(artifacts.get("architecture_upgrades"), Mapping) else {}
    feature_map = architecture_upgrades.get("special_features_map") if isinstance(architecture_upgrades.get("special_features_map"), Mapping) else {}
    if feature_map:
        return {str(key): str(value) for key, value in feature_map.items()}
    return {
        "adaptive_apple_silicon_brain": "Adaptive Apple Silicon Brain: host-aware tuning exists, but the latest feature-proof artifact is missing.",
        "three_mode_switchboard": "Three-Mode Switchboard: switchboard infrastructure exists, but the latest mission-control artifact is missing.",
        "event_to_trade_intelligence": "Event-to-Trade Intelligence: macro/video ingest infrastructure exists, but the latest proof artifact is missing.",
        "self_healing_ops_plane": "Self-Healing Ops Plane: watchdog and autonomy surfaces exist, but the latest feature-proof artifact is missing.",
        "portable_brain_contract": "Portable Brain Contract: native-versus-portable runtime logic exists, but the latest feature-proof artifact is missing.",
    }


def _special_feature_details(artifacts: Mapping[str, Any], special_features: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    portable_brain = artifacts.get("portable_brain") if isinstance(artifacts.get("portable_brain"), Mapping) else {}
    switchboard = artifacts.get("switchboard") if isinstance(artifacts.get("switchboard"), Mapping) else {}
    autonomy_control = artifacts.get("autonomy_control") if isinstance(artifacts.get("autonomy_control"), Mapping) else {}
    process_watchdog = artifacts.get("process_watchdog") if isinstance(artifacts.get("process_watchdog"), Mapping) else {}
    notifications = artifacts.get("notifications") if isinstance(artifacts.get("notifications"), Mapping) else {}
    incident_review = artifacts.get("incident_review") if isinstance(artifacts.get("incident_review"), Mapping) else {}
    chaos_drills = artifacts.get("chaos_drills") if isinstance(artifacts.get("chaos_drills"), Mapping) else {}
    macro_event = artifacts.get("macro_event") if isinstance(artifacts.get("macro_event"), Mapping) else {}

    host_contract = portable_brain.get("host_contract") if isinstance(portable_brain.get("host_contract"), Mapping) else {}
    adaptation_contract = portable_brain.get("adaptation_contract") if isinstance(portable_brain.get("adaptation_contract"), Mapping) else {}
    native_contract = portable_brain.get("native_contract") if isinstance(portable_brain.get("native_contract"), Mapping) else {}
    portable_contract = portable_brain.get("portable_contract") if isinstance(portable_brain.get("portable_contract"), Mapping) else {}
    cross_platform_proof = portable_brain.get("cross_platform_proof_node") if isinstance(portable_brain.get("cross_platform_proof_node"), Mapping) else {}
    nightly_proof = portable_brain.get("nightly_proof_contract") if isinstance(portable_brain.get("nightly_proof_contract"), Mapping) else {}

    switchboard_modes = switchboard.get("modes") if isinstance(switchboard.get("modes"), list) else []
    switchboard_counts = switchboard.get("mode_counts") if isinstance(switchboard.get("mode_counts"), Mapping) else {}
    control_surface = switchboard.get("control_surface") if isinstance(switchboard.get("control_surface"), Mapping) else {}
    active_modes = [str(row.get("mode") or "") for row in switchboard_modes if isinstance(row, Mapping) and bool(row.get("active"))]
    ready_modes = [str(row.get("mode") or "") for row in switchboard_modes if isinstance(row, Mapping) and bool(row.get("ready"))]

    component_statuses = autonomy_control.get("component_statuses") if isinstance(autonomy_control.get("component_statuses"), Mapping) else {}
    lane_recovery = autonomy_control.get("lane_recovery_playbooks") if isinstance(autonomy_control.get("lane_recovery_playbooks"), Mapping) else {}
    drill_program = chaos_drills.get("drill_program") if isinstance(chaos_drills.get("drill_program"), Mapping) else {}

    restart_storms = process_watchdog.get("restart_storms") if isinstance(process_watchdog.get("restart_storms"), list) else []
    watchdog_status = process_watchdog.get("status") if isinstance(process_watchdog.get("status"), list) else []
    watchdog_healthy = sum(
        1
        for row in watchdog_status
        if isinstance(row, Mapping) and (int(_safe_float(row.get("running"), 0.0)) + int(_safe_float(row.get("alt_running"), 0.0)) > 0) and bool(row.get("heartbeat_ok", False))
    )

    feature_labels = {
        "adaptive_apple_silicon_brain": "Adaptive Apple Silicon Brain",
        "three_mode_switchboard": "Three-Mode Switchboard",
        "event_to_trade_intelligence": "Event-to-Trade Intelligence",
        "self_healing_ops_plane": "Self-Healing Ops Plane",
        "portable_brain_contract": "Portable Brain Contract",
    }

    details = {
        "adaptive_apple_silicon_brain": {
            "label": feature_labels["adaptive_apple_silicon_brain"],
            "summary": str(special_features.get("adaptive_apple_silicon_brain") or ""),
            "why_it_matters": "This matters because Apple Silicon unified memory gives the live stack one shared CPU and GPU pool for feature windows, broker-context caches, and MLX inference, so the same code can stay responsive on a MacBook Air and then scale up hard on Max-class machines without copy-heavy rewrites.",
            "current_watch_item": f"Portable posture is still strongest on native Apple Silicon; non-Mac proof is `{cross_platform_proof.get('status', 'unknown')}` and still about replay parity rather than full live parity.",
            "proof_points": [
                f"Recognized host `{host_contract.get('chip', 'unknown chip')}` on `{host_contract.get('system', 'unknown os')}` with profile `{host_contract.get('host_profile', 'unknown')}`.",
                f"Memory architecture is `{host_contract.get('memory_architecture', 'unknown')}` with shared CPU/GPU pool `{host_contract.get('shared_cpu_gpu_memory_pool', False)}`.",
                str(host_contract.get("memory_competitive_advantage") or "Apple Silicon memory advantage details are not yet published in the host contract."),
                f"Recommended runtime posture is `{portable_brain.get('recommended_runtime_mode', 'unknown')}` with backend `{native_contract.get('effective_backend', portable_brain.get('recommended_backend', 'unknown'))}`.",
                f"Host override file is `{('present' if bool(adaptation_contract.get('override_exists', False)) else 'missing')}` at `{adaptation_contract.get('override_path', 'unknown')}`.",
            ],
        },
        "three_mode_switchboard": {
            "label": feature_labels["three_mode_switchboard"],
            "summary": str(special_features.get("three_mode_switchboard") or ""),
            "why_it_matters": "This is the control surface that keeps the same trading brain coherent across shadow, paper, and live instead of forcing three separate systems to drift apart over time.",
            "current_watch_item": f"Runtime clearance is still `{control_surface.get('clearance_state', 'unknown')}`, which means the switchboard is operationally honest about when live should stay read-only.",
            "proof_points": [
                f"Switchboard currently tracks `{_safe_int(switchboard_counts.get('active'), 0)}` active modes and `{_safe_int(switchboard_counts.get('ready'), 0)}` ready modes.",
                f"Active modes: `{', '.join(active_modes) or 'none'}`; ready modes: `{', '.join(ready_modes) or 'none'}`.",
                f"Control surface clearance is `{control_surface.get('clearance_state', 'unknown')}` with live read-only `{control_surface.get('live_lane_should_be_read_only', False)}`.",
            ],
        },
        "event_to_trade_intelligence": {
            "label": feature_labels["event_to_trade_intelligence"],
            "summary": str(special_features.get("event_to_trade_intelligence") or ""),
            "why_it_matters": "It gives the platform a route from macro hearings, policy streams, and transcripts into market-aware stance, relevance, and bulletin surfaces that the rest of the brain can actually use.",
            "current_watch_item": f"Current transcript pipeline is `{macro_event.get('transcript_quality', 'unknown')}` and should keep moving toward fully clean replay-grade transcripts for every event.",
            "proof_points": [
                f"Latest macro event status is `{macro_event.get('overall_status', 'missing')}` from `{macro_event.get('source', 'unknown source')}` with speaker `{macro_event.get('speaker', 'unknown speaker')}`.",
                f"Transcript quality is `{macro_event.get('transcript_quality', 'unknown')}` at `{_safe_float(macro_event.get('transcript_quality_score'), 0.0):.4f}`, cue match `{_safe_float(macro_event.get('cue_match_score'), 0.0):.4f}`.",
                f"Market read is `{macro_event.get('stance', 'unknown')}` with sentiment `{_safe_float(macro_event.get('sentiment_hint'), 0.0):.4f}` and relevance `{macro_event.get('market_relevance', 'unknown')}`.",
            ],
        },
        "self_healing_ops_plane": {
            "label": feature_labels["self_healing_ops_plane"],
            "summary": str(special_features.get("self_healing_ops_plane") or ""),
            "why_it_matters": "It is the difference between a platform that merely runs and one that can diagnose pressure, throttle itself, freeze bad lanes, and preserve operator trust while the rest of the stack keeps moving.",
            "current_watch_item": f"Incident review is currently `{incident_review.get('overall_status', 'unknown') or ('review_required' if incident_review.get('review_required') else 'ready')}`, so the self-healing story is strong but still not fully frictionless.",
            "proof_points": [
                f"Autonomy score is `{_safe_float(autonomy_control.get('autonomy_score'), 0.0):.2f}/100` with `{_safe_int(autonomy_control.get('autonomous_repair_path_count'), 0)}` autonomous repair paths.",
                f"Triggered playbooks: `{_safe_int(lane_recovery.get('triggered_playbook_count'), 0)}`; notification ladder `{notifications.get('overall_status', 'unknown')}`; incident review `{incident_review.get('overall_status', 'unknown') or ('review_required' if incident_review.get('review_required') else 'ready')}`.",
                f"Process watchdog shows `{watchdog_healthy}/{len(watchdog_status)}` healthy targets and `{len(restart_storms)}` restart storms; chaos drill score `{_safe_float(drill_program.get('program_score'), 0.0):.2f}`.",
            ],
        },
        "portable_brain_contract": {
            "label": feature_labels["portable_brain_contract"],
            "summary": str(special_features.get("portable_brain_contract") or ""),
            "why_it_matters": "This is the selling point that keeps the platform from being a dead-end Mac-only build: Apple Silicon stays first-class for the live brain, but the runtime now has an explicit broker-agnostic contract for replay, research, and proof on Linux and Windows.",
            "current_watch_item": f"Next portability milestone is `{portable_brain.get('next_step', 'unknown')}`, which is still the bridge between strong design and undeniable parity proof.",
            "proof_points": [
                f"Native contract is `{native_contract.get('mode', 'unknown')}` on backend `{native_contract.get('effective_backend', 'unknown')}`, portable contract is `{portable_contract.get('mode', 'unknown')}` on `{portable_contract.get('effective_backend', 'unknown')}`.",
                "Broker-specific news, options, and calendar context now sit behind adapter seams instead of being hardwired to one brokerage client.",
                f"Apple Silicon keeps a live-trading edge through `{host_contract.get('memory_architecture', 'unknown')}` memory architecture while the proof node preserves Linux and Windows replay portability.",
                f"Cross-platform proof node is `{cross_platform_proof.get('status', 'unknown')}` and nightly parity support is `{nightly_proof.get('ready', False)}`.",
                f"Linux and Windows deployment matrix entries are present, with next step `{portable_brain.get('next_step', 'unknown')}`.",
            ],
        },
    }
    return details


def _build_snapshot() -> dict[str, Any]:
    bot_summary = _active_bot_summary()
    lane_summary = _live_lane_summary()
    artifacts = _artifact_snapshot()
    special_features = _special_features_map(artifacts)
    special_feature_details = _special_feature_details(artifacts, special_features)
    crypto_ctx = artifacts["crypto_context"]
    divergence = artifacts["divergence"]
    correlation = artifacts["correlation"]
    training = artifacts["training"]
    watchdog = artifacts["watchdog"]
    process_watchdog = artifacts["process_watchdog"]
    live_readiness = artifacts["live_readiness"]
    runtime_separation = artifacts["runtime_separation"]
    platform_control = artifacts["platform_control"]
    pytorch_replay = artifacts["pytorch_replay"]
    autonomy_control = artifacts["autonomy_control"]
    incident_timeline = artifacts["incident_timeline"]
    portable_brain = artifacts["portable_brain"]
    switchboard = artifacts["switchboard"]
    provenance = artifacts["provenance"]
    notifications = artifacts["notifications"]
    incident_review = artifacts["incident_review"]
    architecture_upgrades = artifacts["architecture_upgrades"]
    chaos_drills = artifacts["chaos_drills"]
    promotion_autopilot = artifacts["promotion_autopilot"]
    daily_ops = artifacts["daily_ops"]
    institutional_readiness = platform_control.get("institutional_readiness") if isinstance(platform_control.get("institutional_readiness"), Mapping) else {}
    weakest_domains = institutional_readiness.get("weakest_domains") if isinstance(institutional_readiness.get("weakest_domains"), list) else []
    weakest_domain = weakest_domains[0] if weakest_domains and isinstance(weakest_domains[0], Mapping) else {}
    training_failures = training.get("failure_details") if isinstance(training.get("failure_details"), list) else []
    first_failure = training_failures[0] if training_failures and isinstance(training_failures[0], Mapping) else {}
    training_failure_reason = str(first_failure.get("reason") or training.get("reason") or "")
    training_age_hours = _payload_age_hours(training)
    watchdog_targets = process_watchdog.get("status") if isinstance(process_watchdog.get("status"), list) else []
    watchdog_healthy = sum(
        1
        for row in watchdog_targets
        if isinstance(row, Mapping) and (int(_safe_float(row.get("running"), 0.0)) + int(_safe_float(row.get("alt_running"), 0.0)) > 0) and bool(row.get("heartbeat_ok", False))
    )
    live_watchdog = live_readiness.get("process_watchdog") if isinstance(live_readiness.get("process_watchdog"), Mapping) else {}
    runtime_pressure = runtime_separation.get("shared_host_pressure") if isinstance(runtime_separation.get("shared_host_pressure"), Mapping) else {}
    pytorch_assist = pytorch_replay.get("mlx_shadow_assist") if isinstance(pytorch_replay.get("mlx_shadow_assist"), Mapping) else {}
    pytorch_scoreboard = pytorch_replay.get("scoreboard") if isinstance(pytorch_replay.get("scoreboard"), Mapping) else {}
    pytorch_candidates = pytorch_assist.get("eligible_source_profiles") if isinstance(pytorch_assist.get("eligible_source_profiles"), list) else []
    autonomy_lane = autonomy_control.get("lane_recovery_playbooks") if isinstance(autonomy_control.get("lane_recovery_playbooks"), Mapping) else {}
    promotion_approval = promotion_autopilot.get("approval_record") if isinstance(promotion_autopilot.get("approval_record"), Mapping) else {}
    portable_host = portable_brain.get("host_contract") if isinstance(portable_brain.get("host_contract"), Mapping) else {}
    portable_cross_platform = portable_brain.get("cross_platform_proof_node") if isinstance(portable_brain.get("cross_platform_proof_node"), Mapping) else {}
    switchboard_counts = switchboard.get("mode_counts") if isinstance(switchboard.get("mode_counts"), Mapping) else {}

    readiness_summary = {
        "institutional_status": str(institutional_readiness.get("overall_status") or ""),
        "institutional_score": round(_safe_float(institutional_readiness.get("overall_score"), 0.0), 2),
        "weakest_domain": {
            "slug": str(weakest_domain.get("slug") or ""),
            "title": str(weakest_domain.get("title") or ""),
            "score": round(_safe_float(weakest_domain.get("score"), 0.0), 2),
        },
        "live_status": str(live_readiness.get("overall_status") or ""),
        "live_score": round(_safe_float(live_readiness.get("readiness_score"), 0.0), 2),
        "runtime_status": str(runtime_separation.get("overall_status") or ""),
        "runtime_contention_score": _safe_int(runtime_pressure.get("contention_score"), 0),
        "watchdog_target_count": len(watchdog_targets),
        "watchdog_healthy_targets": int(watchdog_healthy),
        "watchdog_restart_storms": len(process_watchdog.get("restart_storms", []) or []),
        "watchdog_alerts": len(process_watchdog.get("alerts", []) or []),
    }

    training_summary = {
        "confirmed_training_success": bool(training.get("confirmed_training_success", False)),
        "trained_count": _safe_int(training.get("trained_count"), 0),
        "failure_count": _safe_int(training.get("failure_count"), 0),
        "reason": str(training.get("reason") or ""),
        "failure_reason": training_failure_reason,
        "age_hours": training_age_hours,
        "likely_environment_mismatch": "No module named 'mlx'" in training_failure_reason,
    }

    pytorch_summary = {
        "status": str(pytorch_assist.get("status") or ""),
        "assist_candidate_count": len(pytorch_candidates),
        "assist_candidate_profiles": [str((row or {}).get("source_profile") or "") for row in pytorch_candidates if isinstance(row, Mapping)],
        "runs_tracked": _safe_int(pytorch_scoreboard.get("runs_tracked"), 0),
        "positive_calibrated_runs": _safe_int(pytorch_scoreboard.get("positive_calibrated_runs"), 0),
        "active_assist_candidate_runs": _safe_int(pytorch_scoreboard.get("active_assist_candidate_runs"), 0),
        "recommendations": pytorch_replay.get("recommendations") if isinstance(pytorch_replay.get("recommendations"), list) else [],
    }
    autonomy_summary = {
        "overall_status": str(autonomy_control.get("overall_status") or ""),
        "autonomy_score": round(_safe_float(autonomy_control.get("autonomy_score"), 0.0), 2),
        "playbook_count": _safe_int(autonomy_lane.get("triggered_playbook_count"), 0),
        "open_incident_count": _safe_int(incident_timeline.get("open_incident_count"), 0),
        "promotion_state": str(promotion_autopilot.get("autopilot_state") or ""),
        "approval_state": str(promotion_approval.get("approval_state") or ""),
    }
    architecture_summary = {
        "upgrade_count": _safe_int(architecture_upgrades.get("upgrade_count"), 0),
        "ready_count": _safe_int(architecture_upgrades.get("ready_count"), 0),
        "portable_host_profile": str(portable_host.get("host_profile") or ""),
        "portable_proof_status": str(portable_cross_platform.get("status") or ""),
        "switchboard_active_modes": _safe_int(switchboard_counts.get("active"), 0),
        "provenance_card_count": _safe_int(provenance.get("card_count"), 0),
        "remote_pager_ready": bool(notifications.get("remote_pager_ready", False)),
        "incident_review_required": bool(incident_review.get("review_required", False)),
        "drill_program_score": round(_safe_float(((chaos_drills.get("drill_program") or {}).get("program_score")), 0.0), 2),
    }

    highlights = [
        (
            f"Registry currently tracks {bot_summary['total_registered']} bots with {bot_summary['active_count']} active "
            f"across {', '.join(sorted(bot_summary['active_roles'])) or 'no active roles'} lanes."
        ),
        (
            f"Live ingestion is wired across {lane_summary['lane_count']} lane artifacts with "
            f"{lane_summary['running_count']} currently reporting `running`."
        ),
        (
            f"Institutional-readiness score is `{readiness_summary['institutional_score']:.2f}/100` with status "
            f"`{readiness_summary['institutional_status'] or 'unknown'}` across "
            f"{_safe_int(institutional_readiness.get('domain_count'), 0)} governance domains."
        ),
        (
            f"Live readiness is `{readiness_summary['live_status'] or 'unknown'}` at "
            f"`{readiness_summary['live_score']:.2f}/100`, with broker/session ready="
            f"`{bool(live_readiness.get('broker_ready', False))}/{bool(live_readiness.get('session_ready', False))}` "
            f"and watchdog healthy=`{bool(live_watchdog.get('healthy', False))}`."
        ),
        (
            f"Runtime separation is `{readiness_summary['runtime_status'] or 'unknown'}` with contention score "
            f"`{readiness_summary['runtime_contention_score']}` and live-read-only="
            f"`{bool(((runtime_separation.get('release_contract') or {}).get('live_lane_should_be_read_only', False)))}`."
        ),
        (
            f"Autonomy control plane is `{autonomy_summary['overall_status'] or 'unknown'}` at "
            f"`{autonomy_summary['autonomy_score']:.2f}/100`, with `{autonomy_summary['playbook_count']}` triggered playbooks, "
            f"`{autonomy_summary['open_incident_count']}` open incidents, and promotion state "
            f"`{autonomy_summary['promotion_state'] or 'unknown'}`."
        ),
        (
            f"Architecture upgrade scoreboard tracks `{architecture_summary['ready_count']}` ready proof surfaces "
            f"out of `{architecture_summary['upgrade_count']}`, with host profile "
            f"`{architecture_summary['portable_host_profile'] or 'unknown'}` and portable proof "
            f"`{architecture_summary['portable_proof_status'] or 'unknown'}`."
        ),
        (
            f"Crypto context is aggregating {int(_safe_float(crypto_ctx.get('ok_source_count'), 0.0))}/"
            f"{int(_safe_float(crypto_ctx.get('source_count'), 0.0))} healthy sources and "
            f"{int(_safe_float(crypto_ctx.get('news_ok_source_count'), 0.0))}/"
            f"{int(_safe_float(crypto_ctx.get('news_source_count'), 0.0))} healthy crypto news feeds."
        ),
        (
            f"Latest divergence check is `ok={bool(divergence.get('ok', False))}` with worst relative spread "
            f"{_fmt_ratio_pct(_safe_float(divergence.get('worst_relative_spread'), 0.0))}."
        ),
        (
            f"Market/crypto correlation overlay is running in `{str(correlation.get('mode') or 'exact')}` mode with "
            f"{int(_safe_float(correlation.get('aligned_pairs'), 0.0))} aligned pairs "
            f"and cache hits/misses {int(_safe_float(correlation.get('cache_hits'), 0.0))}/"
            f"{int(_safe_float(correlation.get('cache_misses'), 0.0))}."
        ),
        (
            f"Latest training summary is {_fmt_age_hours(training_summary['age_hours'])}: "
            f"{training_summary['trained_count']} trained, {training_summary['failure_count']} failed, "
            f"`confirmed_training_success={training_summary['confirmed_training_success']}` "
            f"with reason `{training_summary['reason'] or 'unknown'}`."
        ),
        (
            f"Process watchdog currently tracks `{readiness_summary['watchdog_target_count']}` services with "
            f"`{readiness_summary['watchdog_healthy_targets']}` healthy targets, "
            f"`{readiness_summary['watchdog_restart_storms']}` restart storms, and "
            f"`{readiness_summary['watchdog_alerts']}` alerts. Tripwire event flag remains "
            f"`active={bool(watchdog.get('active', False))}`."
        ),
        (
            f"PyTorch sidecar stays observation-only, but it now carries "
            f"`{pytorch_summary['assist_candidate_count']}` active shadow-assist candidate profiles "
            f"across `{pytorch_summary['runs_tracked']}` tracked runs."
        ),
        (
            f"Latest daily ops quality score is {daily_ops.get('quality', {}).get('data_quality_score', 'n/a')} "
            f"and the weakest institutional domain is "
            f"`{readiness_summary['weakest_domain']['slug'] or 'unknown'}` "
            f"({readiness_summary['weakest_domain']['score']:.2f})."
        ),
    ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bot_summary": bot_summary,
        "lane_summary": lane_summary,
        "artifacts": artifacts,
        "readiness_summary": readiness_summary,
        "training_summary": training_summary,
        "pytorch_summary": pytorch_summary,
        "autonomy_summary": autonomy_summary,
        "architecture_summary": architecture_summary,
        "special_features": special_features,
        "special_feature_details": special_feature_details,
        "highlights": highlights,
    }


def _render_highlights_markdown(snapshot: Mapping[str, Any]) -> str:
    bot_summary = snapshot["bot_summary"]
    lane_summary = snapshot["lane_summary"]
    artifacts = snapshot["artifacts"]
    special_features = snapshot.get("special_features", {})
    special_feature_details = snapshot.get("special_feature_details", {}) if isinstance(snapshot.get("special_feature_details"), Mapping) else {}
    readiness_summary = snapshot.get("readiness_summary", {}) if isinstance(snapshot.get("readiness_summary"), Mapping) else {}
    training_summary = snapshot.get("training_summary", {}) if isinstance(snapshot.get("training_summary"), Mapping) else {}
    pytorch_summary = snapshot.get("pytorch_summary", {}) if isinstance(snapshot.get("pytorch_summary"), Mapping) else {}
    autonomy_summary = snapshot.get("autonomy_summary", {}) if isinstance(snapshot.get("autonomy_summary"), Mapping) else {}
    architecture_summary = snapshot.get("architecture_summary", {}) if isinstance(snapshot.get("architecture_summary"), Mapping) else {}
    training = artifacts["training"]
    correlation = artifacts["correlation"]
    crypto_ctx = artifacts["crypto_context"]
    top_bots = bot_summary["top_active_bots"]

    lines = [
        "# Auto-Refreshed Highlights",
        "",
        f"_Generated at {_fmt_compact_timestamp(snapshot.get('generated_at_utc'))}_",
        "",
        "## Platform Snapshot",
        "",
        f"- Registered bots: `{bot_summary['total_registered']}`",
        f"- Active bots: `{bot_summary['active_count']}`",
        f"- Live lane artifacts tracked: `{lane_summary['lane_count']}`",
        f"- Running lane artifacts: `{lane_summary['running_count']}`",
        f"- Institutional readiness: `{readiness_summary.get('institutional_score', 0.0):.2f}/100` (`{readiness_summary.get('institutional_status', '')}`)",
        f"- Live readiness: `{readiness_summary.get('live_score', 0.0):.2f}/100` (`{readiness_summary.get('live_status', '')}`)",
        f"- Runtime separation: `{readiness_summary.get('runtime_status', '')}`",
        f"- Crypto source coverage: `{int(_safe_float(crypto_ctx.get('ok_source_count'), 0.0))}/{int(_safe_float(crypto_ctx.get('source_count'), 0.0))}`",
        f"- Crypto news coverage: `{int(_safe_float(crypto_ctx.get('news_ok_source_count'), 0.0))}/{int(_safe_float(crypto_ctx.get('news_source_count'), 0.0))}`",
        f"- Correlation mode: `{str(correlation.get('mode') or 'exact')}`",
        f"- Last training result: `{int(_safe_float(training.get('trained_count'), 0.0))} trained / {int(_safe_float(training.get('failure_count'), 0.0))} failed`",
        f"- PyTorch shadow-assist candidates: `{_safe_int(pytorch_summary.get('assist_candidate_count'), 0)}`",
        f"- Autonomy control plane: `{autonomy_summary.get('autonomy_score', 0.0):.2f}/100` (`{autonomy_summary.get('overall_status', '')}`)",
        f"- Architecture upgrades: `{_safe_int(architecture_summary.get('ready_count'), 0)}/{_safe_int(architecture_summary.get('upgrade_count'), 0)}` ready proof surfaces",
        "",
        "## Key Highlights",
        "",
    ]
    lines.extend(f"- {row}" for row in snapshot["highlights"])
    lines.extend(
        [
            "",
            "## Executive Summary",
            "",
            "- These features matter because they describe the platform’s real differentiators: unified-memory-aware runtime tuning on Apple Silicon, one control surface across shadow/paper/live, event-to-trade intelligence, self-healing operational control, and a broker-agnostic portability contract.",
            "- The proof is intended to be operational rather than promotional. If a feature is still blocked, replay-only, or waiting on better parity proof, the document says so directly.",
            "- Read the feature proof notes as both a strength map and a watch list: they show what is already impressive and what still needs to mature to make the feature impossible to hand-wave away.",
            "",
            "## Real-World Readiness",
            "",
            f"- Institutional posture: `{readiness_summary.get('institutional_status', 'unknown')}` at `{readiness_summary.get('institutional_score', 0.0):.2f}/100`.",
            f"- Live operating posture: `{readiness_summary.get('live_status', 'unknown')}` at `{readiness_summary.get('live_score', 0.0):.2f}/100` with runtime separation `{readiness_summary.get('runtime_status', 'unknown')}`.",
            f"- Watchdog coverage: `{_safe_int(readiness_summary.get('watchdog_healthy_targets'), 0)}/{_safe_int(readiness_summary.get('watchdog_target_count'), 0)}` healthy targets, restart storms `{_safe_int(readiness_summary.get('watchdog_restart_storms'), 0)}`, alerts `{_safe_int(readiness_summary.get('watchdog_alerts'), 0)}`.",
            f"- Training lane: `{training_summary.get('trained_count', 0)}` trained / `{training_summary.get('failure_count', 0)}` failed, artifact `{_fmt_age_hours(training_summary.get('age_hours'))}`.",
            f"- PyTorch research lane: `{_safe_int(pytorch_summary.get('assist_candidate_count'), 0)}` assist candidates over `{_safe_int(pytorch_summary.get('runs_tracked'), 0)}` tracked runs.",
            f"- Autonomy posture: `{autonomy_summary.get('overall_status', 'unknown')}` at `{autonomy_summary.get('autonomy_score', 0.0):.2f}/100`, triggered playbooks `{_safe_int(autonomy_summary.get('playbook_count'), 0)}`, open incidents `{_safe_int(autonomy_summary.get('open_incident_count'), 0)}`.",
            f"- Architecture posture: `{_safe_int(architecture_summary.get('ready_count'), 0)}/{_safe_int(architecture_summary.get('upgrade_count'), 0)}` proof surfaces ready, host profile `{architecture_summary.get('portable_host_profile', 'unknown')}`, portable proof `{architecture_summary.get('portable_proof_status', 'unknown')}`.",
        ]
    )
    lines.extend(["", "## Special Features", ""])
    if isinstance(special_features, Mapping) and special_features:
        lines.extend(f"- {detail}" for detail in special_features.values())
    else:
        lines.append("- No special feature snapshot is currently available.")
    if isinstance(special_feature_details, Mapping) and special_feature_details:
        lines.extend(["", "## Special Feature Proof Notes", ""])
        for feature in special_feature_details.values():
            if not isinstance(feature, Mapping):
                continue
            lines.append(f"### {feature.get('label', 'Feature')}")
            summary = str(feature.get("summary") or "").strip()
            if summary:
                lines.append(f"- {summary}")
            why_it_matters = str(feature.get("why_it_matters") or "").strip()
            if why_it_matters:
                lines.append(f"- Why it matters: {why_it_matters}")
            for point in feature.get("proof_points", []) if isinstance(feature.get("proof_points"), list) else []:
                lines.append(f"- {point}")
            current_watch_item = str(feature.get("current_watch_item") or "").strip()
            if current_watch_item:
                lines.append(f"- Current watch item: {current_watch_item}")
            lines.append("")
    lines.extend(
        [
            "## Next Proof Targets",
            "",
            "- Reduce the data-plane drag so autonomy and runtime-separation proofs are not still competing with queue pressure.",
            "- Keep pushing portability from strong design into undeniable parity by running more non-Mac replay and parity checks.",
            "- Tighten transcript quality and event replay quality so Event-to-Trade Intelligence stays convincing on both live and replay paths.",
            "- Turn the current proof surfaces into a stronger portfolio of stable, repeatable reports rather than one-off wins.",
            "",
        ]
    )
    lines.extend(["", "## Current Active Lineup", ""])
    if top_bots:
        lines.extend(
            [
                "| Bot | Role | Test Accuracy | Quality Score |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for row in top_bots:
            lines.append(
                "| {bot} | {role} | {acc} | {quality:.3f} |".format(
                    bot=row.get("bot_id"),
                    role=row.get("bot_role"),
                    acc=_fmt_pct(_safe_float(row.get("test_accuracy"), 0.0)),
                    quality=_safe_float(row.get("quality_score"), 0.0),
                )
            )
    else:
        lines.append("- No active bots were found in the registry snapshot.")

    lines.extend(
        [
            "",
            "## Showcase Links",
            "",
            "- [Showcase Index](../README.md)",
            "- [Live Multi-Asset Paper Trading Platform](../projects/01-live-multi-asset-paper-platform.md)",
            "- [Quant Research and Model Training System](../projects/02-quant-research-and-model-training.md)",
            "- [Data Fusion and Verification Pipeline](../projects/03-data-fusion-and-verification-pipeline.md)",
            "- [Reliability, Safety, and Ops Automation](../projects/04-reliability-safety-and-ops-automation.md)",
            "- [Cross-Market Crypto and Macro Intelligence](../projects/05-cross-market-crypto-and-macro-intelligence.md)",
            "",
        ]
    )
    return "\n".join(lines)


def _render_special_features_html(snapshot: Mapping[str, Any]) -> str:
    bot_summary = snapshot["bot_summary"]
    lane_summary = snapshot["lane_summary"]
    readiness_summary = snapshot.get("readiness_summary", {}) if isinstance(snapshot.get("readiness_summary"), Mapping) else {}
    training_summary = snapshot.get("training_summary", {}) if isinstance(snapshot.get("training_summary"), Mapping) else {}
    pytorch_summary = snapshot.get("pytorch_summary", {}) if isinstance(snapshot.get("pytorch_summary"), Mapping) else {}
    autonomy_summary = snapshot.get("autonomy_summary", {}) if isinstance(snapshot.get("autonomy_summary"), Mapping) else {}
    architecture_summary = snapshot.get("architecture_summary", {}) if isinstance(snapshot.get("architecture_summary"), Mapping) else {}
    special_features = snapshot.get("special_features", {}) if isinstance(snapshot.get("special_features"), Mapping) else {}
    special_feature_details = snapshot.get("special_feature_details", {}) if isinstance(snapshot.get("special_feature_details"), Mapping) else {}
    highlights = snapshot.get("highlights", []) if isinstance(snapshot.get("highlights"), list) else []
    top_bots = bot_summary.get("top_active_bots", []) if isinstance(bot_summary.get("top_active_bots"), list) else []

    feature_styles = ["teal", "blue", "gold", "purple", "green", "red"]
    feature_labels = {
        "adaptive_apple_silicon_brain": "Adaptive Apple Silicon Brain",
        "three_mode_switchboard": "Three-Mode Switchboard",
        "event_to_trade_intelligence": "Event-to-Trade Intelligence",
        "self_healing_ops_plane": "Self-Healing Ops Plane",
        "portable_brain_contract": "Portable Brain Contract",
    }
    feature_cards: list[str] = []
    for idx, (key, detail) in enumerate(special_features.items()):
        label = feature_labels.get(str(key), str(key).replace("_", " ").title())
        style = feature_styles[idx % len(feature_styles)]
        feature_detail = special_feature_details.get(str(key)) if isinstance(special_feature_details.get(str(key)), Mapping) else {}
        proof_points = feature_detail.get("proof_points") if isinstance(feature_detail.get("proof_points"), list) else []
        why_it_matters = str(feature_detail.get("why_it_matters") or "").strip()
        current_watch_item = str(feature_detail.get("current_watch_item") or "").strip()
        proof_html = ""
        if proof_points:
            proof_html = "<ul class='feature-points'>{}</ul>".format(
                "".join(f"<li>{html.escape(str(point))}</li>" for point in proof_points[:4])
            )
        why_html = ""
        if why_it_matters:
            why_html = (
                "<div class='feature-callout'>"
                "<strong>Why it matters</strong>"
                f"<p>{html.escape(why_it_matters)}</p>"
                "</div>"
            )
        watch_html = ""
        if current_watch_item:
            watch_html = (
                "<div class='feature-watch'>"
                "<strong>Current watch item</strong>"
                f"<p>{html.escape(current_watch_item)}</p>"
                "</div>"
            )
        feature_cards.append(
            "<section class='box {style}'>"
            "<h3>{label}</h3>"
            "<p>{detail}</p>"
            "{why_html}"
            "{proof_html}"
            "{watch_html}"
            "</section>".format(
                style=style,
                label=html.escape(label),
                detail=html.escape(str(detail)),
                why_html=why_html,
                proof_html=proof_html,
                watch_html=watch_html,
            )
        )

    proof_rows = [
        ("Live Readiness", f"{readiness_summary.get('live_score', 0.0):.2f}/100", readiness_summary.get("live_status", "unknown")),
        ("Institutional Readiness", f"{readiness_summary.get('institutional_score', 0.0):.2f}/100", readiness_summary.get("institutional_status", "unknown")),
        ("Autonomy Control", f"{autonomy_summary.get('autonomy_score', 0.0):.2f}/100", autonomy_summary.get("overall_status", "unknown")),
        (
            "Architecture Upgrades",
            f"{_safe_int(architecture_summary.get('ready_count'), 0)}/{_safe_int(architecture_summary.get('upgrade_count'), 0)}",
            f"host {architecture_summary.get('portable_host_profile', 'unknown')}",
        ),
        ("Active Bots", str(bot_summary.get("active_count", 0)), f"of {bot_summary.get('total_registered', 0)} registered"),
        ("Running Lanes", str(lane_summary.get("running_count", 0)), f"of {lane_summary.get('lane_count', 0)} tracked"),
        ("Training Lane", f"{training_summary.get('trained_count', 0)} trained / {training_summary.get('failure_count', 0)} failed", _fmt_age_hours(training_summary.get("age_hours"))),
        ("PyTorch Sidecar", str(_safe_int(pytorch_summary.get("assist_candidate_count"), 0)), f"{_safe_int(pytorch_summary.get('runs_tracked'), 0)} tracked runs"),
    ]
    proof_cards = []
    for idx, (label, value, detail) in enumerate(proof_rows):
        style = feature_styles[idx % len(feature_styles)]
        proof_cards.append(
            "<section class='mini-box {style}'>"
            "<h3>{label}</h3>"
            "<div class='metric'>{value}</div>"
            "<p>{detail}</p>"
            "</section>".format(
                style=style,
                label=html.escape(str(label)),
                value=html.escape(str(value)),
                detail=html.escape(str(detail)),
            )
        )

    executive_cards = [
        (
            "Why These Features Matter",
            "The point is not that the platform has clever modules. The point is that the feature layer explains why the system is differentiated operationally: unified-memory-aware Apple Silicon tuning, one switchboard across modes, event-to-trade intelligence, self-healing ops, and a real broker-portable runtime contract.",
        ),
        (
            "What Makes The Proof Worthy",
            "Each feature is anchored to a live artifact, current score, or runtime contract. That keeps the report honest: if a surface is blocked, degraded, or still only proven in replay, the packet says so instead of pretending the feature is fully complete.",
        ),
        (
            "How To Read This Packet",
            "The proof strip gives current posture, the feature cards explain practical value, and the watch items tell you where the next quality bar still sits. That is what turns a feature list into an actual report.",
        ),
    ]
    executive_html = "".join(
        "<section class='brief-card'><h3>{}</h3><p>{}</p></section>".format(html.escape(title), html.escape(body))
        for title, body in executive_cards
    )
    interpretation_cards = [
        (
            "Differentiation",
            "The strongest features are the ones that combine architecture with lived proof: host-aware runtime tuning, a real multi-mode switchboard, and event ingestion that actually reaches market-relevant stance and relevance surfaces.",
        ),
        (
            "Operational Honesty",
            "The packet intentionally leaves the rough edges visible. If autonomy is still blocked or a portability surface is only replay-proven, that honesty makes the feature story more credible, not less.",
        ),
        (
            "What Still Raises The Bar",
            "The next level comes from steadier control-plane freedom, deeper portability proof, and better training/promotion health so the features read as durable operating strengths rather than ambitious modules.",
        ),
    ]
    interpretation_html = "".join(
        "<section class='brief-card'><h3>{}</h3><p>{}</p></section>".format(html.escape(title), html.escape(body))
        for title, body in interpretation_cards
    )
    recommendations = [
        (
            "Make Portability Hard To Dismiss",
            "Keep the Apple Silicon story, but add more frequent replay and parity evidence on non-Mac nodes so the portable-brain contract reads as a proven pathway rather than a smart intention.",
        ),
        (
            "Reduce Operational Friction",
            "Drive the autonomy blockers and incident-review drag down so the self-healing ops story is as strong in practice as it is in architecture.",
        ),
        (
            "Make Macro Intelligence Cleaner",
            "Push transcript cleanup and replay quality further so event-to-trade intelligence can hold up under scrutiny when live captions are messy or speaker flow changes quickly.",
        ),
    ]
    recommendation_html = "".join(
        "<section class='brief-card'><h3>{}</h3><p>{}</p></section>".format(html.escape(title), html.escape(body))
        for title, body in recommendations
    )
    highlight_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in highlights[:8]) or "<li>No highlight snapshot is currently available.</li>"
    lineup_rows = []
    for row in top_bots[:5]:
        lineup_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('bot_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('bot_role', '')))}</td>"
            f"<td>{html.escape(_fmt_pct(_safe_float(row.get('test_accuracy'), 0.0)))}</td>"
            f"<td>{html.escape(f'{_safe_float(row.get('quality_score'), 0.0):.3f}')}</td>"
            "</tr>"
        )
    lineup_html = (
        "<table><thead><tr><th>Bot</th><th>Role</th><th>Test Accuracy</th><th>Quality Score</th></tr></thead><tbody>"
        + "".join(lineup_rows)
        + "</tbody></table>"
    ) if lineup_rows else "<p>No active lineup snapshot is currently available.</p>"

    generated = _fmt_compact_timestamp(snapshot.get("generated_at_utc"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Special Features And Highlights</title>
  <style>
    :root {{
      --bg: #eef2f3;
      --card: #ffffff;
      --ink: #1f2937;
      --muted: #5b6471;
      --line: #d7e0e6;
      --teal: #2ca6a4;
      --blue: #7fa8d1;
      --gold: #d8a93a;
      --red: #c65a5a;
      --green: #1e8e5a;
      --purple: #6b5bd2;
      --shadow: 0 18px 40px rgba(21, 33, 52, 0.08);
      --hero: linear-gradient(145deg, rgba(27, 146, 146, 0.12), rgba(107, 91, 210, 0.08) 55%, rgba(216, 169, 58, 0.12));
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: radial-gradient(circle at top left, #f8fbfc 0, var(--bg) 40%, #e8eeef 100%); color: var(--ink); font: 15px/1.6 "Avenir Next", "Segoe UI", sans-serif; }}
    .page {{ padding: 28px 30px 36px; }}
    .hero {{ background: var(--hero), var(--card); border: 1px solid rgba(128, 151, 166, 0.22); border-radius: 24px; padding: 24px 26px; box-shadow: var(--shadow); }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{ font: 700 31px/1.12 "Iowan Old Style", "Georgia", serif; letter-spacing: -0.02em; }}
    h2 {{ font: 700 22px/1.18 "Iowan Old Style", "Georgia", serif; margin-bottom: 12px; }}
    h3 {{ font-size: 18px; margin-bottom: 10px; }}
    .eyebrow {{ display: inline-block; margin-bottom: 10px; color: var(--purple); font-size: 12px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }}
    .sub {{ margin-top: 8px; color: var(--muted); max-width: 920px; }}
    .section-card {{ margin-top: 18px; background: var(--card); border: 1px solid rgba(128, 151, 166, 0.2); border-radius: 22px; padding: 18px; box-shadow: var(--shadow); }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .feature-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .brief-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .brief-card {{ background: linear-gradient(180deg, #ffffff 0%, #fafcfd 100%); border-radius: 18px; border: 1px solid rgba(128, 151, 166, 0.18); padding: 18px; }}
    .brief-card p {{ margin: 0; color: var(--muted); }}
    .section-lead {{ color: var(--muted); margin-bottom: 12px; }}
    .box, .mini-box {{ background: linear-gradient(180deg, #ffffff 0%, #fbfcfd 100%); border-radius: 18px; border: 2px solid var(--line); padding: 14px; }}
    .box p, .mini-box p {{ margin: 0; color: var(--muted); }}
    .mini-box .metric {{ font-size: 22px; font-weight: 700; margin-bottom: 8px; }}
    .teal {{ border-color: var(--teal); }}
    .blue {{ border-color: var(--blue); }}
    .gold {{ border-color: var(--gold); }}
    .purple {{ border-color: var(--purple); }}
    .green {{ border-color: var(--green); }}
    .red {{ border-color: var(--red); }}
    ul {{ margin: 0; padding-left: 20px; }}
    li {{ margin: 8px 0; }}
    .feature-points {{ margin-top: 12px; padding-left: 18px; color: var(--ink); }}
    .feature-points li {{ margin: 6px 0; color: var(--ink); }}
    .feature-callout, .feature-watch {{ margin-top: 12px; padding: 12px 13px; border-radius: 14px; }}
    .feature-callout {{ background: rgba(44, 166, 164, 0.08); }}
    .feature-watch {{ background: rgba(198, 90, 90, 0.08); }}
    .feature-callout strong, .feature-watch strong {{ display: block; margin-bottom: 6px; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
    .feature-callout p, .feature-watch p {{ color: var(--ink); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 8px; text-align: left; vertical-align: top; }}
    th {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">Executive Feature Report</div>
      <h1>Special Features And Highlights</h1>
      <p class="sub">Generated {html.escape(generated)}. This packet treats special features as operating capabilities, not marketing bullets. Each section explains why the feature matters, what evidence proves it is real today, and what still needs attention before the capability is fully mature.</p>
      <p class="sub">The goal is to make the feature story worthy of the platform itself: differentiated where the architecture is actually uncommon, honest where the proofs are still incomplete, and useful to an operator who wants to understand why the system deserves trust.</p>
    </section>
    <section class="section-card">
      <h2>Executive Summary</h2>
      <div class="brief-grid">
        {executive_html}
      </div>
    </section>
    <section class="section-card">
      <h2>Feature Proof Surface</h2>
      <div class="feature-grid">
        {''.join(feature_cards)}
      </div>
    </section>
    <section class="section-card">
      <h2>Platform Proof Snapshot</h2>
      <div class="grid">
        {''.join(proof_cards)}
      </div>
    </section>
    <section class="section-card">
      <h2>Interpretation Notes</h2>
      <div class="section-lead">These notes explain how to read the feature set as a real capability stack rather than a list of isolated modules.</div>
      <div class="brief-grid">
        {interpretation_html}
      </div>
    </section>
    <section class="section-card">
      <h2>Current Highlights</h2>
      <ul>{highlight_items}</ul>
    </section>
    <section class="section-card">
      <h2>Current Active Lineup</h2>
      {lineup_html}
    </section>
    <section class="section-card">
      <h2>Recommendations</h2>
      <div class="section-lead">These are the next upgrades most likely to make the feature story feel undeniable from top to bottom.</div>
      <div class="brief-grid">
        {recommendation_html}
      </div>
    </section>
  </div>
</body>
</html>
"""


def _render_readme_snippet(snapshot: Mapping[str, Any]) -> str:
    bot_summary = snapshot["bot_summary"]
    lane_summary = snapshot["lane_summary"]
    artifacts = snapshot["artifacts"]
    readiness_summary = snapshot.get("readiness_summary", {}) if isinstance(snapshot.get("readiness_summary"), Mapping) else {}
    pytorch_summary = snapshot.get("pytorch_summary", {}) if isinstance(snapshot.get("pytorch_summary"), Mapping) else {}
    autonomy_summary = snapshot.get("autonomy_summary", {}) if isinstance(snapshot.get("autonomy_summary"), Mapping) else {}
    architecture_summary = snapshot.get("architecture_summary", {}) if isinstance(snapshot.get("architecture_summary"), Mapping) else {}
    correlation = artifacts["correlation"]
    crypto_ctx = artifacts["crypto_context"]
    top_bots = bot_summary["top_active_bots"][:3]
    lines = [
        f"_Generated at {_fmt_compact_timestamp(snapshot.get('generated_at_utc'))}_",
        "",
        f"- Active registry lineup: `{bot_summary['active_count']}` of `{bot_summary['total_registered']}` bots are active.",
        f"- Live collection snapshot: `{lane_summary['running_count']}/{lane_summary['lane_count']}` lane artifacts are reporting `running`.",
        f"- Institutional readiness: `{readiness_summary.get('institutional_score', 0.0):.2f}/100` with status `{readiness_summary.get('institutional_status', 'unknown')}`.",
        f"- Live/runtime posture: live readiness `{readiness_summary.get('live_status', 'unknown')}` at `{readiness_summary.get('live_score', 0.0):.2f}/100`, runtime separation `{readiness_summary.get('runtime_status', 'unknown')}`.",
        f"- Autonomy posture: `{autonomy_summary.get('autonomy_score', 0.0):.2f}/100` with status `{autonomy_summary.get('overall_status', 'unknown')}`, playbooks `{_safe_int(autonomy_summary.get('playbook_count'), 0)}`, open incidents `{_safe_int(autonomy_summary.get('open_incident_count'), 0)}`.",
        f"- Architecture upgrades: `{_safe_int(architecture_summary.get('ready_count'), 0)}/{_safe_int(architecture_summary.get('upgrade_count'), 0)}` ready proof surfaces, host profile `{architecture_summary.get('portable_host_profile', 'unknown')}`, portable proof `{architecture_summary.get('portable_proof_status', 'unknown')}`.",
        f"- Crypto context: `{int(_safe_float(crypto_ctx.get('ok_source_count'), 0.0))}/{int(_safe_float(crypto_ctx.get('source_count'), 0.0))}` healthy sources and `{int(_safe_float(crypto_ctx.get('news_ok_source_count'), 0.0))}/{int(_safe_float(crypto_ctx.get('news_source_count'), 0.0))}` healthy news feeds.",
        f"- Correlation overlay: mode `{str(correlation.get('mode') or 'exact')}`, aligned pairs `{int(_safe_float(correlation.get('aligned_pairs'), 0.0))}`.",
        f"- PyTorch sidecar: `{_safe_int(pytorch_summary.get('assist_candidate_count'), 0)}` active assist candidates across `{_safe_int(pytorch_summary.get('runs_tracked'), 0)}` tracked runs.",
    ]
    if top_bots:
        formatted = ", ".join(f"`{row['bot_id']}` ({_fmt_pct(_safe_float(row['test_accuracy'], 0.0))})" for row in top_bots)
        lines.append(f"- Top active lineup by test accuracy: {formatted}.")
    lines.append("")
    lines.append("Full generated detail lives in [docs/showcase/generated/highlights_latest.md](docs/showcase/generated/highlights_latest.md).")
    return "\n".join(lines)


def _update_readme(snippet: str) -> None:
    text = README_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(README_START)}.*?{re.escape(README_END)}",
        flags=re.DOTALL,
    )
    replacement = f"{README_START}\n{snippet}\n{README_END}"
    if pattern.search(text):
        text = pattern.sub(replacement, text)
    else:
        text = text.rstrip() + "\n\n## Auto-Refreshed Highlights\n\n" + replacement + "\n"
    README_PATH.write_text(text, encoding="utf-8")


def main() -> int:
    snapshot = _build_snapshot()
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    HIGHLIGHTS_JSON.write_text(json.dumps(snapshot, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    HIGHLIGHTS_MD.write_text(_render_highlights_markdown(snapshot) + "\n", encoding="utf-8")
    SPECIAL_FEATURES_HTML.write_text(_render_special_features_html(snapshot) + "\n", encoding="utf-8")
    _update_readme(_render_readme_snippet(snapshot))
    print(json.dumps({"ok": True, "generated_at_utc": snapshot["generated_at_utc"], "output": str(HIGHLIGHTS_MD)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
