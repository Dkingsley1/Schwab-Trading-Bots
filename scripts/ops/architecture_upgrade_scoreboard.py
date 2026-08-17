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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, status_rank, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, status_rank, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "architecture_upgrade_scoreboard_latest.json"


def _score_row(slug: str, title: str, status: str, proof: str, source_path: str) -> dict[str, Any]:
    return {
        "slug": slug,
        "title": title,
        "status": status,
        "proof": proof,
        "source_path": source_path,
    }


def _capability_ready(status: str, *, ready_when: bool) -> str:
    normalized = str(status or "").strip().lower()
    if ready_when and normalized not in {"blocked", "critical"}:
        return "ready"
    return status


def _capability_recovering(status: str, *, recovering_when: bool) -> str:
    normalized = str(status or "").strip().lower()
    if recovering_when and normalized in {"blocked", "critical"}:
        return "degraded"
    return status


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _guarded_paper_strict_clear(health_fast: dict[str, Any]) -> bool:
    operational = _as_dict(health_fast.get("operational_readiness"))
    guarded_paper = _as_dict(operational.get("guarded_paper"))
    live_execution = _as_dict(operational.get("live_execution"))
    guarded_ready = bool(guarded_paper.get("ok", False)) and str(guarded_paper.get("status") or "").strip().lower() in {
        "ready",
        "armed",
        "guarded_ready",
    }
    live_locked = str(live_execution.get("status") or "").strip().lower() in {
        "blocked_read_only",
        "locked",
        "read_only",
        "disabled",
    }
    return bool(health_fast.get("strict_all_clear", False) and guarded_ready and live_locked)


def _event_trade_status(project_root: Path) -> tuple[str, str, str]:
    intelligence_path = project_root / "governance" / "health" / "macro_event_intelligence_latest.json"
    payload = load_json(intelligence_path)
    if payload:
        replay_contract = payload.get("replay_contract") if isinstance(payload.get("replay_contract"), dict) else {}
        live_detected = bool(payload.get("live_detected", False))
        idle_ready = bool(
            not live_detected
            and str(payload.get("market_relevance") or "").strip().lower() in {"low", "idle", "none"}
            and not bool(replay_contract.get("replay_pending", False))
            and not bool(replay_contract.get("full_video_required", False))
        )
        overall = _capability_ready(str(payload.get("overall_status") or "degraded"), ready_when=idle_ready)
        proof = (
            f"relevance={str(payload.get('market_relevance') or 'unknown')} "
            f"transcript_quality={str(payload.get('transcript_quality') or 'missing')} "
            f"media_status={str(payload.get('media_status') or 'missing')} "
            f"idle_ready={int(idle_ready)}"
        )
        return overall, proof, str(intelligence_path)
    status_path = project_root / "governance" / "health" / "macro_auto_watch_status.json"
    media_path = project_root / "governance" / "health" / "live_macro_media_status.json"
    status = load_json(status_path)
    media = load_json(media_path)
    live_detected = bool(status.get("live_detected", False))
    media_running = str(media.get("status") or "").strip().lower() in {"running", "active", "ready"}
    overall = "ready" if live_detected or media_running else "degraded"
    proof = f"live_detected={int(live_detected)} media_status={str(media.get('status') or 'missing')}"
    return overall, proof, str(status_path if status else media_path)


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    champion_root = project_root / "governance" / "champion_challenger"
    walk_root = project_root / "governance" / "walk_forward"

    runtime = load_json(health_root / "live_runtime_separation_control_latest.json")
    coverage = load_json(walk_root / "coverage_gap_closer_latest.json")
    promotion = load_json(champion_root / "promotion_autopilot_packet_latest.json")
    portable = load_json(health_root / "portable_brain_contract_latest.json")
    apple = load_json(health_root / "apple_silicon_profile_latest.json")
    switchboard = load_json(health_root / "mode_switchboard_mission_control_latest.json")
    autonomy = load_json(health_root / "autonomy_control_plane_latest.json")
    provenance = load_json(health_root / "decision_provenance_cards_latest.json")
    notifications = load_json(health_root / "notification_escalation_ladder_latest.json")
    drills = load_json(health_root / "chaos_drill_coordinator_latest.json")
    incident_review = load_json(health_root / "incident_review_packet_latest.json")
    incident_closeout = load_json(health_root / "incident_closeout_autopilot_latest.json")
    lane_thaw = load_json(health_root / "lane_thaw_controller_latest.json")
    data_plane = load_json(health_root / "data_plane_recovery_controller_latest.json")
    health_fast = load_json(health_root / "health_fast_latest.json")

    event_status, event_proof, event_source = _event_trade_status(project_root)
    portable_host = portable.get("host_contract") if isinstance(portable.get("host_contract"), dict) else {}
    cross_platform = portable.get("cross_platform_proof_node") if isinstance(portable.get("cross_platform_proof_node"), dict) else {}
    switch_modes = switchboard.get("mode_counts") if isinstance(switchboard.get("mode_counts"), dict) else {}
    coverage_contract = coverage.get("autopilot_contract") if isinstance(coverage.get("autopilot_contract"), dict) else {}
    signed_bundle = promotion.get("signed_bundle_contract") if isinstance(promotion.get("signed_bundle_contract"), dict) else {}
    runtime_clearance = runtime.get("clearance_plan") if isinstance(runtime.get("clearance_plan"), dict) else {}
    runtime_live_plane = runtime.get("live_plane") if isinstance(runtime.get("live_plane"), dict) else {}
    runtime_ready_contract = bool(
        runtime_live_plane.get("ready", False)
        and str(runtime_clearance.get("clearance_state") or "").strip().lower()
        in {
            "ready",
            "awaiting_coverage_cycles",
            "coverage_cycles_ready",
            "off_hours_cold_lane_launch_ready",
            "scheduled_off_hours_launch",
            "staged_preclearance",
        }
    )
    coverage_ready_contract = bool(
        bool(coverage_contract.get("can_apply_stage", False))
        and bool(coverage_contract.get("cold_lane_ready", False) or coverage_contract.get("snapshot_ready", False))
        and _status_is_not_blocked(str(coverage_contract.get("overall_status") or coverage.get("overall_status") or ""))
    )
    signed_bundle_ready = bool(
        bool(signed_bundle.get("signature_verified", False))
        and bool(signed_bundle.get("rollback_ready", False))
    )
    autonomy_score = float(autonomy.get("autonomy_score", 0.0) or 0.0)
    autonomy_recovering_contract = bool(
        (
            autonomy_score >= 75.0
            and int((autonomy.get("autonomous_repair_path_count", 0) or 0)) > 0
            and str(data_plane.get("overall_status") or "").strip().lower() in {"ready", "degraded"}
        )
        or (
            _guarded_paper_strict_clear(health_fast)
            and int((autonomy.get("autonomous_repair_path_count", 0) or 0)) > 0
        )
    )
    bounded_incident_closeout = bool(
        incident_closeout.get("bounded_closeout_path_ready", False)
        or (
            str(incident_closeout.get("overall_status") or "").strip().lower() == "degraded"
            and int(incident_closeout.get("open_incident_count", 0) or 0) <= 3
            and not any(
                str(row.get("severity") or "").strip().lower() == "critical"
                for row in incident_closeout.get("blocking_surfaces") or []
                if isinstance(row, dict)
            )
        )
        or (
            _guarded_paper_strict_clear(health_fast)
            and int(incident_closeout.get("open_incident_count", 0) or 0) <= 3
            and not any(
                str(row.get("severity") or "").strip().lower() == "critical"
                for row in incident_closeout.get("blocking_surfaces") or []
                if isinstance(row, dict)
            )
        )
    )

    rows = [
        _score_row(
            "true_live_enclave",
            "True Live Enclave",
            _capability_ready(str(runtime.get("overall_status") or "missing"), ready_when=runtime_ready_contract),
            f"clearance={str(((runtime.get('clearance_plan') or {}).get('clearance_state') or 'unknown'))} contention={int(((runtime.get('shared_host_pressure') or {}).get('contention_score', 0) or 0))}",
            str(health_root / "live_runtime_separation_control_latest.json"),
        ),
        _score_row(
            "continuous_coverage_autopilot",
            "Continuous Coverage Autopilot",
            _capability_ready(
                str(coverage_contract.get("overall_status") or coverage.get("overall_status") or "missing"),
                ready_when=coverage_ready_contract,
            ),
            f"launch_state={str(coverage_contract.get('launch_state') or 'unknown')} staged={int(coverage_contract.get('stage_candidate_count', 0) or 0)}",
            str(walk_root / "coverage_gap_closer_latest.json"),
        ),
        _score_row(
            "signed_promotion_bundles",
            "Signed Promotion Bundles",
            _capability_ready(str(promotion.get("overall_status") or "missing"), ready_when=signed_bundle_ready),
            f"signature_verified={int(bool(signed_bundle.get('signature_verified', False)))} rollback_ready={int(bool(signed_bundle.get('rollback_ready', False)))}",
            str(champion_root / "promotion_autopilot_packet_latest.json"),
        ),
        _score_row(
            "cross_platform_proof_node",
            "Cross-Platform Proof Node",
            str(cross_platform.get("status") or "missing"),
            f"backend={str(cross_platform.get('effective_backend') or 'unknown')} shadow_replay_supported={int(bool(cross_platform.get('shadow_replay_supported', False)))}",
            str(health_root / "portable_brain_contract_latest.json"),
        ),
        _score_row(
            "adaptive_apple_silicon_brain",
            "Adaptive Apple Silicon Brain",
            str(apple.get("overall_status") or "ready"),
            f"host_profile={str(portable_host.get('host_profile') or apple.get('applied_tier') or 'unknown')} chip={str(portable_host.get('chip') or 'unknown')} memory_architecture={str(portable_host.get('memory_architecture') or 'unknown')}",
            str(health_root / "apple_silicon_profile_latest.json"),
        ),
        _score_row(
            "three_mode_switchboard",
            "Three-Mode Switchboard",
            str(switchboard.get("overall_status") or "missing"),
            f"active_modes={int(switch_modes.get('active', 0) or 0)} ready_modes={int(switch_modes.get('ready', 0) or 0)}",
            str(health_root / "mode_switchboard_mission_control_latest.json"),
        ),
        _score_row(
            "event_to_trade_intelligence",
            "Event-to-Trade Intelligence",
            event_status,
            event_proof,
            event_source,
        ),
        _score_row(
            "self_healing_ops_plane",
            "Self-Healing Ops Plane",
            _capability_recovering(str(autonomy.get("overall_status") or "missing"), recovering_when=autonomy_recovering_contract),
            f"autonomy_score={autonomy_score:.2f} playbooks={int(((autonomy.get('lane_recovery_playbooks') or {}).get('triggered_playbook_count', 0) or 0))} thaw_candidates={int(lane_thaw.get('candidate_count', 0) or 0)} data_plane={str(data_plane.get('overall_status') or 'missing')}",
            str(health_root / "autonomy_control_plane_latest.json"),
        ),
        _score_row(
            "decision_provenance_cards",
            "Decision Provenance Cards",
            str(provenance.get("overall_status") or "missing"),
            f"cards={int(provenance.get('card_count', 0) or 0)} modes={int(provenance.get('mode_count', 0) or 0)}",
            str(health_root / "decision_provenance_cards_latest.json"),
        ),
        _score_row(
            "notification_escalation_ladder",
            "Notification Escalation Ladder",
            str(notifications.get("overall_status") or "missing"),
            f"attended={int(bool(notifications.get('attended_runtime_ready', False)))} remote_pager_ready={int(bool(notifications.get('remote_pager_ready', False)))} unacked={int(((notifications.get('critical_backlog') or {}).get('unacked_count', 0) or 0))}",
            str(health_root / "notification_escalation_ladder_latest.json"),
        ),
        _score_row(
            "autonomous_drill_program",
            "Autonomous Drill Program",
            str(drills.get("overall_status") or "missing"),
            f"overdue={len(drills.get('overdue_drills') or [])} program_score={float(((drills.get('drill_program') or {}).get('program_score', 0.0) or 0.0)):.2f}",
            str(health_root / "chaos_drill_coordinator_latest.json"),
        ),
        _score_row(
            "immutable_incident_review",
            "Immutable Incident Review",
            _capability_recovering(
                str(incident_review.get("overall_status") or "missing"),
                recovering_when=bounded_incident_closeout,
            ),
            f"review_required={int(bool(incident_review.get('review_required', False)))} "
            f"packet_sha256={str(incident_review.get('packet_sha256') or '')[:12]} "
            f"bounded_closeout={int(bounded_incident_closeout)}",
            str(health_root / "incident_review_packet_latest.json"),
        ),
    ]

    worst_rank = max(status_rank(str(row.get("status") or "ready")) for row in rows) if rows else status_rank("ready")
    overall_status = "ready"
    if worst_rank >= status_rank("blocked"):
        overall_status = "blocked"
    elif worst_rank >= status_rank("degraded"):
        overall_status = "degraded"

    special_features_map = {
        "adaptive_apple_silicon_brain": (
            f"Adaptive Apple Silicon Brain: host-aware tuning now recognizes `{str(portable_host.get('chip') or 'unknown')}`, "
            f"sees memory architecture `{str(portable_host.get('memory_architecture') or 'unknown')}`, and lands on "
            f"`{str(portable_host.get('host_profile') or 'unknown')}` before the stack starts."
        ),
        "three_mode_switchboard": (
            f"Three-Mode Switchboard: mission control now tracks shadow/paper/live with `{int(switch_modes.get('active', 0) or 0)}` active modes "
            f"and runtime clearance `{str(((switchboard.get('control_surface') or {}).get('clearance_state') or 'unknown'))}`."
        ),
        "event_to_trade_intelligence": (
            f"Event-to-Trade Intelligence: the macro lane now surfaces live-detection and media ingest proof as `{event_status}` "
            f"with `{event_proof}`."
        ),
        "self_healing_ops_plane": (
            f"Self-Healing Ops Plane: autonomy currently sits at `{float(autonomy.get('autonomy_score', 0.0) or 0.0):.2f}/100` "
            f"with `{int(((autonomy.get('lane_recovery_playbooks') or {}).get('triggered_playbook_count', 0) or 0))}` triggered playbooks."
        ),
        "portable_brain_contract": (
            f"Portable Brain Contract: the host contract now recommends `{str((portable.get('adaptation_contract') or {}).get('recommended_runtime_access_mode') or 'unknown')}` "
            f"mode with proof-node status `{str(cross_platform.get('status') or 'unknown')}`, backend `{str(cross_platform.get('effective_backend') or 'unknown')}`, "
            f"and parity focus `{str((portable.get('parity_contract') or {}).get('parity_focus') or 'unknown')}` while keeping the broker/runtime seam portable."
        ),
    }

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "upgrade_count": len(rows),
        "ready_count": sum(1 for row in rows if str(row.get("status") or "") in {"ready", "active", "ok", "awaiting_approval", "active_host_candidate"}),
        "rows": rows,
        "special_features_map": special_features_map,
    }


def _status_is_not_blocked(status: str) -> bool:
    return str(status or "").strip().lower() not in {"blocked", "critical", "missing"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the current architecture-upgrade proof surfaces and special features.")
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
            "architecture_upgrade_scoreboard "
            f"overall_status={payload.get('overall_status', '')} "
            f"upgrade_count={int(payload.get('upgrade_count', 0) or 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
