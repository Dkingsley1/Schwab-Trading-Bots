import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import a_plus_operating_packet as packet


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_sources(root: Path, overrides: dict[str, dict] | None = None) -> None:
    overrides = overrides or {}
    base = {
        "health_fast": {
            "ok": True,
            "overall_status": "ready",
            "strict_all_clear": True,
            "operational_readiness": {"guarded_paper": {"ok": True, "status": "ready", "blockers": []}},
            "process_watchdog": {"alert_summary": {"total_count": 0}},
            "runtime_pressure": {"overall_status": "advisory"},
            "storage": {"severity": "stable"},
        },
        "process_watchdog": {"overall_status": "ready", "alerts": []},
        "rolling_restart": {"ok": True, "overall_status": "ready", "restart_due": False, "recommended_scope": "none"},
        "ingestion_storage": {"ok": True, "overall_status": "ready", "severity": "stable", "pressure_index": 0.1, "backpressure": {"total_pending_lines": 100, "oldest_pending_age_seconds": 1.0}},
        "runtime_throttle": {"ok": True, "overall_status": "advisory", "host_saturation_score": 40.0, "compute_pressure_level": "elevated", "memory_pressure_level": "normal"},
        "process_fanout": {"ok": True, "overall_status": "ready"},
        "command_validity": {"ok": True, "overall_status": "ready", "smoke_failures": []},
        "writer_cycle": {"ok": True, "overall_status": "idle"},
        "paper_performance": {"sleeve_latest": [{"profile": "default"}]},
        "paper_profitability": {"ok": True, "overall_status": "ready", "paper_summary": {"executions": 10, "ending_net_pnl_total": 1.25}},
        "sleeve_profitability": {"ok": True, "overall_status": "ready"},
        "paper_ramp": {"ok": True, "overall_status": "ready", "stage": "armed", "armed": True, "blockers": []},
        "runtime_paper_guard": {"ok": True, "overall_status": "ready"},
        "memory_efficiency": {
            "ok": True,
            "overall_status": "ready",
            "compressed_memory_relief_contract": {"managed": False},
        },
        "account_position": {"ok": True, "account_count": 3, "position_count": 9, "covered_call_roll_watch": {"overall_status": "watch", "covered_call_count": 3, "alert_count": 0}},
        "account_policy": {"ok": True, "overall_status": "ready"},
        "account_snapshot": {"ok": True, "overall_status": "ready"},
        "livefeed_local": {"status": "running", "alive": True, "source": "all", "heavy": 1},
        "livefeed_refresh_guard": {"ok": True, "overall_status": "ready", "route_ok_count": 6, "route_count": 6},
        "spacex_ipo_watch": {"ok": True, "overall_status": "ready", "symbol": "SPCX", "policy": "monitoring_only_no_order_instruction", "quote": {"ok": True}, "alert": {"triggered": False}, "proxy_symbols": ["TSLA"]},
        "macro_event": {"ok": True, "overall_status": "ready"},
        "event_store": {"ok": True, "overall_status": "ready"},
        "notification_watch": {"status": "running", "last_delivery": {"imessage_attempted": True, "imessage": {"returncode": 0}, "mac": {"returncode": 0}}},
        "notification_ladder": {"ok": True, "overall_status": "ready"},
        "promotion_quality": {"ok": True, "overall_status": "ready"},
        "promotion_packet": {"overall_status": "ready", "promotion_ready": True, "readiness_repair_contract": {"critical_repair_gate_count": 0, "warning_repair_gate_count": 0}},
        "promotion_pipeline": {"ok": True, "overall_status": "ready"},
        "release_freeze": {"ok": True, "overall_status": "ready"},
        "runtime_dependency": {"ok": True, "overall_status": "ready"},
        "library_router": {"ok": True, "overall_status": "ready"},
        "mlx_router": {"ok": True, "overall_status": "ready"},
        "mlx_upgrade": {"ok": True, "overall_status": "ready"},
        "storage_dr": {"ok": True, "overall_status": "ready"},
        "post_restart": {"ok": True, "overall_status": "ready"},
        "paper_replay_drill": {"ok": True, "overall_status": "ready"},
        "runtime_dashboard": {"overall": {"ok": True, "status": "ok", "attention": []}},
        "unattended_soak": {"ok": True, "overall_status": "ready", "overall_grade": "A+", "safe_to_leave_unattended": True, "blockers": [], "warnings": []},
    }
    base.update(overrides)
    for name, payload in base.items():
        _write_json(root / packet.SOURCE_FILES[name], payload)
    lock_path = root / "config" / "requirements.lock.txt"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("python==3.14.5\n", encoding="utf-8")


def test_a_plus_packet_all_lanes_ready(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    payload = packet.build_payload(tmp_path)

    assert payload["a_plus_ready"] is True
    assert payload["overall_grade"] in {"A+", "A+"}
    assert payload["a_plus_lane_count"] == 10
    assert payload["blocker_count"] == 0


def test_a_plus_packet_treats_safe_local_fallback_dr_as_managed(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        {
            "storage_dr": {
                "ok": True,
                "overall_status": "degraded",
                "current_storage_mode": "local_fallback",
                "storage_probe": {"external_available": True},
                "curated_restore": {"skipped_reason": "writer_not_quiet"},
            }
        },
    )

    payload = packet.build_payload(tmp_path)
    lane = next(row for row in payload["lanes"] if row["id"] == "disaster_recovery")

    assert lane["a_plus"] is True
    assert lane["score"] == 100.0
    assert lane["evidence"]["storage_dr_managed_advisory"] is True
    assert "storage_dr_managed_local_fallback" in lane["warnings"]


def test_a_plus_packet_treats_creative_hold_restart_due_as_managed(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        {
            "process_watchdog": {
                "overall_status": "ready",
                "alerts": [],
                "creative_cotenant_pause": {"active": True, "reason": "music_playback"},
                "status": [
                    {
                        "name": "all_sleeves",
                        "paused_by_creative_cotenant_guard": True,
                        "reason": "music_playback",
                    }
                ],
            },
            "rolling_restart": {
                "ok": False,
                "overall_status": "degraded",
                "restart_due": True,
                "recommended_scope": "full_stack",
                "due_signals": {
                    "session_stale": True,
                    "shadow_heartbeat_stale": False,
                    "swap_pressure_high": False,
                    "restart_storm_present": False,
                    "checkpoint_missing_or_stale": False,
                },
                "runtime_signals": {"restart_storms": 0},
                "checkpoint_resume": {"checkpoint_fresh": True, "missing_files": []},
            },
        },
    )

    payload = packet.build_payload(tmp_path)
    health_lane = next(row for row in payload["lanes"] if row["id"] == "health_scorecard")
    dr_lane = next(row for row in payload["lanes"] if row["id"] == "disaster_recovery")

    assert health_lane["a_plus"] is True
    assert dr_lane["a_plus"] is True
    assert health_lane["evidence"]["rolling_restart_managed_advisory"] is True
    assert dr_lane["evidence"]["rolling_restart_managed_advisory"] is True
    assert "rolling_restart_managed_creative_hold" in health_lane["warnings"]
    assert "rolling_restart_managed_creative_hold" in dr_lane["warnings"]


def test_a_plus_packet_does_not_penalize_guarded_ready_runtime_heat(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        {
            "runtime_throttle": {
                "ok": True,
                "overall_status": "ready",
                "host_saturation_score": 66.4,
                "compute_pressure_level": "elevated",
                "memory_pressure_level": "normal",
                "soft_cap_advisory_reclassification": {
                    "active": True,
                    "to_status": "ready",
                    "measurements": {"runtime_ready_guarded": True},
                },
            },
        },
    )

    payload = packet.build_payload(tmp_path)
    lane = next(row for row in payload["lanes"] if row["id"] == "anti_degradation_guardrails")

    assert lane["a_plus"] is True
    assert lane["score"] == 100.0
    assert lane["evidence"]["runtime_guarded_ready"] is True
    assert "host_saturation_guarded_or_hot" in lane["warnings"]


def test_a_plus_packet_treats_managed_sql_quota_and_external_host_pressure_as_ready(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        {
            "ingestion_storage": {
                "ok": True,
                "overall_status": "ready",
                "severity": "stable",
                "pressure_index": 0.667,
                "backpressure": {
                    "core_pending_lines": 598,
                    "total_pending_lines": 598,
                    "oldest_pending_age_seconds": 160.133,
                },
            },
            "runtime_throttle": {
                "ok": True,
                "overall_status": "advisory",
                "host_saturation_score": 63.62,
                "soft_cap_advisory_reclassification": {
                    "active": True,
                    "to_status": "advisory",
                    "reason": "external_high_compute_with_bounded_storage_overlay_is_capacity_limited_advisory",
                    "thresholds": {"max_guarded_external_high_compute_host_saturation_score": 75.0},
                    "measurements": {
                        "host_saturation_score": 63.62,
                        "external_high_compute_guarded": True,
                        "bounded_storage_overlay_guarded": True,
                        "paper_ramp_memory_guarded": True,
                        "paper_execution_allowed": True,
                        "paper_execution_paused": False,
                        "paper_execution_hot": False,
                        "research_training_hot": False,
                        "bot_owned_pressure_dominant": False,
                    },
                },
                "paper_execution_policy": {
                    "paper_execution_allowed": True,
                    "pause_paper_execution": False,
                    "stage": "armed",
                    "armed": True,
                    "blockers": [],
                },
            },
            "memory_efficiency": {
                "ok": True,
                "overall_status": "advisory",
                "storage_snapshot": {
                    "stateful_sql_soft_quota_relief": {
                        "managed": True,
                        "status": "managed",
                        "hard_breaches": 0,
                        "soft_breaches": 1,
                        "sql_link_hard_ratio": 0.879,
                        "max_managed_hard_ratio": 0.92,
                        "continuous_run_ready": True,
                        "quota_ready": True,
                        "storage_stable": True,
                        "raw_live_backlog": {
                            "clear": True,
                            "core_pending_lines": 598,
                            "total_pending_lines": 598,
                            "oldest_pending_age_seconds": 160.133,
                            "max_core_pending_lines": 10000,
                            "max_total_pending_lines": 15000,
                            "max_oldest_pending_age_seconds": 900,
                        },
                    }
                },
            },
        },
    )

    payload = packet.build_payload(tmp_path)
    lane = next(row for row in payload["lanes"] if row["id"] == "anti_degradation_guardrails")

    assert lane["a_plus"] is True
    assert lane["score"] == 100.0
    assert lane["evidence"]["stateful_sql_soft_quota_relief"]["managed"] is True
    assert lane["evidence"]["host_saturation_relief"]["managed"] is True
    assert "storage_pressure_managed_stateful_sql_soft_quota" in lane["warnings"]
    assert "host_saturation_managed_external_capacity" in lane["warnings"]
    assert "storage_pressure_index_above_target" not in lane["warnings"]
    assert "host_saturation_guarded_or_hot" not in lane["warnings"]


def test_a_plus_packet_keeps_hot_host_saturation_actionable_above_managed_ceiling(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        {
            "runtime_throttle": {
                "ok": True,
                "overall_status": "advisory",
                "host_saturation_score": 78.0,
                "soft_cap_advisory_reclassification": {
                    "active": True,
                    "to_status": "advisory",
                    "reason": "external_high_compute_with_bounded_storage_overlay_is_capacity_limited_advisory",
                    "thresholds": {"max_guarded_external_high_compute_host_saturation_score": 75.0},
                    "measurements": {
                        "host_saturation_score": 78.0,
                        "external_high_compute_guarded": True,
                        "bounded_storage_overlay_guarded": True,
                        "paper_execution_allowed": True,
                        "paper_execution_paused": False,
                        "paper_execution_hot": False,
                        "bot_owned_pressure_dominant": False,
                    },
                },
                "paper_execution_policy": {
                    "paper_execution_allowed": True,
                    "pause_paper_execution": False,
                    "stage": "armed",
                    "armed": True,
                    "blockers": [],
                },
            }
        },
    )

    payload = packet.build_payload(tmp_path)
    lane = next(row for row in payload["lanes"] if row["id"] == "anti_degradation_guardrails")

    assert lane["a_plus"] is False
    assert lane["score"] == 85.0
    assert lane["evidence"]["host_saturation_relief"]["managed"] is False
    assert "host_saturation_guarded_or_hot" in lane["warnings"]
    assert "host_saturation_managed_external_capacity" not in lane["warnings"]


def test_a_plus_packet_treats_paper_ramp_memory_hold_as_managed(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        {
            "paper_ramp": {
                "ok": False,
                "stage": "blocked",
                "armed": False,
                "blockers": ["memory_pressure_above_paper_400_gate"],
                "gates": {
                    "runtime": {"ok": True, "live_execution_locked": True},
                    "global_halt": {"ok": True},
                    "memory": {"ok": False, "overall_status": "advisory"},
                },
            },
            "memory_efficiency": {
                "ok": True,
                "overall_status": "advisory",
                "compressed_memory_relief_contract": {
                    "managed": True,
                    "status": "managed",
                    "pressure_clear": True,
                    "storage_clear": True,
                    "compressor_bounded": True,
                    "auto_apply_ready": True,
                },
            },
            "paper_profitability": {
                "ok": True,
                "overall_status": "protective_tightening",
                "paper_summary": {"executions": 10, "ending_net_pnl_total": -10.0},
                "a_plus_target_contract": {"operational_control_a_plus_ready": True},
            },
        },
    )

    payload = packet.build_payload(tmp_path)
    lane = next(row for row in payload["lanes"] if row["id"] == "paper_performance_attribution")

    assert lane["a_plus"] is True
    assert lane["evidence"]["paper_ramp_managed_memory_hold"] is True
    assert "paper_ramp_not_armed" not in lane["blockers"]
    assert "paper_ramp_managed_memory_hold" in lane["warnings"]
    assert "raw_profitability_protected_by_a_plus_controls" in lane["warnings"]


def test_a_plus_packet_treats_inactive_event_watch_as_managed_cold_lane(tmp_path: Path) -> None:
    _write_sources(tmp_path, {"spacex_ipo_watch": {}})

    payload = packet.build_payload(tmp_path)
    lane = next(row for row in payload["lanes"] if row["id"] == "event_mode")

    assert lane["a_plus"] is True
    assert lane["score"] == 95.0
    assert lane["evidence"]["event_watch_managed_cold_lane"] is True
    assert "event_watch_cold_lane_managed_by_unattended_soak" in lane["warnings"]
    assert "event_policy_not_explicitly_monitoring_only" not in lane["warnings"]


def test_a_plus_packet_defers_promotion_repairs_while_paper_soak_green(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        {
            "promotion_packet": {
                "overall_status": "degraded",
                "promotion_ready": False,
                "canary_packet_ready": False,
                "autopilot_state": "assembling_packet",
                "blockers": [
                    "coverage_shortfall_bots=4",
                    "promotion_readiness:insufficient_walk_forward_coverage",
                    "promotion_pipeline_failed",
                    "promotion_packet_incomplete",
                    "promotion_packet_signature_unverified",
                ],
                "readiness_repair_contract": {
                    "critical_repair_gate_count": 1,
                    "warning_repair_gate_count": 1,
                },
            },
            "promotion_pipeline": {"ok": False, "overall_status": "degraded"},
        },
    )

    payload = packet.build_payload(tmp_path)
    lane = next(row for row in payload["lanes"] if row["id"] == "promotion_discipline")

    assert lane["a_plus"] is True
    assert lane["score"] == 100.0
    assert lane["blockers"] == []
    assert lane["evidence"]["promotion_deferred_while_paper_soak_green"] is True
    assert "promotion_deferred_while_paper_soak_green" in lane["warnings"]
    assert "promotion_repairs_deferred_while_live_money_locked" in lane["warnings"]


def test_a_plus_packet_blocks_on_missing_imessage_delivery(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        {
            "notification_watch": {
                "status": "running",
                "last_delivery": {"imessage_attempted": True, "imessage": {"returncode": 1}, "mac": {"returncode": 0}},
            }
        },
    )

    payload = packet.build_payload(tmp_path)
    notification_lane = next(row for row in payload["lanes"] if row["id"] == "notification_reliability")

    assert payload["a_plus_ready"] is False
    assert notification_lane["a_plus"] is False
    assert "imessage_last_delivery_not_confirmed" in notification_lane["blockers"]
