import json
import sys
from pathlib import Path

from scripts.ops import coordination_state_control as src


FRESH_TS = "2099-01-01T00:00:00+00:00"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_base_artifacts(project_root: Path) -> None:
    health = project_root / "governance" / "health"
    alerts = project_root / "governance" / "alerts"
    _write_json(
        health / "halt_trigger_control_plane_latest.json",
        {
            "timestamp_utc": FRESH_TS,
            "overall_status": "blocked",
            "effective_state": "live_read_only",
            "manual_flags": {
                "paper_trade_lock": {
                    "active": True,
                    "reason": "live_data_paper_trade_only",
                    "payload": {"paper_execution_allowed": True},
                }
            },
            "blockers": {
                "halt_clear": [],
                "live_execution": ["paper_trade_lock_active", "runtime_release_live_read_only"],
                "heavy_viewer": [],
            },
            "execution_policy": {
                "control_plane_allows_live_orders": False,
                "environment_expects_live_orders": False,
                "effective_live_order_execution_allowed": False,
                "paper_trade_lock_active": True,
                "operator_stop_active": False,
            },
            "viewer_policy": {
                "light_livefeed_allowed": True,
                "heavy_livefeed_allowed": True,
                "heavy_livefeed_wait_reasons": [],
            },
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": FRESH_TS,
            "overall_status": "advisory",
            "throttle_profile": "soft_cap",
            "mac_fluidity_contract": {
                "foreground_first": True,
                "foreground_active": False,
                "fluidity_band": "guarded_smooth",
            },
            "release_contract": {"paper_trade_lock_active": True},
        },
    )
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "timestamp_utc": FRESH_TS,
            "overall_status": "ready",
            "watchdog_intelligence": {
                "target_count": 4,
                "healthy_target_count": 4,
                "active_issue_count": 0,
                "missing_targets": [],
                "stale_targets": [],
            },
            "restarts": [],
            "restart_storms": [],
            "recent_restart_storms": [],
        },
    )
    _write_json(
        health / "shadow_watchdog_tripwire_latest.json",
        {"timestamp_utc": FRESH_TS, "enabled": True, "active": False, "active_incidents": []},
    )
    _write_json(
        health / "guardrail_triprate_latest.json",
        {"timestamp_utc": FRESH_TS, "ok": True, "trip_count": 0, "trip_rate": 0.0},
    )
    _write_json(
        health / "remote_alert_control_latest.json",
        {
            "timestamp_utc": FRESH_TS,
            "overall_status": "ready",
            "critical_backlog": {"unacked_count": 0, "unsent_count": 0},
        },
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "timestamp_utc": FRESH_TS,
            "overall_status": "constrained",
            "training_launch_contract": {
                "mode": "prep_only",
                "launch_allowed": False,
                "prep_allowed": True,
                "launch_blockers": ["host_training_headroom_not_clear"],
                "prep_blockers": [],
                "recommended_batch_size": 0,
            },
        },
    )
    _write_json(
        health / "livefeed_local_latest.json",
        {"timestamp_utc": FRESH_TS, "status": "running", "alive": True, "contract": "launchd_local_livefeed_mirror"},
    )
    _write_json(
        health / "live_feed_heavy_guarded_latest.json",
        {"timestamp_utc": FRESH_TS, "status": "running", "allowed": True, "heavy_pid": 1234},
    )
    _write_json(
        health / "operator_control_latest.json",
        {"timestamp_utc": FRESH_TS, "operator_stop": False, "global_halt": False},
    )
    _write_json(
        alerts / "incident_auto_halt_latest.json",
        {"timestamp_utc": FRESH_TS, "ok": True, "halt": False, "failed_checks": []},
    )


def test_coordination_state_unifies_read_only_paper_livefeed_and_training(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _write_base_artifacts(project_root)
    monkeypatch.setattr(src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["coordination_state_control.py", "--json"])

    rc = src.main()
    payload = json.loads((project_root / "governance" / "health" / "coordination_state_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["overall_status"] == "guarded"
    assert payload["policies"]["live_orders"]["allowed"] is False
    assert payload["policies"]["paper_execution"]["allowed"] is True
    assert payload["policies"]["heavy_viewer"]["allowed"] is True
    assert payload["policies"]["training_prep"]["allowed"] is True
    assert payload["policies"]["training_launch"]["allowed"] is False
    assert payload["policies"]["terminal_restart"]["safe"] is True


def test_coordination_tripwire_triage_splits_suppressed_expected_and_actionable(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_base_artifacts(project_root)
    _write_json(
        health / "shadow_watchdog_tripwire_latest.json",
        {
            "timestamp_utc": FRESH_TS,
            "enabled": True,
            "active": True,
            "active_incidents": [
                {
                    "target": "coinbase_shadow",
                    "required": True,
                    "process_live": False,
                    "heartbeat_lost": True,
                    "action": "suppressed",
                    "note": "process_missing,creative_audio_pause_guard_active",
                },
                {
                    "target": "schwab_parallel",
                    "required": True,
                    "process_live": False,
                    "heartbeat_lost": True,
                    "action": "page",
                    "note": "process_missing",
                },
            ],
        },
    )
    monkeypatch.setattr(src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["coordination_state_control.py", "--json"])

    rc = src.main()
    payload = json.loads((health / "coordination_state_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    triage = payload["tripwire_triage"]
    assert triage["overall_status"] == "active_actionable"
    assert triage["counts"]["suppressed_by_guard"] == 1
    assert triage["counts"]["expected_offline"] == 1
    assert triage["counts"]["needs_operator"] == 1
    assert payload["policies"]["terminal_restart"]["safe"] is False


def test_coordination_crashloop_classifier_tags_common_causes(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_base_artifacts(project_root)
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "timestamp_utc": FRESH_TS,
            "overall_status": "ready",
            "watchdog_intelligence": {
                "target_count": 4,
                "healthy_target_count": 4,
                "active_issue_count": 0,
                "missing_targets": [],
                "stale_targets": [],
            },
            "restarts": [
                {"target": "sql_link_writer", "message": "restart storm budget exhausted"},
                {"target": "auth_refresher", "message": "OAuth token expired"},
                {"target": "storage_guard", "message": "BOT_LOGS mount missing"},
            ],
        },
    )
    monkeypatch.setattr(src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["coordination_state_control.py", "--json"])

    rc = src.main()
    payload = json.loads((health / "coordination_state_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    causes = payload["crashloop_cause_classifier"]["cause_counts"]
    assert causes["restart_storm_or_budget"] == 1
    assert causes["auth_or_token_failure"] == 1
    assert causes["storage_mount_or_backpressure"] == 1
    assert payload["policies"]["terminal_restart"]["safe"] is False


def test_coordination_foreground_protect_defers_heavy_viewer(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_base_artifacts(project_root)
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": FRESH_TS,
            "overall_status": "blocked",
            "throttle_profile": "protect_live",
            "mac_fluidity_contract": {
                "foreground_first": True,
                "foreground_active": True,
                "fluidity_band": "protect",
            },
        },
    )
    monkeypatch.setattr(src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["coordination_state_control.py", "--json"])

    rc = src.main()
    payload = json.loads((health / "coordination_state_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["coordination_mode"] == "foreground_first"
    assert payload["operator_intent"]["protected"] is True
    assert payload["policies"]["heavy_viewer"]["allowed"] is False
    assert "livefeed_heavy" in payload["priority_arbiter"]["deferred_lanes"]
