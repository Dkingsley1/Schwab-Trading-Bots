from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import codex_operator_bridge as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_codex_operator_bridge_builds_trade_and_gate_attention_packet(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "paper_performance_latest.json",
        {
            "day": {
                "day_utc": "20260614",
                "available": True,
                "executions": 120,
                "buy_count": 80,
                "sell_count": 40,
                "unique_symbols": 8,
                "ending_net_pnl_total": -42.5,
                "change_vs_previous_day": -55.0,
                "top_profiles": [{"name": "default", "executions": 120}],
                "top_symbols": [{"name": "NVDA", "executions": 12}],
                "top_strategies": [{"name": "paper_mirror::alpha", "executions": 10}],
            },
            "week": {
                "available": True,
                "executions": 300,
                "ending_net_pnl_total": 11.0,
                "week_to_date_change": 11.0,
                "rolling_change": 11.0,
            },
            "sleeve_latest": [
                {
                    "profile": "default",
                    "day_utc": "20260614",
                    "current_day_available": True,
                    "executions": 120,
                    "ending_net_pnl_total": -42.5,
                    "win_rate": 0.25,
                    "data_status": "current",
                    "top_loss_causes": [{"cause": "fill_quality:unknown", "loss_total": 42.5}],
                }
            ],
        },
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "constrained",
            "training_quality": {"training_quality_score": 100.0},
            "training_launch_contract": {
                "mode": "prep_only",
                "launch_allowed": False,
                "launch_blockers": ["host_training_headroom_not_clear"],
                "requested_batch_size": 30,
                "recommended_batch_size": 0,
                "recommended_retrain_command": [],
                "recommended_prep_commands": [["./scripts/ops/opsctl.sh", "memory-pressure-intelligence", "--apply", "--json"]],
            },
            "host_training_headroom_gate": {
                "status": "blocked",
                "safe_for_training": False,
                "batch_cap": 0,
                "memory_status": "soft_guard",
                "memory_decision": "cooldown_probe_only",
                "next_reentry_stage": {"stage": "micro_canary", "allowed_now": False},
            },
            "pretraining_drain_buffer": {"status": "ready", "safe_to_launch_now": True, "pending_lines": 1},
        },
    )
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "waiting_for_writer",
            "writer_state_before": {
                "active": True,
                "current_step": "shard_linking",
                "completed_shard_count": 2,
                "planned_shard_count": 4,
            },
            "drain_effectiveness": {"pending_after": 10},
        },
    )
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "soft_guard", "decision": "cooldown_probe_only"},
            "reopen_gate": {
                "safe_for_training": False,
                "training_batch_cap": 0,
                "consecutive_memory_clear_samples": 0,
                "memory_clear_required_samples_for_training": 3,
            },
            "snapshot": {"swap_used_gb": 1.0, "compressed_store_gb": 4.0},
        },
    )
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "ready", "throttle_profile": "soft_cap"})
    _write_json(health / "livefeed_local_latest.json", {"status": "running", "alive": True, "source": "all"})
    _write_json(health / "notification_escalation_ladder_latest.json", {"overall_status": "ready"})
    _write_json(health / "remote_alert_control_latest.json", {"overall_status": "ready"})
    _write_json(
        health / "mac_notification_watch_state.json",
        {
            "timestamp_utc": "2099-01-01T00:00:00+00:00",
            "imessage_enabled": True,
            "imessage_recipient_configured": True,
            "max_alert_age_seconds": 900.0,
            "last_sent_at": {"tripwire:demo": "2099-01-01T00:00:00+00:00"},
            "last_delivery": {"imessage": {"returncode": 0, "stderr": ""}},
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "advisory"
    assert payload["sections"]["paper_trading"]["day"]["executions"] == 120
    assert payload["sections"]["training"]["launch_allowed"] is False
    assert "watch_training_gate_and_only_launch_from_recommended_command" in payload["attention_packet"]["needs_codex"]
    assert "explain_negative_paper_day_and_check_loss_causes" in payload["attention_packet"]["needs_codex"]
    assert payload["attention_packet"]["active_blockers"] == ["host_training_headroom_not_clear"]
    assert ["./scripts/ops/opsctl.sh", "memory-pressure-intelligence", "--apply", "--json"] in payload["attention_packet"]["safe_next_commands"]
    assert payload["sections"]["notifications"]["mac_watch_status"] == "ready"
    assert payload["sections"]["notifications"]["mac_watch_imessage_enabled"] is True
    assert payload["sections"]["notifications"]["last_error"] is None


def test_codex_operator_bridge_writes_json_and_markdown(tmp_path: Path) -> None:
    payload = src.build_payload(tmp_path)
    out_path = tmp_path / "governance" / "health" / "codex_operator_bridge_latest.json"
    md_path = tmp_path / "exports" / "reports" / "operator" / "codex_operator_bridge_latest.md"

    src.write_outputs(payload, out_path=out_path, markdown_path=md_path)

    assert out_path.exists()
    assert md_path.exists()
    assert "# Codex Operator Bridge" in md_path.read_text(encoding="utf-8")
