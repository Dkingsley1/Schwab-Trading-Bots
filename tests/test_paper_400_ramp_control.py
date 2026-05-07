import json
from datetime import date
from pathlib import Path

from scripts.ops import paper_400_ramp_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_ready_project(project_root: Path, *, bot_count: int = 700) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "needs_work",
            "recommended_profile": "constrained",
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_free_pct": 52.0,
                "swap_used_gb": 0.5,
                "compressed_store_gb": 18.5,
                "compressor_gb": 9.5,
            },
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "degraded",
            "throttle_profile": "sustain",
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "elevated",
            "paper_capacity_contract": {
                "ready_for_700_bot_paper": True,
                "pressure_limited": False,
                "active_bot_count": bot_count,
                "paper_tagged_count": bot_count,
            },
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.02,
            "backpressure": {
                "core_pending_lines": 120,
                "total_pending_lines": 2500,
            },
        },
    )
    _write_json(
        health / "global_killswitch_latest.json",
        {
            "halt": False,
            "clear_ready": True,
            "clear_blockers": [],
        },
    )
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": f"brain_refinery_v{i}_paper_ready_bot",
                    "active": True,
                    "paper_trade_enabled": True,
                    "lifecycle_state": "active",
                }
                for i in range(bot_count)
            ]
        },
    )


def test_paper_400_ramp_plans_before_activation_without_high_caps(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 4),
        registry_path=tmp_path / "master_bot_registry.json",
    )
    payload = src.apply_payload(
        tmp_path,
        payload,
        out_path=tmp_path / "governance" / "health" / "paper_400_ramp_latest.json",
        override_path=tmp_path / "config" / ".env.paper_400_ramp_override",
    )

    override_text = (tmp_path / "config" / ".env.paper_400_ramp_override").read_text(encoding="utf-8")
    assert payload["stage"] == "planned"
    assert payload["armed"] is False
    assert payload["blockers"] == ["calendar_wait_until_2026-05-11"]
    assert "PAPER_400_RAMP_ARMED=0" in override_text
    assert "SCHWAB_TOP_BOT_PAPER_TRADING_TOP_N" not in override_text
    assert "COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N" not in override_text


def test_paper_400_ramp_arms_after_activation_when_gates_are_clean(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )
    payload = src.apply_payload(
        tmp_path,
        payload,
        out_path=tmp_path / "governance" / "health" / "paper_400_ramp_latest.json",
        override_path=tmp_path / "config" / ".env.paper_400_ramp_override",
    )

    override_text = (tmp_path / "config" / ".env.paper_400_ramp_override").read_text(encoding="utf-8")
    allocation_total = sum(int(row["target"]) for row in payload["paper_allocation"]["lanes"].values())
    assert payload["stage"] == "armed"
    assert payload["armed"] is True
    assert allocation_total == 400
    assert "PAPER_400_RAMP_ARMED=1" in override_text
    assert "PAPER_400_RAMP_AGGREGATE_TOP_N=400" in override_text
    assert "SCHWAB_TOP_BOT_PAPER_TRADING_TOP_N=200" in override_text
    assert "SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_TOP_N=40" in override_text
    assert "SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N=80" in override_text
    assert "COINBASE_TOP_BOT_PAPER_TRADING_TOP_N=50" in override_text
    assert "COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N=30" in override_text


def test_paper_400_ramp_removes_high_caps_when_memory_blocks(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "memory_efficiency_control_latest.json",
        {
            "overall_status": "blocked",
            "recommended_profile": "constrained",
            "memory_snapshot": {
                "memory_pressure_state": "red",
                "memory_free_pct": 8.0,
                "swap_used_gb": 13.0,
                "compressed_store_gb": 31.0,
                "compressor_gb": 17.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )
    payload = src.apply_payload(
        tmp_path,
        payload,
        out_path=tmp_path / "governance" / "health" / "paper_400_ramp_latest.json",
        override_path=tmp_path / "config" / ".env.paper_400_ramp_override",
    )

    override_text = (tmp_path / "config" / ".env.paper_400_ramp_override").read_text(encoding="utf-8")
    assert payload["stage"] == "blocked"
    assert payload["armed"] is False
    assert "memory_pressure_above_paper_400_gate" in payload["blockers"]
    assert "SCHWAB_TOP_BOT_PAPER_TRADING_TOP_N" not in override_text
    assert "PAPER_400_RAMP_ARMED=0" in override_text


def test_paper_400_ramp_counts_only_explicit_paper_live_data_bots(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "collector_stability_only",
                    "active": True,
                    "lifecycle_state": "data_collection_only",
                    "paper_runtime_stability_mode": "full_force_guarded",
                },
                {
                    "bot_id": "legacy_paper_live_data",
                    "active": True,
                    "lifecycle_state": "active",
                    "paper_live_data_enabled": True,
                },
                {
                    "bot_id": "legacy_paper_trade",
                    "active": True,
                    "lifecycle_state": "active",
                    "paper_trade_enabled": True,
                },
                {
                    "bot_id": "inactive_paper",
                    "active": False,
                    "lifecycle_state": "inactive",
                    "paper_live_data_enabled": True,
                },
            ]
        },
    )

    counts = src._registry_counts(tmp_path, tmp_path / "master_bot_registry.json")

    assert counts["active_bot_count"] == 3
    assert counts["paper_tagged_count"] == 2
