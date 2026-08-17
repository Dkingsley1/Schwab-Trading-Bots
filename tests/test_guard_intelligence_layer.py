import json
from pathlib import Path

from scripts.ops import process_fanout_guard
from scripts.ops import guard_intelligence_layer as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_guard_intelligence_relaxes_stale_throttle_when_fanout_is_calm(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "process_fanout_guard_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "triggered": False,
            "thresholds": {"max_count": 120, "target_count": 80, "max_rss_mb": 12288.0, "target_rss_mb": 8192.0},
            "fanout": {"process_count": 36, "total_rss_mb": 3800.0},
            "override": {"active": True, "hold_active": True},
        },
    )
    _write_json(
        health / "resource_guard_latest.json",
        {"overall_status": "ready", "ok": True, "memory_pressure_state": "green", "memory_free_pct": 34.0},
    )

    payload = src.build_payload(
        tmp_path,
        apply=True,
        collect_live=False,
        out_path=health / "guard_intelligence_latest.json",
        state_path=health / "guard_intelligence_state.json",
        override_path=tmp_path / "config" / ".env.guard_intelligence_override",
    )

    assert payload["policy_mode"] == "full_schwab_observe"
    assert payload["recommended_env_overrides"]["RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES"] == "1"
    assert payload["recommended_env_overrides"]["PROCESS_FANOUT_GUARD_ACTIVE"] == "0"
    override = (tmp_path / "config" / ".env.guard_intelligence_override").read_text(encoding="utf-8")
    assert "PROCESS_FANOUT_GUARD_MAX_RSS_MB=12288.0" in override
    assert "OPS_WATCHDOG_ALL_SLEEVES_WITH_AGGRESSIVE=1" in override


def test_guard_intelligence_downshifts_when_memory_or_fanout_is_hot(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "process_fanout_guard_latest.json",
        {
            "overall_status": "degraded",
            "ok": False,
            "triggered": True,
            "thresholds": {"max_count": 120, "target_count": 80, "max_rss_mb": 6144.0, "target_rss_mb": 4096.0},
            "fanout": {"process_count": 141, "total_rss_mb": 7900.0},
        },
    )
    _write_json(
        health / "resource_guard_latest.json",
        {"overall_status": "blocked", "ok": False, "memory_pressure_state": "red", "memory_free_pct": 5.0, "swap_used_gb": 28.0},
    )

    payload = src.build_payload(
        tmp_path,
        apply=True,
        collect_live=False,
        out_path=health / "guard_intelligence_latest.json",
        state_path=health / "guard_intelligence_state.json",
        override_path=tmp_path / "config" / ".env.guard_intelligence_override",
    )

    assert payload["policy_mode"] == "protective_throttle"
    assert payload["recommended_env_overrides"]["RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES"] == "0"
    assert payload["recommended_env_overrides"]["TRAINING_RUNTIME_PAUSED_FOR_FANOUT"] == "1"
    assert payload["signals"]["guard_status_counts"]["blocker_count"] >= 1


def test_guard_intelligence_live_budget_overrides_stale_trigger_bit(monkeypatch, tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "process_fanout_guard_latest.json",
        {
            "overall_status": "active",
            "ok": True,
            "triggered": True,
            "thresholds": {"max_count": 120, "target_count": 80, "max_rss_mb": 6144.0, "target_rss_mb": 4096.0},
            "fanout": {"process_count": 72, "total_rss_mb": 7800.0},
        },
    )
    _write_json(
        health / "resource_guard_latest.json",
        {"overall_status": "ready", "ok": True, "memory_pressure_state": "green", "memory_free_pct": 40.0},
    )
    monkeypatch.setattr(
        process_fanout_guard,
        "collect_processes",
        lambda project_marker=process_fanout_guard.DEFAULT_PROJECT_MARKER: [
            process_fanout_guard.ProcRow(101, 1, 0.0, 7800.0, 60, "/repo/scripts/run_shadow_training_loop.py --broker schwab")
        ],
    )

    payload = src.build_payload(
        tmp_path,
        apply=False,
        collect_live=True,
        out_path=health / "guard_intelligence_latest.json",
        state_path=health / "guard_intelligence_state.json",
        override_path=tmp_path / "config" / ".env.guard_intelligence_override",
    )

    assert payload["signals"]["fanout"]["triggered"] is False
    assert payload["policy_mode"] == "full_schwab_observe"


def test_guard_intelligence_keeps_paper_soak_sleeves_on_during_guarded_pressure_relief(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "process_fanout_guard_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "triggered": False,
            "thresholds": {"max_count": 120, "target_count": 80, "max_rss_mb": 12288.0, "target_rss_mb": 8192.0},
            "fanout": {"process_count": 34, "total_rss_mb": 2600.0},
        },
    )
    _write_json(
        health / "resource_guard_latest.json",
        {"overall_status": "ready", "ok": True, "memory_pressure_state": "green", "memory_free_pct": 90.0},
    )
    _write_json(
        health / "pressure_relief_control_latest.json",
        {
            "overall_status": "degraded",
            "ok": True,
            "tier": "guarded_relief",
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "normal",
            "storage_pressure": {"severity": "stable", "pressure_index": 0.0},
            "swap_pressure": {"tier": "normal", "raw_tier": "normal"},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "advisory",
            "paper_execution_policy": {
                "paper_execution_allowed": True,
                "pause_paper_execution": False,
                "stage": "armed",
                "armed": True,
            },
            "paper_capacity_contract": {
                "ready_for_700_bot_paper": True,
                "runtime_policy": {"live_execution_blocked": True},
            },
        },
    )
    _write_json(health / "paper_400_ramp_latest.json", {"overall_status": "ready", "armed": True, "stage": "armed"})

    payload = src.build_payload(
        tmp_path,
        apply=True,
        collect_live=False,
        out_path=health / "guard_intelligence_latest.json",
        state_path=health / "guard_intelligence_state.json",
        override_path=tmp_path / "config" / ".env.guard_intelligence_override",
    )

    assert payload["policy_mode"] == "full_schwab_observe"
    assert payload["signals"]["guard_status_counts"]["blockers"] == []
    assert "pressure_relief" in payload["signals"]["guard_status_counts"]["warnings"]
    assert payload["signals"]["storage_pressure"]["details"]["pressure_relief"]["paper_soak_advisory"] is True
    assert payload["recommended_env_overrides"]["RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES"] == "1"
    assert payload["recommended_env_overrides"]["PROCESS_FANOUT_GUARD_ACTIVE"] == "0"


def test_guard_intelligence_treats_deep_relief_pressure_only_paper_bypass_as_advisory(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "process_fanout_guard_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "triggered": False,
            "thresholds": {"max_count": 180, "target_count": 140, "max_rss_mb": 12288.0, "target_rss_mb": 8192.0},
            "fanout": {"process_count": 67, "total_rss_mb": 5450.0},
        },
    )
    _write_json(
        health / "resource_guard_latest.json",
        {"overall_status": "ready", "ok": True, "memory_pressure_state": "green", "memory_free_pct": 79.0, "swap_used_gb": 1.0},
    )
    _write_json(
        health / "pressure_relief_control_latest.json",
        {
            "overall_status": "blocked",
            "ok": False,
            "tier": "deep_relief",
            "compute_pressure_level": "high",
            "memory_pressure_level": "normal",
            "storage_pressure": {"severity": "stable", "pressure_index": 0.0},
            "swap_pressure": {"tier": "normal", "raw_tier": "normal"},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "ready",
            "paper_execution_policy": {
                "paper_execution_allowed": True,
                "pause_paper_execution": False,
                "reason": "paper_ramp_pressure_only_blocker_bypassed_for_full_force_soak",
                "pressure_pause_bypassed": True,
            },
            "paper_capacity_contract": {
                "ready_for_700_bot_paper": True,
                "attribution_capacity_advisory": True,
                "runtime_policy": {"live_execution_blocked": True},
            },
        },
    )
    _write_json(health / "paper_400_ramp_latest.json", {"overall_status": "ready", "armed": True, "stage": "armed"})

    payload = src.build_payload(
        tmp_path,
        apply=True,
        collect_live=False,
        out_path=health / "guard_intelligence_latest.json",
        state_path=health / "guard_intelligence_state.json",
        override_path=tmp_path / "config" / ".env.guard_intelligence_override",
    )

    assert payload["policy_mode"] == "full_schwab_observe"
    assert payload["pressure_score"] == 0.65
    assert payload["signals"]["guard_status_counts"]["blockers"] == []
    assert "pressure_relief" in payload["signals"]["guard_status_counts"]["warnings"]
    assert payload["signals"]["storage_pressure"]["details"]["pressure_relief"]["paper_soak_advisory"] is True
    assert payload["recommended_env_overrides"]["RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES"] == "1"


def test_guard_intelligence_is_loaded_after_pressure_relief_override() -> None:
    text = (src.PROJECT_ROOT / "scripts" / "ops" / "load_runtime_env.sh").read_text(encoding="utf-8")

    assert "config/.env.pressure_relief_override" in text
    assert "config/.env.guard_intelligence_override" in text
    assert text.index("config/.env.pressure_relief_override") < text.index("config/.env.guard_intelligence_override")
