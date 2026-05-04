import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import swap_pressure_governor as src


def test_swap_pressure_governor_pauses_shadow_training_loops_under_swap() -> None:
    assert "scripts/run_shadow_training_loop.py" in src.HEAVY_RESEARCH_PATTERNS


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _base_project(tmp_path: Path, resource_snapshot: dict) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "resource_guard_latest.json", resource_snapshot)
    _write_json(
        health / "apple_silicon_profile_latest.json",
        {
            "applied_tier": "max_throughput",
            "hardware": {"memory_gb": 64.0, "is_apple_silicon": True},
            "unified_memory_telemetry": {"shared_cpu_gpu_memory_pool": True},
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {"severity": "stable", "pressure_index": 0.05, "backpressure": {"estimated_core_drain_minutes": 2.0}},
    )
    rows = [
        {
            "bot_id": f"brain_refinery_v{i}",
            "active": True,
            "data_collection_active": True,
            "lifecycle_state": "data_collection_only",
            "sleeve_profile": f"sleeve_{i % 70}",
        }
        for i in range(560)
    ]
    _write_json(tmp_path / "master_bot_registry.json", {"sub_bots": rows})


def test_swap_pressure_governor_pauses_research_and_keeps_restarts_advisory_only(monkeypatch, tmp_path: Path) -> None:
    resource_snapshot = {
        "memory_pressure_state": "yellow",
        "memory_pressure_kind": "swap_only",
        "memory_free_pct": 80.0,
        "swap_used_gb": 18.4,
        "compressed_store_gb": 20.0,
        "compressor_gb": 2.5,
    }
    _base_project(tmp_path, resource_snapshot)
    monkeypatch.setattr(src, "_refresh_resource_guard", lambda project_root: resource_snapshot)
    monkeypatch.setattr(
        src,
        "_parse_process_rows",
        lambda: [
            {
                "pid": 100,
                "cpu_percent": 5.0,
                "mem_percent": 5.0,
                "rss_mb": 2300.0,
                "command": "/Users/dankingsley/Applications/PyCharm.app/Contents/MacOS/pycharm",
            }
        ],
    )
    monkeypatch.setattr(
        src,
        "_pause_heavy_research",
        lambda tier, apply, patterns=None: {
            "active": src._research_pause_active(tier),
            "apply": apply,
            "action": "observe",
            "match_count": 1,
            "terminated_count": 0,
            "matches": [],
            "terminated": [],
        },
    )

    payload = src.build_payload(
        tmp_path,
        apply=False,
        state_path=tmp_path / "governance" / "health" / "swap_pressure_governor_state.json",
        override_path=tmp_path / "config" / ".env.swap_pressure_override",
        memory_override_path=tmp_path / "config" / ".env.memory_efficiency_override",
        runtime_override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
    )

    assert payload["swap_pressure"]["tier"] == "pause_research"
    assert payload["env_overrides"]["SWAP_PRESSURE_HEAVY_RESEARCH_PAUSED"] == "1"
    assert payload["env_overrides"]["QUANT_RESEARCH_PAUSED_FOR_SWAP"] == "1"
    assert payload["env_overrides"]["MLX_LAZY_IMPORTS"] == "1"
    assert payload["restart_advisory"]["active"] is True
    assert payload["applied_actions"]["restart_big_apps"] == "notify_only_no_force_quit"
    assert payload["applied_actions"]["reboot"] == "notify_only_no_automatic_reboot"


def test_swap_pressure_governor_recommends_reboot_after_persistent_massive_pressure(monkeypatch, tmp_path: Path) -> None:
    now = datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc)
    resource_snapshot = {
        "memory_pressure_state": "yellow",
        "memory_pressure_kind": "swap_only",
        "memory_free_pct": 74.0,
        "swap_used_gb": 19.0,
        "compressed_store_gb": 24.0,
        "compressor_gb": 2.8,
    }
    _base_project(tmp_path, resource_snapshot)
    _write_json(
        tmp_path / "governance" / "health" / "swap_pressure_governor_state.json",
        {
            "current_tier": "pause_research",
            "pressure_started_utc": (now - timedelta(minutes=45)).isoformat(),
        },
    )
    monkeypatch.setattr(src, "_refresh_resource_guard", lambda project_root: resource_snapshot)
    monkeypatch.setattr(src, "_parse_process_rows", lambda: [])
    monkeypatch.setattr(
        src,
        "_pause_heavy_research",
        lambda tier, apply, patterns=None: {
            "active": src._research_pause_active(tier),
            "apply": apply,
            "action": "observe",
            "match_count": 0,
            "terminated_count": 0,
            "matches": [],
            "terminated": [],
        },
    )

    payload = src.build_payload(
        tmp_path,
        apply=False,
        state_path=tmp_path / "governance" / "health" / "swap_pressure_governor_state.json",
        override_path=tmp_path / "config" / ".env.swap_pressure_override",
        memory_override_path=tmp_path / "config" / ".env.memory_efficiency_override",
        runtime_override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        now=now,
    )

    assert payload["reboot_advisory"]["active"] is True
    assert payload["notification"]["event"] == "swap_pressure_reboot_recommended"
    assert "reboot when your work is saved" in payload["notification"]["message"]


def test_swap_pressure_governor_relaxes_stale_swap_allocation_with_headroom(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SWAP_PRESSURE_STALE_SWAP_RELIEF_ENABLED", "1")
    resource_snapshot = {
        "memory_pressure_state": "red",
        "memory_pressure_kind": "swap_only",
        "memory_free_pct": 90.0,
        "swap_used_gb": 19.0,
        "compressed_store_gb": 21.0,
        "compressor_gb": 0.4,
        "pages_throttled": 0,
    }
    _base_project(tmp_path, resource_snapshot)
    monkeypatch.setattr(src, "_refresh_resource_guard", lambda project_root: resource_snapshot)
    monkeypatch.setattr(src, "_parse_process_rows", lambda: [])
    monkeypatch.setattr(
        src,
        "_pause_heavy_research",
        lambda tier, apply, patterns=None: {
            "active": src._research_pause_active(tier),
            "apply": apply,
            "action": "none",
            "match_count": 0,
            "terminated_count": 0,
            "matches": [],
            "terminated": [],
        },
    )

    payload = src.build_payload(
        tmp_path,
        apply=False,
        state_path=tmp_path / "governance" / "health" / "swap_pressure_governor_state.json",
        override_path=tmp_path / "config" / ".env.swap_pressure_override",
        memory_override_path=tmp_path / "config" / ".env.memory_efficiency_override",
        runtime_override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
    )

    assert payload["swap_pressure"]["raw_tier"] == "pause_research"
    assert payload["swap_pressure"]["tier"] == "calm"
    assert payload["swap_pressure"]["stale_swap_allocation_relief"]["active"] is True
    assert payload["env_overrides"]["SWAP_PRESSURE_HEAVY_RESEARCH_PAUSED"] == "0"


def test_swap_pressure_governor_apply_writes_swap_override(monkeypatch, tmp_path: Path) -> None:
    resource_snapshot = {
        "memory_pressure_state": "yellow",
        "memory_pressure_kind": "swap_only",
        "memory_free_pct": 78.0,
        "swap_used_gb": 14.5,
        "compressed_store_gb": 12.0,
        "compressor_gb": 1.8,
    }
    _base_project(tmp_path, resource_snapshot)
    monkeypatch.setattr(src, "_refresh_resource_guard", lambda project_root: resource_snapshot)
    monkeypatch.setattr(src, "_parse_process_rows", lambda: [])
    monkeypatch.setattr(
        src.runtime_src,
        "apply_runtime_guard",
        lambda *args, **kwargs: {"applied": True, "override_changed": False, "collector_guard": {"changed_count": 0}},
    )
    monkeypatch.setattr(
        src,
        "_pause_heavy_research",
        lambda tier, apply, patterns=None: {
            "active": src._research_pause_active(tier),
            "apply": apply,
            "action": "none",
            "match_count": 0,
            "terminated_count": 0,
            "matches": [],
            "terminated": [],
        },
    )
    override = tmp_path / "config" / ".env.swap_pressure_override"

    payload = src.build_payload(
        tmp_path,
        apply=True,
        state_path=tmp_path / "governance" / "health" / "swap_pressure_governor_state.json",
        override_path=override,
        memory_override_path=tmp_path / "config" / ".env.memory_efficiency_override",
        runtime_override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
    )

    assert payload["swap_pressure"]["tier"] == "constrained"
    assert payload["apply_result"]["applied"] is True
    text = override.read_text(encoding="utf-8")
    assert "SWAP_PRESSURE_TIER=constrained" in text
    assert "RUNTIME_FEATURE_CACHE_MAX_ENTRIES=32" in text
