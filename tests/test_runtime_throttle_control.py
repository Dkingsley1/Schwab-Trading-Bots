import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import runtime_throttle_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_runtime_throttle_control_protects_core_lanes_and_flags_support_jobs(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(
        health_root / "resource_guard_latest.json",
        {
            "memory_pressure_state": "yellow",
            "memory_pressure_kind": "none",
            "swap_used_gb": 9.5,
        },
    )
    _write_json(
        health_root / "memory_efficiency_control_latest.json",
        {
            "overall_status": "degraded",
            "recommended_profile": "pro_balanced",
        },
    )
    _write_json(
        health_root / "live_runtime_separation_control_latest.json",
        {
            "release_contract": {
                "live_lane_should_be_read_only": True,
                "promotions_should_wait_for_cold_lane": True,
                "shared_host_training_resume_allowed": False,
            }
        },
    )
    _write_json(
        health_root / "portable_brain_contract_latest.json",
        {
            "host_contract": {
                "chip": "Apple M5 Max",
                "memory_architecture": "unified",
                "shared_cpu_gpu_memory_pool": True,
            }
        },
    )

    runtime_snapshot = {
        "cpu_count": 12,
        "load_averages": {"one_minute": 13.2, "five_minutes": 11.4, "fifteen_minutes": 9.8},
        "thermal": {
            "thermal_warning_active": False,
            "performance_warning_active": False,
            "cpu_power_warning_active": False,
        },
        "vm_stat": {"pages_throttled": 0},
        "top_processes": [
            {
                "pid": 101,
                "cpu_percent": 86.5,
                "mem_percent": 2.0,
                "elapsed": "00:42:11",
                "command": "python scripts/run_execution_lane.py --mode paper",
                "category": "live_execution",
                "priority_tier": "protected",
                "throttle_candidate": False,
            },
            {
                "pid": 202,
                "cpu_percent": 70.2,
                "mem_percent": 1.4,
                "elapsed": "01:15:03",
                "command": "python scripts/ops/sql_queue_retention.py --vacuum",
                "category": "support_maintenance",
                "priority_tier": "throttle_first",
                "throttle_candidate": True,
            },
            {
                "pid": 303,
                "cpu_percent": 26.1,
                "mem_percent": 3.8,
                "elapsed": "02:01:33",
                "command": "python scripts/run_shadow_training_loop.py --broker schwab --profile dividend",
                "category": "research_training",
                "priority_tier": "protected",
                "throttle_candidate": False,
            },
            {
                "pid": 404,
                "cpu_percent": 39.2,
                "mem_percent": 0.6,
                "elapsed": "10:10:10",
                "command": "WindowServer",
                "category": "interactive_cotenant",
                "priority_tier": "external_cotenant",
                "throttle_candidate": False,
            },
        ],
        "category_cpu": {
            "live_execution": 86.5,
            "support_maintenance": 70.2,
            "research_training": 26.1,
            "interactive_cotenant": 39.2,
        },
        "category_counts": {
            "live_execution": 1,
            "support_maintenance": 1,
            "research_training": 1,
            "interactive_cotenant": 1,
        },
    }

    payload = src.build_payload(tmp_path, runtime_snapshot=runtime_snapshot)

    assert payload["overall_status"] == "degraded"
    assert payload["throttle_profile"] == "sustain"
    assert payload["host_saturation_score"] >= 56.0
    assert payload["protected_workloads"]["categories"] == ["live_execution", "research_training"]
    assert payload["support_trim_candidates"][0]["category"] == "support_maintenance"
    assert payload["upgrade_track"]["upgradeable"] is True
    assert any("off-hours throttle windows" in action for action in payload["recommended_actions"])


def test_runtime_throttle_control_escalates_to_protect_live_when_thermal_pressure_hits(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(
        health_root / "resource_guard_latest.json",
        {
            "memory_pressure_state": "red",
            "memory_pressure_kind": "throttled",
            "swap_used_gb": 24.0,
        },
    )
    _write_json(
        health_root / "memory_efficiency_control_latest.json",
        {
            "overall_status": "blocked",
            "recommended_profile": "constrained",
        },
    )
    _write_json(
        health_root / "live_runtime_separation_control_latest.json",
        {
            "release_contract": {
                "live_lane_should_be_read_only": True,
                "promotions_should_wait_for_cold_lane": True,
                "shared_host_training_resume_allowed": False,
            }
        },
    )

    runtime_snapshot = {
        "cpu_count": 10,
        "load_averages": {"one_minute": 15.0, "five_minutes": 13.5, "fifteen_minutes": 11.8},
        "thermal": {
            "thermal_warning_active": True,
            "performance_warning_active": True,
            "cpu_power_warning_active": False,
        },
        "vm_stat": {"pages_throttled": 0},
        "top_processes": [],
        "category_cpu": {},
        "category_counts": {},
    }

    payload = src.build_payload(tmp_path, runtime_snapshot=runtime_snapshot)

    assert payload["overall_status"] == "blocked"
    assert payload["throttle_profile"] == "protect_live"
    assert payload["memory_pressure_level"] == "high"
    assert payload["release_contract"]["live_lane_should_be_read_only"] is True


def test_heavy_livefeed_is_operator_observability_not_support_trim() -> None:
    row = src._classify_process(
        "/bin/zsh /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/live_feed_tail.sh --source all --heavy"
    )

    assert row["category"] == "operator_observability"
    assert row["priority_tier"] == "operator_visible"
    assert row["throttle_candidate"] is False

    tail_row = src._classify_process("tail -n 120 -F logs/a.log logs/b.log")

    assert tail_row["category"] == "operator_observability"
    assert tail_row["priority_tier"] == "operator_visible"
    assert tail_row["throttle_candidate"] is False
