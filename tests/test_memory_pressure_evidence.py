from datetime import datetime, timedelta, timezone

import pytest

from core.memory_pressure_evidence import allocation_only_memory_evidence
from scripts.ops import memory_efficiency_control as efficiency
from scripts.ops import runtime_throttle_control as runtime


@pytest.fixture
def readings():
    stamp = datetime.now(timezone.utc).isoformat()
    return (
        {"timestamp_utc": stamp, "memory_pressure_state": "green", "memory_pressure_kind": "none",
         "memory_free_pct": 92.0, "swap_used_gb": 9.7, "compressed_store_gb": 18.8,
         "compressor_gb": 0.04, "pages_throttled": 0, "local_disk_free_gb": 54.0},
        {"timestamp_utc": stamp, "swap_pressure": {
            "tier": "normal", "raw_tier": "normal", "swap_used_gb": 9.7,
            "memory_pressure_state": "green", "memory_pressure_kind": "none",
            "thresholds": {"calm_swap_gb": 10.0},
            "stale_swap_allocation_relief": {"memory_free_pct": 92.0, "compressor_gb": 0.04, "pages_throttled": 0},
        }},
    )


def test_actual_green_readings_corroborate_allocated_not_resident_pressure(readings):
    evidence = allocation_only_memory_evidence(*readings)
    assert evidence["ready"]
    assert evidence["swap_used_gb"] == 9.7
    assert evidence["resident_compressor_gb"] == 0.04


@pytest.mark.parametrize("key,value", [
    ("memory_pressure_state", "red"), ("memory_pressure_state", "yellow"),
    ("memory_pressure_kind", "throttled"), ("memory_pressure_kind", "disk_swap_headroom"),
    ("memory_free_pct", 84), ("memory_free_pct", 101), ("memory_free_pct", float("nan")),
    ("compressor_gb", 1.01), ("compressor_gb", -1), ("compressor_gb", None),
    ("pages_throttled", 1), ("pages_throttled", True),
    ("local_disk_free_gb", 31.9), ("swap_used_gb", 10.0), ("swap_used_gb", 20.0),
    ("swap_used_gb", float("inf")),
])
def test_real_pressure_and_invalid_resource_measurements_do_not_get_relief(readings, key, value):
    resource, swap = readings
    resource[key] = value
    assert not allocation_only_memory_evidence(resource, swap)["ready"]


@pytest.mark.parametrize("owner", [0, 1])
@pytest.mark.parametrize("age", [-60, 121])
def test_stale_future_or_rewritten_old_evidence_never_admits(readings, owner, age):
    readings[owner]["source_timestamp_utc"] = (datetime.now(timezone.utc) - timedelta(seconds=age)).isoformat()
    assert not allocation_only_memory_evidence(*readings)["ready"]


@pytest.mark.parametrize("owner", [0, 1])
def test_missing_timestamp_fails_closed(readings, owner):
    del readings[owner]["timestamp_utc"]
    assert not allocation_only_memory_evidence(*readings)["ready"]


@pytest.mark.parametrize("key,value", [
    ("tier", "calm"), ("raw_tier", "pause_research"),
    ("memory_pressure_state", "yellow"), ("swap_used_gb", 8.0),
])
def test_disagreeing_swap_owner_prevents_relief(readings, key, value):
    readings[1]["swap_pressure"][key] = value
    assert not allocation_only_memory_evidence(*readings)["ready"]


def test_missing_or_bad_second_resident_measurement_fails_closed(readings):
    observed = readings[1]["swap_pressure"]["stale_swap_allocation_relief"]
    observed["memory_free_pct"] = 1000
    assert not allocation_only_memory_evidence(*readings)["ready"]
    del observed["memory_free_pct"]
    assert not allocation_only_memory_evidence(*readings)["ready"]


def test_memory_owner_keeps_raw_counters_but_does_not_report_false_compression_pressure(readings):
    resource, swap = readings
    effective, truth = efficiency._memory_truth_reconciliation(resource, swap)
    assert effective["swap_used_gb"] == resource["swap_used_gb"]
    assert effective["compressed_store_gb"] == resource["compressed_store_gb"]
    assert truth["allocation_only_memory_evidence"]["ready"]
    assert efficiency._memory_pressure_clear(effective)
    profile, reasons, status, *_ = efficiency._recommended_profile("pro_balanced", effective, {})
    assert "compressed_memory_high" not in reasons
    assert "compressed_memory_critical" not in reasons
    assert status == "ready"
    assert profile == "pro_balanced"


def test_runtime_does_not_reintroduce_legacy_eight_gb_alarm(readings):
    resource, swap = readings
    memory = {"overall_status": "needs_work", "reasons": ["compressed_memory_high"],
              "memory_snapshot": dict(resource), "cotenant_awareness": {"memory_pressure_clear": False}}
    assert runtime._memory_pressure_level(resource, memory) == "elevated"
    assert runtime._memory_pressure_level(resource, memory, swap) == "normal"
    memory["reasons"].append("memory_pressure_red")
    memory["overall_status"] = "blocked"
    assert runtime._memory_pressure_level(resource, memory, swap) == "high"


def test_stale_relief_cannot_open_runtime_training(readings):
    resource, swap = readings
    memory = {"overall_status": "ready", "memory_snapshot": dict(resource),
              "cotenant_awareness": {"memory_pressure_clear": True}}
    swap["timestamp_utc"] = (datetime.now(timezone.utc) - timedelta(seconds=121)).isoformat()
    assert runtime._memory_pressure_level(resource, memory, swap) == "elevated"


def test_compositor_is_system_work_but_codex_cpu_is_still_counted():
    compositor = runtime._classify_process("/System/Library/PrivateFrameworks/SkyLight.framework/Resources/WindowServer -daemon")
    assert compositor["category"] == "system_cotenant"
    assert not compositor["throttle_candidate"]
    codex = runtime._classify_process("/Applications/ChatGPT.app/Contents/Helpers/Codex (Renderer)")
    assert codex["category"] == "interactive_cotenant"
    assert not codex["throttle_candidate"]


@pytest.mark.parametrize("name", ["training_research_batch", "training_dataset_preflight", "training_dataset_evaluation"])
def test_research_workers_are_not_unknown_foreground_apps(name):
    result = runtime._classify_process(f"python /repo/scripts/ops/{name}.py --json")
    assert result["category"] == "research_training"
    assert result["throttle_candidate"]


def test_hardening_watch_is_platform_observability_not_an_external_app():
    result = runtime._classify_process("python /repo/scripts/ops/production_hardening_watch.py")
    assert result["category"] == "operator_observability"
    assert not result["throttle_candidate"]
