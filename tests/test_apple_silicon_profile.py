import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import apple_silicon_profile as src


def test_detect_profile_tier_prefers_air_safe_for_lower_memory_apple_silicon() -> None:
    hardware = {
        "system": "Darwin",
        "machine": "arm64",
        "chip": "Apple M2",
        "memory_gb": 16.0,
        "is_apple_silicon": True,
    }

    assert src.detect_profile_tier(hardware) == "air_safe"


def test_build_payload_includes_detected_and_applied_tiers(tmp_path: Path) -> None:
    override_path = tmp_path / ".env.apple_silicon_override"
    override_path.write_text("BOT_APPLE_SILICON_TIER=max_throughput\n", encoding="utf-8")
    hardware = {
        "system": "Darwin",
        "machine": "arm64",
        "chip": "Apple M4 Max",
        "memory_gb": 64.0,
        "is_apple_silicon": True,
        "physical_core_count": 10,
        "logical_core_count": 10,
        "performance_core_count": 8,
        "efficiency_core_count": 2,
    }

    payload = src.build_payload(
        action="status",
        tier="max_throughput",
        hardware=hardware,
        override_path=override_path,
        changed=False,
    )

    assert payload["detected_tier"] == "max_throughput"
    assert payload["applied_tier"] == "max_throughput"
    assert payload["override_exists"] is True
    assert payload["env_overrides"]["SQL_LINK_SERVICE_INTERVAL_SECONDS"] == "45"
    assert payload["unified_memory_telemetry"]["memory_architecture"] == "unified"
    assert payload["unified_memory_telemetry"]["competitive_advantage"] == "high"
    assert payload["env_overrides"]["COINBASE_SNAPSHOT_MAX_WORKERS"] == "4"
    assert payload["env_overrides"]["ASYNC_PIPELINE_WORKERS"] == "6"
    assert payload["env_overrides"]["BOT_CPU_ALLOCATION_POLICY"] == "performance_core_primary"
    assert payload["env_overrides"]["BOT_CPU_HARD_AFFINITY_SUPPORTED"] == "0"
    assert payload["env_overrides"]["BOT_CPU_QOS_POLICY"] == "performance_core_primary_no_background_writer"
    assert payload["env_overrides"]["BOT_CPU_EFFICIENCY_SATURATION_GUARD"] == "1"
    assert payload["env_overrides"]["BOT_PERFORMANCE_CORE_TARGET"] == "8"
    assert payload["env_overrides"]["BOT_EFFICIENCY_CORE_SPILLOVER_COUNT"] == "2"
    assert payload["env_overrides"]["SQL_LINK_WRITER_BACKGROUND_POLICY"] == "0"
    assert payload["env_overrides"]["SQL_LINK_WRITER_NICE"] == "0"
    assert payload["env_overrides"]["SLEEVE_WORKERS_BASELINE"] == "8"
    assert payload["env_overrides"]["SLEEVE_NICE_BASELINE"] == "0"
    assert payload["env_overrides"]["SLEEVE_NICE_SPECIALIZED"] == "6"
    assert payload["env_overrides"]["RUNTIME_TRAIN_MAX_SAMPLES"] == "32000"
    assert payload["env_overrides"]["RESOURCE_GUARD_CREATIVE_HOT_CPU_THRESHOLD"] == "135"
    assert payload["env_overrides"]["RESOURCE_GUARD_OPTIONAL_BLOCK_ON_CREATIVE_SESSION_LEVELS"] == "dual_pro,hot"
    assert payload["env_overrides"]["MEMORY_EFFICIENCY_CREATIVE_HOT_PROFILE"] == "air_safe"
    assert payload["env_overrides"]["CREATIVE_AUDIO_SAMPLE_RATE_HZ"] == "96000"
    assert payload["creative_audio_contract"]["target_sample_rate_hz"] == 96000
    assert payload["creative_audio_contract"]["require_matched_input_output"] is True
    assert payload["performance_core_contract"]["policy"] == "performance_core_primary"
    assert payload["performance_core_contract"]["hard_affinity_supported"] is False
    assert payload["performance_core_contract"]["primary_performance_core_budget"] == 8
    assert payload["performance_core_contract"]["efficiency_spillover_core_budget"] == 2
    assert payload["performance_core_contract"]["worker_budget_contract"]["baseline_sleeve_workers"] == 8
    json.dumps(payload, ensure_ascii=True)


def test_override_lines_quote_values_with_spaces() -> None:
    hardware = {
        "system": "Darwin",
        "machine": "arm64",
        "chip": "Apple M4 Max",
        "memory_gb": 64.0,
        "is_apple_silicon": True,
        "physical_core_count": 10,
        "logical_core_count": 10,
        "performance_core_count": 8,
        "efficiency_core_count": 2,
    }

    lines = src.override_lines_for_tier("max_throughput", hardware)

    assert "RESOURCE_GUARD_CREATIVE_APP_NAMES='Final Cut Pro,Logic Pro,Music,iTunes'" in lines
    assert "BOT_APPLE_SILICON_CHIP=Apple_M4_Max" in lines
    assert "BOT_CPU_ALLOCATION_POLICY=performance_core_primary" in lines
    assert "BOT_CPU_QOS_POLICY=performance_core_primary_no_background_writer" in lines
    assert "BOT_PERFORMANCE_CORE_TARGET=8" in lines
    assert "BOT_EFFICIENCY_CORE_SPILLOVER_COUNT=2" in lines
    assert "SQL_LINK_WRITER_BACKGROUND_POLICY=0" in lines
    assert "SLEEVE_NICE_BASELINE=0" in lines
    assert "SLEEVE_WORKERS_BASELINE=8" in lines
    assert "LOGIC_PRO_AUDIO_SAMPLE_RATE_HZ=96000" in lines
