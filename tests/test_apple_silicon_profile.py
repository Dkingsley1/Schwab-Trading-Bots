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
    assert payload["env_overrides"]["COINBASE_SNAPSHOT_MAX_WORKERS"] == "4"
    assert payload["env_overrides"]["ASYNC_PIPELINE_WORKERS"] == "6"
    assert payload["env_overrides"]["RUNTIME_TRAIN_MAX_SAMPLES"] == "32000"
    assert payload["env_overrides"]["RESOURCE_GUARD_CREATIVE_HOT_CPU_THRESHOLD"] == "135"
    assert payload["env_overrides"]["RESOURCE_GUARD_OPTIONAL_BLOCK_ON_CREATIVE_SESSION_LEVELS"] == "dual_pro,hot"
    assert payload["env_overrides"]["MEMORY_EFFICIENCY_CREATIVE_HOT_PROFILE"] == "air_safe"
    json.dumps(payload, ensure_ascii=True)


def test_override_lines_quote_values_with_spaces() -> None:
    hardware = {
        "system": "Darwin",
        "machine": "arm64",
        "chip": "Apple M4 Max",
        "memory_gb": 64.0,
        "is_apple_silicon": True,
    }

    lines = src.override_lines_for_tier("max_throughput", hardware)

    assert "RESOURCE_GUARD_CREATIVE_APP_NAMES='Final Cut Pro,Logic Pro'" in lines
    assert "BOT_APPLE_SILICON_CHIP=Apple_M4_Max" in lines
