import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import memory_efficiency_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_memory_efficiency_control_recommends_constrained_profile_under_pressure(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "resource_guard_latest.json",
        {
            "memory_pressure_state": "red",
            "memory_pressure_kind": "throttled",
            "memory_free_pct": 7.0,
            "swap_used_gb": 26.0,
            "compressed_store_gb": 20.0,
            "compressor_gb": 14.0,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "apple_silicon_profile_latest.json",
        {
            "applied_tier": "max_throughput",
            "env_overrides": {
                "COINBASE_SNAPSHOT_MAX_WORKERS": "4",
                "TRADE_BEHAVIOR_BATCH_SIZE": "1536",
                "MEMORY_EFFICIENCY_CREATIVE_ACTIVE_MAX_PROFILE": "pro_balanced",
            },
            "hardware": {"memory_gb": 32.0},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {"severity": "critical", "pressure_index": 3.2, "backpressure": {"estimated_core_drain_minutes": 42.0}},
    )

    payload = src.build_payload(tmp_path, action="status", override_path=tmp_path / "config" / ".env.memory_efficiency_override")

    assert payload["overall_status"] == "blocked"
    assert payload["recommended_profile"] == "constrained"
    assert payload["recommended_env_overrides"]["TRADE_BEHAVIOR_BATCH_SIZE"] == "512"
    assert payload["recommended_env_overrides"]["COINBASE_CACHE_MAX_ENTRIES"] == "96"
    assert payload["recommended_env_overrides"]["RUNTIME_TRAIN_BATCH_SIZE_CAP"] == "48"
    assert payload["recommended_env_overrides"]["SQLITE_TEMP_STORE_MODE"] == "FILE"
    assert payload["recommended_env_overrides"]["BOT_OPS_SQLITE_CACHE_SIZE_KB"] == "2048"
    assert payload["recommended_env_overrides"]["QUANT_MODEL_TRANSFORMER_SEQUENCE"] == "32"
    assert payload["recommended_env_overrides"]["QUANT_MODEL_DML_CROSSFIT_FOLDS"] == "2"
    assert payload["recommended_env_overrides"]["QUANT_MODEL_DMS_STEPS"] == "12"
    assert payload["unified_memory_telemetry"]["competitive_advantage_state"] == "eroding_under_swap"


def test_memory_efficiency_control_downshifts_for_dual_creative_session(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "resource_guard_latest.json",
        {
            "memory_pressure_state": "green",
            "memory_pressure_kind": "none",
            "memory_free_pct": 28.0,
            "swap_used_gb": 4.0,
            "compressed_store_gb": 2.0,
            "compressor_gb": 1.0,
            "creative_apps_active": True,
            "creative_app_count": 2,
            "creative_apps": ["Final Cut Pro", "Logic Pro"],
            "creative_session_level": "dual_pro",
            "editing_app_cpu_sum": 162.0,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "apple_silicon_profile_latest.json",
        {
            "applied_tier": "max_throughput",
            "env_overrides": {"COINBASE_SNAPSHOT_MAX_WORKERS": "4", "TRADE_BEHAVIOR_BATCH_SIZE": "1536"},
            "hardware": {"memory_gb": 64.0},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {"severity": "ready", "pressure_index": 0.2, "backpressure": {"estimated_core_drain_minutes": 6.0}},
    )

    payload = src.build_payload(tmp_path, action="status", override_path=tmp_path / "config" / ".env.memory_efficiency_override")

    assert payload["overall_status"] == "needs_work"
    assert payload["recommended_profile"] == "constrained"
    assert payload["creative_session"]["level"] == "dual_pro"
    assert payload["recommended_env_overrides"]["ONE_NUMBERS_REFRESH_INTERVAL_SECONDS"] == "1800"
    assert payload["recommended_env_overrides"]["TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N"] == "0"
    assert payload["recommended_env_overrides"]["COINBASE_SNAPSHOT_MAX_WORKERS"] == "1"
    assert payload["recommended_env_overrides"]["SQLITE_ANALYZE_ENABLED"] == "0"
    assert payload["recommended_env_overrides"]["BOT_OPS_SQLITE_TEMP_STORE_MODE"] == "FILE"


def test_memory_efficiency_control_keeps_single_creative_app_looser_on_max_tier(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "resource_guard_latest.json",
        {
            "memory_pressure_state": "green",
            "memory_pressure_kind": "none",
            "memory_free_pct": 30.0,
            "swap_used_gb": 3.0,
            "compressed_store_gb": 1.5,
            "compressor_gb": 0.8,
            "creative_apps_active": True,
            "creative_app_count": 1,
            "creative_apps": ["Final Cut Pro"],
            "creative_session_level": "active",
            "editing_app_cpu_sum": 58.0,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "apple_silicon_profile_latest.json",
        {
            "applied_tier": "max_throughput",
            "env_overrides": {
                "COINBASE_SNAPSHOT_MAX_WORKERS": "4",
                "TRADE_BEHAVIOR_BATCH_SIZE": "1536",
                "MEMORY_EFFICIENCY_CREATIVE_ACTIVE_MAX_PROFILE": "pro_balanced",
            },
            "hardware": {"memory_gb": 32.0},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {"severity": "ready", "pressure_index": 0.1, "backpressure": {"estimated_core_drain_minutes": 4.0}},
    )

    payload = src.build_payload(tmp_path, action="status", override_path=tmp_path / "config" / ".env.memory_efficiency_override")

    assert payload["overall_status"] == "needs_work"
    assert payload["recommended_profile"] == "pro_balanced"
    assert payload["creative_session"]["level"] == "active"
    assert payload["recommended_env_overrides"]["COINBASE_SNAPSHOT_MAX_WORKERS"] == "3"
    assert payload["recommended_env_overrides"]["ONE_NUMBERS_REFRESH_INTERVAL_SECONDS"] == "420"
    assert payload["recommended_env_overrides"]["ASYNC_PIPELINE_WORKERS"] == "4"
    assert payload["recommended_env_overrides"]["SQLITE_TEMP_STORE_MODE"] == "FILE"
    assert payload["recommended_env_overrides"]["BOT_OPS_SQLITE_CACHE_SIZE_KB"] == "4096"
    assert payload["unified_memory_telemetry"]["memory_architecture"] == "unified"


def test_write_override_shell_quotes_values_with_spaces(tmp_path: Path) -> None:
    override_path = tmp_path / "config" / ".env.memory_efficiency_override"

    changed = src._write_override(
        override_path,
        "constrained",
        {"RESOURCE_GUARD_CREATIVE_APP_NAMES": "Final Cut Pro,Logic Pro"},
    )

    assert changed is True
    text = override_path.read_text(encoding="utf-8")
    assert "RESOURCE_GUARD_CREATIVE_APP_NAMES='Final Cut Pro,Logic Pro'" in text

    completed = subprocess.run(
        [
            "zsh",
            "-c",
            f"source {override_path} && print -r -- \"$RESOURCE_GUARD_CREATIVE_APP_NAMES\"",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "Final Cut Pro,Logic Pro"


def test_memory_efficiency_control_downshifts_for_interactive_co_running_apps(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "resource_guard_latest.json",
        {
            "memory_pressure_state": "green",
            "memory_pressure_kind": "none",
            "memory_free_pct": 26.0,
            "swap_used_gb": 2.0,
            "co_running_apps_active": True,
            "co_running_class_count": 2,
            "co_running_classes": ["browser", "developer"],
            "co_running_apps": ["Google Chrome", "PyCharm"],
            "co_running_session_level": "interactive",
            "co_running_cpu_sum": 124.0,
            "co_running_class_cpu": {"browser": 66.0, "developer": 58.0},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "apple_silicon_profile_latest.json",
        {
            "applied_tier": "max_throughput",
            "env_overrides": {
                "COINBASE_SNAPSHOT_MAX_WORKERS": "4",
                "TRADE_BEHAVIOR_BATCH_SIZE": "1536",
                "MEMORY_EFFICIENCY_COTENANT_INTERACTIVE_MAX_PROFILE": "pro_balanced",
            },
            "hardware": {"memory_gb": 32.0},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {"severity": "ready", "pressure_index": 0.1, "backpressure": {"estimated_core_drain_minutes": 5.0}},
    )

    payload = src.build_payload(tmp_path, action="status", override_path=tmp_path / "config" / ".env.memory_efficiency_override")

    assert payload["overall_status"] == "needs_work"
    assert payload["recommended_profile"] == "pro_balanced"
    assert payload["co_running_session"]["level"] == "interactive"
    assert payload["recommended_env_overrides"]["COINBASE_SNAPSHOT_MAX_WORKERS"] == "3"
    assert payload["recommended_env_overrides"]["ONE_NUMBERS_REFRESH_INTERVAL_SECONDS"] == "420"
