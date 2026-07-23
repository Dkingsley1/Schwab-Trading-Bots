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
    assert payload["recommended_env_overrides"]["QUANT_MODEL_DAINN_LAYERS"] == "2"
    assert payload["recommended_env_overrides"]["QUANT_MODEL_DIFF_BACKTEST_STEPS"] == "12"
    assert payload["recommended_env_overrides"]["QUANT_MODEL_FORMAL_CHECKS"] == "6"
    assert payload["unified_memory_telemetry"]["competitive_advantage_state"] == "eroding_under_swap"


def test_memory_efficiency_control_treats_bounded_overlay_storage_recovery_as_advisory(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "resource_guard_latest.json",
        {
            "memory_pressure_state": "green",
            "memory_pressure_kind": "normal",
            "memory_free_pct": 70.0,
            "swap_used_gb": 1.3,
            "compressed_store_gb": 18.33,
            "compressor_gb": 6.482,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "apple_silicon_profile_latest.json",
        {
            "applied_tier": "max_throughput",
            "env_overrides": {"ASYNC_PIPELINE_WORKERS": "6"},
            "hardware": {"memory_gb": 64.0},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {
            "severity": "critical",
            "pressure_index": 32.377,
            "recovery_state": "stabilized_recovery",
            "storage": {"backlog_drain_status": "drain_active"},
            "backpressure": {
                "core_pending_lines": 672,
                "total_pending_lines": 1581,
                "overlay_adjusted": True,
                "effective_raw_live": {
                    "core_pending_lines": 672,
                    "total_pending_lines": 1581,
                    "oldest_pending_age_seconds": 7770.598,
                    "source": "sql_ingestion_overlay_pressure",
                    "raw_live_estimate": {
                        "core_pending_lines": 22,
                        "total_pending_lines": 929,
                        "oldest_pending_age_seconds": 0.0,
                    },
                },
                "effective_raw_live_source": "sql_ingestion_overlay_pressure",
            },
        },
    )

    payload = src.build_payload(tmp_path, action="status", override_path=tmp_path / "config" / ".env.memory_efficiency_override")

    assert payload["overall_status"] == "needs_work"
    assert payload["recommended_profile"] == "constrained"
    assert "storage_pressure_critical_overlay_recovery_advisory" in payload["reasons"]
    assert payload["storage_snapshot"]["bounded_overlay_relief"]["active"] is True


def test_memory_efficiency_control_reconciles_stale_allocation_high_water(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "resource_guard_latest.json",
        {
            "memory_pressure_state": "green",
            "memory_pressure_kind": "none",
            "memory_free_pct": 91.0,
            "swap_used_gb": 7.645,
            "compressed_store_gb": 19.552,
            "compressor_gb": 0.386,
            "pages_throttled": 0,
        },
    )
    _write_json(
        health / "swap_pressure_governor_latest.json",
        {
            "swap_pressure": {
                "tier": "normal",
                "swap_used_gb": 0.185,
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
            }
        },
    )
    _write_json(
        health / "apple_silicon_profile_latest.json",
        {"applied_tier": "max_throughput", "env_overrides": {}, "hardware": {"memory_gb": 32.0}},
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"severity": "stable", "pressure_index": 0.0})

    payload = src.build_payload(tmp_path, action="status", override_path=tmp_path / "config" / ".env.memory_efficiency_override")

    assert payload["overall_status"] == "ready"
    assert payload["recommended_profile"] == "max_throughput"
    assert "compressed_memory_high" not in payload["reasons"]
    assert "swap_usage_high" not in payload["reasons"]
    assert payload["memory_truth_reconciliation"]["active"] is True
    assert payload["memory_truth_reconciliation"]["stale_swap_relief"] is True
    assert payload["memory_truth_reconciliation"]["stale_compression_relief"] is True
    assert payload["raw_memory_snapshot"]["swap_used_gb"] == 7.645
    assert payload["memory_snapshot"]["swap_used_gb"] == 0.185
    assert payload["raw_memory_snapshot"]["compressed_store_gb"] == 19.552
    assert payload["memory_snapshot"]["compressed_store_gb"] == 8.0


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
    assert payload["recommended_env_overrides"]["COINBASE_SNAPSHOT_MAX_WORKERS"] == "1"
    assert payload["recommended_env_overrides"]["ONE_NUMBERS_REFRESH_INTERVAL_SECONDS"] == "1800"
    assert payload["recommended_env_overrides"]["ASYNC_PIPELINE_WORKERS"] == "1"
    assert payload["recommended_env_overrides"]["SQLITE_TEMP_STORE_MODE"] == "FILE"
    assert payload["recommended_env_overrides"]["BOT_OPS_SQLITE_CACHE_SIZE_KB"] == "1024"
    assert payload["unified_memory_telemetry"]["memory_architecture"] == "unified"


def test_memory_efficiency_control_downshifts_for_music_playback(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "resource_guard_latest.json",
        {
            "memory_pressure_state": "green",
            "memory_pressure_kind": "none",
            "memory_free_pct": 31.0,
            "swap_used_gb": 3.0,
            "compressed_store_gb": 1.5,
            "compressor_gb": 0.8,
            "creative_apps_active": True,
            "creative_app_count": 1,
            "creative_apps": ["Music"],
            "creative_session_level": "active",
            "creative_session_kind": "music_playback",
            "editing_app_cpu_sum": 4.5,
            "music_playback_cpu": 4.5,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "apple_silicon_profile_latest.json",
        {
            "applied_tier": "max_throughput",
            "env_overrides": {
                "COINBASE_SNAPSHOT_MAX_WORKERS": "4",
                "TRADE_BEHAVIOR_BATCH_SIZE": "1536",
                "MEMORY_EFFICIENCY_CREATIVE_MUSIC_PROFILE": "air_safe",
            },
            "hardware": {"memory_gb": 32.0},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {"severity": "ready", "pressure_index": 0.1, "backpressure": {"estimated_core_drain_minutes": 4.0}},
    )

    payload = src.build_payload(tmp_path, action="status", override_path=tmp_path / "config" / ".env.memory_efficiency_override")

    assert payload["overall_status"] == "advisory"
    assert payload["recommended_profile"] == "air_safe"
    assert payload["creative_session"]["kind"] == "music_playback"
    assert payload["cotenant_awareness"]["mode"] == "managed_media_cotenant"
    assert payload["cotenant_awareness"]["status_adjusted"] is True
    assert payload["recommended_env_overrides"]["COINBASE_SNAPSHOT_MAX_WORKERS"] == "1"
    assert payload["recommended_env_overrides"]["ASYNC_PIPELINE_WORKERS"] == "1"
    assert payload["recommended_env_overrides"]["AUDIO_PLAYBACK_PRIORITY"] == "1"
    assert payload["recommended_env_overrides"]["MUSIC_PLAYBACK_PRIORITY"] == "1"
    assert payload["recommended_env_overrides"]["CREATIVE_HEAVY_RESEARCH_PAUSED"] == "1"


def test_memory_efficiency_control_marks_music_and_green_high_compression_as_advisory(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "resource_guard_latest.json",
        {
            "memory_pressure_state": "green",
            "memory_pressure_kind": "normal",
            "memory_free_pct": 70.0,
            "swap_used_gb": 1.9,
            "compressed_store_gb": 20.9,
            "compressor_gb": 7.0,
            "creative_apps_active": True,
            "creative_app_count": 1,
            "creative_apps": ["Music"],
            "creative_session_level": "active",
            "creative_session_kind": "music_playback",
            "editing_app_cpu_sum": 4.5,
            "music_playback_cpu": 4.5,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "apple_silicon_profile_latest.json",
        {
            "applied_tier": "max_throughput",
            "env_overrides": {
                "MEMORY_EFFICIENCY_CREATIVE_MUSIC_PROFILE": "air_safe",
            },
            "hardware": {"memory_gb": 32.0},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {"severity": "stable", "pressure_index": 0.017},
    )

    payload = src.build_payload(tmp_path, action="status", override_path=tmp_path / "config" / ".env.memory_efficiency_override")

    assert payload["overall_status"] == "advisory"
    assert payload["recommended_profile"] == "air_safe"
    assert set(payload["reasons"]) == {"compressed_memory_high", "creative_session_music_playback"}
    assert payload["cotenant_awareness"]["mode"] == "managed_media_cotenant"
    assert payload["cotenant_awareness"]["memory_pressure_clear"] is True
    assert payload["cotenant_awareness"]["storage_pressure_clear"] is True


def test_write_override_shell_quotes_values_with_spaces(tmp_path: Path) -> None:
    override_path = tmp_path / "config" / ".env.memory_efficiency_override"

    changed = src._write_override(
        override_path,
        "constrained",
        {"RESOURCE_GUARD_CREATIVE_APP_NAMES": "Final Cut Pro,Logic Pro,Music,iTunes"},
    )

    assert changed is True
    text = override_path.read_text(encoding="utf-8")
    assert "RESOURCE_GUARD_CREATIVE_APP_NAMES='Final Cut Pro,Logic Pro,Music,iTunes'" in text

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

    assert completed.stdout.strip() == "Final Cut Pro,Logic Pro,Music,iTunes"


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

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    assert payload["recommended_profile"] == "pro_balanced"
    assert payload["co_running_session"]["level"] == "interactive"
    assert payload["cotenant_awareness"]["mode"] == "managed_cotenant"
    assert payload["cotenant_awareness"]["status_adjusted"] is True
    assert payload["cotenant_awareness"]["open_apps"] == ["Google Chrome", "PyCharm"]
    assert payload["recommended_env_overrides"]["COINBASE_SNAPSHOT_MAX_WORKERS"] == "3"
    assert payload["recommended_env_overrides"]["ONE_NUMBERS_REFRESH_INTERVAL_SECONDS"] == "420"
    assert payload["recommended_env_overrides"]["MEMORY_GUARD_COTENANT_MODE"] == "managed_cotenant"
    assert payload["recommended_env_overrides"]["MEMORY_GUARD_OPEN_APP_COUNT"] == "2"


def test_memory_efficiency_control_treats_light_cotenant_with_stable_storage_as_ready(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "resource_guard_latest.json",
        {
            "memory_pressure_state": "green",
            "memory_pressure_kind": "normal",
            "memory_free_pct": 85.0,
            "swap_used_gb": 0.8,
            "co_running_apps_active": True,
            "co_running_class_count": 1,
            "co_running_classes": ["browser"],
            "co_running_apps": ["Google Chrome"],
            "co_running_session_level": "light_competition",
            "co_running_cpu_sum": 38.0,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "apple_silicon_profile_latest.json",
        {
            "applied_tier": "max_throughput",
            "env_overrides": {"COINBASE_SNAPSHOT_MAX_WORKERS": "4"},
            "hardware": {"memory_gb": 32.0},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {"severity": "stable", "pressure_index": 0.307, "backpressure": {"estimated_core_drain_minutes": 12.0}},
    )

    payload = src.build_payload(tmp_path, action="status", override_path=tmp_path / "config" / ".env.memory_efficiency_override")

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    assert payload["cotenant_awareness"]["mode"] == "managed_cotenant"
    assert payload["cotenant_awareness"]["storage_pressure_clear"] is True


def test_memory_efficiency_control_treats_music_plus_light_cotenant_as_advisory(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "resource_guard_latest.json",
        {
            "memory_pressure_state": "green",
            "memory_pressure_kind": "normal",
            "memory_free_pct": 88.0,
            "swap_used_gb": 0.002,
            "compressed_store_gb": 3.886,
            "compressor_gb": 1.432,
            "creative_apps_active": True,
            "creative_app_count": 1,
            "creative_apps": ["Music"],
            "creative_session_level": "active",
            "creative_session_kind": "music_playback",
            "music_playback_cpu": 4.5,
            "co_running_apps_active": True,
            "co_running_class_count": 1,
            "co_running_classes": ["browser"],
            "co_running_apps": ["Google Chrome"],
            "co_running_session_level": "light_competition",
            "co_running_cpu_sum": 38.0,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "apple_silicon_profile_latest.json",
        {
            "applied_tier": "max_throughput",
            "env_overrides": {"MEMORY_EFFICIENCY_CREATIVE_MUSIC_PROFILE": "air_safe"},
            "hardware": {"memory_gb": 32.0},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {"severity": "stable", "pressure_index": 0.313},
    )

    payload = src.build_payload(tmp_path, action="status", override_path=tmp_path / "config" / ".env.memory_efficiency_override")

    assert payload["overall_status"] == "advisory"
    assert payload["ok"] is True
    assert set(payload["reasons"]) == {"creative_session_music_playback", "co_running_light_competition"}
    assert payload["cotenant_awareness"]["mode"] == "managed_media_cotenant"
    assert payload["cotenant_awareness"]["status_adjusted"] is True
    assert payload["cotenant_awareness"]["open_apps"] == ["Google Chrome", "Music"]


def test_memory_efficiency_control_keeps_cotenant_warning_when_memory_pressure_is_real(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "resource_guard_latest.json",
        {
            "memory_pressure_state": "yellow",
            "memory_pressure_kind": "swap_only",
            "memory_free_pct": 18.0,
            "swap_used_gb": 12.0,
            "co_running_apps_active": True,
            "co_running_class_count": 2,
            "co_running_classes": ["browser", "developer"],
            "co_running_apps": ["Google Chrome", "PyCharm"],
            "co_running_session_level": "interactive",
            "co_running_cpu_sum": 124.0,
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
    assert payload["cotenant_awareness"]["mode"] == "pressure_aware_cotenant"
    assert payload["cotenant_awareness"]["status_adjusted"] is False
    assert payload["cotenant_awareness"]["memory_pressure_clear"] is False


def test_memory_efficiency_control_keeps_sql_writer_drain_friendly_when_backlog_is_active(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "resource_guard_latest.json",
        {
            "memory_pressure_state": "green",
            "memory_pressure_kind": "none",
            "memory_free_pct": 83.0,
            "swap_used_gb": 0.02,
            "co_running_apps_active": True,
            "co_running_class_count": 2,
            "co_running_classes": ["browser", "developer"],
            "co_running_session_level": "heavy_competition",
            "co_running_cpu_sum": 208.0,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "apple_silicon_profile_latest.json",
        {
            "applied_tier": "max_throughput",
            "env_overrides": {"SQL_LINK_SERVICE_INTERVAL_SECONDS": "120", "ASYNC_PIPELINE_WORKERS": "6"},
            "hardware": {"memory_gb": 64.0},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {
            "severity": "stable",
            "pressure_index": 0.08,
            "recommended_operating_mode": "maintenance_drain_window",
            "storage": {"backlog_drain_status": "drain_active"},
            "backpressure": {"core_pending_lines": 861, "total_pending_lines": 861},
        },
    )

    payload = src.build_payload(tmp_path, action="status", override_path=tmp_path / "config" / ".env.memory_efficiency_override")

    assert payload["recommended_profile"] == "constrained"
    assert payload["storage_snapshot"]["drain_friendly_sql_active"] is True
    assert payload["recommended_env_overrides"]["SQL_LINK_SERVICE_INTERVAL_SECONDS"] == "12"
    assert payload["recommended_env_overrides"]["SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS"] == "30"
    assert payload["recommended_env_overrides"]["SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS"] == "180"
    assert payload["recommended_env_overrides"]["SQL_LINK_SERVICE_HOT_BATCH_SIZE"] == "240000"


def test_memory_efficiency_control_uses_concentrated_sql_drain_contract(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "resource_guard_latest.json",
        {
            "memory_pressure_state": "green",
            "memory_pressure_kind": "none",
            "memory_free_pct": 83.0,
            "swap_used_gb": 0.02,
            "co_running_apps_active": True,
            "co_running_class_count": 2,
            "co_running_classes": ["browser", "developer"],
            "co_running_session_level": "heavy_competition",
            "co_running_cpu_sum": 208.0,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "apple_silicon_profile_latest.json",
        {
            "applied_tier": "max_throughput",
            "env_overrides": {"SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "25"},
            "hardware": {"memory_gb": 64.0},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {
            "severity": "stable",
            "pressure_index": 0.08,
            "recommended_operating_mode": "maintenance_drain_window",
            "storage": {"backlog_drain_status": "handoff_requested"},
            "backpressure": {"core_pending_lines": 33631, "total_pending_lines": 33651},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "backpressure_drainer_fleet_latest.json",
        {
            "active_drainer": {
                "name": "core_decision_drainer",
                "concentration": {
                    "total_pending_lines": 33623,
                    "top1_share": 0.544806,
                    "top3_share": 0.92871,
                    "concentrated": True,
                },
            },
            "service_request": {"env_overrides": {"SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN": "1"}},
        },
    )

    payload = src.build_payload(tmp_path, action="status", override_path=tmp_path / "config" / ".env.memory_efficiency_override")
    coordination = payload["storage_snapshot"]["sql_writer_coordination"]

    assert coordination["concentrated_core_drain"] is True
    assert coordination["recommended_merge_max_seconds_per_cycle"] == 90
    assert payload["recommended_env_overrides"]["SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN"] == "1"
    assert payload["recommended_env_overrides"]["SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS"] == "420"
    assert payload["recommended_env_overrides"]["SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"] == "90"
    assert payload["recommended_env_overrides"]["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE"] == "12000"


def test_memory_efficiency_control_massive_expansion_enables_sleeve_rollups(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "resource_guard_latest.json",
        {
            "memory_pressure_state": "green",
            "memory_pressure_kind": "none",
            "memory_free_pct": 34.0,
            "swap_used_gb": 2.0,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "apple_silicon_profile_latest.json",
        {
            "applied_tier": "max_throughput",
            "env_overrides": {"ASYNC_PIPELINE_WORKERS": "6"},
            "hardware": {"memory_gb": 64.0, "is_apple_silicon": True},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {"severity": "ready", "pressure_index": 0.1},
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

    payload = src.build_payload(tmp_path, action="status", override_path=tmp_path / "config" / ".env.memory_efficiency_override")

    assert payload["expansion_session"]["pressure_level"] == "massive"
    assert payload["recommended_env_overrides"]["SLEEVE_MASTER_ROLLUP_ENABLED"] == "1"
    assert payload["recommended_env_overrides"]["GRAND_MASTER_READS_SLEEVE_ROLLUPS"] == "1"
    assert payload["recommended_env_overrides"]["SPECIALIZED_SLEEVE_INTERVAL"] == "210"
    assert payload["recommended_env_overrides"]["SLEEVE_WORKERS_SPECIALIZED"] == "1"
    assert payload["recommended_env_overrides"]["QUANT_MODEL_MLX_COMPILE_ENABLED"] == "0"
