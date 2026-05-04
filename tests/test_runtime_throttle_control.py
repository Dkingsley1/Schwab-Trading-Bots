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


def test_runtime_throttle_apply_keeps_sql_writer_drain_friendly(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(
        health_root / "resource_guard_latest.json",
        {
            "memory_pressure_state": "red",
            "memory_pressure_kind": "throttled",
            "swap_used_gb": 24.0,
        },
    )
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "blocked"})
    _write_json(
        health_root / "live_runtime_separation_control_latest.json",
        {"release_contract": {"live_lane_should_be_read_only": True}},
    )
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "degraded",
            "recommended_operating_mode": "maintenance_drain_window",
            "pressure_index": 0.081,
            "severity": "stable",
            "storage": {"backlog_drain_status": "drain_active"},
            "backpressure": {"core_pending_lines": 586, "total_pending_lines": 586},
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 3.0, "five_minutes": 3.0, "fifteen_minutes": 3.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=0,
    )

    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")

    assert payload["throttle_profile"] == "protect_live"
    assert payload["storage_stabilization"]["drain_friendly_sql_required"] is True
    assert result["storage_drain_active"] is True
    assert "SQL_LINK_SERVICE_INTERVAL_SECONDS=12" in override
    assert "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS=30" in override
    assert "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS=180" in override
    assert "SQL_LINK_SERVICE_INTERVAL_SECONDS=180" not in override


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


def test_project_path_does_not_misclassify_maintenance_as_pycharm() -> None:
    row = src._classify_process(
        "/opt/homebrew/Cellar/python@3.12/3.12.12_2/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python "
        "/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/sql_hot_retention.py --vacuum"
    )

    assert row["category"] == "support_maintenance"
    assert row["priority_tier"] == "throttle_first"
    assert row["throttle_candidate"] is True


def test_pycharm_app_is_still_interactive_cotenant() -> None:
    row = src._classify_process("/Applications/PyCharm.app/Contents/MacOS/pycharm")

    assert row["category"] == "interactive_cotenant"
    assert row["priority_tier"] == "external_cotenant"
    assert row["throttle_candidate"] is False


def test_storage_control_plane_helpers_are_support_throttle_candidates() -> None:
    row = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/storage_resilience_control.py --fast --json"
    )

    assert row["category"] == "support_maintenance"
    assert row["priority_tier"] == "throttle_first"
    assert row["throttle_candidate"] is True

    coverage_row = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/snapshot_coverage_sentinel.py --json"
    )

    assert coverage_row["category"] == "support_maintenance"
    assert coverage_row["priority_tier"] == "throttle_first"
    assert coverage_row["throttle_candidate"] is True


def test_schwab_auth_supervisor_is_protected_not_support_throttle() -> None:
    row = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/schwab_auth_supervisor.py --apply --json"
    )

    assert row["category"] == "live_execution"
    assert row["priority_tier"] == "protected_if_live"
    assert row["throttle_candidate"] is False


def test_heavy_diagnostics_are_support_throttle_candidates() -> None:
    divergence = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/data_source_divergence_bot.py --json"
    )
    assert divergence["category"] == "support_maintenance"
    assert divergence["priority_tier"] == "throttle_first"
    assert divergence["throttle_candidate"] is True

    mlx_audit = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/mlx_runtime_audit.py --json"
    )
    assert mlx_audit["category"] == "support_maintenance"
    assert mlx_audit["priority_tier"] == "throttle_first"
    assert mlx_audit["throttle_candidate"] is True

    mlx_router = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/mlx_intelligence_router.py --json"
    )
    assert mlx_router["category"] == "support_maintenance"
    assert mlx_router["priority_tier"] == "throttle_first"
    assert mlx_router["throttle_candidate"] is True

    library_router = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/library_utilization_router.py --json"
    )
    assert library_router["category"] == "support_maintenance"
    assert library_router["priority_tier"] == "throttle_first"
    assert library_router["throttle_candidate"] is True


def test_report_and_cleanup_helpers_are_support_throttle_candidates() -> None:
    report = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/paper_performance_report.py --day 20260503 --json-only"
    )
    assert report["category"] == "support_maintenance"
    assert report["priority_tier"] == "throttle_first"
    assert report["throttle_candidate"] is True

    sqlite_maintenance = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/sqlite_performance_maintenance.py --checkpoint-only --json"
    )
    assert sqlite_maintenance["category"] == "support_maintenance"
    assert sqlite_maintenance["priority_tier"] == "throttle_first"
    assert sqlite_maintenance["throttle_candidate"] is True

    collector_contracts = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/collector_contracts.py --json"
    )
    assert collector_contracts["category"] == "support_maintenance"
    assert collector_contracts["priority_tier"] == "throttle_first"
    assert collector_contracts["throttle_candidate"] is True

    infra_autofix = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/infrastructure_autofix_bot.py --apply --json"
    )
    assert infra_autofix["category"] == "support_maintenance"
    assert infra_autofix["priority_tier"] == "throttle_first"
    assert infra_autofix["throttle_candidate"] is True

    coverage_gap = src._classify_process(
        "python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/coverage_gap_closer.py --apply-stage --json"
    )
    assert coverage_gap["category"] == "support_maintenance"
    assert coverage_gap["priority_tier"] == "throttle_first"
    assert coverage_gap["throttle_candidate"] is True


def test_full_force_paper_capacity_adds_buffered_runtime_overrides(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "shadow_only",
            "pressure_index": 0.08,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {"core_pending_lines": 1200, "total_pending_lines": 1200},
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": f"brain_refinery_v{i}_paper_capacity_test_bot",
                    "active": True,
                    "lifecycle_state": "data_collection_only" if i % 2 else "shadow_candidate",
                    "sleeve_profile": "options" if i % 7 == 0 else "intraday_aggressive",
                }
                for i in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 12,
            "load_averages": {"one_minute": 2.4, "five_minutes": 2.0, "fifteen_minutes": 1.8},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=0,
    )
    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))

    assert payload["paper_capacity_contract"]["full_force_stabilization_required"] is True
    assert payload["paper_capacity_contract"]["ready_for_700_bot_paper"] is True
    assert "PAPER_RUNTIME_CONTROL_REFRESH_SECONDS=240" in override
    assert "JSONL_BUFFER_MAX_ITEMS=240" in override
    assert "SQL_LINK_SERVICE_INTERVAL_SECONDS=12" in override
    assert result["collector_guard"]["full_force_paper_stabilization"] is True
    assert registry["sub_bots"][0]["paper_execution_queue_policy"] == "buffered_jsonl_batching"


def test_runtime_throttle_preserves_lower_capability_pack_sampling(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "yellow", "swap_used_gb": 9.0})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "degraded"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": True}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.08,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {"core_pending_lines": 1200, "total_pending_lines": 1200},
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v849_coordination_lineage_genome_mapper_bot",
                    "active": True,
                    "lifecycle_state": "data_collection_only",
                    "data_collection_sample_rate": 0.9,
                    "data_collection_max_daily_mb": 150,
                    "freshness_slo_seconds": 60,
                    "capability_pack_contract": {
                        "storage_retention_rule": {
                            "sample_rate": 0.18,
                            "max_daily_mb_per_bot": 28,
                        }
                    },
                }
            ]
        },
    )
    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 8.0, "five_minutes": 7.0, "fifteen_minutes": 6.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {"interactive_cotenant": 40.0},
            "category_counts": {"interactive_cotenant": 1},
        },
    )
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=0,
    )
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    row = registry["sub_bots"][0]

    assert result["collector_guard"]["policy"]["sample_rate"] == 0.3
    assert row["data_collection_sample_rate"] == 0.18
    assert row["data_collection_max_daily_mb"] == 28


def test_runtime_throttle_consumes_memory_cotenant_awareness(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(
        health_root / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "cotenant_awareness": {
                "active": True,
                "mode": "managed_cotenant",
                "co_running_level": "interactive",
                "creative_level": "none",
                "open_app_count": 2,
                "open_apps": ["PyCharm", "Final Cut Pro"],
                "co_running_classes": ["developer", "creative"],
                "memory_pressure_clear": True,
                "storage_pressure_clear": True,
            },
        },
    )
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.01,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {"core_pending_lines": 0, "total_pending_lines": 0},
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 12,
            "load_averages": {"one_minute": 1.2, "five_minutes": 1.0, "fifteen_minutes": 1.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=0,
    )
    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")

    assert payload["overall_status"] == "advisory"
    assert payload["throttle_profile"] == "soft_cap"
    assert payload["cotenant_awareness_contract"]["profile_adjusted"] is True
    assert "RUNTIME_COTENANT_AWARE=1" in override
    assert result["env_override_count"] >= 5


def test_runtime_throttle_consumes_mlx_intelligence_router_caps(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.01,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {"core_pending_lines": 0, "total_pending_lines": 0},
        },
    )
    _write_json(
        health_root / "mlx_intelligence_router_latest.json",
        {
            "overall_status": "advisory",
            "library_coverage": {"coverage_ratio": 1.0},
            "route_coverage": {"route_coverage_ratio": 1.0},
            "runtime_caps": {
                "profile": "foreground_safe",
                "max_concurrent_mlx_jobs": 2,
                "tensor_batch_cap": 48,
                "embedding_batch_cap": 96,
                "graph_node_cap": 9000,
                "audio_minutes_per_job_cap": 30,
                "heavy_vlm_enabled": True,
                "compile_mode": "canary_first",
            },
            "recommended_runtime_env": {
                "MLX_INTELLIGENCE_ROUTER_ENABLED": "1",
                "MLX_INTELLIGENCE_PROFILE": "foreground_safe",
                "MLX_INTELLIGENCE_MAX_CONCURRENT_JOBS": "2",
                "MLX_INTELLIGENCE_TENSOR_BATCH_CAP": "48",
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 12,
            "load_averages": {"one_minute": 1.2, "five_minutes": 1.0, "fifteen_minutes": 1.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=0,
    )
    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")

    assert payload["mlx_intelligence_contract"]["active"] is True
    assert payload["mlx_intelligence_contract"]["library_coverage_ratio"] == 1.0
    assert "MLX_INTELLIGENCE_ROUTER_ENABLED=1" in override
    assert "MLX_INTELLIGENCE_TENSOR_BATCH_CAP=48" in override
    assert result["env_override_count"] >= 4


def test_runtime_throttle_consumes_library_utilization_router_caps_and_keeps_mlx_default(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "resource_guard_latest.json", {"memory_pressure_state": "green", "swap_used_gb": 0.1})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"release_contract": {"live_lane_should_be_read_only": False}})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_operating_mode": "live_full",
            "pressure_index": 0.01,
            "severity": "stable",
            "storage": {"backlog_drain_status": "steady_state"},
            "backpressure": {"core_pending_lines": 0, "total_pending_lines": 0},
        },
    )
    _write_json(
        health_root / "library_utilization_router_latest.json",
        {
            "overall_status": "advisory",
            "coverage": {
                "coverage_ratio": 1.0,
                "locked_runtime_ok_ratio": 1.0,
                "managed_non_mlx_package_count": 80,
            },
            "runtime_caps": {
                "profile": "foreground_safe",
                "max_async_request_concurrency": 8,
                "max_sql_writer_workers": 2,
                "max_dataframe_workers": 2,
                "max_portable_model_replay_jobs": 0,
                "max_report_render_jobs": 1,
            },
            "recommended_runtime_env": {
                "LIBRARY_UTILIZATION_ROUTER_ENABLED": "1",
                "LIBRARY_UTILIZATION_PROFILE": "foreground_safe",
                "PRIMARY_ML_RUNTIME_BACKEND": "mlx",
                "LIBRARY_DEFAULT_ML_BACKEND": "mlx",
                "PORTABLE_MODEL_REPLAY_POLICY": "canary_or_off_hours_only",
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        runtime_snapshot={
            "cpu_count": 12,
            "load_averages": {"one_minute": 1.2, "five_minutes": 1.0, "fifteen_minutes": 1.0},
            "thermal": {"thermal_warning_active": False, "performance_warning_active": False},
            "vm_stat": {},
            "top_processes": [],
            "category_cpu": {},
            "category_counts": {},
        },
    )
    result = src.apply_runtime_guard(
        tmp_path,
        payload,
        override_path=tmp_path / "config" / ".env.runtime_resource_guard_override",
        registry_path=tmp_path / "master_bot_registry.json",
        max_renice_processes=0,
    )
    override = (tmp_path / "config" / ".env.runtime_resource_guard_override").read_text(encoding="utf-8")

    assert payload["library_utilization_contract"]["active"] is True
    assert payload["library_utilization_contract"]["default_ml_backend"] == "mlx"
    assert "LIBRARY_UTILIZATION_ROUTER_ENABLED=1" in override
    assert "PRIMARY_ML_RUNTIME_BACKEND=mlx" in override
    assert "PORTABLE_MODEL_REPLAY_POLICY=canary_or_off_hours_only" in override
    assert result["env_override_count"] >= 5
