import argparse
import json
import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PROJECT_ROOT / "core"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.weekly_retrain as retrain


def test_lifecycle_hygiene_mutations_require_committed_master_update() -> None:
    assert not retrain._lifecycle_hygiene_mutations_allowed(
        skip_master_update=True,
        master_update_status="updated",
    )
    assert not retrain._lifecycle_hygiene_mutations_allowed(
        skip_master_update=False,
        master_update_status="skipped_by_flag",
    )
    assert not retrain._lifecycle_hygiene_mutations_allowed(
        skip_master_update=False,
        master_update_status="precheck_failed",
    )
    assert not retrain._lifecycle_hygiene_mutations_allowed(
        skip_master_update=False,
        master_update_status="rolled_back",
    )
    assert retrain._lifecycle_hygiene_mutations_allowed(
        skip_master_update=False,
        master_update_status="updated",
    )
    assert retrain._lifecycle_hygiene_mutations_allowed(
        skip_master_update=False,
        master_update_status="updated_precheck_override",
    )


def test_trade_behavior_holdout_is_not_a_trainer_failure() -> None:
    assert retrain._trade_behavior_trainer_outcome(0) == ("promoted", True, False)
    assert retrain._trade_behavior_trainer_outcome(4) == ("held_out", False, False)
    assert retrain._trade_behavior_trainer_outcome(2) == ("failed", False, True)


def test_weekly_retrain_child_env_pins_project_root_on_pythonpath(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/tmp/existing_path")

    env = retrain._build_child_env(2)

    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(PROJECT_ROOT)
    assert "/tmp/existing_path" in parts


def test_v43_script_bootstrap_adds_project_root_for_direct_execution(monkeypatch) -> None:
    filtered = [entry for entry in sys.path if str(entry).strip() != str(PROJECT_ROOT)]
    monkeypatch.setattr(sys, "path", [str(CORE_ROOT)] + filtered)

    module_globals = runpy.run_path(str(CORE_ROOT / "brain_refinery_v43_intraday_ultrafast_proxy.py"), run_name="not_main")

    assert "train_brain" in module_globals
    assert str(PROJECT_ROOT) in sys.path


def test_retrain_launch_record_prefers_explicit_trigger_source(monkeypatch) -> None:
    monkeypatch.setenv("RETRAIN_TRIGGER_SOURCE", "shadow_training_loop_auto_retrain")
    monkeypatch.setenv("RETRAIN_TRIGGER_BROKER", "schwab")
    monkeypatch.setenv("RETRAIN_TRIGGER_PROFILE", "aggressive")
    monkeypatch.setenv("RETRAIN_LAUNCH_LOG_PATH", "/tmp/retrain_launch.log")
    monkeypatch.setenv("CORRELATION_RUN_ID", "run-1")
    monkeypatch.setenv("CORRELATION_ITER_ID", "run-1:7")
    monkeypatch.setattr(retrain, "_safe_parent_command", lambda pid: "python scripts/run_shadow_training_loop.py")

    args = argparse.Namespace(
        include_bot_ids="brain_refinery_v43_intraday_ultrafast_proxy",
        exclude_bot_ids="",
        regime_focus="",
        active_only=False,
        max_targets=12,
        min_model_age_hours=18.0,
        skip_master_update=True,
        continue_on_error=True,
        dry_run=False,
    )

    payload = retrain._build_retrain_launch_record(args, "fast_daytime")

    assert payload["source"] == "shadow_training_loop_auto_retrain"
    assert payload["source_broker"] == "schwab"
    assert payload["source_profile"] == "aggressive"
    assert payload["launch_log_path"] == "/tmp/retrain_launch.log"
    assert payload["run_mode"] == "targeted"
    assert payload["selector_summary"]["include_bot_ids"] == ["brain_refinery_v43_intraday_ultrafast_proxy"]
    assert payload["correlation_run_id"] == "run-1"


def test_persist_retrain_launch_record_writes_source_latest_alias(tmp_path: Path, monkeypatch) -> None:
    artifact_dir = tmp_path / "artifacts"
    latest_path = tmp_path / "health" / "retrain_launch_latest.json"
    source_latest_path = tmp_path / "health" / "retrain_launch_latest_coverage_gap_closer.json"

    monkeypatch.setattr(retrain, "_retrain_launch_artifact_dir", lambda: str(artifact_dir))
    monkeypatch.setattr(retrain, "_retrain_launch_latest_path", lambda *, dry_run: str(latest_path))
    monkeypatch.setattr(
        retrain,
        "_retrain_launch_source_latest_path",
        lambda *, dry_run, source: str(source_latest_path),
    )

    payload = retrain._persist_retrain_launch_record(
        {
            "launch_slug": "20260416_000000",
            "pid": 123,
            "source": "coverage_gap_closer",
        },
        dry_run=False,
    )

    assert latest_path.exists()
    assert source_latest_path.exists()
    assert payload["source_latest_path"] == str(source_latest_path)
    assert str(source_latest_path) in payload["latest_alias_paths"]


def test_configured_runtime_snapshot_summary_prefers_explicit_snapshot(tmp_path: Path) -> None:
    explicit_path = tmp_path / "explicit_snapshot.json"
    fallback_path = tmp_path / "fallback_snapshot.json"
    explicit_path.write_text('{"health_path": "explicit"}', encoding="utf-8")
    fallback_path.write_text('{"health_path": "fallback"}', encoding="utf-8")

    path, summary = retrain._configured_runtime_snapshot_summary(
        {
            "RUNTIME_TRAIN_SNAPSHOT_FILE": str(explicit_path),
            "RETRAIN_COVERAGE_CANARY_SNAPSHOT_FILE": str(fallback_path),
        }
    )

    assert path == str(explicit_path)
    assert summary["health_path"] == "explicit"


def test_configured_runtime_snapshot_summary_falls_back_to_coverage_canary_snapshot(tmp_path: Path) -> None:
    fallback_path = tmp_path / "fallback_snapshot.json"
    fallback_path.write_text('{"health_path": "fallback"}', encoding="utf-8")

    path, summary = retrain._configured_runtime_snapshot_summary(
        {
            "RUNTIME_TRAIN_SNAPSHOT_FILE": "",
            "RETRAIN_COVERAGE_CANARY_SNAPSHOT_FILE": str(fallback_path),
        }
    )

    assert path == str(fallback_path)
    assert summary["health_path"] == "fallback"


def test_held_out_training_refreshes_walk_forward_evidence(monkeypatch, tmp_path: Path) -> None:
    validator = tmp_path / "walk_forward_validate.py"
    validator.write_text("", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(retrain, "WALK_FORWARD_VALIDATE_SCRIPT", str(validator))
    monkeypatch.setattr(retrain, "VENV_PY", "/runtime/python")
    monkeypatch.setattr(
        retrain,
        "run_cmd",
        lambda command, dry_run, env, extra_nice=0: calls.append(command) or 0,
    )

    result = retrain._refresh_held_out_walk_forward_evidence(
        target_outcomes=[{"bot_id": "brain_refinery_v58", "status": "trained"}],
        enabled=True,
        dry_run=False,
        env={},
        extra_nice=0,
    )

    assert result["status"] == "refreshed"
    assert result["trained_bot_ids"] == ["brain_refinery_v58"]
    assert calls == [["/runtime/python", str(validator)]]


def test_weekly_retrain_memory_gate_allows_green_advisory_swap_relief(monkeypatch, tmp_path: Path) -> None:
    resource_guard = tmp_path / "resource_guard_latest.json"
    resource_guard.write_text(
        json.dumps(
            {
                "resource_guard_ok": False,
                "memory_pressure_state": "green",
                "resource_guard_reasons": ["support_maintenance_frozen_for_mac_fluidity"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(retrain, "RESOURCE_GUARD_LATEST", str(resource_guard))
    monkeypatch.setenv("RETRAIN_GREEN_MEMORY_SWAP_RELIEF", "1")
    monkeypatch.setattr(
        retrain,
        "_memory_snapshot",
        lambda: {"free_pct": 22.0, "available_pct": 40.0, "swap_used_gb": 8.0},
    )

    ok, reason, snapshot = retrain._memory_ready(min_free_pct=18.0, max_swap_gb=2.5)

    assert ok is True
    assert "swap_relaxed_by_resource_guard" in reason
    assert snapshot["swap_relief_by_resource_guard"] == 1.0


def test_coverage_micro_profile_caps_memory_and_ops_waits(monkeypatch) -> None:
    monkeypatch.setattr(retrain.os.path, "exists", lambda path: True if path == retrain.RUNTIME_TRAINING_SNAPSHOT_LATEST else False)
    args = argparse.Namespace(
        retrain_profile="coverage_micro_canary",
        counterfactual_replay=True,
        paper_hard_example_pack=True,
        require_sample_quotas=True,
        new_bot_boost=True,
        build_runtime_training_snapshot=True,
        runtime_training_snapshot_prefer_sqlite=True,
        runtime_train_use_snapshot=False,
        runtime_train_prefer_sqlite=False,
        runtime_train_fast_fail_zero_sample_attempts=0,
        target_timeout_seconds=0,
        memory_max_wait_seconds=1800,
        ops_timeout_seconds=900,
        cold_lane_retrain_extras=True,
        auto_insufficient_data_retry=True,
    )

    profile = retrain._apply_retrain_profile_defaults(args)

    assert profile == "coverage_micro_canary"
    assert args.target_timeout_seconds == 600
    assert args.memory_max_wait_seconds == 120
    assert args.ops_timeout_seconds == 120
    assert args.cold_lane_retrain_extras is False
