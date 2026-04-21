import argparse
import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PROJECT_ROOT / "core"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.weekly_retrain as retrain


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
