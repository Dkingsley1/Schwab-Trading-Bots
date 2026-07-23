from pathlib import Path

from core.runtime_layers import CheckpointStore, TelemetryEmitter


def test_runtime_layers_fall_back_when_governance_profile_dir_is_broken_symlink(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    governance = project_root / "governance"
    governance.mkdir(parents=True)
    profile_dir = governance / "shadow_example_equities"
    profile_dir.symlink_to(project_root / "missing_bot_logs_target")

    emitter = TelemetryEmitter(str(profile_dir / "runtime_telemetry.jsonl"))
    emitter.emit({"ok": True})

    expected = project_root / "local_fallback_storage" / "governance" / "shadow_example_equities" / "runtime_telemetry.jsonl"
    assert Path(emitter.path) == expected
    assert expected.read_text(encoding="utf-8").strip() == '{"ok": true}'


def test_checkpoint_store_falls_back_when_governance_profile_dir_is_broken_symlink(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    governance = project_root / "governance"
    governance.mkdir(parents=True)
    profile_dir = governance / "shadow_example_equities"
    profile_dir.symlink_to(project_root / "missing_bot_logs_target")

    checkpoint = CheckpointStore(str(profile_dir / "runtime_checkpoint.json"))
    checkpoint.save({"iteration": 7})

    expected = project_root / "local_fallback_storage" / "governance" / "shadow_example_equities" / "runtime_checkpoint.json"
    assert Path(checkpoint.path) == expected
    assert checkpoint.load() == {"iteration": 7}
