import json
from pathlib import Path

from scripts.ops import training_drain_autopilot as src


def test_training_drain_autopilot_finalizes_timed_out_retrain_launch(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    artifact = project_root / "governance" / "training_diagnostics" / "retrain_launches" / "launch.json"
    latest = health / "retrain_launch_latest.json"
    alias = health / "retrain_launch_latest_opsctl.json"
    payload = {
        "state": "running",
        "pid": 0,
        "phase": "memory_gate",
        "progress": {"bot_id": "brain_refinery_v10_seasonal"},
        "artifact_path": str(artifact),
        "latest_path": str(latest),
        "latest_alias_paths": [str(latest), str(alias)],
    }
    for path in (artifact, latest, alias):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    result = src._finalize_timed_out_retrain_launch(
        project_root,
        {"timed_out": True, "returncode": 124, "stdout_tail": "out", "stderr_tail": "err"},
    )

    assert result["status"] == "finalized_timeout"
    updated = json.loads(latest.read_text(encoding="utf-8"))
    assert updated["state"] == "completed"
    assert updated["final_status"] == "timed_out_by_training_drain_autopilot"
    assert updated["exit_code"] == 124
    assert updated["timeout_phase"] == "memory_gate"
    assert updated["timeout_progress"]["bot_id"] == "brain_refinery_v10_seasonal"
    assert json.loads(artifact.read_text(encoding="utf-8"))["final_status"] == "timed_out_by_training_drain_autopilot"
    assert json.loads(alias.read_text(encoding="utf-8"))["final_status"] == "timed_out_by_training_drain_autopilot"
