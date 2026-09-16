import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import run_scheduled_lifecycle_job as runner  # noqa: E402


def test_competing_risk_refresh_defers_without_counting_a_failure():
    reason = runner.infer_deferred_reason(
        stdout="one_numbers lock busy lock_path=one_numbers.lock owner=pid=123",
        stderr="", rc=1,
    )
    assert reason == "already_running"
    assert runner._terminal_status(rc=1, timed_out=False, artifact_present_after=True, deferred_reason=reason) == ("deferred", "", True)


def test_scheduler_uses_process_tree_cleanup_on_timeout(tmp_path, monkeypatch):
    calls = []

    def bounded(command, **kwargs):
        calls.append((command, kwargs))
        return {
            "rc": 124,
            "stdout": "",
            "stderr": "",
            "timed_out": True,
            "timeout_cleanup": {"reaped": True},
        }

    monkeypatch.setattr(runner, "run_bounded_process_group", bounded)
    result = runner.run_command(["test-job"], cwd=tmp_path, timeout_seconds=7)
    assert calls == [(["test-job"], {"cwd": tmp_path, "timeout_seconds": 7})]
    assert result["rc"] == 124
    assert result["timeout_cleanup"]["reaped"] is True


def test_scheduler_timeout_stops_descendant_work(tmp_path):
    marker = tmp_path / "child_completed"
    child_code = (
        "import time; from pathlib import Path; time.sleep(2); Path("
        + repr(str(marker))
        + ").touch()"
    )
    parent_code = (
        "import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', "
        + repr(child_code)
        + "]); time.sleep(30)"
    )
    result = runner.run_command(
        [sys.executable, "-c", parent_code], cwd=tmp_path, timeout_seconds=1
    )
    assert result["timed_out"] is True
    assert result["timeout_cleanup"]["reaped"] is True
    time.sleep(1.5)
    assert not marker.exists()


def test_scheduled_lifecycle_runner_stamps_existing_artifact_without_touching_producer_time(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    artifact = project_root / "governance" / "health" / "job_latest.json"
    producer_timestamp = "2026-09-09T00:00:00+00:00"
    command = [
        sys.executable,
        "-c",
        (
            "import json,sys; "
            "from pathlib import Path; "
            "path=Path(sys.argv[1]); "
            "path.parent.mkdir(parents=True, exist_ok=True); "
            "path.write_text(json.dumps({'timestamp_utc':'2026-09-09T00:00:00+00:00','ok':True,'overall_status':'ready'}), encoding='utf-8')"
        ),
        str(artifact),
    ]

    payload, rc = runner.build_payload(
        project_root=project_root,
        job_id="test_job",
        artifact=artifact,
        schedule_interval_seconds=300,
        deadline_seconds=30,
        command=command,
    )

    assert rc == 0
    assert payload["timestamp_utc"] == producer_timestamp
    assert payload["overall_status"] == "ready"
    lifecycle = payload["job_lifecycle"]
    assert lifecycle["job_id"] == "test_job"
    assert lifecycle["scheduled"] is True
    assert lifecycle["completed"] is True
    assert lifecycle["failed"] is False
    assert lifecycle["artifact_present_before"] is False
    assert lifecycle["artifact_present_after"] is True
    assert lifecycle["next_eligible_utc"]


def test_scheduled_lifecycle_runner_fails_closed_when_artifact_is_missing(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    artifact = project_root / "governance" / "health" / "missing_latest.json"

    payload, rc = runner.build_payload(
        project_root=project_root,
        job_id="missing_job",
        artifact=artifact,
        schedule_interval_seconds=120,
        deadline_seconds=30,
        command=[sys.executable, "-c", "pass"],
    )

    assert rc == 2
    assert artifact.exists()
    assert payload["ok"] is False
    assert payload["overall_status"] == "artifact_missing_after_run"
    lifecycle = payload["job_lifecycle"]
    assert lifecycle["failed"] is True
    assert lifecycle["failure_reason"] == "artifact_missing_after_run"
    assert lifecycle["artifact_present_after"] is False


def test_scheduled_lifecycle_runner_accepts_report_findings_exit_code(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    artifact = project_root / "governance" / "health" / "findings_latest.json"
    command = [
        sys.executable,
        "-c",
        (
            "import json,sys; "
            "from pathlib import Path; "
            "path=Path(sys.argv[1]); "
            "path.parent.mkdir(parents=True, exist_ok=True); "
            "path.write_text(json.dumps({'timestamp_utc':'2026-09-09T00:00:00+00:00','ok':False,'overall_status':'blocked'}), encoding='utf-8'); "
            "raise SystemExit(2)"
        ),
        str(artifact),
    ]

    payload, rc = runner.build_payload(
        project_root=project_root,
        job_id="findings_job",
        artifact=artifact,
        schedule_interval_seconds=300,
        deadline_seconds=30,
        command=command,
    )

    assert rc == 0
    assert payload["overall_status"] == "blocked"
    lifecycle = payload["job_lifecycle"]
    assert lifecycle["failed"] is False
    assert lifecycle["terminal_status"] == "completed_with_findings"
    assert lifecycle["failure_reason"] == ""
    assert lifecycle["rc"] == 2


def test_scheduled_lifecycle_runner_marks_writer_busy_as_deferred(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    artifact = project_root / "governance" / "health" / "writer_latest.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-09-09T00:00:00+00:00",
                "ok": True,
                "overall_status": "ready",
            }
        ),
        encoding="utf-8",
    )

    payload, rc = runner.build_payload(
        project_root=project_root,
        job_id="sql_link_writer",
        artifact=artifact,
        schedule_interval_seconds=45,
        deadline_seconds=30,
        command=[
            sys.executable,
            "-c",
            "print('sql_link_shard_manager busy owner=pid=123 cmd=sql_link_shard_manager')",
        ],
    )

    assert rc == 0
    lifecycle = payload["job_lifecycle"]
    assert lifecycle["failed"] is False
    assert lifecycle["deferred"] is True
    assert lifecycle["deferred_reason"] == "writer_lock_busy"
    assert lifecycle["terminal_status"] == "deferred"
