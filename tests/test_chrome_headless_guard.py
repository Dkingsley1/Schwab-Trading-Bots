import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import chrome_headless_guard as src


def test_chrome_headless_guard_suppresses_stale_orphan_helpers(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    rows = [
        {"pid": 101, "ppid": 1, "elapsed": "00:20:00", "elapsed_seconds": 1200, "command": "Google Chrome Helper --headless --type=renderer"},
        {"pid": 202, "ppid": 55, "elapsed": "00:00:30", "elapsed_seconds": 30, "command": "Google Chrome"},
    ]

    payload = src.build_payload(project_root, process_rows=rows, apply=False)

    assert payload["overall_status"] == "blocked"
    assert payload["timeline_pdf_policy"] == "suppress"
    assert payload["interactive_protection_active"] is True
    assert payload["timeline_autorender_suppressed"] is True
    assert payload["upgrade_track"]["current_generation"] == "chrome_headless_guard_v3"


def test_chrome_headless_guard_prefers_headless_only_when_foreground_chrome_is_active(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    lock_path = project_root / "governance" / "locks" / "project_timeline_report.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("lock", encoding="utf-8")
    rows = [
        {"pid": 101, "ppid": 90, "elapsed": "00:00:20", "elapsed_seconds": 20, "command": "Google Chrome Helper --headless --type=renderer"},
        {"pid": 202, "ppid": 55, "elapsed": "00:00:30", "elapsed_seconds": 30, "command": "Google Chrome"},
    ]

    payload = src.build_payload(project_root, process_rows=rows, apply=False)

    assert payload["overall_status"] == "degraded"
    assert payload["timeline_pdf_policy"] == "headless_only"
    assert payload["policy_reason"] == "interactive_chrome_protected"


def test_chrome_headless_guard_quiet_mode_suppresses_and_cleans_timeline_runner(tmp_path: Path, monkeypatch) -> None:
    killed: list[int] = []
    project_root = tmp_path / "project"
    rows = [
        {
            "pid": 90,
            "ppid": 1,
            "elapsed": "00:00:20",
            "elapsed_seconds": 20,
            "command": "python scripts/ops/project_timeline_report.py --auto --json",
        },
        {
            "pid": 101,
            "ppid": 90,
            "elapsed": "00:00:10",
            "elapsed_seconds": 10,
            "command": "Google Chrome --headless --user-data-dir=/tmp/project-timeline-pdf-a",
        },
    ]
    monkeypatch.setenv("CHROME_HEADLESS_QUIET_MODE", "1")

    payload = src.build_payload(
        project_root,
        process_rows=rows,
        apply=True,
        kill_runner=lambda pid: not killed.append(pid),
    )

    assert payload["overall_status"] == "ready"
    assert payload["quiet_mode_active"] is True
    assert payload["timeline_pdf_policy"] == "suppress"
    assert payload["policy_reason"] == "quiet_mode_suppressed"
    assert payload["kill_candidates"] == [101, 90]
    assert killed == [101, 90]


def test_chrome_headless_guard_counts_temp_profile_helpers_as_headless_not_interactive(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    lock_path = project_root / "governance" / "locks" / "project_timeline_report.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("lock", encoding="utf-8")
    rows = [
        {"pid": 55, "ppid": 1, "elapsed": "00:03:20", "elapsed_seconds": 200, "command": "python render_owner.py"},
        {
            "pid": 101,
            "ppid": 55,
            "elapsed": "00:03:20",
            "elapsed_seconds": 200,
            "command": "Google Chrome --user-data-dir=/tmp/report-bundle-pdf-open-a --print-to-pdf=out.pdf",
        },
        {
            "pid": 102,
            "ppid": 101,
            "elapsed": "00:03:19",
            "elapsed_seconds": 199,
            "command": "Google Chrome Helper --type=utility --user-data-dir=/tmp/report-bundle-pdf-open-a",
        },
        {"pid": 202, "ppid": 55, "elapsed": "00:00:30", "elapsed_seconds": 30, "command": "Google Chrome"},
    ]

    payload = src.build_payload(
        project_root,
        process_rows=rows,
        apply=False,
        max_headless_count=1,
        runaway_headless_age_seconds=30,
    )

    assert payload["overall_status"] == "blocked"
    assert payload["headless_process_count"] == 2
    assert payload["temp_headless_profile_count"] == 2
    assert payload["interactive_chrome_process_count"] == 1
    assert payload["timeline_pdf_policy"] == "suppress"
    assert payload["kill_candidates"] == [101, 102]


def test_chrome_headless_guard_applies_runaway_cleanup_with_recent_lock(tmp_path: Path) -> None:
    killed: list[int] = []
    project_root = tmp_path / "project"
    lock_path = project_root / "governance" / "locks" / "project_timeline_report.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("lock", encoding="utf-8")
    rows = [
        {"pid": 55, "ppid": 1, "elapsed": "00:03:20", "elapsed_seconds": 200, "command": "python render_owner.py"},
        {
            "pid": 101,
            "ppid": 55,
            "elapsed": "00:03:20",
            "elapsed_seconds": 200,
            "command": "Google Chrome --headless --user-data-dir=/tmp/report-bundle-pdf-open-a",
        },
        {
            "pid": 102,
            "ppid": 101,
            "elapsed": "00:03:19",
            "elapsed_seconds": 199,
            "command": "Google Chrome Helper --type=utility --user-data-dir=/tmp/report-bundle-pdf-open-a",
        },
    ]

    payload = src.build_payload(
        project_root,
        process_rows=rows,
        apply=True,
        max_headless_count=1,
        runaway_headless_age_seconds=30,
        kill_runner=lambda pid: not killed.append(pid),
    )

    assert payload["runaway_detected"] is True
    assert payload["killed_pid_count"] == 2
    assert killed == [101, 102]


def test_chrome_headless_guard_cleans_runaway_timeline_parent(tmp_path: Path) -> None:
    killed: list[int] = []
    project_root = tmp_path / "project"
    lock_path = project_root / "governance" / "locks" / "project_timeline_report.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("lock", encoding="utf-8")
    rows = [
        {
            "pid": 90,
            "ppid": 1,
            "elapsed": "00:01:00",
            "elapsed_seconds": 60,
            "command": "python scripts/ops/project_timeline_report.py --auto --json",
        },
        {
            "pid": 101,
            "ppid": 90,
            "elapsed": "00:01:00",
            "elapsed_seconds": 60,
            "command": "Google Chrome --headless --user-data-dir=/tmp/project-timeline-pdf-a",
        },
        {
            "pid": 102,
            "ppid": 101,
            "elapsed": "00:00:59",
            "elapsed_seconds": 59,
            "command": "Google Chrome Helper --type=utility --user-data-dir=/tmp/project-timeline-pdf-a",
        },
    ]

    payload = src.build_payload(
        project_root,
        process_rows=rows,
        apply=True,
        max_headless_count=1,
        runaway_headless_age_seconds=30,
        kill_runner=lambda pid: not killed.append(pid),
    )

    assert payload["timeline_cleanup_count"] == 1
    assert payload["kill_candidates"] == [101, 102, 90]
    assert killed == [101, 102, 90]
