import sys
import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import ops_scheduled_job_catalog as catalog  # noqa: E402


def test_ops_launchd_installer_wraps_bounded_scheduled_jobs() -> None:
    installer = PROJECT_ROOT / "scripts" / "ops" / "install_ops_automation_launchd.sh"
    text = installer.read_text(encoding="utf-8")
    expected_wrapped = [
        spec.job_id
        for spec in catalog.DEFAULT_JOB_SPECS
        if spec.install_policy == catalog.ACTIVE_INSTALL_POLICY
        and (spec.resource_class != "single_writer" or spec.deadline_seconds > 0)
    ]

    assert "SCHEDULED_LIFECYCLE_RUNNER" in text
    assert "run_scheduled_lifecycle_job.py" in text
    for job_id in expected_wrapped:
        assert f"scheduled_program_arguments {job_id} " in text
    assert "sql_link_writer" in expected_wrapped


@pytest.mark.parametrize("shards", ["", "3"])
@pytest.mark.parametrize("hold", ["", "storage", "maintenance"])
def test_native_sql_writer_is_bounded_and_respects_holds(tmp_path, shards, hold):
    scripts = tmp_path / "scripts/ops"
    scripts.mkdir(parents=True)
    python = tmp_path / ".venv314/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    runner = scripts / "run_sql_link_writer_launchd.sh"
    shutil.copy2(PROJECT_ROOT / "scripts/ops" / runner.name, runner)
    (scripts / "runtime_maintenance_hold.py").write_text(
        'import json, os; print(json.dumps({"active": os.environ.get("TEST_HOLD") == "maintenance"}))'
    )
    (scripts / "load_runtime_env.sh").write_text(
        "# Keep the fixture's explicit environment.\n"
    )
    (scripts / "backpressure_drainer_fleet.py").write_text("pass\n")
    (scripts / "soak_self_healing_control.py").write_text("pass\n")
    guarded = scripts / "run_guarded_maintenance.sh"
    guarded.write_text('#!/bin/zsh\nprintf "%s\\n" "$@"\n')
    guarded.chmod(0o755)
    env = dict(
        os.environ,
        SQL_LINK_SERVICE_SHARDS=shards,
        SQL_LINK_WRITER_ONCE="0",
        SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE="1" if hold == "storage" else "0",
        TEST_HOLD=hold,
    )
    result = subprocess.run(
        ["/bin/zsh", str(runner)], env=env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    if hold:
        assert "status=deferred" in result.stdout
        assert "--once" not in result.stdout
    else:
        assert "--once" in result.stdout.splitlines()
        assert ("--scheduled-drain" in result.stdout.splitlines()) == bool(shards)
        assert (
            "sql_link_shard_manager.py" if shards else "sql_link_writer_service.py"
        ) in result.stdout
