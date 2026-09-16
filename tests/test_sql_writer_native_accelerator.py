import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("scenario", ["clear", "held", "recovered", "operator_hold"])
def test_native_accelerator_planning_precedes_storage_exit_and_replans_after_relief(
    tmp_path, scenario
):
    zsh = shutil.which("zsh")
    if not zsh:
        pytest.skip("native launcher requires zsh")
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/ops/run_sql_link_writer_launchd.sh"
    )
    ops = tmp_path / "scripts/ops"
    ops.mkdir(parents=True)
    launcher = ops / source.name
    shutil.copyfile(source, launcher)
    python = tmp_path / ".venv314/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    (tmp_path / "pause").write_text("0" if scenario == "clear" else "1")
    (ops / "load_runtime_env.sh").write_text(
        'export SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE="$(<"$PROJECT_ROOT/pause")"\n'
        "export SQL_LINK_SERVICE_SHARDS=trading\n"
    )
    (ops / "runtime_maintenance_hold.py").write_text(
        f"print({json.dumps({'active': scenario == 'operator_hold'})!r})\n"
    )
    prelude = (
        "import json,sys\nfrom pathlib import Path\n"
        "root=Path(__file__).resolve().parents[2]\n"
        "def record(name):\n"
        "    with (root/'events').open('a') as handle:\n"
        "        handle.write(json.dumps({'name':name,'args':sys.argv[1:]})+'\\n')\n"
    )
    (ops / "backpressure_drainer_fleet.py").write_text(prelude + "record('plan')\n")
    (ops / "soak_self_healing_control.py").write_text(
        prelude
        + "record('recovery')\n"
        + ("(root/'pause').write_text('0')\n" if scenario == "recovered" else "")
    )
    (ops / "sql_link_shard_manager.py").write_text(prelude + "record('writer')\n")
    guard = ops / "run_guarded_maintenance.sh"
    guard.write_text('#!/bin/zsh\nshift\nexec "$@"\n')
    guard.chmod(0o755)
    completed = subprocess.run(
        [zsh, str(launcher)], cwd=tmp_path, capture_output=True, text=True, timeout=10
    )
    assert completed.returncode == 0, completed.stderr
    events_path = tmp_path / "events"
    events = (
        [json.loads(line) for line in events_path.read_text().splitlines()]
        if events_path.exists()
        else []
    )
    assert [row["name"] for row in events] == {
        "clear": ["plan", "writer"],
        "held": ["plan", "recovery"],
        "recovered": ["plan", "recovery", "plan", "writer"],
        "operator_hold": [],
    }[scenario]
    for row in events:
        if row["name"] == "plan":
            assert row["args"] == [
                "--apply",
                "--refresh-backlog",
                "--refresh-accelerator",
                "--ttl-seconds",
                "120",
                "--json",
            ]
        elif row["name"] == "writer":
            assert row["args"] == ["--once", "--scheduled-drain"]
