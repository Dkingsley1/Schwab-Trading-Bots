import json
import plistlib

from scripts.ops import system_power as power


def fixture_agents(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    agents = tmp_path / "agents"
    agents.mkdir()
    for label, project in (
        ("owned.test", root),
        ("unrelated.test", tmp_path / "other"),
    ):
        with (agents / f"{label}.plist").open("wb") as handle:
            plistlib.dump(
                {
                    "Label": label,
                    "ProgramArguments": ["python", str(project / "scripts/job.py")],
                },
                handle,
            )
    return root, agents


def test_off_is_persistent_and_only_owns_repo_agents(tmp_path):
    root, agents = fixture_agents(tmp_path)
    loaded = {"owned.test", "unrelated.test"}
    calls = []

    def runner(root, argv, timeout):
        calls.append(argv)
        if argv[:2] == ["launchctl", "print"]:
            return {"rc": 0 if argv[2].split("/")[-1] in loaded else 1}
        if argv[:2] == ["launchctl", "bootout"]:
            loaded.remove(argv[2].split("/")[-1])
        return {"rc": 0}

    result = power.run(root, "off", agent_dir=agents, runner=runner)
    assert result["ok"] and result["requested_state"] == "off"
    assert loaded == {"unrelated.test"}
    assert power.load(root)["agents"][0]["label"] == "owned.test"
    assert all("unrelated.test" not in " ".join(call) for call in calls)
    power.run(root, "off", agent_dir=agents, runner=runner)
    assert len(power.load(root)["agents"]) == 1
    assert not power.run(root, "clear-halts", runner=runner)["ok"]


def test_on_cannot_bypass_failed_safe_clear(tmp_path):
    root, agents = fixture_agents(tmp_path)
    power.write(root, power.OFF, {"reason": "off"})
    calls = []

    def runner(root, argv, timeout):
        calls.append(argv)
        return {"rc": 2 if "global-halt-auto-clear" in argv else 0}

    result = power.run(root, "on", agent_dir=agents, runner=runner)
    assert not result["ok"]
    assert power.local(root, power.OFF).exists()
    assert not any("start" in call for call in calls)


def test_on_restores_recorded_agents_only_and_never_live(tmp_path):
    root, agents = fixture_agents(tmp_path)
    power.write(root, power.OFF, {"reason": "off"})
    power.write(
        root,
        power.STATE,
        {
            "agents": [
                {"label": "owned.test", "plist": str(agents / "owned.test.plist")}
            ]
        },
    )
    calls = []

    def runner(root, argv, timeout):
        calls.append(argv)
        return {"rc": 1 if argv[:2] == ["launchctl", "print"] else 0}

    result = power.run(root, "on", agent_dir=agents, runner=runner)
    assert result["ok"] and result["live_execution_authority"] is False
    assert [str(root / "scripts/ops/opsctl.sh"), "start", "--paper"] in calls
    assert not power.local(root, power.OFF).exists()


def test_off_failure_does_not_claim_success(tmp_path):
    root, agents = fixture_agents(tmp_path)
    result = power.run(root, "off", agent_dir=agents, runner=lambda *a: {"rc": 0})
    assert not result["ok"] and result["last_transition"] == "off_incomplete"


def test_status_does_not_invoke_commands(tmp_path):
    result = power.run(
        tmp_path, "status", runner=lambda *a: (_ for _ in ()).throw(AssertionError())
    )
    assert result["requested_state"] == "not_set"
    assert result["running_processes_verified"] is False


def test_failed_start_restores_off_and_operator_stop(tmp_path):
    root, agents = fixture_agents(tmp_path)
    calls = []

    def runner(root, argv, timeout):
        calls.append(argv)
        return {"rc": 2 if "start" in argv else 0}

    result = power.run(root, "on", agent_dir=agents, runner=runner)
    assert not result["ok"] and result["requested_state"] == "off"
    assert any("system_power_start_incomplete" in call for call in calls)


def test_failed_operator_release_never_starts(tmp_path):
    root, agents = fixture_agents(tmp_path)
    calls = []

    def runner(root, argv, timeout):
        calls.append(argv)
        return {"rc": 2}

    assert not power.run(root, "on", agent_dir=agents, runner=runner)["ok"]
    assert len(calls) == 1


def test_scheduler_off_never_executes_or_stamps_producer(tmp_path, monkeypatch):
    from scripts.ops import run_scheduled_lifecycle_job as job

    power.write(tmp_path, power.OFF, {"reason": "off"})
    artifact = tmp_path / "latest.json"
    artifact.write_text('{"timestamp_utc":"old"}')
    monkeypatch.setattr(
        job, "run_command", lambda *a, **kw: (_ for _ in ()).throw(AssertionError())
    )
    result, rc = job.build_payload(
        project_root=tmp_path,
        job_id="test",
        artifact=artifact,
        schedule_interval_seconds=60,
        deadline_seconds=30,
        command=["unused"],
    )
    assert rc == 0 and result["command_executed"] is False
    assert json.loads(artifact.read_text())["timestamp_utc"] == "old"
