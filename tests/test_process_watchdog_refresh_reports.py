import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from scripts.ops import process_watchdog as pw


def test_live_process_matching_ignores_watchdog_wrapper_commands(monkeypatch) -> None:
    ps_out = "\n".join(
        [
            "python scripts/shadow_watchdog.py --schwab-match scripts/run_all_sleeves.py",
            "python scripts/failover_hot_standby.py --primary-match scripts/run_all_sleeves.py --standby-cmd scripts/run_parallel_shadows.py --simulate",
            "python scripts/run_all_sleeves.py --broker schwab",
        ]
    )

    monkeypatch.setattr(
        pw.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=ps_out),
    )

    assert pw._proc_running("scripts/run_all_sleeves.py") == 1
    assert pw._proc_running("scripts/run_parallel_shadows.py") == 0


def test_live_data_target_excludes_simulated_alt_coverage_by_default(monkeypatch) -> None:
    monkeypatch.delenv("OPS_WATCHDOG_ALL_SLEEVES_SIMULATE", raising=False)

    target = pw._build_all_sleeves_target(heartbeat_max_age_seconds=300)

    assert "--simulate" in target["exclude_patterns"]
    assert "--simulate" not in target["cmd"]

    monkeypatch.setenv("OPS_WATCHDOG_ALL_SLEEVES_SIMULATE", "1")

    simulated_target = pw._build_all_sleeves_target(heartbeat_max_age_seconds=300)

    assert "--simulate" not in simulated_target["exclude_patterns"]
    assert "--simulate" in simulated_target["cmd"]


def test_live_data_excludes_preserve_profile_exclusions() -> None:
    assert pw._live_data_excludes(False, ["--profile crypto_futures"]) == [
        "--profile crypto_futures",
        "--simulate",
    ]
    assert pw._live_data_excludes(True, ["--profile crypto_futures"]) == ["--profile crypto_futures"]


def test_refresh_runtime_reports_uses_full_one_numbers_command(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    one_numbers = project_root / "exports" / "one_numbers" / "one_numbers_summary.json"
    one_numbers.parent.mkdir(parents=True, exist_ok=True)
    paper_performance = project_root / "governance" / "health" / "paper_performance_latest.json"
    paper_performance.parent.mkdir(parents=True, exist_ok=True)
    backpressure = project_root / "governance" / "health" / "ingestion_backpressure_latest.json"
    divergence = project_root / "governance" / "health" / "data_source_divergence_latest.json"
    daily_summary = project_root / "exports" / "sql_reports" / f"daily_runtime_summary_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    daily_summary.parent.mkdir(parents=True, exist_ok=True)
    daily_summary.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pw, "PROJECT_ROOT", project_root)

    calls: list[list[str]] = []

    def _fake_file_age(path: Path) -> float:
        if path in {one_numbers, paper_performance, backpressure, divergence}:
            return 999999.0
        return 0.0

    def _fake_run(cmd: list[str]):
        calls.append(cmd)
        if "build_one_numbers_report.py" in str(cmd[1:]):
            one_numbers.write_text(json.dumps({"generated_utc": "2026-03-31T21:00:00+00:00"}), encoding="utf-8")
            return 0, "", ""
        if "paper_performance_report.py" in str(cmd[1:]):
            paper_performance.write_text(json.dumps({"timestamp_utc": "2026-03-31T21:00:00+00:00"}), encoding="utf-8")
            return 0, "", ""
        if "ingestion_backpressure_guard.py" in str(cmd[1:]):
            backpressure.write_text(json.dumps({"timestamp_utc": "2026-03-31T21:00:00+00:00"}), encoding="utf-8")
            return 0, "", ""
        if "data_source_divergence_bot.py" in str(cmd[1:]):
            divergence.write_text(json.dumps({"timestamp_utc": "2026-03-31T21:00:00+00:00", "ok": True}), encoding="utf-8")
            return 0, "", ""
        return 0, "{}", ""

    monkeypatch.setattr(pw, "_file_age_seconds", _fake_file_age)
    monkeypatch.setattr(pw, "_run", _fake_run)
    monkeypatch.setattr(pw, "_proc_running", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(pw, "_resource_guard_allows_job", lambda job_name, profile="optional": (True, f"{job_name}:{profile}:ok"))

    out = pw._refresh_runtime_reports(max_age_seconds=60)

    assert out["one_numbers"]["refreshed"] is True
    assert out["paper_performance"]["refreshed"] is True
    assert out["ingestion_backpressure"]["refreshed"] is True
    assert out["data_source_divergence"]["refreshed"] is True
    assert any(
        "build_one_numbers_report.py" in " ".join(cmd)
        and "--lightweight" not in cmd
        and "--no-sql-write" not in cmd
        for cmd in calls
    )
    assert any("paper_performance_report.py" in " ".join(cmd) and "--json-only" in cmd for cmd in calls)
    assert any("ingestion_backpressure_guard.py" in " ".join(cmd) and "--json" in cmd for cmd in calls)
    assert any("data_source_divergence_bot.py" in " ".join(cmd) and "--json" in cmd for cmd in calls)


def test_refresh_runtime_reports_flags_stuck_refresh_process(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    one_numbers = project_root / "exports" / "one_numbers" / "one_numbers_summary.json"
    one_numbers.parent.mkdir(parents=True, exist_ok=True)
    paper_performance = project_root / "governance" / "health" / "paper_performance_latest.json"
    paper_performance.parent.mkdir(parents=True, exist_ok=True)
    backpressure = project_root / "governance" / "health" / "ingestion_backpressure_latest.json"
    backpressure.write_text("{}", encoding="utf-8")
    divergence = project_root / "governance" / "health" / "data_source_divergence_latest.json"
    divergence.write_text("{}", encoding="utf-8")
    daily_summary = project_root / "exports" / "sql_reports" / f"daily_runtime_summary_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    daily_summary.parent.mkdir(parents=True, exist_ok=True)
    daily_summary.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pw, "PROJECT_ROOT", project_root)
    monkeypatch.setenv("OPS_WATCHDOG_REFRESH_STUCK_SECONDS", "300")
    monkeypatch.setattr(
        pw,
        "_file_age_seconds",
        lambda path: 999999.0 if path == one_numbers else 0.0,
    )
    monkeypatch.setattr(pw, "_resource_guard_allows_job", lambda job_name, profile="optional": (True, f"{job_name}:{profile}:ok"))
    monkeypatch.setattr(pw, "_proc_running", lambda pattern, exclude_patterns=None: 1 if "build_one_numbers_report.py" in pattern else 0)
    monkeypatch.setattr(pw, "_proc_elapsed_seconds", lambda pattern, exclude_patterns=None: 1200.0 if "build_one_numbers_report.py" in pattern else None)
    monkeypatch.setattr(pw, "_run", lambda cmd: (_ for _ in ()).throw(AssertionError("refresh should not rerun while a stuck process is still present")))

    out = pw._refresh_runtime_reports(max_age_seconds=60)

    assert out["one_numbers"]["refreshed"] is False
    assert out["one_numbers"]["error"] == "refresh_stuck_suspected"
    assert out["one_numbers"]["running_seconds"] == 1200.0


def test_refresh_runtime_reports_keeps_health_fast_inside_livefeed_budget(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True)
    health_fast = health / "health_fast_latest.json"
    monkeypatch.setattr(pw, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        pw,
        "_file_age_seconds",
        lambda path: 420.0 if path == health_fast else 0.0,
    )
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str]):
        calls.append(cmd)
        return 2, json.dumps({"overall_status": "degraded"}), ""

    monkeypatch.setattr(pw, "_run", _fake_run)

    out = pw._refresh_runtime_reports(max_age_seconds=7200, health_fast_max_age_seconds=300)

    assert out["health_fast"]["refreshed"] is True
    assert out["health_fast"]["rc"] == 2
    assert out["health_fast"]["freshness_budget_seconds"] == 300
    assert out["health_fast"]["lightweight_always_on"] is True
    assert any("health_fast.py" in " ".join(cmd) for cmd in calls)


def test_lightweight_health_fast_refresh_does_not_depend_on_heavy_reports(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health_fast = project_root / "governance" / "health" / "health_fast_latest.json"
    monkeypatch.setattr(pw, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(pw, "_file_age_seconds", lambda path: 301.0 if path == health_fast else 0.0)
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str]):
        calls.append(cmd)
        return 0, "{}", ""

    monkeypatch.setattr(pw, "_run", _fake_run)

    row = pw._refresh_health_fast(300)

    assert row["refreshed"] is True
    assert row["lightweight_always_on"] is True
    assert len(calls) == 1
    assert "health_fast.py" in " ".join(calls[0])


def test_refresh_runtime_reports_blocks_daily_summary_when_resource_guard_denies(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    one_numbers = project_root / "exports" / "one_numbers" / "one_numbers_summary.json"
    one_numbers.parent.mkdir(parents=True, exist_ok=True)
    paper_performance = project_root / "governance" / "health" / "paper_performance_latest.json"
    paper_performance.parent.mkdir(parents=True, exist_ok=True)
    backpressure = project_root / "governance" / "health" / "ingestion_backpressure_latest.json"
    divergence = project_root / "governance" / "health" / "data_source_divergence_latest.json"
    backpressure.write_text("{}", encoding="utf-8")
    divergence.write_text("{}", encoding="utf-8")
    daily_summary = project_root / "exports" / "sql_reports" / f"daily_runtime_summary_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    daily_summary.parent.mkdir(parents=True, exist_ok=True)
    daily_summary.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pw, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        pw,
        "_file_age_seconds",
        lambda path: 999999.0 if path == daily_summary else 0.0,
    )
    monkeypatch.setattr(
        pw,
        "_resource_guard_allows_job",
        lambda job_name, profile="optional": (False, f"{job_name}:{profile}:creative_session_dual_pro"),
    )
    monkeypatch.setattr(
        pw,
        "_run",
        lambda cmd: (_ for _ in ()).throw(AssertionError("daily summary refresh should be skipped when resource guard blocks")),
    )

    out = pw._refresh_runtime_reports(max_age_seconds=60)

    assert out["daily_runtime_summary"]["refreshed"] is False
    assert out["daily_runtime_summary"]["resource_guard_ok"] is False
    assert out["daily_runtime_summary"]["error"] == "resource_guard_blocked"


def test_run_returns_timeout_payload(monkeypatch) -> None:
    class _FakeProcess:
        pid = 4321
        returncode = None
        stdout = None
        stderr = None

        def __init__(self, *_args, **_kwargs):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                exc = subprocess.TimeoutExpired(cmd=["fake-helper", "--json"], timeout=timeout)
                exc.output = "partial stdout\n"
                exc.stderr = "partial stderr\n"
                raise exc
            self.returncode = -15
            return "partial stdout\n", "partial stderr\n"

        def send_signal(self, _sig):
            return None

    monkeypatch.setattr(pw.subprocess, "Popen", _FakeProcess)
    monkeypatch.setattr(pw.os, "killpg", lambda *_args: None)

    rc, stdout, stderr = pw._run(["fake-helper", "--json"], timeout_seconds=12.0)

    assert rc == 124
    assert stdout == "partial stdout"
    assert "timeout_after_seconds=12.0" in stderr
    assert "partial stderr" in stderr


def test_process_watchdog_singleton_rejects_duplicate_instance(tmp_path) -> None:
    lock_path = tmp_path / "process_watchdog.lock"
    first, _owner = pw._acquire_singleton_lock(lock_path)
    assert first is not None
    second, owner = pw._acquire_singleton_lock(lock_path)
    assert second is None
    assert "pid=" in owner
    first.close()


def test_parse_ps_etime_seconds_handles_macos_formats() -> None:
    assert pw._parse_ps_etime_seconds("00:01") == 1.0
    assert pw._parse_ps_etime_seconds("01:02:03") == 3723.0
    assert pw._parse_ps_etime_seconds("2-03:04:05") == 183845.0


def test_build_execution_lane_target_uses_paper_health_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pw, "PROJECT_ROOT", tmp_path)

    target = pw._build_execution_lane_target("paper", heartbeat_max_age_seconds=240)

    assert target["name"] == "execution_lane_paper"
    assert target["pattern"] == "scripts/run_execution_lane.py --mode paper"
    assert target["cmd"][-2:] == ["--mode", "paper"]
    assert str(target["heartbeat_glob"]).endswith("execution_lane_paper_latest.json")
    assert target["heartbeat_max_age_seconds"] == 240
    assert target["heartbeat_startup_grace_seconds"] == 240
    assert target["max_running"] == 1
    assert target["restart_storm_impact"] == "execution_lane"
    assert target["restart_storm_quarantine_allowed"] is False
    assert target["live_execution_critical"] is True


def test_paper_execution_runtime_pause_state_prefers_override_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED", "1")
    monkeypatch.setenv("PAPER_EXECUTION_RUNTIME_PAUSED_FOR_PRESSURE", "0")
    monkeypatch.setenv("PAPER_400_RAMP_BLOCKED_RUNTIME_PAUSE", "0")
    override = tmp_path / ".env.runtime_resource_guard_override"
    override.write_text(
        "\n".join(
            [
                "PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED=0",
                "PAPER_EXECUTION_RUNTIME_PAUSED_FOR_PRESSURE=1",
                "PAPER_400_RAMP_BLOCKED_RUNTIME_PAUSE=1",
            ]
        ),
        encoding="utf-8",
    )

    state = pw._paper_execution_runtime_pause_state(override)

    assert state["paused"] is True
    assert state["consumer_enabled"] is False
    assert state["runtime_paused_for_pressure"] is True
    assert state["paper_400_ramp_blocked_runtime_pause"] is True
    assert state["reason"] == "paper_queue_consumer_disabled+runtime_pressure_pause+paper_400_ramp_blocked"


def test_row_intentionally_held_for_runtime_paper_pause() -> None:
    row = {
        "name": "execution_lane_paper",
        "heartbeat_ok": False,
        "process_live": False,
        "paused_by_runtime_gate": True,
        "restart_skipped": "runtime_paper_execution_paused",
        "runtime_pause_reason": "paper_400_ramp_blocked",
    }

    need = pw._watchdog_need_for_row(row)

    assert pw._row_intentionally_held(row) is True
    assert need is not None
    assert need["status"] == "intentional_hold"
    assert need["severity"] == "info"


def test_trim_duplicate_processes_keeps_newest(monkeypatch) -> None:
    killed: list[int] = []

    monkeypatch.setattr(pw, "_matching_pids", lambda *_args, **_kwargs: [101, 202, 303])

    def _fake_kill(pid: int, sig: int) -> None:
        if sig == 0:
            raise ProcessLookupError
        if sig == pw.signal.SIGCONT:
            return
        killed.append(pid)

    monkeypatch.setattr(pw.os, "kill", _fake_kill)

    payload = pw._trim_duplicate_processes(
        "scripts/run_execution_lane.py --mode paper",
        max_running=1,
    )

    assert payload["attempted"] is True
    assert payload["kept_pids"] == [303]
    assert payload["terminated_pids"] == [101, 202]
    assert killed == [101, 202]
    assert payload["still_running_pids"] == []


def test_default_require_paper_executor_disabled_when_all_sleeves_owns_it(monkeypatch) -> None:
    monkeypatch.delenv("OPS_WATCHDOG_REQUIRE_PAPER_EXECUTOR", raising=False)
    monkeypatch.setenv("OPS_WATCHDOG_REQUIRE_ALL_SLEEVES", "1")
    monkeypatch.setenv("RUN_ALL_SLEEVES_WITH_PAPER_EXECUTOR", "1")

    assert pw._default_require_paper_executor() is False


def test_all_sleeves_target_has_child_fanout_floor(monkeypatch) -> None:
    monkeypatch.delenv("OPS_WATCHDOG_ALL_SLEEVES_MIN_CHILDREN", raising=False)
    monkeypatch.delenv("OPS_WATCHDOG_ALL_SLEEVES_CHILD_GRACE_SECONDS", raising=False)

    target = pw._build_all_sleeves_target(heartbeat_max_age_seconds=300)

    assert target["parent_process_required"] is True
    assert target["min_child_processes"] >= 4
    assert target["child_fanout_grace_seconds"] >= 60
    assert target["heartbeat_startup_grace_seconds"] >= target["child_fanout_grace_seconds"]
    assert str(target["launcher_health_path"]).endswith("all_sleeves_launcher_latest.json")
    assert "sleeve_launcher_parent_watchdog" in target["repair_infrabots"]
    assert "sleeve_child_recycler" in target["repair_infrabots"]
    assert target["repair_policy"] == "restart_read_only_sleeve_collection_and_clean_orphans_without_enabling_live_execution"
    assert target["restart_storm_impact"] == "read_only_collection"
    assert target["restart_storm_quarantine_allowed"] is True
    assert target["live_execution_critical"] is False
    assert any("process_watchdog.py" in " ".join(command) for command in target["repair_commands"])
    assert "scripts/run_shadow_training_loop.py --broker schwab" in target["alt_patterns"]
    assert "scripts/run_shadow_training_loop.py --broker schwab" in target["orphan_cleanup_patterns"]
    assert target["restart_storm_settle_seconds"] == 180
    assert target["restart_storm_min_healthy_seconds"] == 90


def test_process_watchdog_cleans_all_sleeves_orphans_before_startup_ready_gate() -> None:
    text = Path(pw.__file__).read_text(encoding="utf-8")

    cleanup_idx = text.index("parent_process_required and running <= 0 and alt_running > 0")
    ready_idx = text.index("ready, reason = _all_sleeves_start_ready")
    assert cleanup_idx < ready_idx


def test_child_fanout_health_respects_startup_grace() -> None:
    target = {"min_child_processes": 4, "child_fanout_grace_seconds": 180}

    health = pw._child_fanout_health(
        target,
        running=1,
        alt_running=0,
        parent_elapsed_seconds=30.0,
    )

    assert health["ok"] is True
    assert health["reason"] == "startup_grace"


def test_child_fanout_health_flags_hollow_launcher_after_grace() -> None:
    target = {"min_child_processes": 4, "child_fanout_grace_seconds": 180}

    health = pw._child_fanout_health(
        target,
        running=1,
        alt_running=0,
        parent_elapsed_seconds=300.0,
    )

    assert health["ok"] is False
    assert health["reason"] == "child_fanout_below_floor"
    assert health["child_process_count"] == 0


def test_child_fanout_health_scores_parent_missing_children_against_floor() -> None:
    target = {"min_child_processes": 4, "child_fanout_grace_seconds": 180}

    healthy = pw._child_fanout_health(
        target,
        running=0,
        alt_running=6,
        parent_elapsed_seconds=None,
    )
    thin = pw._child_fanout_health(
        target,
        running=0,
        alt_running=2,
        parent_elapsed_seconds=None,
    )

    assert healthy["ok"] is True
    assert healthy["reason"] == "parent_missing_child_fanout_present"
    assert thin["ok"] is False
    assert thin["reason"] == "parent_missing_child_fanout_below_floor"


def test_all_sleeves_launcher_artifact_health_certifies_fresh_full_fanout(tmp_path: Path) -> None:
    launcher = tmp_path / "all_sleeves_launcher_latest.json"
    launcher.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.fromtimestamp(100.0, timezone.utc).isoformat(),
                "overall_status": "ready",
                "phase": "running",
                "expected_job_count": 100,
                "running_job_count": 100,
                "missing_job_count": 0,
                "exited_job_count": 0,
                "exact_needs": [],
            }
        ),
        encoding="utf-8",
    )

    health = pw._all_sleeves_launcher_artifact_health(
        {
            "launcher_health_path": str(launcher),
            "heartbeat_max_age_seconds": 360,
            "child_fanout_grace_seconds": 180,
        },
        now_epoch=120.0,
    )

    assert health["ok"] is True
    assert health["reason"] == "fresh_launcher_artifact_certifies_full_fanout"
    assert health["running_job_count"] == 100


def test_all_sleeves_launcher_artifact_health_certifies_policy_parked_fanout(tmp_path: Path) -> None:
    launcher = tmp_path / "all_sleeves_launcher_latest.json"
    launcher.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.fromtimestamp(100.0, timezone.utc).isoformat(),
                "overall_status": "ready",
                "phase": "running",
                "expected_job_count": 101,
                "running_job_count": 5,
                "missing_job_count": 0,
                "exited_job_count": 96,
                "policy_parked_job_count": 96,
                "clean_exited_job_count": 0,
                "repair_packet": {"problem_job_count": 0},
                "launcher_readiness_contract": {"exact_needs": []},
            }
        ),
        encoding="utf-8",
    )

    health = pw._all_sleeves_launcher_artifact_health(
        {
            "launcher_health_path": str(launcher),
            "heartbeat_max_age_seconds": 360,
            "child_fanout_grace_seconds": 180,
        },
        now_epoch=120.0,
    )

    assert health["ok"] is True
    assert health["reason"] == "fresh_launcher_artifact_certifies_stable_fanout"
    assert health["policy_parked_job_count"] == 96
    assert health["problem_job_count"] == 0


def test_all_sleeves_launcher_artifact_uses_repair_packet_problem_count(tmp_path: Path) -> None:
    launcher = tmp_path / "all_sleeves_launcher_latest.json"
    launcher.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.fromtimestamp(100.0, timezone.utc).isoformat(),
                "overall_status": "blocked",
                "phase": "running",
                "expected_job_count": 101,
                "running_job_count": 0,
                "missing_job_count": 0,
                "exited_job_count": 101,
                "repair_packet": {"problem_job_count": 26},
                "launcher_readiness_contract": {"exact_needs": [{"target": "baseline_parallel"}]},
            }
        ),
        encoding="utf-8",
    )

    health = pw._all_sleeves_launcher_artifact_health(
        {
            "launcher_health_path": str(launcher),
            "heartbeat_max_age_seconds": 360,
            "child_fanout_grace_seconds": 180,
        },
        now_epoch=120.0,
    )

    assert health["ok"] is False
    assert health["reason"] == "launcher_artifact_jobs_not_all_running"
    assert health["problem_job_count"] == 26
    assert health["exact_need_count"] == 1


def test_default_require_paper_executor_honors_explicit_override(monkeypatch) -> None:
    monkeypatch.setenv("OPS_WATCHDOG_REQUIRE_PAPER_EXECUTOR", "1")
    monkeypatch.setenv("OPS_WATCHDOG_REQUIRE_ALL_SLEEVES", "1")
    monkeypatch.setenv("RUN_ALL_SLEEVES_WITH_PAPER_EXECUTOR", "1")

    assert pw._default_require_paper_executor() is True


def test_all_sleeves_start_ready_allows_core_restart_when_fanout_guard_has_no_targetable_workers(tmp_path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    (health / "process_fanout_guard_latest.json").write_text(
        json.dumps(
            {
                "triggered": True,
                "fanout": {"targetable_count": 0},
                "kill_plan": [],
                "startup_policy": {"core_sleeve_restart_allowed": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pw, "HEALTH_DIR", health)
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_ACTIVE", "1")
    monkeypatch.setenv("TRAINING_RUNTIME_PAUSED_FOR_FANOUT", "1")
    monkeypatch.setenv("SHADOW_SYMBOLS_CORE", "SPY")
    monkeypatch.setenv("SHADOW_SYMBOLS_VOLATILE", "QQQ")
    monkeypatch.setenv("SHADOW_SYMBOLS_DEFENSIVE", "TLT")

    ready, reason = pw._all_sleeves_start_ready("schwab", True)

    assert ready is True
    assert reason == "process_fanout_guard_core_sleeve_pressure_mode"


def test_all_sleeves_start_ready_blocks_when_fanout_guard_has_targetable_workers(tmp_path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    (health / "process_fanout_guard_latest.json").write_text(
        json.dumps(
            {
                "triggered": True,
                "fanout": {"targetable_count": 1},
                "kill_plan": [{"pid": 101}],
                "startup_policy": {"core_sleeve_restart_allowed": False},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pw, "HEALTH_DIR", health)
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_ACTIVE", "1")
    monkeypatch.setenv("TRAINING_RUNTIME_PAUSED_FOR_FANOUT", "1")

    ready, reason = pw._all_sleeves_start_ready("schwab", True)

    assert ready is False
    assert reason == "process_fanout_guard_active"


def test_all_sleeves_start_ready_allows_readonly_restart_when_training_paused(monkeypatch) -> None:
    monkeypatch.setenv("TRAINING_RUNTIME_PAUSED_BY_OPERATOR_MODE", "1")
    monkeypatch.setenv("SHADOW_SYMBOLS_CORE", "SPY")
    monkeypatch.setenv("SHADOW_SYMBOLS_VOLATILE", "QQQ")
    monkeypatch.setenv("SHADOW_SYMBOLS_DEFENSIVE", "TLT")

    ready, reason = pw._all_sleeves_start_ready("schwab", True)

    assert ready is True
    assert reason == "training_or_research_paused_but_readonly_sleeve_restart_allowed"


def test_all_sleeves_start_ready_respects_explicit_sleeve_pause(monkeypatch) -> None:
    monkeypatch.setenv("ALL_SLEEVES_PAUSED_BY_OPERATOR_MODE", "1")

    ready, reason = pw._all_sleeves_start_ready("schwab", True)

    assert ready is False
    assert reason == "all_sleeves_explicitly_paused_by_operator_mode"


def test_resolved_restart_storms_drop_healthy_settled_services() -> None:
    active, recent = pw._resolved_restart_storms(
        events=[
            {"event": "restart", "name": "execution_lane_paper", "ts_epoch": 100.0},
            {"event": "restart", "name": "execution_lane_paper", "ts_epoch": 200.0},
            {"event": "restart", "name": "execution_lane_paper", "ts_epoch": 300.0},
            {"event": "restart", "name": "execution_lane_paper", "ts_epoch": 400.0},
        ],
        status_rows=[{"name": "execution_lane_paper", "running": 1, "heartbeat_ok": True}],
        restart_window_seconds=3600,
        restart_storm_threshold=4,
        settle_seconds=120,
        now_epoch=1000.0,
    )

    assert active == []
    assert len(recent) == 1
    assert recent[0]["resolved"] is True


def test_resolved_restart_storms_accepts_idle_complete_sql_writer() -> None:
    active, recent = pw._resolved_restart_storms(
        events=[
            {"event": "restart", "name": "sql_link_writer", "ts_epoch": 100.0},
            {"event": "restart", "name": "sql_link_writer", "ts_epoch": 200.0},
            {"event": "restart", "name": "sql_link_writer", "ts_epoch": 300.0},
            {"event": "restart", "name": "sql_link_writer", "ts_epoch": 400.0},
        ],
        status_rows=[
            {
                "name": "sql_link_writer",
                "running": 0,
                "heartbeat_ok": True,
                "process_live": True,
                "writer_idle_ok": True,
                "live_execution_critical": False,
                "restart_storm_impact": "storage_writer",
                "restart_storm_quarantine_allowed": False,
            }
        ],
        restart_window_seconds=3600,
        restart_storm_threshold=4,
        settle_seconds=900,
        now_epoch=450.0,
    )

    assert active == []
    assert len(recent) == 1
    assert recent[0]["resolved"] is True
    assert recent[0]["resolution_reason"] == "sql_writer_on_demand_idle_complete"


def test_resolved_restart_storms_accepts_active_progressing_sql_writer() -> None:
    active, recent = pw._resolved_restart_storms(
        events=[
            {"event": "restart", "name": "sql_link_writer", "ts_epoch": 100.0},
            {"event": "restart", "name": "sql_link_writer", "ts_epoch": 200.0},
            {"event": "restart", "name": "sql_link_writer", "ts_epoch": 300.0},
            {"event": "restart", "name": "sql_link_writer", "ts_epoch": 400.0},
        ],
        status_rows=[
            {
                "name": "sql_link_writer",
                "running": 0,
                "heartbeat_ok": True,
                "process_live": False,
                "writer_recovered_ok": True,
                "live_execution_critical": False,
                "restart_storm_impact": "storage_writer",
                "restart_storm_quarantine_allowed": False,
            }
        ],
        restart_window_seconds=3600,
        restart_storm_threshold=4,
        settle_seconds=900,
        now_epoch=450.0,
    )

    assert active == []
    assert len(recent) == 1
    assert recent[0]["resolved"] is True
    assert recent[0]["resolution_reason"] == "sql_writer_active_progress_recovered"
    assert recent[0]["blocks_execution_clear"] is False


def test_resolved_restart_storms_requires_parent_when_marked_parent_required() -> None:
    active, recent = pw._resolved_restart_storms(
        events=[
            {"event": "restart", "name": "all_sleeves", "ts_epoch": 100.0},
            {"event": "restart", "name": "all_sleeves", "ts_epoch": 200.0},
            {"event": "restart", "name": "all_sleeves", "ts_epoch": 300.0},
            {"event": "restart", "name": "all_sleeves", "ts_epoch": 400.0},
        ],
        status_rows=[
            {
                "name": "all_sleeves",
                "running": 0,
                "alt_running": 6,
                "heartbeat_ok": True,
                "parent_process_required": True,
            }
        ],
        restart_window_seconds=3600,
        restart_storm_threshold=4,
        settle_seconds=120,
        now_epoch=1000.0,
    )

    assert len(active) == 1
    assert active[0]["name"] == "all_sleeves"
    assert active[0]["resolved"] is False
    assert len(recent) == 1
    assert recent[0]["resolved"] is False


def test_resolved_restart_storms_accepts_certified_all_sleeves_fanout_without_parent() -> None:
    active, recent = pw._resolved_restart_storms(
        events=[
            {"event": "restart", "name": "all_sleeves", "ts_epoch": 100.0},
            {"event": "restart", "name": "all_sleeves", "ts_epoch": 200.0},
            {"event": "restart", "name": "all_sleeves", "ts_epoch": 300.0},
            {"event": "restart", "name": "all_sleeves", "ts_epoch": 400.0},
        ],
        status_rows=[
            {
                "name": "all_sleeves",
                "running": 0,
                "alt_running": 100,
                "heartbeat_ok": True,
                "parent_process_required": True,
                "effective_process_live": True,
                "launcher_artifact_certified_fanout": True,
                "heartbeat_age_seconds": 10.0,
                "heartbeat_max_age_seconds": 360.0,
            }
        ],
        restart_window_seconds=3600,
        restart_storm_threshold=4,
        settle_seconds=120,
        now_epoch=1000.0,
    )

    assert active == []
    assert len(recent) == 1
    assert recent[0]["resolved"] is True


def test_resolved_restart_storms_marks_read_only_collection_as_quarantinable() -> None:
    active, recent = pw._resolved_restart_storms(
        events=[
            {"event": "restart", "name": "all_sleeves", "ts_epoch": 100.0},
            {"event": "restart", "name": "all_sleeves", "ts_epoch": 200.0},
            {"event": "restart", "name": "all_sleeves", "ts_epoch": 300.0},
            {"event": "restart", "name": "all_sleeves", "ts_epoch": 400.0},
        ],
        status_rows=[
            {
                "name": "all_sleeves",
                "running": 0,
                "alt_running": 6,
                "heartbeat_ok": False,
                "parent_process_required": True,
                "restart_storm_impact": "read_only_collection",
                "restart_storm_quarantine_allowed": True,
                "live_execution_critical": False,
            }
        ],
        restart_window_seconds=3600,
        restart_storm_threshold=4,
        settle_seconds=120,
        now_epoch=450.0,
    )

    assert len(active) == 1
    assert len(recent) == 1
    assert active[0]["impact"] == "read_only_collection"
    assert active[0]["quarantinable"] is True
    assert active[0]["quarantine_state"] == "isolated_read_only_collection"
    assert active[0]["blocks_execution_clear"] is False
    assert active[0]["live_execution_critical"] is False


def test_watchdog_intelligence_downgrades_isolated_restart_storm_budget() -> None:
    status_rows = [
        {
            "name": "all_sleeves",
            "running": 0,
            "heartbeat_ok": False,
            "process_live": False,
            "restart_skipped": "budget_exhausted",
            "restart_storm_impact": "read_only_collection",
            "restart_storm_quarantine_allowed": True,
            "live_execution_critical": False,
        }
    ]
    restart_storms = [
        {
            "name": "all_sleeves",
            "count": 4,
            "resolved": False,
            "impact": "read_only_collection",
            "quarantinable": True,
            "blocks_execution_clear": False,
        }
    ]

    payload = pw._watchdog_intelligence_contract(
        status_rows=status_rows,
        restarts=[],
        restart_storms=restart_storms,
        recent_restart_storms=restart_storms,
        alerts=[],
        safety_pause={"active": False},
        creative_pause={"active": False},
        network_payload={"outage_active": False},
    )

    assert payload["overall_status"] == "degraded"
    assert payload["restart_storm_isolation"]["all_active_storms_isolated"] is True
    assert payload["restart_budget_isolated_blocks"] == ["all_sleeves"]
    assert payload["restart_budget_execution_blocks"] == []
    assert payload["exact_needs"][0]["severity"] == "warn"


def test_restart_budget_alert_metadata_downgrades_isolated_collectors() -> None:
    severity, event = pw._restart_budget_alert_metadata(
        "coinbase_loop",
        {
            "name": "coinbase_loop",
            "restart_storm_impact": "read_only_collection",
            "restart_storm_quarantine_allowed": True,
            "live_execution_critical": False,
        },
    )

    assert severity == "warn"
    assert event == "watchdog_restart_budget_exhausted_isolated"


def test_restart_budget_repair_probe_allows_coinbase_after_cooldown(monkeypatch) -> None:
    monkeypatch.setenv("OPS_WATCHDOG_READONLY_BUDGET_REPAIR_PROBE", "1")
    events = [
        {"event": "restart", "name": "coinbase_loop", "ts_epoch": 100.0},
        {"event": "restart", "name": "coinbase_loop", "ts_epoch": 200.0},
        {"event": "restart", "name": "coinbase_loop", "ts_epoch": 300.0},
        {"event": "restart", "name": "coinbase_loop", "ts_epoch": 400.0},
        {"event": "restart", "name": "coinbase_loop", "ts_epoch": 500.0},
        {"event": "restart", "name": "coinbase_loop", "ts_epoch": 600.0},
    ]

    probe = pw._restart_budget_repair_probe(
        events=events,
        name="coinbase_loop",
        row={
            "name": "coinbase_loop",
            "restart_storm_impact": "read_only_collection",
            "restart_storm_quarantine_allowed": True,
            "live_execution_critical": False,
        },
        cooldown_seconds=900,
        now_epoch=1600.0,
    )

    assert probe["allowed"] is True
    assert probe["reason"] == "read_only_collector_repair_probe_after_restart_budget_exhausted"
    assert probe["last_restart_age_seconds"] == 1000.0


def test_restart_budget_repair_probe_denies_execution_lane_and_cooldown(monkeypatch) -> None:
    monkeypatch.setenv("OPS_WATCHDOG_READONLY_BUDGET_REPAIR_PROBE", "1")
    events = [{"event": "restart", "name": "coinbase_loop", "ts_epoch": 1000.0}]

    cooldown_probe = pw._restart_budget_repair_probe(
        events=events,
        name="coinbase_loop",
        row={
            "name": "coinbase_loop",
            "restart_storm_impact": "read_only_collection",
            "restart_storm_quarantine_allowed": True,
            "live_execution_critical": False,
        },
        cooldown_seconds=900,
        now_epoch=1200.0,
    )
    execution_probe = pw._restart_budget_repair_probe(
        events=[{"event": "restart", "name": "execution_lane_live", "ts_epoch": 100.0}],
        name="execution_lane_live",
        row={
            "name": "execution_lane_live",
            "restart_storm_impact": "execution_lane",
            "restart_storm_quarantine_allowed": False,
            "live_execution_critical": True,
        },
        cooldown_seconds=900,
        now_epoch=1600.0,
    )

    assert cooldown_probe["allowed"] is False
    assert cooldown_probe["reason"] == "repair_probe_cooldown"
    assert execution_probe["allowed"] is False
    assert execution_probe["reason"] == "not_read_only_quarantinable"


def test_resolved_restart_storms_respect_target_specific_settle_window() -> None:
    active, recent = pw._resolved_restart_storms(
        events=[
            {"event": "restart", "name": "execution_lane_paper", "ts_epoch": 100.0},
            {"event": "restart", "name": "execution_lane_paper", "ts_epoch": 200.0},
            {"event": "restart", "name": "execution_lane_paper", "ts_epoch": 300.0},
            {"event": "restart", "name": "execution_lane_paper", "ts_epoch": 400.0},
        ],
        status_rows=[
            {
                "name": "execution_lane_paper",
                "running": 1,
                "heartbeat_ok": True,
                "heartbeat_age_seconds": 45.0,
                "heartbeat_max_age_seconds": 240.0,
                "restart_storm_settle_seconds": 120,
                "restart_storm_min_healthy_seconds": 120,
            }
        ],
        restart_window_seconds=3600,
        restart_storm_threshold=4,
        settle_seconds=900,
        now_epoch=531.0,
    )

    assert active == []
    assert recent[0]["settle_seconds"] == 120
    assert recent[0]["resolved"] is True


def test_resolved_restart_storms_resolve_when_target_is_paused_by_safety_flags() -> None:
    active, recent = pw._resolved_restart_storms(
        events=[
            {"event": "restart", "name": "all_sleeves", "ts_epoch": 100.0},
            {"event": "restart", "name": "all_sleeves", "ts_epoch": 200.0},
            {"event": "restart", "name": "all_sleeves", "ts_epoch": 300.0},
            {"event": "restart", "name": "all_sleeves", "ts_epoch": 400.0},
        ],
        status_rows=[
            {
                "name": "all_sleeves",
                "running": 0,
                "heartbeat_ok": False,
                "paused_by_safety_flags": True,
                "safety_pause_reason": "operator_stop_active",
            }
        ],
        restart_window_seconds=3600,
        restart_storm_threshold=4,
        settle_seconds=900,
        now_epoch=450.0,
    )

    assert active == []
    assert recent[0]["resolved"] is True
    assert recent[0]["resolution_reason"] == "operator_stop_active"


def test_resolved_restart_storms_resolve_when_paper_paused_by_runtime_gate() -> None:
    active, recent = pw._resolved_restart_storms(
        events=[
            {"event": "restart", "name": "execution_lane_paper", "ts_epoch": 100.0},
            {"event": "restart", "name": "execution_lane_paper", "ts_epoch": 200.0},
            {"event": "restart", "name": "execution_lane_paper", "ts_epoch": 300.0},
            {"event": "restart", "name": "execution_lane_paper", "ts_epoch": 400.0},
        ],
        status_rows=[
            {
                "name": "execution_lane_paper",
                "running": 0,
                "heartbeat_ok": False,
                "paused_by_runtime_gate": True,
                "runtime_pause_reason": "paper_400_ramp_blocked",
                "restart_skipped": "runtime_paper_execution_paused",
                "restart_storm_impact": "execution_lane",
                "restart_storm_quarantine_allowed": False,
                "live_execution_critical": True,
            }
        ],
        restart_window_seconds=3600,
        restart_storm_threshold=4,
        settle_seconds=900,
        now_epoch=450.0,
    )

    assert active == []
    assert recent[0]["resolved"] is True
    assert recent[0]["resolution_reason"] == "paper_400_ramp_blocked"


def test_forgive_resolved_restart_debt_removes_only_recovered_target_events() -> None:
    events = [
        {"event": "restart", "name": "coinbase_loop", "ts_epoch": 100.0},
        {"event": "restart", "name": "coinbase_futures_loop", "ts_epoch": 101.0},
        {"event": "restart", "name": "sql_link_writer", "ts_epoch": 102.0},
        {"event": "note", "name": "coinbase_loop", "ts_epoch": 103.0},
    ]

    kept, forgiveness = pw._forgive_resolved_restart_debt(
        events,
        [
            {"name": "coinbase_loop", "resolved": True},
            {"name": "coinbase_futures_loop", "resolved": False},
        ],
    )

    assert forgiveness["active"] is True
    assert forgiveness["forgiven_names"] == ["coinbase_loop"]
    assert forgiveness["removed_event_count"] == 1
    assert kept == [
        {"event": "restart", "name": "coinbase_futures_loop", "ts_epoch": 101.0},
        {"event": "restart", "name": "sql_link_writer", "ts_epoch": 102.0},
        {"event": "note", "name": "coinbase_loop", "ts_epoch": 103.0},
    ]
