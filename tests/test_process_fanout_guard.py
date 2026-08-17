from pathlib import Path

from scripts.ops import process_fanout_guard as src


def test_process_fanout_guard_trims_only_optional_schwab_workers(monkeypatch, tmp_path: Path) -> None:
    rows = [
        src.ProcRow(101, 1, 1.0, 900.0, 120, "/repo/scripts/run_execution_lane.py --mode paper"),
        src.ProcRow(201, 1, 0.0, 700.0, 120, "/repo/scripts/run_shadow_training_loop.py --broker schwab --profile fx"),
        src.ProcRow(301, 1, 0.0, 800.0, 120, "/repo/scripts/run_shadow_training_loop.py --broker coinbase"),
        src.ProcRow(401, 1, 0.0, 600.0, 120, "/repo/scripts/run_shadow_training_loop.py --broker schwab --profile qemc_path_volatility"),
        src.ProcRow(402, 1, 0.0, 500.0, 120, "/repo/scripts/run_specialized_sleeve_shadow.py --broker schwab --profile pairs_correlation"),
        src.ProcRow(403, 1, 0.0, 400.0, 120, "/repo/scripts/run_parallel_aggressive_modes.py"),
    ]
    monkeypatch.setattr(src, "collect_processes", lambda project_marker=src.DEFAULT_PROJECT_MARKER: rows)
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_COUNT", "3")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_TARGET_COUNT", "3")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_RSS_MB", "1000")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_TARGET_RSS_MB", "2500")

    payload = src.build_payload(out_path=tmp_path / "out.json", state_path=tmp_path / "state.json", override_path=tmp_path / "override.env")

    assert payload["triggered"] is True
    planned = {row["pid"] for row in payload["kill_plan"]}
    assert planned == {401, 402, 403}
    protected = {row["pid"] for row in payload["top_processes"] if row["protected"]}
    assert {101, 201, 301}.issubset(protected)
    assert "TRAINING_RUNTIME_PAUSED_FOR_FANOUT=1" in (tmp_path / "override.env").read_text(encoding="utf-8")


def test_process_fanout_guard_trims_optional_workers_on_cpu_pressure(monkeypatch, tmp_path: Path) -> None:
    rows = [
        src.ProcRow(101, 1, 2.0, 90.0, 120, "/repo/scripts/run_execution_lane.py --mode paper"),
        src.ProcRow(201, 1, 65.0, 120.0, 120, "/repo/scripts/run_shadow_training_loop.py --broker schwab --profile fx"),
        src.ProcRow(
            301,
            1,
            70.0,
            120.0,
            120,
            "/repo/scripts/run_shadow_training_loop.py --broker schwab --profile commodity_inflation",
        ),
        src.ProcRow(
            302,
            1,
            60.0,
            120.0,
            120,
            "/repo/scripts/run_shadow_training_loop.py --broker schwab --profile earnings_event",
        ),
        src.ProcRow(
            303,
            1,
            40.0,
            120.0,
            120,
            "/repo/scripts/run_specialized_sleeve_shadow.py --broker schwab --profile earnings_event",
        ),
    ]
    monkeypatch.setattr(src, "collect_processes", lambda project_marker=src.DEFAULT_PROJECT_MARKER: rows)
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_COUNT", "20")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_TARGET_COUNT", "20")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_RSS_MB", "1000")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_TARGET_RSS_MB", "1000")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_CPU_PERCENT", "120")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_TARGET_CPU_PERCENT", "75")

    payload = src.build_payload(out_path=tmp_path / "out.json", state_path=tmp_path / "state.json", override_path=tmp_path / "override.env")

    assert payload["triggered"] is True
    assert payload["trigger_reasons"]["targetable_cpu"] is True
    assert payload["fanout"]["targetable_cpu_percent"] == 170.0
    assert {row["pid"] for row in payload["kill_plan"]} == {301, 302}
    protected = {row["pid"] for row in payload["top_processes"] if row["protected"]}
    assert {101, 201}.issubset(protected)
    override = (tmp_path / "override.env").read_text(encoding="utf-8")
    assert "PROCESS_FANOUT_GUARD_REASON=runtime_cpu_pressure" in override
    assert "TRAINING_RUNTIME_PAUSED_FOR_FANOUT=1" in override


def test_process_fanout_guard_trims_orphaned_replay_sanity_checks(monkeypatch, tmp_path: Path) -> None:
    rows = [
        src.ProcRow(101, 1, 1.0, 90.0, 120, "/repo/scripts/run_execution_lane.py --mode paper"),
        src.ProcRow(201, 1, 25.0, 40.0, 18_000, "/repo/scripts/replay_preopen_sanity_check.py --hours 24 --json"),
        src.ProcRow(202, 1, 20.0, 40.0, 18_000, "/repo/scripts/replay_preopen_sanity_check.py --hours 24 --json"),
        src.ProcRow(203, 1, 15.0, 40.0, 18_000, "/repo/scripts/replay_preopen_sanity_check.py --hours 24 --json"),
    ]
    monkeypatch.setattr(src, "collect_processes", lambda project_marker=src.DEFAULT_PROJECT_MARKER: rows)
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_COUNT", "2")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_TARGET_COUNT", "2")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_RSS_MB", "1000")

    payload = src.build_payload(out_path=tmp_path / "out.json", state_path=tmp_path / "state.json", override_path=tmp_path / "override.env")

    assert payload["triggered"] is True
    assert payload["startup_policy"]["core_sleeve_restart_allowed"] is False
    assert {row["pid"] for row in payload["kill_plan"]} == {201, 202}
    assert payload["orphan_cleanup"]["planned_count"] == 3


def test_process_fanout_guard_clears_orphaned_one_shot_helpers_under_budget(monkeypatch, tmp_path: Path) -> None:
    rows = [
        src.ProcRow(101, 1, 1.0, 90.0, 120, "/repo/scripts/run_execution_lane.py --mode paper"),
        src.ProcRow(201, 1, 8.0, 60.0, 3_600, "/repo/scripts/data_source_divergence_bot.py --json"),
    ]
    monkeypatch.setattr(src, "collect_processes", lambda project_marker=src.DEFAULT_PROJECT_MARKER: rows)
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_COUNT", "10")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_RSS_MB", "1000")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_ORPHAN_GRACE_SECONDS", "900")

    payload = src.build_payload(
        apply=True,
        out_path=tmp_path / "out.json",
        state_path=tmp_path / "state.json",
        override_path=tmp_path / "override.env",
    )

    assert payload["triggered"] is False
    assert payload["overall_status"] == "active"
    assert payload["orphan_cleanup"]["candidate_count"] == 1
    assert payload["orphan_cleanup"]["terminated_count"] == 1
    assert payload["override"]["active"] is False
    assert "stale orphaned helper processes were cleared" in payload["recommended_actions"]


def test_process_fanout_guard_does_not_orphan_trim_live_loops(monkeypatch, tmp_path: Path) -> None:
    rows = [
        src.ProcRow(101, 1, 2.0, 250.0, 3_600, "/repo/scripts/run_shadow_training_loop.py --broker coinbase --profile crypto"),
        src.ProcRow(102, 1, 2.0, 120.0, 3_600, "/repo/scripts/ops/sql_link_shard_manager.py --once --json"),
    ]
    monkeypatch.setattr(src, "collect_processes", lambda project_marker=src.DEFAULT_PROJECT_MARKER: rows)
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_COUNT", "10")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_RSS_MB", "1000")

    payload = src.build_payload(out_path=tmp_path / "out.json", state_path=tmp_path / "state.json", override_path=tmp_path / "override.env")

    assert payload["overall_status"] == "ready"
    assert payload["orphan_cleanup"]["candidate_count"] == 0
    assert payload["kill_plan"] == []


def test_process_fanout_guard_clears_override_when_within_budget(monkeypatch, tmp_path: Path) -> None:
    rows = [
        src.ProcRow(101, 1, 1.0, 90.0, 120, "/repo/scripts/run_execution_lane.py --mode paper"),
        src.ProcRow(201, 1, 0.0, 80.0, 120, "/repo/scripts/run_shadow_training_loop.py --broker schwab --profile fx"),
    ]
    monkeypatch.setattr(src, "collect_processes", lambda project_marker=src.DEFAULT_PROJECT_MARKER: rows)
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_COUNT", "10")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_RSS_MB", "1000")

    payload = src.build_payload(out_path=tmp_path / "out.json", state_path=tmp_path / "state.json", override_path=tmp_path / "override.env")

    assert payload["overall_status"] == "ready"
    assert payload["triggered"] is False
    override = (tmp_path / "override.env").read_text(encoding="utf-8")
    assert "PROCESS_FANOUT_GUARD_MAX_COUNT=10" in override
    assert "PROCESS_FANOUT_GUARD_MAX_RSS_MB=1000.0" in override
    assert "TRAINING_RUNTIME_PAUSED_FOR_FANOUT=0" in override


def test_process_fanout_guard_clear_hold_removes_cooldown_override(monkeypatch, tmp_path: Path) -> None:
    rows = [
        src.ProcRow(101, 1, 1.0, 90.0, 120, "/repo/scripts/run_execution_lane.py --mode paper"),
        src.ProcRow(201, 1, 0.0, 80.0, 120, "/repo/scripts/run_shadow_training_loop.py --broker schwab --profile fx"),
    ]
    state = tmp_path / "state.json"
    state.write_text('{"hold_until_utc": "2999-01-01T00:00:00+00:00"}', encoding="utf-8")
    monkeypatch.setattr(src, "collect_processes", lambda project_marker=src.DEFAULT_PROJECT_MARKER: rows)
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_COUNT", "10")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_RSS_MB", "1000")

    payload = src.build_payload(
        clear_hold=True,
        out_path=tmp_path / "out.json",
        state_path=state,
        override_path=tmp_path / "override.env",
    )

    assert payload["triggered"] is False
    assert payload["override"]["hold_active"] is False
    assert payload["override"]["hold_cleared"] is True
    assert "OPS_WATCHDOG_ALL_SLEEVES_WITH_AGGRESSIVE=1" in (tmp_path / "override.env").read_text(encoding="utf-8")


def test_process_fanout_guard_preserves_clear_cooldown_by_default(monkeypatch, tmp_path: Path) -> None:
    rows = [
        src.ProcRow(101, 1, 1.0, 90.0, 120, "/repo/scripts/run_execution_lane.py --mode paper"),
        src.ProcRow(201, 1, 0.0, 80.0, 120, "/repo/scripts/run_shadow_training_loop.py --broker schwab --profile fx"),
    ]
    state = tmp_path / "state.json"
    state.write_text('{"hold_until_utc": "2999-01-01T00:00:00+00:00"}', encoding="utf-8")
    monkeypatch.setattr(src, "collect_processes", lambda project_marker=src.DEFAULT_PROJECT_MARKER: rows)
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_COUNT", "10")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_RSS_MB", "1000")

    payload = src.build_payload(out_path=tmp_path / "out.json", state_path=state, override_path=tmp_path / "override.env")

    assert payload["triggered"] is False
    assert payload["override"]["hold_active"] is True
    assert "RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES=0" in (tmp_path / "override.env").read_text(encoding="utf-8")


def test_process_fanout_guard_restores_recent_cooldown_from_last_trigger(monkeypatch, tmp_path: Path) -> None:
    rows = [
        src.ProcRow(101, 1, 1.0, 90.0, 120, "/repo/scripts/run_execution_lane.py --mode paper"),
        src.ProcRow(201, 1, 0.0, 80.0, 120, "/repo/scripts/run_shadow_training_loop.py --broker schwab --profile fx"),
    ]
    state = tmp_path / "state.json"
    state.write_text(
        '{"last_triggered_utc": "2999-01-01T00:00:00+00:00", "hold_until_utc": ""}',
        encoding="utf-8",
    )
    monkeypatch.setattr(src, "collect_processes", lambda project_marker=src.DEFAULT_PROJECT_MARKER: rows)
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_COUNT", "10")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_RSS_MB", "1000")

    payload = src.build_payload(out_path=tmp_path / "out.json", state_path=state, override_path=tmp_path / "override.env")

    assert payload["triggered"] is False
    assert payload["override"]["hold_active"] is True
    assert "PROCESS_FANOUT_GUARD_ACTIVE=1" in (tmp_path / "override.env").read_text(encoding="utf-8")


def test_process_fanout_guard_allows_core_sleeve_restart_when_no_targetable_workers(monkeypatch, tmp_path: Path) -> None:
    rows = [
        src.ProcRow(101, 1, 50.0, 4600.0, 120, "/repo/scripts/sql_hot_retention.py --vacuum"),
        src.ProcRow(201, 1, 1.0, 90.0, 120, "/repo/scripts/ops/sql_link_shard_manager.py"),
    ]
    monkeypatch.setattr(src, "collect_processes", lambda project_marker=src.DEFAULT_PROJECT_MARKER: rows)
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_RSS_MB", "1000")

    payload = src.build_payload(out_path=tmp_path / "out.json", state_path=tmp_path / "state.json", override_path=tmp_path / "override.env")
    override = (tmp_path / "override.env").read_text(encoding="utf-8")

    assert payload["triggered"] is True
    assert payload["fanout"]["targetable_count"] == 0
    assert payload["startup_policy"]["core_sleeve_restart_allowed"] is True
    assert "PROCESS_FANOUT_GUARD_CORE_SLEEVE_RESTART_ALLOWED=1" in override
    assert "RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES=0" in override
    assert "RUN_ALL_SLEEVES_WITH_DIVIDEND_CAPTURE=0" in override


def test_process_fanout_guard_never_trims_all_sleeves_parent(monkeypatch, tmp_path: Path) -> None:
    rows = [
        src.ProcRow(101, 1, 50.0, 4600.0, 120, "/repo/scripts/sql_hot_retention.py --vacuum"),
        src.ProcRow(201, 1, 1.0, 120.0, 120, "/repo/scripts/run_all_sleeves.py --broker schwab"),
        src.ProcRow(202, 201, 1.0, 500.0, 120, "/repo/scripts/run_specialized_sleeve_shadow.py --broker schwab --profile qemc_path_volatility"),
        src.ProcRow(203, 201, 1.0, 400.0, 120, "/repo/scripts/run_shadow_training_loop.py --broker schwab --profile qemc_path_volatility"),
    ]
    monkeypatch.setattr(src, "collect_processes", lambda project_marker=src.DEFAULT_PROJECT_MARKER: rows)
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_RSS_MB", "1000")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_TARGET_RSS_MB", "4300")

    payload = src.build_payload(out_path=tmp_path / "out.json", state_path=tmp_path / "state.json", override_path=tmp_path / "override.env")

    assert payload["triggered"] is True
    assert {row["pid"] for row in payload["kill_plan"]} == {202, 203}
    protected = {row["pid"] for row in payload["top_processes"] if row["protected"]}
    assert 201 in protected


def test_process_fanout_guard_protects_pressure_core_sleeves_after_restart(monkeypatch, tmp_path: Path) -> None:
    rows = [
        src.ProcRow(101, 1, 50.0, 4600.0, 120, "/repo/scripts/sql_hot_retention.py --vacuum"),
        src.ProcRow(201, 1, 1.0, 120.0, 120, "/repo/scripts/run_all_sleeves.py --broker schwab"),
        src.ProcRow(202, 201, 1.0, 120.0, 120, "/repo/scripts/run_parallel_shadows.py --broker schwab"),
        src.ProcRow(203, 201, 1.0, 120.0, 120, "/repo/scripts/run_shadow_training_loop.py --broker schwab --profile dividend"),
    ]
    monkeypatch.setattr(src, "collect_processes", lambda project_marker=src.DEFAULT_PROJECT_MARKER: rows)
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_RSS_MB", "1000")

    payload = src.build_payload(out_path=tmp_path / "out.json", state_path=tmp_path / "state.json", override_path=tmp_path / "override.env")

    assert payload["triggered"] is True
    assert payload["fanout"]["targetable_count"] == 0
    assert payload["kill_plan"] == []
    protected = {row["pid"] for row in payload["top_processes"] if row["protected"]}
    assert {201, 202, 203}.issubset(protected)
