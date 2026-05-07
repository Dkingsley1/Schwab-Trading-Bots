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


def test_process_fanout_guard_does_not_preserve_clear_cooldown_by_default(monkeypatch, tmp_path: Path) -> None:
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
    assert payload["override"]["hold_active"] is False
    assert "RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES=1" in (tmp_path / "override.env").read_text(encoding="utf-8")


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


def test_process_fanout_guard_protects_pressure_core_sleeves_after_restart(monkeypatch, tmp_path: Path) -> None:
    rows = [
        src.ProcRow(101, 1, 50.0, 4600.0, 120, "/repo/scripts/sql_hot_retention.py --vacuum"),
        src.ProcRow(201, 1, 1.0, 120.0, 120, "/repo/scripts/run_all_sleeves.py --broker schwab"),
        src.ProcRow(202, 201, 1.0, 120.0, 120, "/repo/scripts/run_parallel_shadows.py --broker schwab"),
        src.ProcRow(203, 201, 1.0, 120.0, 120, "/repo/scripts/run_shadow_training_loop.py --broker schwab --profile dividend"),
    ]
    monkeypatch.setattr(src, "collect_processes", lambda project_marker=src.DEFAULT_PROJECT_MARKER: rows)
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_MAX_RSS_MB", "1000")
    monkeypatch.setenv("PROCESS_FANOUT_GUARD_CORE_SLEEVE_RESTART_ALLOWED", "1")

    payload = src.build_payload(out_path=tmp_path / "out.json", state_path=tmp_path / "state.json", override_path=tmp_path / "override.env")

    assert payload["triggered"] is True
    assert payload["fanout"]["targetable_count"] == 0
    assert payload["kill_plan"] == []
    protected = {row["pid"] for row in payload["top_processes"] if row["protected"]}
    assert {201, 202, 203}.issubset(protected)
