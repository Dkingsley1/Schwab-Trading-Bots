import scripts.sqlite_performance_maintenance as maint


def test_checkpoint_mode_for_wal_uses_truncate_for_small_wal() -> None:
    assert maint._checkpoint_mode_for_wal(1.5, "auto", 8.0) == "truncate"


def test_checkpoint_mode_for_wal_uses_passive_for_large_wal() -> None:
    assert maint._checkpoint_mode_for_wal(12.0, "auto", 8.0) == "passive"


def test_checkpoint_mode_for_wal_respects_explicit_mode() -> None:
    assert maint._checkpoint_mode_for_wal(12.0, "restart", 8.0) == "restart"


def test_resolve_runtime_settings_downshifts_under_red_memory_pressure(tmp_path) -> None:
    health_root = tmp_path / "governance" / "health"
    health_root.mkdir(parents=True, exist_ok=True)
    (health_root / "resource_guard_latest.json").write_text(
        """
        {
          "memory_pressure_state": "red",
          "memory_pressure_kind": "throttled",
          "memory_free_pct": 7.5,
          "swap_used_gb": 22.0
        }
        """.strip(),
        encoding="utf-8",
    )

    settings = maint.resolve_runtime_settings(tmp_path)

    assert settings["pressure_level"] == "red"
    assert settings["temp_store_mode"] == "FILE"
    assert settings["cache_size_kb"] == 4096
    assert settings["mmap_size_mb"] == 32
    assert settings["analyze_enabled"] is False
    assert settings["auto_vacuum_allowed"] is False


def test_sqlite_maintenance_heartbeat_marks_running_step(tmp_path) -> None:
    out_path = tmp_path / "sqlite_maintenance_latest.json"
    payload = {"timestamp_utc": "2026-06-28T00:00:00+00:00", "ok": False}

    maint._write_heartbeat(
        payload,
        out_path,
        current_step="wal_checkpoint",
        started_monotonic=maint.time.monotonic(),
    )

    written = maint._read_json(out_path)
    assert payload["running"] is True
    assert payload["current_step"] == "wal_checkpoint"
    assert written["running"] is True
    assert written["current_step"] == "wal_checkpoint"
    assert written["pid"] > 0


def test_sqlite_maintenance_deadline_helper_raises_after_deadline() -> None:
    try:
        maint._raise_if_deadline_expired(maint.time.monotonic() - 1.0)
    except maint.MaintenanceDeadlineExceeded as exc:
        assert "runtime_exceeded" in str(exc)
    else:
        raise AssertionError("expected MaintenanceDeadlineExceeded")
