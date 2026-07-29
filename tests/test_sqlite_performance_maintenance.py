import scripts.sqlite_performance_maintenance as maint


def test_checkpoint_mode_for_wal_uses_truncate_for_small_wal() -> None:
    assert maint._checkpoint_mode_for_wal(1.5, "auto", 8.0) == "truncate"


def test_checkpoint_mode_for_wal_uses_passive_for_large_wal() -> None:
    assert maint._checkpoint_mode_for_wal(12.0, "auto", 8.0) == "passive"


def test_checkpoint_mode_for_wal_respects_explicit_mode() -> None:
    assert maint._checkpoint_mode_for_wal(12.0, "restart", 8.0) == "restart"


def test_row_count_skip_reason_skips_checkpoint_only() -> None:
    assert (
        maint._row_count_skip_reason(
            checkpoint_only=True,
            skip_row_count=False,
            db_size_gb=1.0,
            skip_over_gb=50.0,
        )
        == "checkpoint_only"
    )


def test_analyze_skip_reason_skips_large_database() -> None:
    reason = maint._analyze_skip_reason(
        skip_analyze=False,
        db_size_gb=182.214,
        skip_over_gb=50.0,
    )

    assert reason == "db_size_over_analyze_skip_threshold:182.214>=50.000"


def test_analyze_skip_reason_respects_operator_skip() -> None:
    assert (
        maint._analyze_skip_reason(
            skip_analyze=True,
            db_size_gb=1.0,
            skip_over_gb=50.0,
        )
        == "operator_skip_analyze"
    )


def test_row_count_skip_reason_skips_large_database() -> None:
    reason = maint._row_count_skip_reason(
        checkpoint_only=False,
        skip_row_count=False,
        db_size_gb=182.214,
        skip_over_gb=50.0,
    )

    assert reason == "db_size_over_row_count_skip_threshold:182.214>=50.000"


def test_row_count_skip_reason_allows_small_database() -> None:
    assert (
        maint._row_count_skip_reason(
            checkpoint_only=False,
            skip_row_count=False,
            db_size_gb=1.0,
            skip_over_gb=50.0,
        )
        == ""
    )


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
    assert settings["mmap_requested_mb"] == 0
    assert settings["mmap_size_mb"] == 0
    assert settings["mmap_enabled"] is False
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


def test_select_vacuum_temp_dir_uses_first_candidate_with_headroom(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "data" / "jsonl_link.sqlite3"
    db_path.parent.mkdir(parents=True)
    explicit = tmp_path / "small_tmp"
    db_tmp = db_path.parent / ".sqlite_tmp"
    project_tmp = tmp_path / ".tmp" / "sqlite_vacuum"

    def fake_free(path):
        text = str(path)
        if text == str(explicit):
            return 25.0
        if text == str(db_tmp):
            return 260.0
        if text == str(project_tmp):
            return 500.0
        return 0.0

    monkeypatch.setattr(maint, "_disk_free_gb", fake_free)

    selected = maint._select_vacuum_temp_dir(
        db_path=db_path,
        project_root=tmp_path,
        db_size_gb=200.0,
        explicit=str(explicit),
        min_free_ratio=1.15,
        min_free_gb=8.0,
    )

    assert selected["selected"] is True
    assert selected["selected_dir"] == str(db_tmp)
    assert selected["selected_source"] == "db_volume_tmpdir"
    assert selected["required_gb"] == 230.0
    assert selected["candidate_evaluations"][0]["reason"] == "insufficient_free_space"


def test_select_vacuum_temp_dir_refuses_all_small_candidates(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "data" / "jsonl_link.sqlite3"
    db_path.parent.mkdir(parents=True)
    monkeypatch.setattr(maint, "_disk_free_gb", lambda _path: 25.0)

    selected = maint._select_vacuum_temp_dir(
        db_path=db_path,
        project_root=tmp_path,
        db_size_gb=200.0,
        explicit="",
        min_free_ratio=1.15,
        min_free_gb=8.0,
    )

    assert selected["selected"] is False
    assert selected["reason"] == "insufficient_vacuum_temp_headroom"
    assert all(not row["usable"] for row in selected["candidate_evaluations"])
