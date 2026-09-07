import scripts.sqlite_performance_maintenance as maint
from pathlib import Path


def test_vacuum_temp_candidates_never_probe_protected_volume(tmp_path, monkeypatch):
    original_exists = Path.exists

    def checked_exists(path):
        assert not str(path).startswith("/Volumes/VIDEO")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", checked_exists)
    candidates = maint._vacuum_temp_dir_candidates(tmp_path / "db.sqlite3", tmp_path)
    assert all(source != "video_volume_tmpdir" for _, source in candidates)


def test_vacuum_temp_selection_rejects_protected_path_before_io(tmp_path, monkeypatch):
    protected = Path("/Volumes/VIDEO/sqlite_tmp")
    monkeypatch.setattr(maint, "_vacuum_temp_dir_candidates", lambda *args: [(protected, "explicit")])

    def unexpected(*args, **kwargs):
        raise AssertionError("protected volume must not be probed")

    monkeypatch.setattr(Path, "mkdir", unexpected)
    monkeypatch.setattr(Path, "resolve", unexpected)
    result = maint._select_vacuum_temp_dir(db_path=tmp_path / "db.sqlite3", project_root=tmp_path, db_size_gb=1)
    assert result["selected"] is False
    assert result["candidate_evaluations"][0]["reason"] == "protected_volume"


def test_vacuum_temp_selection_rejects_symlink_to_protected_volume(tmp_path):
    link = tmp_path / "media_alias"
    link.symlink_to("/Volumes/VIDEO", target_is_directory=True)
    assert maint._protected_storage_path(link)


def test_sqlite_maintenance_owned_hold_allows_checkpoint(tmp_path, monkeypatch):
    import sqlite3
    from core.runtime_maintenance import MAINTENANCE_HOLD_TOKEN_ENV

    db = tmp_path / "test.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE evidence (id INTEGER PRIMARY KEY)")
    out = tmp_path / "maintenance.json"
    monkeypatch.setattr(maint, "maintenance_hold_snapshot", lambda root: {"active": True, "valid": True, "token": "test-owner-token"})
    monkeypatch.setenv(MAINTENANCE_HOLD_TOKEN_ENV, "test-owner-token")
    monkeypatch.setattr(maint.sys, "argv", ["maintenance", "--db", str(db), "--out-file", str(out), "--checkpoint-only", "--json"])
    assert maint.main() == 0
    assert maint._read_json(out)["current_step"] == "complete"


def test_sqlite_maintenance_hold_exits_before_opening_database(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "missing.sqlite3"
    out_path = tmp_path / "sqlite_maintenance_latest.json"
    monkeypatch.setattr(
        maint,
        "maintenance_hold_snapshot",
        lambda project_root: {"active": True, "reason": "test_cutover"},
    )
    monkeypatch.setattr(
        maint.sys,
        "argv",
        [
            "sqlite_performance_maintenance.py",
            "--db",
            str(db_path),
            "--out-file",
            str(out_path),
            "--checkpoint-only",
            "--json",
        ],
    )

    assert maint.main() == 0
    payload = maint._read_json(out_path)
    assert payload["overall_status"] == "runtime_maintenance_hold"
    assert payload["route_mutation_performed"] is False
    assert not db_path.exists()


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
