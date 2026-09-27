import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ops import sql_writer_admission as admission
from scripts.ops import writer_cycle_coordinator as coordinator

FIELDS = ("target", "pressure", "hard", "emergency")
THRESHOLDS = dict(zip((f"{field}_free_gb" for field in FIELDS), (125, 64, 32, 16)))


@pytest.fixture
def disk(monkeypatch):
    for field in FIELDS:
        monkeypatch.delenv(f"BOT_LOCAL_STORAGE_{field.upper()}_FREE_GB", raising=False)
    monkeypatch.delenv("SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE", raising=False)
    state = {"free": 200}
    contract = admission.reserve.local_storage_reserve_contract

    def snapshot(root, **policy):
        gib = 1024**3
        return contract(
            root,
            **policy,
            disk_usage_fn=lambda _: SimpleNamespace(
                total=1000 * gib,
                used=(1000 - state["free"]) * gib,
                free=state["free"] * gib,
            ),
        )

    monkeypatch.setattr(admission.reserve, "local_storage_reserve_contract", snapshot)
    return state


def owner_receipt(root: Path, **updates):
    path = root / "governance/health/local_storage_reserve_guard_latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"local_storage_reserve": {**THRESHOLDS, **updates}}))
    return path


def test_live_low_disk_overrules_old_green_receipt(tmp_path, disk):
    owner_receipt(tmp_path, free_gb=250, pause_nonessential_writers=False)
    disk["free"] = 15
    result = admission.storage_admission(tmp_path)
    assert result["writer_start_allowed"] is False
    assert result["blockers"] == ["local_storage_reserve_pause"]
    assert result["local_storage_reserve"]["status"] == "emergency"
    assert result["local_storage_reserve"]["free_gb"] == 15


def test_fresh_deferral_does_not_refresh_old_writer_progress(tmp_path, disk):
    owner_receipt(tmp_path)
    progress = tmp_path / "governance/health/sql_link_service_progress_latest.json"
    original = '{"timestamp_utc":"2025-01-01T00:00:00+00:00","status":"complete"}'
    progress.write_text(original)
    disk["free"] = 15
    result = admission.publish_writer_observation(tmp_path, owner="test")
    assert result["overall_status"] == "deferred"
    assert result["writer_progress_fresh"] is False
    assert result["current_sql_failure_count"] is None
    assert progress.read_text() == original


def test_fresh_disk_clears_stale_observation_but_keeps_owner_policy(tmp_path, disk):
    owner_receipt(tmp_path, free_gb=1, pause_nonessential_writers=True)
    result = admission.storage_admission(tmp_path)
    assert result["writer_start_allowed"] is True
    assert all(
        result["local_storage_reserve"][key] == value
        for key, value in THRESHOLDS.items()
    )
    disk["free"] = 40
    assert admission.storage_admission(tmp_path)["writer_start_allowed"] is False


@pytest.mark.parametrize("pause", ["1", "true", "yes", "on", " TRUE "])
def test_explicit_pause_cannot_be_bypassed(tmp_path, disk, monkeypatch, pause):
    monkeypatch.setenv("SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE", pause)
    assert admission.storage_admission(tmp_path)["blockers"] == [
        "sql_writer_storage_pause_requested"
    ]


def test_lower_environment_thresholds_do_not_relax_owner_policy(
    tmp_path, disk, monkeypatch
):
    owner_receipt(tmp_path)
    for field in FIELDS:
        monkeypatch.setenv(f"BOT_LOCAL_STORAGE_{field.upper()}_FREE_GB", "1")
    disk["free"] = 40
    assert admission.storage_admission(tmp_path)["writer_start_allowed"] is False


def test_missing_receipt_uses_current_configured_policy(tmp_path, disk, monkeypatch):
    monkeypatch.setenv("BOT_LOCAL_STORAGE_PRESSURE_FREE_GB", "60")
    disk["free"] = 50
    result = admission.storage_admission(tmp_path)
    assert result["writer_start_allowed"] is False
    assert result["local_storage_reserve"]["pressure_free_gb"] == 60


@pytest.mark.parametrize(
    "value", [True, None, "32", -1, 0, float("nan"), float("inf"), {}, []]
)
def test_invalid_owner_policy_fails_closed(tmp_path, disk, value):
    owner_receipt(tmp_path, hard_free_gb=value)
    assert admission.storage_admission(tmp_path)["writer_start_allowed"] is False


@pytest.mark.parametrize("raw", ["{", "{}", "[]", "x" * (2 * 1024 * 1024 + 1)])
def test_invalid_receipt_fails_closed(tmp_path, disk, raw):
    owner_receipt(tmp_path).write_text(raw)
    assert admission.storage_admission(tmp_path)["writer_start_allowed"] is False


@pytest.mark.parametrize("value", ["nan", "inf", "invalid", "-1", "0"])
def test_invalid_environment_policy_fails_closed(tmp_path, disk, monkeypatch, value):
    monkeypatch.setenv("BOT_LOCAL_STORAGE_HARD_FREE_GB", value)
    assert admission.storage_admission(tmp_path)["writer_start_allowed"] is False


def test_unordered_policy_fails_closed(tmp_path, disk):
    owner_receipt(tmp_path, hard_free_gb=500)
    assert admission.storage_admission(tmp_path)["writer_start_allowed"] is False


def test_unsafe_route_is_not_opened(tmp_path, disk, monkeypatch):
    monkeypatch.setattr(
        admission, "inspect_storage_path", lambda _: {"status": "protected"}
    )
    monkeypatch.setattr(
        Path, "open", lambda *a, **k: pytest.fail("unsafe route opened")
    )
    assert admission.storage_admission(tmp_path)["writer_start_allowed"] is False


def test_unknown_disk_fails_closed(tmp_path, disk, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(
        admission.reserve, "local_storage_reserve_contract", unavailable
    )
    assert admission.storage_admission(tmp_path)["writer_start_allowed"] is False


def test_defer_receipt_does_not_clobber_active_writer_progress(tmp_path, disk):
    owner_receipt(tmp_path)
    active = tmp_path / "governance/health/sql_link_service_latest.json"
    active.write_text('{"running": true, "owner": "existing"}')
    disk["free"] = 15
    result = admission.defer_storage_writer(tmp_path, owner="test")
    assert result["rc"] == 75
    assert json.loads(active.read_text()) == {"running": True, "owner": "existing"}
    assert (
        json.loads(
            (active.parent / "sql_link_storage_admission_latest.json").read_text()
        )
        == result
    )
    disk["free"] = 200
    assert admission.defer_storage_writer(tmp_path, owner="test") == {}
    cleared = json.loads(
        (active.parent / "sql_link_storage_admission_latest.json").read_text()
    )
    assert cleared["writer_start_allowed"] is True
    assert cleared["deferred"] is False
    assert cleared["running"] is False
    assert cleared["blockers"] == []
    assert json.loads(active.read_text()) == {"running": True, "owner": "existing"}


@pytest.mark.parametrize(
    "module_name", ["sql_link_shard_manager", "sql_link_writer_service"]
)
def test_direct_entry_points_refuse_before_lock_or_children(
    tmp_path, disk, monkeypatch, capsys, module_name
):
    from importlib import import_module

    module = import_module(f"scripts.ops.{module_name}")
    owner_receipt(tmp_path)
    disk["free"] = 15
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "maintenance_hold_snapshot",
        lambda *a, **k: pytest.fail(
            "maintenance token reached before storage admission"
        ),
    )
    monkeypatch.setattr(
        module.fcntl, "flock", lambda *a, **k: pytest.fail("writer lock acquired")
    )
    monkeypatch.setattr(
        module.subprocess, "run", lambda *a, **k: pytest.fail("child launched")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [module_name, "--once", "--json", "--lock-path", str(tmp_path / "writer.lock")],
    )
    monkeypatch.setenv("SQL_LINK_SERVICE_MAINTENANCE_HOLD_TOKEN", "authorized-token")
    assert module.main() == 75
    assert json.loads(capsys.readouterr().out)["writer_start_allowed"] is False
    assert not (tmp_path / "writer.lock").exists()


def test_coordinator_recognizes_only_valid_storage_deferral():
    payload = {
        "reason": "local_storage_writer_admission",
        "deferred": True,
        "writer_start_allowed": False,
    }
    result = {"rc": 75, "payload": payload}
    assert coordinator._step_status(result) == "deferred"
    assert coordinator._step_record(result)["reason"] == payload["reason"]
    assert coordinator._step_status({**result, "timed_out": True}) == "timed_out"
    assert coordinator._step_status({**result, "rc": 2}) == "error"
    for key in payload:
        incomplete = {name: value for name, value in payload.items() if name != key}
        assert coordinator._step_status({**result, "payload": incomplete}) == "error"


@pytest.mark.parametrize("successful_waves", [0, 1])
def test_coordinator_stops_catch_up_and_reports_storage_hold(
    tmp_path, monkeypatch, successful_waves
):
    health = tmp_path / "governance/health"
    health.mkdir(parents=True)
    (health / "ingestion_storage_control_latest.json").write_text(
        json.dumps(
            {
                "backlog_relief_contract": {
                    "active": True,
                    "active_issue_ids": ["single_writer_merge_speed"],
                    "p_core_backlog_allocation_contract": {
                        "catch_up_wave_controller": {"enabled": True, "max_waves": 5}
                    },
                }
            }
        )
    )
    monkeypatch.setattr(
        coordinator,
        "writer_state_snapshot",
        lambda *a, **k: {"active": False, "running": False, "current_step": "complete"},
    )
    monkeypatch.setattr(
        coordinator.drain_src,
        "build_payload",
        lambda *a, **k: {"recommended_now": False},
    )
    monkeypatch.setattr(
        coordinator.drainer_src,
        "build_payload",
        lambda *a, **k: {
            "overall_status": "ready",
            "ready_drainer_count": 1,
            "active_drainer": {
                "name": "core_decision_drainer",
                "status": "ready",
                "live_window_safe": True,
            },
        },
    )
    monkeypatch.setattr(
        coordinator.maintenance_src,
        "_priority_retention_focus",
        lambda *a, **k: {"enabled": False},
    )
    monkeypatch.setattr(coordinator, "_refresh_surface_artifacts", lambda *a: {})
    waves = []

    def run(cmd, **kwargs):
        rc = 0
        if "backpressure_drainer_fleet.py" in " ".join(cmd):
            payload = {
                "ok": True,
                "overall_status": "handoff_requested",
                "service_request": {
                    "active": True,
                    "env_overrides": {"SQL_LINK_SERVICE_SHARDS": "trading"},
                },
            }
        elif "sql_link_shard_manager.py" in " ".join(cmd):
            waves.append(cmd)
            if len(waves) <= successful_waves:
                payload = {
                    "ok": True,
                    "merged_rows_this_cycle": 1200,
                    "merge_followup": {"followup_needed": True},
                }
            else:
                rc = 75
                payload = {
                    "ok": False,
                    "reason": "local_storage_writer_admission",
                    "deferred": True,
                    "writer_start_allowed": False,
                }
        else:
            pytest.fail(f"unexpected command {cmd}")
        return {"cmd": cmd, "rc": rc, "payload": payload}

    monkeypatch.setattr(coordinator, "_run_json_command", run)
    result = coordinator.build_payload(
        tmp_path, apply=True, poll_seconds=0, wait_timeout_seconds=1
    )
    assert len(waves) == successful_waves + 1
    assert result["overall_status"] == "deferred_storage_pressure"
    assert result["ok"] is False
    assert result["deferred"] is True
    assert result["summary"]["live_drainer_applied"] is bool(successful_waves)
    assert result["writer_follow_through_contract"]["followup_remaining"] is False
    assert "do not force" in result["recommended_actions"][0]


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("corrupt", [False, True])
def test_shards_recheck_live_storage_before_each_child(
    tmp_path, disk, monkeypatch, workers, corrupt
):
    from scripts.ops import sql_link_shard_manager as manager

    owner_receipt(tmp_path)
    monkeypatch.setattr(manager, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(manager, "_fresh_idle_shard_skip_record", lambda *a, **k: None)
    monkeypatch.setattr(
        manager,
        "_quarantine_shard_artifacts",
        lambda **k: {"triggered": bool(k.get("force_reason"))},
    )
    launched = []

    def run(cmd, **kwargs):
        launched.append(cmd)
        disk["free"] = 15
        return SimpleNamespace(
            returncode=1 if corrupt else 0,
            stdout="",
            stderr="file is not a database" if corrupt else "",
        )

    monkeypatch.setattr(manager.subprocess, "run", run)
    shards = [
        {
            "name": name,
            **{
                key: str(tmp_path / f"{name}_{key}")
                for key in (
                    "sqlite_db",
                    "state_file",
                    "health_file",
                    "journal_file",
                    "journal_events_file",
                    "invalid_log_file",
                )
            },
        }
        for name in ("trading", "runtime", "governance")
    ]
    results = manager._run_shard_links(
        shards=shards,
        link_mode="sqlite",
        sqlite_timeout_seconds=1,
        sqlite_lock_retries=0,
        sqlite_lock_retry_delay_seconds=0,
        shard_link_timeout_seconds=1,
        preprocess_workers=workers,
    )
    assert len(launched) < len(shards)
    deferred = [row for row in results if row.get("storage_deferred")]
    assert deferred
    assert all(row["rc"] in ({1, 75} if corrupt else {75}) for row in deferred)
    if corrupt:
        failures = [row for row in deferred if row["rc"] == 1]
        assert len(failures) == len(launched)
        assert all("not a database" in row["stderr_tail"] for row in failures)
    assert all(not row["health"] for row in deferred)
    assert all(not row["storage_admission"]["writer_start_allowed"] for row in deferred)


@pytest.mark.parametrize(
    "stage",
    ["after_shards", "after_shards_error", "between_merges", "after_last_merge"],
)
def test_mid_cycle_storage_hold_preserves_progress_without_completion_credit(
    tmp_path, disk, monkeypatch, capsys, stage
):
    from scripts.ops import sql_link_shard_manager as manager

    after_shards = stage.startswith("after_shards")
    prior_error = stage == "after_shards_error"
    owner_receipt(tmp_path)
    monkeypatch.setattr(manager, "PROJECT_ROOT", tmp_path)
    for name in ("SHARD_DB_ROOT", "SHARD_STATE_ROOT", "HEALTH_ROOT", "EVENT_ROOT"):
        monkeypatch.setattr(manager, name, tmp_path / name)
    for name in (
        "MAINTENANCE_STATE_PATH",
        "LATEST_HEALTH",
        "PROGRESS_HEALTH",
        "REQUEST_PATH",
    ):
        monkeypatch.setattr(manager, name, tmp_path / f"{name}.json")
    monkeypatch.setattr(
        manager, "_configured_primary_db_path", lambda value: Path(value)
    )
    monkeypatch.setattr(manager, "maintenance_hold_snapshot", lambda *a: {})
    monkeypatch.setattr(manager, "_cycle_boundary_maintenance_hold", lambda *a, **k: {})
    monkeypatch.setattr(
        manager, "_load_active_request", lambda *a: {"request_id": "keep-me"}
    )
    monkeypatch.setattr(manager, "_cycle_runtime_overrides", lambda *a: {})
    names = ["trading"] if stage == "after_last_merge" else ["trading", "runtime"]
    shards = [
        {"name": name, "sqlite_db": str(tmp_path / f"{name}.sqlite3")} for name in names
    ]
    monkeypatch.setattr(manager, "_build_shards", lambda *a: shards)
    monkeypatch.setattr(
        manager, "_apply_raw_live_priority_focus", lambda rows: (rows, {})
    )
    monkeypatch.setattr(
        manager, "_prioritize_shards_for_linking", lambda rows: (rows, {})
    )
    monkeypatch.setattr(manager, "_shard_writer_lane_contract", lambda *a, **k: {})
    monkeypatch.setattr(
        manager, "_probe_shard_merge_state", lambda **k: {"merge_required": True}
    )
    monkeypatch.setattr(
        manager,
        "_once_inline_retention_enabled",
        lambda *a: pytest.fail("maintenance after pressure"),
    )
    monkeypatch.setattr(
        manager,
        "_record_consumed_focused_request",
        lambda *a, **k: pytest.fail("request incorrectly consumed"),
    )

    def link(**kwargs):
        if after_shards:
            disk["free"] = 15
            return [
                {"shard": names[0], "rc": 1 if prior_error else 0},
                {
                    "shard": names[1],
                    "rc": 75,
                    "storage_deferred": True,
                    "storage_admission": admission.storage_admission(tmp_path),
                },
            ]
        return [{"shard": name, "rc": 0} for name in names]

    merged = []

    def merge(**kwargs):
        merged.append(kwargs["shard_name"])
        disk["free"] = 15
        return {"shard": kwargs["shard_name"], "ok": True, "jsonl_rows_inserted": 3}

    monkeypatch.setattr(manager, "_run_shard_links", link)
    monkeypatch.setattr(manager, "_merge_shard_into_primary", merge)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manager",
            "--once",
            "--json",
            "--shards",
            ",".join(names),
            "--lock-path",
            str(tmp_path / "writer.lock"),
            "--primary-db",
            str(tmp_path / "primary.sqlite3"),
        ],
    )
    assert manager.main() == (1 if prior_error else 75)
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall_status"] == ("error" if prior_error else "deferred")
    assert payload["ok"] is False
    assert payload["running"] is False
    assert payload["active_request"]["request_id"] == "keep-me"
    assert len(merged) == (0 if after_shards else 1)
    assert payload["merged_rows_this_cycle"] == len(merged) * 3
    progress = json.loads(manager.PROGRESS_HEALTH.read_text())
    assert progress["status"] == "deferred"
    assert progress["current_step"] == "storage_deferred"
    assert progress["completed_shard_count"] == (1 if after_shards else len(names))
    assert progress["pending_shards"] == ([names[1]] if after_shards else [])
    state = json.loads(manager.MAINTENANCE_STATE_PATH.read_text())
    assert state["wal_checkpoint"]["rows_since_last_run"] == len(merged) * 3
