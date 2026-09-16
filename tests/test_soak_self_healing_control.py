import json
import sys
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.ops.soak_self_healing_control as src


def test_storage_recovery_cli_rejects_reserved_alias_before_resolve(tmp_path, monkeypatch):
    alias = tmp_path / "reserved"
    alias.symlink_to("/Volumes/VIDEO")
    real_resolve = Path.resolve

    def checked_resolve(path, *args, **kwargs):
        assert not str(path).startswith((str(alias), "/Volumes/VIDEO"))
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", checked_resolve)
    with pytest.raises(SystemExit, match="2"):
        src.main(["--project-root", str(alias), "--storage-recovery-only", "--apply"])


def _write_daily(project_root: Path, *, ok: bool, failed_checks: list[str]) -> None:
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    (health / "daily_auto_verify_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "ok": ok,
                "overall_status": "ready" if ok else "blocked",
                "failed_checks": failed_checks,
            }
        ),
        encoding="utf-8",
    )


def _result(cmd: list[str], parsed: dict) -> dict:
    return {
        "command": cmd,
        "rc": 0 if parsed.get("ok", True) else 2,
        "timed_out": False,
        "duration_seconds": 0.001,
        "parsed": parsed,
        "ok": bool(parsed.get("ok", True)),
        "stdout_tail": "",
        "stderr_tail": "",
    }


def _base_fake_runner(calls: list[str], *, soak_payload: dict | None = None, daily_sequence: list[dict] | None = None):
    daily_rows = list(daily_sequence or [])
    default_soak = {
        "ok": True,
        "overall_status": "ready",
        "overall_grade": "A+",
        "safe_to_leave_unattended": True,
        "blockers": [],
        "sections": {
            "storage": {
                "current_external_free_gb": 150.0,
                "required_external_free_gb": 111.0,
                "available_margin_gb": 39.0,
            }
        },
    }
    raw_soak_rows = soak_payload if soak_payload is not None else default_soak
    soak_rows = list(raw_soak_rows) if isinstance(raw_soak_rows, list) else [raw_soak_rows]

    def _fake_run(cmd: list[str], *, project_root: Path, timeout_sec: int, env: dict[str, str]) -> dict:
        text = " ".join(str(item) for item in cmd)
        calls.append(text)
        if "daily_auto_verify.py" in text:
            parsed = daily_rows.pop(0) if daily_rows else {"ok": True, "overall_status": "ready", "failed_checks": []}
            return _result(cmd, parsed)
        if "daily_verify_auto_remediation_bot.py" in text:
            return _result(
                cmd,
                {
                    "ok": True,
                    "overall_status": "ready",
                    "resolved_checks": ["nightly_resilience_check"],
                    "unresolved_checks": [],
                },
            )
        if "promotion_quality_gate.py" in text:
            return _result(cmd, {"ok": False, "overall_status": "blocked", "failed_checks": ["promotion_quality_gate"]})
        if "unattended_soak_readiness.py" in text:
            soak = soak_rows.pop(0) if len(soak_rows) > 1 else soak_rows[0]
            return _result(cmd, soak)
        if "storage-retention-unison" in text:
            return _result(cmd, {"ok": True, "overall_status": "ready", "actions": [{"name": "bounded_cleanup"}]})
        return _result(cmd, {"ok": True, "overall_status": "ready", "status": "ready"})

    return _fake_run


def test_promotion_only_daily_failure_is_managed_without_remediation(tmp_path: Path, monkeypatch) -> None:
    _write_daily(tmp_path, ok=False, failed_checks=["promotion_quality_gate"])
    calls: list[str] = []
    monkeypatch.setattr(src, "_run_command", _base_fake_runner(calls))

    payload = src.build_payload(tmp_path, apply=True, respect_cooldowns=False)

    assert payload["ok"] is True
    assert payload["daily_verify"]["managed_failed_checks"] == ["promotion_quality_gate"]
    assert payload["daily_verify"]["repairable_failed_checks"] == []
    assert payload["daily_verify"]["remediation"]["attempted"] is False
    assert not any("daily_verify_auto_remediation_bot.py" in call for call in calls)
    assert payload["safety_contract"]["promotion_gate_autounlock_allowed"] is False


def test_soak_manages_promotion_evidence_family_daily_failures(tmp_path: Path, monkeypatch) -> None:
    managed_failures = [
        "snapshot_coverage_sentinel",
        "feature_store_manifest",
        "retrain_schema_compatibility_guard",
        "promotion_packet_builder",
        "promotion_quality_gate",
    ]
    _write_daily(tmp_path, ok=False, failed_checks=managed_failures)
    calls: list[str] = []
    monkeypatch.setattr(src, "_run_command", _base_fake_runner(calls))

    payload = src.build_payload(tmp_path, apply=True, respect_cooldowns=False)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["daily_verify"]["managed_failed_checks"] == managed_failures
    assert payload["daily_verify"]["repairable_failed_checks"] == []
    assert payload["promotion_quality"]["managed_as_evidence_lock"] is True
    assert "keep_live_money_and_promotion_locked_until_promotion_quality_gate_clears" in payload["recommended_actions"]
    assert not any("daily_verify_auto_remediation_bot.py" in call for call in calls)


def test_repairable_daily_failure_runs_remediation_and_recheck(tmp_path: Path, monkeypatch) -> None:
    _write_daily(tmp_path, ok=False, failed_checks=["nightly_resilience_check"])
    calls: list[str] = []
    monkeypatch.setattr(
        src,
        "_run_command",
        _base_fake_runner(
            calls,
            daily_sequence=[
                {"ok": False, "overall_status": "blocked", "failed_checks": ["nightly_resilience_check"]},
                {"ok": True, "overall_status": "ready", "failed_checks": []},
            ],
        ),
    )

    payload = src.build_payload(tmp_path, apply=True, respect_cooldowns=False)

    assert payload["ok"] is True
    assert payload["daily_verify"]["remediation"]["attempted"] is True
    assert payload["daily_verify"]["repairable_failed_checks"] == []
    assert any("daily_verify_auto_remediation_bot.py" in call for call in calls)
    assert sum(1 for call in calls if "daily_auto_verify.py" in call) == 2


def test_storage_soak_blocker_runs_compaction_retention_and_cold_offload(tmp_path: Path, monkeypatch) -> None:
    _write_daily(tmp_path, ok=True, failed_checks=[])
    monkeypatch.setenv("BOT_SECOND_COLD_ROOT", str(tmp_path / "BOT_COLD" / "schwab_trading_bot"))
    calls: list[str] = []
    monkeypatch.setattr(
        src,
        "_run_command",
        _base_fake_runner(
            calls,
            soak_payload={
                "ok": False,
                "overall_status": "blocked",
                "overall_grade": "C",
                "safe_to_leave_unattended": False,
                "blockers": ["storage_margin_not_30_day_ready"],
                "sections": {
                    "storage": {
                        "current_external_free_gb": 57.0,
                        "required_external_free_gb": 111.0,
                        "available_margin_gb": -54.0,
                    }
                },
            },
        ),
    )

    payload = src.build_payload(
        tmp_path,
        apply=True,
        storage_cleanup_max_delete_gb=16.0,
        storage_target_free_gb=125.0,
        respect_cooldowns=False,
    )

    retention_calls = [call for call in calls if "storage-retention-unison" in call]
    raw_compaction_calls = [call for call in calls if "raw-training-compaction" in call]
    evidence_calls = [call for call in calls if "cold-evidence-compactor" in call]
    offload_calls = [call for call in calls if "manifest-backed-offload" in call]
    assert payload["ok"] is True
    assert payload["overall_status"] == "guarded_storage_capacity"
    assert payload["storage"]["retention_attempted"] is True
    assert payload["storage"]["recovery"]["raw_compaction_attempted"] is True
    assert payload["storage"]["recovery"]["manifest_cold_offload_attempted"] is True
    assert raw_compaction_calls
    assert evidence_calls
    assert "--target-free-gb 125.0" in evidence_calls[0]
    assert "--seconds 540" in evidence_calls[0]
    assert calls.index(evidence_calls[0]) < calls.index(raw_compaction_calls[0])
    assert "--jumbo-gb 12.0" in raw_compaction_calls[0]
    assert offload_calls
    assert "--release-source-after-verify" in offload_calls[0]
    assert retention_calls
    assert "--cleanup-max-delete-gb 16.0" in retention_calls[0]
    assert "--target-free-gb 125.0" in retention_calls[0]
    assert "add_or_free_external_storage_capacity_for_30_day_soak" in payload["self_healing"]["operator_followups"]


def test_oversized_local_compatibility_cache_triggers_transactional_rebuild(tmp_path: Path, monkeypatch) -> None:
    _write_daily(tmp_path, ok=True, failed_checks=[])
    cache = tmp_path / "local_fallback_storage" / "data" / "jsonl_link.sqlite3"
    cache.parent.mkdir(parents=True)
    with cache.open("wb") as handle:
        handle.seek((2 * 1024**3) - 1)
        handle.write(b"\0")
    cold_root = tmp_path / "BOT_COLD"
    monkeypatch.setenv("BOT_SECOND_COLD_ROOT", str(cold_root))
    monkeypatch.setenv("BOT_LOGS_SQLITE_LOCAL_CACHE_REBUILD_THRESHOLD_GB", "1")
    monkeypatch.setenv("BOT_LOGS_SQLITE_LOCAL_CACHE_HARD_ENVELOPE_GB", "1.5")
    monkeypatch.setenv("BOT_LOGS_SQLITE_LOCAL_CACHE_TARGET_FREE_GB", "99999")
    calls: list[str] = []
    base_runner = _base_fake_runner(calls)

    def _runner(cmd: list[str], *, project_root: Path, timeout_sec: int, env: dict[str, str]) -> dict:
        command_text = " ".join(str(item) for item in cmd)
        if "storage_sqlite_hot_route.py" in command_text:
            calls.append(command_text)
            cache.write_bytes(b"bounded-cache")
            return _result(cmd, {"ok": True, "overall_status": "rebuilt_pruned", "reclaimed_bytes": 2 * 1024**3})
        return base_runner(cmd, project_root=project_root, timeout_sec=timeout_sec, env=env)

    monkeypatch.setattr(src, "_run_command", _runner)

    payload = src.build_payload(tmp_path, apply=True, respect_cooldowns=False)

    cache_rebuild = payload["application_memory_protection"]["compatibility_cache_rebuild"]
    assert cache_rebuild["initial"]["active"] is True
    assert cache_rebuild["attempted"] is True
    assert cache_rebuild["final"]["active"] is False
    assert cache_rebuild["transactional"] is True
    assert cache_rebuild["resumable"] is True
    assert any("storage_sqlite_hot_route.py" in call and "--rebuild-local-cache" in call for call in calls)


def test_local_storage_target_warning_triggers_bounded_storage_recovery(tmp_path: Path, monkeypatch) -> None:
    _write_daily(tmp_path, ok=True, failed_checks=[])
    calls: list[str] = []
    monkeypatch.setattr(
        src,
        "_run_command",
        _base_fake_runner(
            calls,
            soak_payload={
                "ok": True,
                "overall_status": "watch",
                "overall_grade": "A",
                "safe_to_leave_unattended": False,
                "blockers": [],
                "warnings": ["local_hot_storage_below_unattended_target"],
                "sections": {
                    "storage": {
                        "current_external_free_gb": 150.0,
                        "required_external_free_gb": 111.0,
                        "available_margin_gb": 39.0,
                    }
                },
            },
        ),
    )

    payload = src.build_payload(
        tmp_path,
        apply=True,
        storage_target_free_gb=125.0,
        respect_cooldowns=False,
    )

    retention_calls = [call for call in calls if "storage-retention-unison" in call]
    assert retention_calls
    assert "--target-free-gb 125.0" in retention_calls[0]
    assert payload["storage"]["retention_attempted"] is True
    assert any("cold-evidence-compactor --apply --target-free-gb 125.0" in call for call in calls)


def test_ingestion_soak_blocker_runs_bounded_repair_and_rechecks(tmp_path: Path, monkeypatch) -> None:
    _write_daily(tmp_path, ok=True, failed_checks=[])
    calls: list[str] = []
    blocked_soak = {
        "ok": False,
        "overall_status": "blocked",
        "overall_grade": "B",
        "safe_to_leave_unattended": False,
        "blockers": ["ingestion_soak_contract_not_ready"],
        "sections": {
            "storage": {
                "current_external_free_gb": 728.0,
                "required_external_free_gb": 111.0,
                "available_margin_gb": 617.0,
            }
        },
    }
    ready_soak = {
        "ok": True,
        "overall_status": "ready",
        "overall_grade": "A+",
        "safe_to_leave_unattended": True,
        "blockers": [],
        "sections": {
            "storage": {
                "current_external_free_gb": 728.0,
                "required_external_free_gb": 111.0,
                "available_margin_gb": 617.0,
            }
        },
    }
    monkeypatch.setattr(
        src,
        "_run_command",
        _base_fake_runner(calls, soak_payload=[blocked_soak, ready_soak]),
    )

    payload = src.build_payload(tmp_path, apply=True, respect_cooldowns=False)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["safe_to_leave_unattended"] is True
    assert payload["ingestion_soak_repair"]["attempted"] is True
    assert payload["ingestion_soak_repair"]["blockers"] == []
    assert any("storage-transition-coordinator" in call for call in calls)
    assert any("storage-backpressure-autopilot" in call and "--quick-bounded" in call for call in calls)
    assert any("ingestion_storage_control.py" in call for call in calls)


def test_critical_local_disk_headroom_runs_bounded_application_memory_recovery(tmp_path: Path, monkeypatch) -> None:
    _write_daily(tmp_path, ok=True, failed_checks=[])
    monkeypatch.setenv("BOT_SECOND_COLD_ROOT", str(tmp_path / "VIDEO" / "schwab_trading_bot_cold"))
    calls: list[str] = []
    memory_calls = 0

    def fake_run(cmd: list[str], *, project_root: Path, timeout_sec: int, env: dict[str, str]) -> dict:
        nonlocal memory_calls
        text = " ".join(str(item) for item in cmd)
        calls.append(text)
        if "memory_efficiency_control.py" in text:
            memory_calls += 1
            if memory_calls == 1:
                return _result(
                    cmd,
                    {
                        "ok": False,
                        "overall_status": "blocked",
                        "reasons": ["local_disk_swap_temp_headroom_low", "memory_pressure_red"],
                        "memory_snapshot": {
                            "memory_pressure_state": "red",
                            "memory_pressure_kind": "disk_swap_headroom",
                            "memory_free_pct": 83.0,
                            "swap_used_gb": 1.6,
                            "local_disk_free_gb": 0.25,
                        },
                        "local_disk_headroom_contract": {
                            "active": True,
                            "severity": "critical",
                            "local_disk_free_gb": 0.25,
                            "warning_free_gb": 32.0,
                            "critical_free_gb": 8.0,
                        },
                    },
                )
            return _result(
                cmd,
                {
                    "ok": True,
                    "overall_status": "ready",
                    "reasons": ["memory_headroom_ok"],
                    "memory_snapshot": {
                        "memory_pressure_state": "green",
                        "memory_pressure_kind": "none",
                        "memory_free_pct": 83.0,
                        "swap_used_gb": 1.6,
                        "local_disk_free_gb": 96.0,
                    },
                    "local_disk_headroom_contract": {
                        "active": False,
                        "severity": "clear",
                        "local_disk_free_gb": 96.0,
                        "warning_free_gb": 32.0,
                        "critical_free_gb": 8.0,
                    },
                },
            )
        if "sql_queue_retention.py" in text:
            return _result(cmd, {"ok": True, "deleted_acked_rows": 500000})
        if "unattended_soak_readiness.py" in text:
            return _result(
                cmd,
                {
                    "ok": True,
                    "overall_status": "ready",
                    "overall_grade": "A+",
                    "safe_to_leave_unattended": True,
                    "blockers": [],
                    "sections": {
                        "storage": {
                            "current_external_free_gb": 400.0,
                            "required_external_free_gb": 125.0,
                            "available_margin_gb": 275.0,
                        }
                    },
                },
            )
        if "promotion_quality_gate.py" in text:
            return _result(cmd, {"ok": True, "overall_status": "ready", "failed_checks": []})
        return _result(cmd, {"ok": True, "overall_status": "ready", "status": "ready"})

    monkeypatch.setattr(src, "_run_command", fake_run)

    payload = src.build_payload(tmp_path, apply=True, respect_cooldowns=False)

    protection = payload["application_memory_protection"]
    assert payload["ok"] is True
    assert protection["recovery_attempted"] is True
    assert protection["initial"]["critical"] is True
    assert protection["final"]["active"] is False
    assert protection["acknowledged_queue_rows_deleted"] == 500000
    assert any("storage-transition-coordinator --transition-mode external --apply" in call for call in calls)
    queue_call = next(call for call in calls if "sql_queue_retention.py" in call)
    assert "--acked-hours 1" in queue_call
    assert "--vacuum" not in queue_call
    assert any("governance-telemetry-compactor --apply" in call for call in calls)
    assert any("deep-cold-storage-layer --apply --adaptive --move-to-second-cold" in call for call in calls)
    assert any("storage-pressure-clearance --apply" in call for call in calls)
    assert memory_calls == 2


def test_storage_recovery_starts_at_writer_pause_threshold(monkeypatch):
    monkeypatch.setenv("BOT_LOCAL_STORAGE_PRESSURE_FREE_GB", "64")
    result = src._local_disk_headroom_recovery_contract(
        {
            "local_disk_headroom_contract": {
                "local_disk_free_gb": 40,
                "warning_free_gb": 32,
                "critical_free_gb": 8,
            }
        }
    )
    assert result["active"] and result["storage_pressure_active"]
    assert not result["critical"]
    assert result["warning_free_gb"] == 64
    clear = src._local_disk_headroom_recovery_contract(
        {"memory_snapshot": {"local_disk_free_gb": 70}}
    )
    assert not clear["active"]


def test_cold_archive_configuration_rejects_protected_volume_and_uses_safe_fallback(tmp_path: Path) -> None:
    external = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    external.mkdir(parents=True)
    env = {
        "BOT_SECOND_COLD_ROOT": "/Volumes/VIDEO/schwab_trading_bot_cold",
        "BOT_LOGS_EXTERNAL_PROJECT_ROOT": str(external),
    }

    payload = src._configure_cold_archive_env(env, apply=False)

    assert payload["configured"] is True
    assert payload["auto_selected"] is True
    assert payload["path"] == str(external / "cold_archive")
    assert env["BOT_SECOND_COLD_ROOT"] == str(external / "cold_archive")
    assert env["BOT_NEVER_TOUCH_VIDEO"] == "1"


def test_cold_archive_configuration_fails_closed_without_safe_fallback() -> None:
    env = {"BOT_SECOND_COLD_ROOT": "/Volumes/VIDEO/schwab_trading_bot_cold"}

    payload = src._configure_cold_archive_env(env, apply=False)

    assert payload["configured"] is False
    assert payload["reason"] == "non_protected_second_cold_root_not_configured"
    assert "BOT_SECOND_COLD_ROOT" not in env


def test_cold_archive_rejects_protected_alias_before_metadata(tmp_path, monkeypatch):
    alias = tmp_path / "forbidden_alias"
    alias.symlink_to("/Volumes/VIDEO")
    original_exists = Path.exists

    def checked_exists(path):
        assert not str(path).startswith((str(alias), "/Volumes/VIDEO"))
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", checked_exists)
    monkeypatch.setattr(src, "PROJECT_ROOT", tmp_path)
    payload = src._configure_cold_archive_env(
        {"BOT_LOGS_EXTERNAL_PROJECT_ROOT": str(alias / "bot")}, apply=True
    )
    assert payload["route_state"] == "deferred_until_external_returns"


def _storage_recovery_fixture(tmp_path, monkeypatch, free_gb=40):
    from types import SimpleNamespace

    monkeypatch.setattr(
        src.shutil, "disk_usage", lambda path: SimpleNamespace(free=free_gb * 1024**3)
    )
    monkeypatch.setattr(
        src, "maintenance_hold_snapshot", lambda root: {"active": False}
    )
    monkeypatch.setattr(src.os, "getloadavg", lambda: (0, 0, 0))
    monkeypatch.setattr(
        src,
        "_configure_cold_archive_env",
        lambda env, apply: {
            "configured": True,
            "redundancy_ready": True,
            "path": str(tmp_path / "cold"),
        },
    )
    calls = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        return _result(
            cmd,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "ok": True,
                "overall_status": "ready",
                "memory_snapshot": {
                    "memory_pressure_state": "green",
                    "memory_free_pct": 50,
                    "swap_used_gb": 2,
                },
            },
        )

    monkeypatch.setattr(src, "_run_command", runner)
    return calls


@pytest.mark.parametrize("lease_state", ["fresh", "stale", "denied", "missing"])
def test_adaptive_pressure_entry_is_compression_only(tmp_path, monkeypatch, lease_state):
    calls = _storage_recovery_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(src.os, "cpu_count", lambda: 10)
    monkeypatch.setattr(src.os, "getloadavg", lambda: (7.5, 7.5, 7.5))
    monkeypatch.setattr(src, "_configure_cold_archive_env",
                        lambda *a, **kw: pytest.fail("extra CPU admission cannot invoke external/heavy recovery"))
    now = datetime.now(timezone.utc).isoformat()
    lease = {
        "schema_version": 1, "timestamp_utc": now, "source_timestamp_utc": now,
        "input_evidence_ready": True, "workloads": {"storage_recovery": {"admitted": True}},
    }
    if lease_state == "stale":
        lease["source_timestamp_utc"] = "2000-01-01T00:00:00Z"
    elif lease_state == "denied":
        lease["workloads"]["storage_recovery"]["admitted"] = False
    if lease_state != "missing":
        src.write_payload(tmp_path / "governance/health/runtime_throttle_control_latest.json",
                    {"workload_admission": lease})
    payload = src.build_storage_recovery_payload(tmp_path, apply=True)
    if lease_state != "fresh":
        assert not payload["admitted"]
        assert not calls
        return
    assert payload["admitted"] and payload["adaptive_compression_only"]
    assert not payload["ok"] and not payload["live_execution_authority"]
    assert [cmd[1] for cmd in calls[1:]] == [
        "runtime-training-snapshot", "cold-evidence-compactor", "governance-lifecycle-compactor", "local-storage-reserve-guard"
    ]
    lifecycle = calls[3]
    assert lifecycle[lifecycle.index("--max-files") + 1] == "32"
    assert lifecycle[lifecycle.index("--seconds") + 1] == "180"
    assert "--include-current-day" not in lifecycle


@pytest.mark.parametrize("load", [float("nan"), float("inf"), -1])
def test_invalid_outer_recovery_load_cannot_admit(tmp_path, monkeypatch, load):
    calls = _storage_recovery_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(src.os, "getloadavg", lambda: (load, load, load))
    payload = src.build_storage_recovery_payload(tmp_path, apply=True)
    assert not payload["admitted"] and not calls


def _memory_result(**snapshot_overrides):
    return _result(
        [],
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "overall_status": "blocked",
            "reasons": ["storage_pressure_critical"],
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_free_pct": 89,
                "swap_used_gb": 4.75,
                **snapshot_overrides,
            },
        },
    )


def test_valid_blocked_assessment_does_not_exhaust_repair_budget(tmp_path, monkeypatch):
    _storage_recovery_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        src, "_run_command", lambda *a, **kw: _memory_result(memory_free_pct=20)
    )
    for _ in range(4):
        payload = src.build_storage_recovery_payload(tmp_path, apply=True)
        assert not payload["admitted"]
        assert len(payload["steps"]) == 1
        assert payload["steps"][0]["executed"]
        assert payload["steps"][0]["ok"]
    step = src._load_state(tmp_path)["steps"][src.STORAGE_MEMORY_STEP]
    assert step["failure_count"] == 0
    assert not step["admission_ready"]
    assert step["last_status"] == "blocked"


@pytest.mark.parametrize("valid", [True, False])
def test_legacy_observation_circuit_gets_one_fresh_revalidation(
    tmp_path, monkeypatch, valid
):
    _storage_recovery_fixture(tmp_path, monkeypatch)
    prior = {
        "failure_count": 3,
        "last_rc": 2,
        "last_status": "blocked",
        "circuit_until_utc": (
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat(),
    }
    src._write_state(tmp_path, {"steps": {src.STORAGE_MEMORY_STEP: prior}})
    result = _memory_result()
    if not valid:
        result["parsed"].pop("timestamp_utc")
    monkeypatch.setattr(src, "_run_command", lambda *a, **kw: result)
    first = src.build_storage_recovery_payload(tmp_path, apply=True)
    assert first["admitted"] is valid
    step = src._load_state(tmp_path)["steps"][src.STORAGE_MEMORY_STEP]
    assert step["observation_contract_version"] == 1
    assert step["legacy_circuit_revalidation"]["previous_state"] == prior
    assert step["failure_count"] == (0 if valid else 4)
    if not valid:
        second = src.build_storage_recovery_payload(tmp_path, apply=True)
        assert not second["steps"][0]["executed"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("memory_free_pct", None),
        ("memory_free_pct", True),
        ("memory_free_pct", "89"),
        ("memory_free_pct", float("nan")),
        ("memory_free_pct", float("inf")),
        ("memory_free_pct", 101),
        ("swap_used_gb", -1),
        ("swap_used_gb", False),
        ("swap_used_gb", float("nan")),
        ("memory_pressure_state", "unknown"),
        ("swap_used_gb", 10**1000),
        ("memory_pressure_state", ["green"]),
    ],
)
def test_memory_observation_rejects_invalid_metrics(field, value):
    assert not src._storage_memory_observation(_memory_result(**{field: value}))["ok"]


@pytest.mark.parametrize(
    "timestamp",
    [
        None,
        "garbage",
        "2026-09-08T12:00:00",
        "2000-01-01T00:00:00Z",
        "2100-01-01T00:00:00Z",
    ],
)
def test_memory_observation_requires_fresh_aware_timestamp(timestamp):
    result = _memory_result()
    result["parsed"]["timestamp_utc"] = timestamp
    assert not src._storage_memory_observation(result)["ok"]


@pytest.mark.parametrize(
    "field,value", [("rc", 1), ("rc", 124), ("rc", False), ("timed_out", True)]
)
def test_memory_observation_requires_completed_assessment(field, value):
    result = _memory_result()
    result[field] = value
    assert not src._storage_memory_observation(result)["ok"]


def test_storage_recovery_only_is_bounded_and_does_not_claim_complete(
    tmp_path, monkeypatch
):
    calls = _storage_recovery_fixture(tmp_path, monkeypatch)
    payload = src.build_storage_recovery_payload(tmp_path, apply=True)
    assert payload["admitted"]
    assert not payload["ok"]
    assert not payload["live_execution_authority"]
    assert not payload["heavy_maintenance_allowed"]
    assert calls[1][1] == "runtime-training-snapshot"
    assert "--cleanup-abandoned-builds" in calls[1] and "--apply-cleanup" in calls[1]
    calls = [cmd for cmd in calls if "runtime-training-snapshot" not in cmd]
    assert len(calls) == 7
    assert "memory_efficiency_control.py" in calls[0][1]
    assert [cmd[1] for cmd in calls[1:]] == [
        "cold-evidence-compactor",
        "governance-lifecycle-compactor",
        "governance-telemetry-compactor",
        "cold-archive-compactor",
        "deep-cold-storage-layer",
        "local-storage-reserve-guard",
    ]
    assert calls[1][calls[1].index("--seconds") + 1] == "840"
    assert calls[2][calls[2].index("--seconds") + 1] == "180"
    assert "--include-current-day" not in calls[2]
    assert calls[4][calls[4].index("--max-raw-gb") + 1] == "4"
    assert calls[4][calls[4].index("--filesystem-timeout-seconds") + 1] == "1200"
    assert calls[4][calls[4].index("--maintenance-hold-ttl-seconds") + 1] == "1320"
    assert calls[4][calls[4].index("--filesystem-compressor") + 1] == "auto"
    assert calls[5][calls[5].index("--destination-reserve-gb") + 1] == "125"
    assert "--no-include-local-quarantine" in calls[5]
    assert "--include-registry-backups" in calls[5]
    assert "--closed-history-min-age-hours" not in calls[5]
    assert "--reserve-only" in calls[6]
    assert "--skip-governor-reconcile" in calls[6]


def _disk_only_memory_result():
    result = _memory_result(
        memory_pressure_state="yellow", memory_pressure_kind="disk_swap_headroom"
    )
    result["parsed"].update(
        input_evidence_ready=True,
        storage_recovery_memory_observation={
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input_evidence_ready": True,
            "memory_pressure_state": "yellow",
            "memory_pressure_kind": "disk_swap_headroom",
            "memory_pressure_reasons": ["local_disk_swap_headroom_gb:17<32"],
            "memory_free_pct": 91,
            "swap_used_gb": 6.9,
            "compressor_gb": 0.23,
            "pages_throttled": 0,
        },
    )
    return result


def test_pressure_recovery_keeps_offload_separate_from_apfs_compression(tmp_path, monkeypatch):
    calls = _storage_recovery_fixture(tmp_path, monkeypatch)
    offload = str(tmp_path / "separate_archive")
    monkeypatch.setenv("BOT_DEEP_COLD_OFFLOAD_ROOT", offload)
    monkeypatch.setenv("BOT_DEEP_COLD_MAX_MOVE_FILES", "9999")
    monkeypatch.setenv("BOT_DEEP_COLD_MIN_SIZE_MB", "0.1")
    src.build_storage_recovery_payload(tmp_path, apply=True)
    compression = next(cmd for cmd in calls if "cold-archive-compactor" in cmd)
    move = next(cmd for cmd in calls if "deep-cold-storage-layer" in cmd)
    assert compression[compression.index("--archive-root") + 1] != offload
    assert move[move.index("--second-cold-root") + 1] == offload
    assert move[move.index("--max-move-files") + 1] == "256"
    assert float(move[move.index("--min-size-mb") + 1]) == 1
    assert move[move.index("--destination-reserve-gb") + 1] == "125"


def test_disk_only_pressure_admits_only_bounded_storage_recovery(tmp_path, monkeypatch):
    calls = _storage_recovery_fixture(tmp_path, monkeypatch)
    original_runner = src._run_command

    def runner(cmd, **kwargs):
        if "memory_efficiency_control.py" in cmd[1]:
            calls.append(cmd)
            return _disk_only_memory_result()
        return original_runner(cmd, **kwargs)

    monkeypatch.setattr(src, "_run_command", runner)
    payload = src.build_storage_recovery_payload(tmp_path, apply=True)
    assert payload["admitted"]
    assert not payload["ok"]
    assert not payload["heavy_maintenance_allowed"]
    assert not payload["live_execution_authority"]
    assert len(calls) == 8
    assert "--cleanup-abandoned-builds" in calls[1]
    assert (
        payload["steps"][0]["observation_reason"]
        == "disk_only_pressure_memory_admitted"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("timestamp_utc", "2000-01-01T00:00:00Z"),
        ("timestamp_utc", "2100-01-01T00:00:00Z"),
        ("timestamp_utc", "2026-09-13T20:00:00"),
        ("input_evidence_ready", False),
        ("input_evidence_ready", "true"),
        ("memory_pressure_state", "red"),
        ("memory_pressure_kind", "mixed"),
        ("memory_pressure_reasons", []),
        (
            "memory_pressure_reasons",
            ["local_disk_swap_headroom_gb:17<32", "free_pct:3<8"],
        ),
        ("memory_free_pct", 84),
        ("memory_free_pct", True),
        ("memory_free_pct", float("nan")),
        ("swap_used_gb", 8.1),
        ("swap_used_gb", None),
        ("compressor_gb", 1.1),
        ("compressor_gb", "0.23"),
        ("pages_throttled", 1),
        ("pages_throttled", False),
        ("memory_free_pct", 10**1000),
    ],
)
def test_disk_only_admission_rejects_real_or_unverified_pressure(field, value):
    result = _disk_only_memory_result()
    result["parsed"]["storage_recovery_memory_observation"][field] = value
    assert not src._storage_memory_observation(result)["admission_ready"]


def test_disk_only_admission_requires_raw_evidence_and_completed_observation():
    result = _disk_only_memory_result()
    result["parsed"].pop("storage_recovery_memory_observation")
    assert not src._storage_memory_observation(result)["admission_ready"]
    result = _disk_only_memory_result()
    result["timed_out"] = True
    assert not src._storage_memory_observation(result)["admission_ready"]


def test_quick_storage_recovery_is_compression_only_and_preserves_bounds(
    tmp_path, monkeypatch
):
    calls = _storage_recovery_fixture(tmp_path, monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("quick recovery must not inspect offload routes")

    monkeypatch.setattr(src, "_configure_cold_archive_env", forbidden)
    original = src._run_command
    timeouts = []

    def runner(cmd, **kwargs):
        timeouts.append(kwargs["timeout_sec"])
        return original(cmd, **kwargs)

    monkeypatch.setattr(src, "_run_command", runner)
    payload = src.build_storage_recovery_payload(
        tmp_path, apply=True, quick_bounded=True
    )
    assert payload["quick_bounded"]
    assert payload["shared_deadline_seconds"] == 90
    assert not payload["ok"]
    assert all(timeout <= 30 for timeout in timeouts)
    cleanup = next(cmd for cmd in calls if "runtime-training-snapshot" in cmd)
    assert "--cleanup-abandoned-builds" in cleanup and "--apply-cleanup" in cleanup
    assert cleanup[cleanup.index("--max-runtime-seconds") + 1] == "25"
    compactors = [cmd for cmd in calls if any("compactor" in str(part) for part in cmd)]
    assert len(compactors) == 2
    for cmd in compactors:
        assert cmd[cmd.index("--max-files") + 1] == "4"
        assert cmd[cmd.index("--seconds") + 1] == "25"
    assert not any(
        "--force" in part or "--include-current-day" in part
        for cmd in calls
        for part in cmd
    )


def test_quick_storage_recovery_reconciles_owner_after_external_relief(
    tmp_path, monkeypatch
):
    calls = _storage_recovery_fixture(tmp_path, monkeypatch, free_gb=80)
    payload = src.build_storage_recovery_payload(
        tmp_path, apply=True, quick_bounded=True
    )
    assert payload["ok"]
    assert len(calls) == 1
    assert "local-storage-reserve-guard" in calls[0]
    assert "--reserve-only" in calls[0]
    assert "--skip-governor-reconcile" in calls[0]


def test_quick_storage_recovery_keeps_existing_cooldowns_and_holds(
    tmp_path, monkeypatch
):
    from datetime import timedelta

    calls = _storage_recovery_fixture(tmp_path, monkeypatch)
    until = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    monkeypatch.setattr(
        src,
        "_load_state",
        lambda root: {
            "steps": {
                name: {"cooldown_until_utc": until}
                for name in (
                    "local_disk_cold_evidence_compaction",
                    "local_disk_lifecycle_backup_compaction",
                )
            }
        },
    )
    src.build_storage_recovery_payload(tmp_path, apply=True, quick_bounded=True)
    assert not any("compactor" in part for cmd in calls for part in cmd)
    calls.clear()
    monkeypatch.setattr(src, "maintenance_hold_snapshot", lambda root: {"active": True})
    payload = src.build_storage_recovery_payload(
        tmp_path, apply=True, quick_bounded=True
    )
    assert payload["reason"] == "existing_maintenance_hold"
    assert calls == []


def test_quick_storage_recovery_requires_recovery_only_mode():
    with pytest.raises(SystemExit) as exc:
        src.main(["--quick-storage-recovery"])
    assert exc.value.code == 2


@pytest.mark.parametrize("args", [
    ["--rebuild-reserve"],
    ["--storage-recovery-only", "--rebuild-reserve", "--quick-storage-recovery"],
])
def test_proactive_recovery_requires_its_dedicated_mode(args):
    with pytest.raises(SystemExit) as exc:
        src.main(args)
    assert exc.value.code == 2


def test_storage_recovery_only_noop_and_read_only_do_not_run_repairs(
    tmp_path, monkeypatch
):
    calls = _storage_recovery_fixture(tmp_path, monkeypatch, free_gb=80)
    assert src.build_storage_recovery_payload(tmp_path, apply=True)["ok"]
    assert not calls
    calls = _storage_recovery_fixture(tmp_path, monkeypatch, free_gb=40)
    assert not src.build_storage_recovery_payload(tmp_path, apply=False)["ok"]
    assert not calls


def test_proactive_recovery_starts_before_writer_pressure_and_keeps_floors(tmp_path, monkeypatch):
    calls = _storage_recovery_fixture(tmp_path, monkeypatch, free_gb=80)
    monkeypatch.setenv("BOT_LOCAL_STORAGE_TARGET_FREE_GB", "125")
    monkeypatch.setenv("SOAK_SELF_HEAL_STORAGE_TARGET_FREE_GB", "135")
    payload = src.build_storage_recovery_payload(tmp_path, apply=True, rebuild_reserve=True)
    assert payload["admitted"]
    assert not payload["ok"]
    assert payload["pressure_free_gb"] == 64
    assert payload["recovery_trigger_free_gb"] == 125
    assert payload["recovery_target_free_gb"] == 135
    compactor = next(cmd for cmd in calls if "cold-evidence-compactor" in cmd)
    assert compactor[compactor.index("--target-free-gb") + 1] == "135.0"
    assert not payload["heavy_maintenance_allowed"]


def test_proactive_recovery_does_not_churn_inside_hysteresis_band(tmp_path, monkeypatch):
    calls = _storage_recovery_fixture(tmp_path, monkeypatch, free_gb=130)
    monkeypatch.setenv("BOT_LOCAL_STORAGE_TARGET_FREE_GB", "125")
    monkeypatch.setenv("SOAK_SELF_HEAL_STORAGE_TARGET_FREE_GB", "135")
    payload = src.build_storage_recovery_payload(tmp_path, apply=True, rebuild_reserve=True)
    assert payload["ok"] and not payload["admitted"]
    assert not calls


def test_storage_recovery_only_respects_hold_load_and_unknown_memory(
    tmp_path, monkeypatch
):
    calls = _storage_recovery_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(src, "maintenance_hold_snapshot", lambda root: {"active": True})
    assert (
        src.build_storage_recovery_payload(tmp_path, apply=True)["reason"]
        == "existing_maintenance_hold"
    )
    assert not calls
    monkeypatch.setattr(
        src, "maintenance_hold_snapshot", lambda root: {"active": False}
    )
    monkeypatch.setattr(src.os, "getloadavg", lambda: (1000, 1000, 1000))
    assert (
        src.build_storage_recovery_payload(tmp_path, apply=True)["reason"]
        == "host_load_above_recovery_budget"
    )
    assert not calls
    monkeypatch.setattr(src.os, "getloadavg", lambda: (0, 0, 0))
    monkeypatch.setattr(
        src, "_run_command", lambda cmd, **kw: _result(cmd, {"ok": False})
    )
    payload = src.build_storage_recovery_payload(tmp_path, apply=True)
    assert payload["reason"] == "memory_admission_not_ready"
    assert len(payload["steps"]) == 1


def test_storage_recovery_launcher_precedes_heavy_gate():
    launcher = (
        PROJECT_ROOT / "scripts/ops/run_soak_self_healing_launchd.sh"
    ).read_text()
    assert launcher.index("--storage-recovery-only") < launcher.index(
        "run_guarded_maintenance.sh"
    )
    assert "MAINTENANCE_SLOT_DEFER_OUTSIDE_QUIET_WINDOW=0" not in launcher
    assert "--storage-recovery-only --rebuild-reserve --apply" in launcher


def test_storage_recovery_shared_deadline_reserves_final_assessment(tmp_path, monkeypatch):
    calls = _storage_recovery_fixture(tmp_path, monkeypatch)
    clock = [0]
    monkeypatch.setattr(src.time, "monotonic", lambda: clock[0])
    runner = src._run_command
    def slow(cmd, **kwargs):
        result = runner(cmd, **kwargs)
        if "cold-evidence-compactor" in cmd:
            clock[0] = 1750
        return result
    monkeypatch.setattr(src, "_run_command", slow)
    payload = src.build_storage_recovery_payload(tmp_path, apply=True)
    assert [cmd[1] for cmd in calls[1:]] == ["runtime-training-snapshot", "cold-evidence-compactor", "local-storage-reserve-guard"]
    assert payload["reason"] == "storage_recovery_deadline"
    assert not payload["ok"]


def test_storage_target_stops_more_compression_but_reconciles_reserve(tmp_path, monkeypatch):
    calls = _storage_recovery_fixture(tmp_path, monkeypatch)
    runner = src._run_command
    def recovered(cmd, **kwargs):
        result = runner(cmd, **kwargs)
        if "cold-evidence-compactor" in cmd:
            monkeypatch.setattr(src.shutil, "disk_usage", lambda p: SimpleNamespace(free=80*1024**3))
        return result
    monkeypatch.setattr(src, "_run_command", recovered)
    payload = src.build_storage_recovery_payload(tmp_path, apply=True)
    assert payload["ok"]
    assert [cmd[1] for cmd in calls[1:]] == ["runtime-training-snapshot", "cold-evidence-compactor", "local-storage-reserve-guard"]


def test_sqlite_compression_requires_its_complete_verification_window(tmp_path, monkeypatch):
    calls = _storage_recovery_fixture(tmp_path, monkeypatch)
    clock = [0]
    monkeypatch.setattr(src.time, "monotonic", lambda: clock[0])
    runner = src._run_command

    def slow(cmd, **kwargs):
        result = runner(cmd, **kwargs)
        if "cold-evidence-compactor" in cmd:
            clock[0] = 700
        return result

    monkeypatch.setattr(src, "_run_command", slow)
    payload = src.build_storage_recovery_payload(tmp_path, apply=True)
    assert not any("cold-archive-compactor" in cmd for cmd in calls)
    deferred = next(
        row
        for row in payload["steps"]
        if row["name"] == "local_disk_cold_sqlite_compression"
    )
    assert deferred["executed"] is False
    assert deferred["reason"] == "insufficient_complete_compression_window"
    assert deferred["required_seconds"] == 1300
    assert any("local-storage-reserve-guard" in cmd for cmd in calls)


def test_cold_archive_configuration_defers_locally_when_external_root_is_offline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(src, "PROJECT_ROOT", tmp_path)
    unavailable_external = tmp_path / "offline-volume" / "schwab_trading_bot"
    env = {"BOT_LOGS_EXTERNAL_PROJECT_ROOT": str(unavailable_external)}

    payload = src._configure_cold_archive_env(env, apply=False)

    expected = tmp_path / "local_fallback_storage" / "cold_archive_deferred"
    assert payload["configured"] is True
    assert payload["path"] == str(expected)
    assert payload["route_state"] == "deferred_until_external_returns"
    assert payload["redundancy_ready"] is False
    assert payload["hot_path_blocked"] is False
    assert payload["auto_failback_enabled"] is True
    assert env["BOT_SECOND_COLD_ROOT"] == str(expected)


def test_cold_archive_configuration_defers_explicit_unmounted_external_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(src, "PROJECT_ROOT", tmp_path)
    requested = "/Volumes/OFFLINE_TEST_VOLUME/schwab_trading_bot/cold_archive"
    env = {"BOT_SECOND_COLD_ROOT": requested}

    payload = src._configure_cold_archive_env(env, apply=False)

    expected = tmp_path / "local_fallback_storage" / "cold_archive_deferred"
    assert payload["requested_path"] == requested
    assert payload["path"] == str(expected)
    assert payload["route_state"] == "deferred_until_external_returns"
    assert payload["redundancy_ready"] is False
    assert env["BOT_SECOND_COLD_ROOT"] == str(expected)


def test_stale_profitability_runtime_controls_are_refreshed_and_rechecked(tmp_path: Path, monkeypatch) -> None:
    _write_daily(tmp_path, ok=True, failed_checks=[])
    calls: list[str] = []
    runtime_guard_rows = [
        {
            "ok": False,
            "overall_status": "degraded",
            "failed_guards": ["soak_hot_artifact_freshness_contract"],
            "hard_failed_guard_count": 0,
            "regression_guards": [
                {
                    "name": "soak_hot_artifact_freshness_contract",
                    "ok": False,
                    "actual": {
                        "stale_artifacts": [
                            {
                                "name": "paper_runtime_profitability_controls",
                                "age_minutes": 523.0,
                                "max_age_minutes": 120.0,
                            }
                        ]
                    },
                }
            ],
        },
        {
            "ok": True,
            "overall_status": "ready",
            "failed_guards": [],
            "hard_failed_guard_count": 0,
            "regression_guards": [],
        },
    ]

    def fake_run(cmd: list[str], *, project_root: Path, timeout_sec: int, env: dict[str, str]) -> dict:
        text = " ".join(str(item) for item in cmd)
        calls.append(text)
        if "runtime_paper_regression_guard.py" in text:
            parsed = runtime_guard_rows.pop(0) if runtime_guard_rows else {"ok": True, "overall_status": "ready"}
            return _result(cmd, parsed)
        if "paper-profitability-control" in text:
            return _result(
                cmd,
                {
                    "ok": True,
                    "overall_status": "protective_tightening",
                    "raw_profitability_grade": "D",
                    "controlled_profitability_grade": "A+",
                },
            )
        if "daily_auto_verify.py" in text:
            return _result(cmd, {"ok": True, "overall_status": "ready", "failed_checks": []})
        if "promotion_quality_gate.py" in text:
            return _result(cmd, {"ok": False, "overall_status": "blocked", "failed_checks": ["promotion_quality_gate"]})
        if "unattended_soak_readiness.py" in text:
            return _result(
                cmd,
                {
                    "ok": True,
                    "overall_status": "ready",
                    "overall_grade": "A+",
                    "safe_to_leave_unattended": True,
                    "blockers": [],
                    "sections": {
                        "storage": {
                            "current_external_free_gb": 150.0,
                            "required_external_free_gb": 111.0,
                            "available_margin_gb": 39.0,
                        }
                    },
                },
            )
        return _result(cmd, {"ok": True, "overall_status": "ready", "status": "ready"})

    monkeypatch.setattr(src, "_run_command", fake_run)

    payload = src.build_payload(tmp_path, apply=True, respect_cooldowns=False)

    assert payload["ok"] is True
    assert payload["profitability_control_refresh"]["attempted"] is True
    assert payload["profitability_control_refresh"]["controlled_profitability_grade"] == "A+"
    assert any("paper-profitability-control --apply --json" in call for call in calls)
    assert sum(1 for call in calls if "runtime_paper_regression_guard.py" in call) == 2


def test_production_hardening_cascade_accepts_managed_live_money_lock(tmp_path: Path, monkeypatch) -> None:
    _write_daily(tmp_path, ok=True, failed_checks=[])
    calls: list[str] = []

    def fake_run(cmd: list[str], *, project_root: Path, timeout_sec: int, env: dict[str, str]) -> dict:
        text = " ".join(str(item) for item in cmd)
        calls.append(text)
        if "promotion_packet_builder.py" in text:
            return _result(
                cmd,
                {
                    "ok": False,
                    "promotion_scope": {"target_count": 0, "trained_bot_ids": [], "failure_count": 0},
                    "committee_packet_seed_ready": True,
                    "replayability_contract": {"hash_bundle_complete": True, "exact_replay_ready": True},
                    "gate_results": {"training_success_confirmed": True, "retrain_schema_compatibility_ok": True},
                },
            )
        if "promotion_quality_gate.py" in text:
            return _result(cmd, {"ok": True, "overall_status": "ready", "failed_checks": []})
        if "live-money-readiness" in text:
            return _result(
                cmd,
                {
                    "ok": False,
                    "overall_status": "blocked",
                    "live_money_locked": True,
                    "blocking_reasons": ["target_window_not_complete"],
                    "grade_summary": {
                        "required_section_count": 14,
                        "ready_required_section_count": 14,
                        "below_floor_sections": [],
                        "not_ready_sections": [],
                    },
                },
            )
        if "unattended_soak_readiness.py" in text:
            return _result(
                cmd,
                {
                    "ok": True,
                    "overall_status": "ready",
                    "overall_grade": "A+",
                    "safe_to_leave_unattended": True,
                    "blockers": [],
                    "sections": {
                        "storage": {
                            "current_external_free_gb": 150.0,
                            "required_external_free_gb": 111.0,
                            "available_margin_gb": 39.0,
                        }
                    },
                },
            )
        if "paper-profitability-control" in text:
            return _result(
                cmd,
                {
                    "ok": True,
                    "overall_status": "protective_tightening",
                    "profitability_display_grade": "A+ controlled / D raw",
                    "raw_profitability_grade": "D",
                },
            )
        if "paper-execution-truth" in text:
            return _result(cmd, {"ok": True, "overall_status": "ready", "grade": "A+"})
        if "retrain_schema_compatibility_guard.py" in text:
            return _result(cmd, {"ok": True, "compatibility_seed_ready": True, "failed_checks": []})
        return _result(cmd, {"ok": True, "overall_status": "ready", "status": "ready"})

    monkeypatch.setattr(src, "_run_command", fake_run)

    payload = src.build_payload(tmp_path, apply=True, respect_cooldowns=False)

    assert payload["ok"] is True
    assert payload["production_hard_blockers_clear"] is True
    assert payload["production_hardening"]["ready"] is True
    assert payload["production_hardening"]["hard_blockers"] == []
    assert payload["production_hardening"]["managed_live_money_locks"] == ["target_window_not_complete"]
    assert payload["production_hardening"]["promotion_packet_idle_seed_ready"] is True
    assert any("paper-profitability-control --apply --json" in call for call in calls)
    assert any("paper-execution-truth --json" in call for call in calls)
    assert any("live-money-readiness --json" in call for call in calls)


def test_production_hardening_cascade_blocks_real_live_money_section_failures(tmp_path: Path, monkeypatch) -> None:
    _write_daily(tmp_path, ok=True, failed_checks=[])
    calls: list[str] = []

    def fake_run(cmd: list[str], *, project_root: Path, timeout_sec: int, env: dict[str, str]) -> dict:
        text = " ".join(str(item) for item in cmd)
        calls.append(text)
        if "live-money-readiness" in text:
            return _result(
                cmd,
                {
                    "ok": False,
                    "overall_status": "blocked",
                    "live_money_locked": True,
                    "blocking_reasons": ["paper_profitability_control_not_ready"],
                    "grade_summary": {
                        "required_section_count": 14,
                        "ready_required_section_count": 13,
                        "below_floor_sections": [],
                        "not_ready_sections": ["paper_profitability_control"],
                    },
                },
            )
        if "unattended_soak_readiness.py" in text:
            return _result(
                cmd,
                {
                    "ok": True,
                    "overall_status": "ready",
                    "overall_grade": "A+",
                    "safe_to_leave_unattended": True,
                    "blockers": [],
                    "sections": {
                        "storage": {
                            "current_external_free_gb": 150.0,
                            "required_external_free_gb": 111.0,
                            "available_margin_gb": 39.0,
                        }
                    },
                },
            )
        if "promotion_quality_gate.py" in text:
            return _result(cmd, {"ok": True, "overall_status": "ready", "failed_checks": []})
        return _result(cmd, {"ok": True, "overall_status": "ready", "status": "ready"})

    monkeypatch.setattr(src, "_run_command", fake_run)

    payload = src.build_payload(tmp_path, apply=True, respect_cooldowns=False)

    assert payload["ok"] is False
    assert payload["overall_status"] == "blocked"
    assert payload["production_hard_blockers_clear"] is False
    assert "paper_profitability_control_not_ready" in payload["production_hardening"]["hard_blockers"]
    assert "inspect_production_hard_blocker_cascade" in payload["self_healing"]["operator_followups"]


def test_runtime_continuity_failure_reapplies_runtime_and_paper_ramp(tmp_path: Path, monkeypatch) -> None:
    _write_daily(tmp_path, ok=True, failed_checks=[])
    calls: list[str] = []
    runtime_guard_rows = [
        {
            "ok": False,
            "overall_status": "blocked",
            "failed_guards": ["soak_30_day_continuity_contract"],
            "hard_failed_guard_count": 1,
            "regression_guards": [
                {
                    "name": "soak_30_day_continuity_contract",
                    "ok": False,
                    "actual": {"blockers": ["runtime_not_ready_or_advisory"]},
                }
            ],
        },
        {
            "ok": True,
            "overall_status": "ready",
            "failed_guards": [],
            "hard_failed_guard_count": 0,
            "regression_guards": [],
        },
    ]

    def fake_run(cmd: list[str], *, project_root: Path, timeout_sec: int, env: dict[str, str]) -> dict:
        text = " ".join(str(item) for item in cmd)
        calls.append(text)
        if "runtime_paper_regression_guard.py" in text:
            parsed = runtime_guard_rows.pop(0) if runtime_guard_rows else {"ok": True, "overall_status": "ready"}
            return _result(cmd, parsed)
        if "runtime-throttle --apply" in text:
            return _result(cmd, {"ok": True, "overall_status": "ready"})
        if "paper-400-ramp --apply" in text:
            return _result(cmd, {"ok": True, "stage": "armed", "armed": True, "blockers": []})
        if "daily_auto_verify.py" in text:
            return _result(cmd, {"ok": True, "overall_status": "ready", "failed_checks": []})
        if "promotion_quality_gate.py" in text:
            return _result(cmd, {"ok": False, "overall_status": "blocked", "failed_checks": ["promotion_quality_gate"]})
        if "unattended_soak_readiness.py" in text:
            return _result(
                cmd,
                {
                    "ok": True,
                    "overall_status": "ready",
                    "overall_grade": "A+",
                    "safe_to_leave_unattended": True,
                    "blockers": [],
                    "sections": {
                        "storage": {
                            "current_external_free_gb": 150.0,
                            "required_external_free_gb": 111.0,
                            "available_margin_gb": 39.0,
                        }
                    },
                },
            )
        return _result(cmd, {"ok": True, "overall_status": "ready", "status": "ready"})

    monkeypatch.setattr(src, "_run_command", fake_run)

    payload = src.build_payload(tmp_path, apply=True, respect_cooldowns=False)

    assert payload["ok"] is True
    assert payload["runtime_continuity_refresh"]["attempted"] is True
    assert payload["runtime_continuity_refresh"]["schwab_auth_status"] == "ready"
    assert payload["runtime_continuity_refresh"]["runtime_guard_after_refresh"] == "ready"
    assert payload["runtime_continuity_refresh"]["failed_guards_after_refresh"] == []
    assert any("schwab-auth-supervisor --apply --json" in call for call in calls)
    assert any("global-halt-refresh --json" in call for call in calls)
    assert any("runtime-throttle --apply --max-renice-processes 8 --json" in call for call in calls)
    assert any("paper-400-ramp --apply --json" in call for call in calls)
    assert sum(1 for call in calls if "runtime_paper_regression_guard.py" in call) == 2


def test_production_authority_guard_triggers_runtime_continuity_refresh() -> None:
    payload = {
        "failed_guards": ["production_grade_paper_live_authority_contract"],
    }

    assert src._runtime_continuity_refresh_needed(payload) is True


def test_repeated_repair_failures_open_bounded_circuit() -> None:
    state = {"steps": {}}
    failed = {"ok": False, "rc": 2, "parsed": {"overall_status": "blocked"}}

    for _ in range(3):
        src._update_step_state(
            state,
            "repair",
            failed,
            max_failures_before_circuit=3,
            circuit_open_seconds=60,
        )

    circuit = src._repair_circuit_active(state, "repair")
    assert circuit["active"] is True
    assert circuit["failure_count"] == 3


def test_open_repair_circuit_cannot_be_bypassed_with_no_cooldowns(tmp_path: Path, monkeypatch) -> None:
    state = {
        "steps": {
            "repair": {
                "failure_count": 3,
                "circuit_until_utc": "2099-01-01T00:00:00+00:00",
                "circuit_reason": "bounded_repair_failure_budget_exhausted",
            }
        }
    }

    def unexpected(*args, **kwargs):
        raise AssertionError("repair command must not execute while its circuit is open")

    monkeypatch.setattr(src, "_run_command", unexpected)
    steps: list[dict] = []
    row = src._run_step(
        steps,
        name="repair",
        cmd=["false"],
        project_root=tmp_path,
        timeout_sec=1,
        env={},
        state=state,
        respect_cooldowns=False,
    )

    assert row["executed"] is False
    assert row["skipped_reason"] == "bounded_repair_circuit_open"
    assert row["ok"] is False


def test_raw_profitability_contract_failure_triggers_profitability_refresh() -> None:
    payload = {
        "regression_guards": [
            {
                "name": "production_grade_paper_live_authority_contract",
                "ok": False,
                "actual": {"blockers": ["raw_profitability_improvement_contract_not_ready"]},
            }
        ]
    }

    assert src._stale_profitability_control_from_runtime_guard(payload) is True


def test_latest_hard_failures_treats_green_memory_efficiency_as_managed_throttle() -> None:
    failures = src._latest_hard_failures(
        [
            {
                "name": "memory_efficiency",
                "executed": True,
                "ok": False,
                "parsed": {
                    "overall_status": "needs_work",
                    "reasons": ["storage_pressure_high", "creative_session_music_playback"],
                    "memory_snapshot": {
                        "memory_pressure_state": "green",
                        "memory_pressure_kind": "normal",
                        "memory_free_pct": 87.0,
                        "swap_used_gb": 0.7,
                    },
                    "cotenant_awareness": {"memory_pressure_clear": True},
                },
            }
        ]
    )

    assert failures == []

    advisory_failures = src._latest_hard_failures(
        [
            {
                "name": "memory_efficiency",
                "executed": True,
                "ok": False,
                "parsed": {
                    "overall_status": "advisory",
                    "reasons": ["compressed_memory_high", "creative_session_music_playback"],
                    "memory_snapshot": {
                        "memory_pressure_state": "green",
                        "memory_pressure_kind": "none",
                        "memory_free_pct": 58.0,
                        "swap_used_gb": 1.996,
                    },
                    "cotenant_awareness": {"memory_pressure_clear": True},
                },
            }
        ]
    )

    assert advisory_failures == []

    light_cotenant_failures = src._latest_hard_failures(
        [
            {
                "name": "memory_efficiency",
                "executed": True,
                "ok": False,
                "parsed": {
                    "overall_status": "needs_work",
                    "reasons": ["compressed_memory_high", "co_running_light_competition"],
                    "memory_snapshot": {
                        "memory_pressure_state": "green",
                        "memory_pressure_kind": "none",
                        "memory_free_pct": 64.0,
                        "swap_used_gb": 0.0,
                    },
                    "cotenant_awareness": {"memory_pressure_clear": True},
                },
            }
        ]
    )

    assert light_cotenant_failures == []


def test_latest_hard_failures_keeps_real_memory_pressure_hard() -> None:
    failures = src._latest_hard_failures(
        [
            {
                "name": "memory_efficiency",
                "executed": True,
                "ok": False,
                "parsed": {
                    "overall_status": "blocked",
                    "reasons": ["memory_pressure_red"],
                    "memory_snapshot": {
                        "memory_pressure_state": "red",
                        "memory_pressure_kind": "critical",
                        "memory_free_pct": 5.0,
                        "swap_used_gb": 18.0,
                    },
                },
            }
        ]
    )

    assert failures == ["memory_efficiency"]
