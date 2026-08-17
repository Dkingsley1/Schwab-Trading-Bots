import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.ops.soak_self_healing_control as src


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
    offload_calls = [call for call in calls if "manifest-backed-offload" in call]
    assert payload["ok"] is True
    assert payload["overall_status"] == "guarded_storage_capacity"
    assert payload["storage"]["retention_attempted"] is True
    assert payload["storage"]["recovery"]["raw_compaction_attempted"] is True
    assert payload["storage"]["recovery"]["manifest_cold_offload_attempted"] is True
    assert raw_compaction_calls
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
