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


def test_storage_soak_blocker_runs_bounded_retention_only_in_apply_mode(tmp_path: Path, monkeypatch) -> None:
    _write_daily(tmp_path, ok=True, failed_checks=[])
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
    assert payload["ok"] is True
    assert payload["overall_status"] == "guarded_storage_capacity"
    assert payload["storage"]["retention_attempted"] is True
    assert retention_calls
    assert "--cleanup-max-delete-gb 16.0" in retention_calls[0]
    assert "--target-free-gb 125.0" in retention_calls[0]
    assert "add_or_free_external_storage_capacity_for_30_day_soak" in payload["self_healing"]["operator_followups"]


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
