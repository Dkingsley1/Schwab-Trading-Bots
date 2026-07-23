import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import master_infrastructure_supervisor as supervisor


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_ready_fixture(project_root: Path, *, one_numbers_start: str = "20260422") -> None:
    health = project_root / "governance" / "health"
    ready_stamp = "2099-04-23T20:00:00+00:00"
    one_numbers = project_root / "exports" / "one_numbers"
    one_numbers.mkdir(parents=True, exist_ok=True)
    _write_json(
        one_numbers / "one_numbers_summary.json",
        {
            "requested_day": "20260423",
            "resolved_day": "20260423",
            "report_mode": "full",
            "month_to_date_days_covered": "2",
            "all_time_days_covered": "2",
            "combined_decision_total_rows": "100",
            "combined_governance_total_rows": "10",
            "month_to_date_decision_total_rows": "180",
            "all_time_decision_total_rows": "180",
            "combined_blocked_total": "4",
            "month_to_date_blocked_total": "7",
            "all_time_blocked_total": "7",
            "data_blocked_total": "1",
            "month_to_date_data_blocked_total": "2",
            "all_time_data_blocked_total": "2",
            "risk_blocked_total": "3",
            "month_to_date_risk_blocked_total": "5",
            "all_time_risk_blocked_total": "5",
        },
    )
    (one_numbers / "latest.csv").write_text("label,value\n", encoding="utf-8")
    (one_numbers / "latest_metrics.csv").write_text("section,label,value,metric\n", encoding="utf-8")
    _write_json(
        health / "one_numbers_rollup_history.json",
        {
            "history_by_day": {
                one_numbers_start: {"day_utc": one_numbers_start},
                "20260423": {"day_utc": "20260423"},
            }
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready", "backpressure": {"total_pending_lines": 0}, "storage": {"backlog_drain_status": "idle"}})
    _write_json(
        health / "storage_route_status_latest.json",
        {"mode": "external", "active_root": str(project_root), "split_brain_conflicts": 0, "route_verification": {"verification_state": "ready"}},
    )
    _write_json(health / "chrome_headless_guard_latest.json", {"timestamp_utc": ready_stamp, "overall_status": "ready", "ok": True})
    _write_json(health / "report_pdf_bundle_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(health / "system_summary_autopilot_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(
        health / "system_drift_guard_latest.json",
        {"overall_status": "ready", "ok": True, "metrics": {"blocked_surface_count": 0, "degraded_surface_count": 0, "stale_surface_count": 0, "missing_surface_count": 0}},
    )
    _write_json(
        health / "command_validity_latest.json",
        {
            "timestamp_utc": ready_stamp,
            "overall_status": "ready",
            "ok": True,
            "metrics": {
                "blocked_entry_count": 0,
                "smoke_failure_count": 0,
                "runtime_smoke_failure_count": 0,
                "contract_dispatch_smoke_failure_count": 0,
                "contract_hash_mismatch_count": 0,
                "unprobed_operator_gated_count": 0,
            },
        },
    )
    _write_json(health / "commands_hygiene_latest.json", {"overall_status": "ready", "ok": True, "commands_changed": False, "runbook_changed": False, "metrics": {}})
    _write_json(
        health / "infrastructure_autofix_bot_latest.json",
        {"timestamp_utc": ready_stamp, "overall_status": "ready", "ok": True, "repair_plan": [], "attempts": [], "operator_followups": []},
    )
    _write_json(health / "storage_disaster_recovery_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(health / "chaos_drill_coordinator_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(health / "storage_resilience_control_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(health / "operator_cockpit_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(health / "runtime_gate_dashboard_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(health / "backlog_organizer_latest.json", {"timestamp_utc": ready_stamp, "overall_status": "ready", "ok": True})
    _write_json(health / "platform_control_plane_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(health / "cold_lane_refresh_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(project_root / "governance" / "walk_forward" / "coverage_gap_closer_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(project_root / "governance" / "experiments" / "immutable_experiment_ledger_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(project_root / "governance" / "champion" / "promotion_autopilot_packet_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(project_root / "governance" / "champion_challenger" / "promotion_autopilot_packet_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(health / "point_in_time_event_store_latest.json", {"overall_status": "ready", "ok": True, "event_count": 3})
    _write_json(health / "replay_hash_registry_guard_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(health / "golden_replay_regression_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(health / "replay_end_to_end_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(health / "one_numbers_regression_guard_latest.json", {"timestamp_utc": ready_stamp, "overall_status": "ready", "ok": True})
    _write_json(health / "system_drift_autopilot_latest.json", {"timestamp_utc": ready_stamp, "overall_status": "ready", "ok": True})
    _write_json(health / "storage_backpressure_autopilot_latest.json", {"timestamp_utc": ready_stamp, "overall_status": "ready", "ok": True})
    _write_json(health / "storage_pressure_clearance_latest.json", {"timestamp_utc": ready_stamp, "overall_status": "ready", "ok": True})
    _write_json(
        health / "stateful_storage_regression_guard_latest.json",
        {
            "timestamp_utc": ready_stamp,
            "overall_status": "ready",
            "ok": True,
            "metrics": {"local_stateful_gb": 0.0, "blocked_check_count": 0, "degraded_check_count": 0},
            "checks": [],
        },
    )
    _write_json(
        health / "schwab_auth_supervisor_latest.json",
        {
            "timestamp_utc": ready_stamp,
            "overall_status": "ready",
            "ok": True,
            "findings": [],
            "token": {"ready": True, "expires_in_seconds": 1700},
            "callback": {"port_in_use": False},
            "auth_processes": [],
            "recent_auth_signals": {},
        },
    )
    _write_json(
        health / "coinbase_api_health_latest.json",
        {
            "timestamp_utc": ready_stamp,
            "overall_status": "ready",
            "ok": True,
            "public_market_data": {"ok": True, "symbol": "BTC-USD"},
            "credentials": {"api_key_present": False, "api_secret_present": False, "auth_credentials_complete": False},
        },
    )


def test_master_supervisor_ready_when_child_surfaces_are_coherent(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setenv("ONE_NUMBERS_ORIGINAL_START_DAY", "20260422")
    _write_ready_fixture(project_root)

    payload = supervisor.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["blocked_check_count"] == 0
    assert payload["metrics"]["degraded_check_count"] == 0
    assert [row["number"] for row in payload["envelope_lanes"]] == [str(number) for number in range(1, 9)]
    assert {row["status"] for row in payload["envelope_lanes"]} == {"ready"}
    assert payload["platform_posture"]["operating_posture"] == "coherent"
    assert payload["maturity_scores"]["operational_cleanliness"] >= 8.0
    assert payload["hardening_scorecard"]["process_ownership_canonical"] is True
    assert {row["surface"] for row in payload["regression_control_map"]} >= {
        "commands_and_runbook",
        "one_numbers_original_coverage",
        "storage_and_backpressure",
        "schwab_auth",
    }


def test_master_supervisor_degrades_when_one_numbers_start_is_unpinned(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.delenv("ONE_NUMBERS_ORIGINAL_START_DAY", raising=False)
    monkeypatch.delenv("ONE_NUMBERS_EXPECTED_START_DAY", raising=False)
    monkeypatch.delenv("INFRA_SUPERVISOR_ONE_NUMBERS_START_DAY", raising=False)
    _write_ready_fixture(project_root)

    payload = supervisor.build_payload(project_root)
    one_numbers_check = next(row for row in payload["checks"] if row["name"] == "one_numbers_original_coverage")

    assert payload["overall_status"] == "degraded"
    assert one_numbers_check["status"] == "degraded"
    assert "one_numbers_original_start_unpinned" in one_numbers_check["summary"]
    assert payload["operator_followups"] == [
        "pin the One Numbers original start day in config/one_numbers_start_day.txt or ONE_NUMBERS_ORIGINAL_START_DAY"
    ]


def test_master_supervisor_allows_operator_gated_command_validity(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setenv("ONE_NUMBERS_ORIGINAL_START_DAY", "20260422")
    _write_ready_fixture(project_root)
    _write_json(
        project_root / "governance" / "health" / "command_validity_latest.json",
        {
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
            "overall_status": "degraded",
            "ok": True,
            "metrics": {
                "blocked_entry_count": 0,
                "operator_gated_entry_count": 57,
                "smoke_failure_count": 0,
                "runtime_smoke_failure_count": 0,
                "contract_dispatch_smoke_failure_count": 0,
                "contract_hash_mismatch_count": 0,
                "unprobed_operator_gated_count": 0,
            },
        },
    )

    payload = supervisor.build_payload(project_root)
    command_check = next(row for row in payload["checks"] if row["name"] == "command_docs_vs_opsctl_routes")

    assert payload["overall_status"] == "ready"
    assert command_check["status"] == "ready"


def test_master_supervisor_degrades_timed_out_child_when_recovery_lane_is_active(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setenv("ONE_NUMBERS_ORIGINAL_START_DAY", "20260422")
    _write_ready_fixture(project_root)
    _write_json(
        project_root / "governance" / "health" / "storage_backpressure_autopilot_latest.json",
        {
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
            "schema_version": 1,
            "ok": True,
            "overall_status": "already_running",
            "busy": True,
        },
    )
    _write_json(
        project_root / "governance" / "health" / "infrastructure_autofix_bot_latest.json",
        {
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
            "overall_status": "blocked",
            "ok": False,
            "repair_plan": [
                {
                    "name": "storage_backpressure_autopilot",
                    "cmd": ["/tmp/project/scripts/ops/storage_backpressure_autopilot.py", "--apply", "--json"],
                }
            ],
            "attempts": [
                {
                    "cmd": ["/tmp/project/scripts/ops/storage_backpressure_autopilot.py", "--apply", "--json"],
                    "rc": 124,
                    "timed_out": True,
                }
            ],
            "operator_followups": [],
        },
    )

    payload = supervisor.build_payload(project_root)
    child_check = next(row for row in payload["checks"] if row["name"] == "child_repair_bot_outcomes")

    assert payload["overall_status"] == "degraded"
    assert child_check["status"] == "degraded"
    assert child_check["evidence"]["failed_attempt_count"] == 0
    assert child_check["evidence"]["mitigated_active_recovery_attempt_count"] == 1


def test_master_supervisor_blocks_when_child_bot_quietly_has_followups(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setenv("ONE_NUMBERS_ORIGINAL_START_DAY", "20260422")
    _write_ready_fixture(project_root)
    _write_json(
        project_root / "governance" / "health" / "infrastructure_autofix_bot_latest.json",
        {
            "overall_status": "blocked",
            "ok": False,
            "repair_plan": [],
            "attempts": [{"rc": 124, "timed_out": True}],
            "operator_followups": ["configure remote alert delivery"],
        },
    )

    payload = supervisor.build_payload(project_root)
    child_check = next(row for row in payload["checks"] if row["name"] == "child_repair_bot_outcomes")

    assert payload["overall_status"] == "blocked"
    assert child_check["status"] == "blocked"
    assert "failed_attempts=1" in child_check["summary"]


def test_master_supervisor_degrades_bounded_drift_timeouts_in_self_audit(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setenv("ONE_NUMBERS_ORIGINAL_START_DAY", "20260422")
    _write_ready_fixture(project_root)
    _write_json(
        project_root / "governance" / "health" / "system_drift_autopilot_latest.json",
        {
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
            "overall_status": "blocked",
            "ok": False,
            "repair_plan": [{"name": "adaptive_regression_guard"}],
            "attempts": [{"surface": "adaptive_regression_guard", "rc": 124, "timeout_sec": 90, "timed_out": False}],
            "operator_followups": [],
        },
    )

    payload = supervisor.build_payload(project_root)
    self_check = next(row for row in payload["checks"] if row["name"] == "self_auditing_infra_bots")
    drift_row = next(row for row in self_check["evidence"]["bots"] if row["name"] == "system_drift_autopilot")

    assert payload["overall_status"] == "degraded"
    assert self_check["status"] == "degraded"
    assert drift_row["status"] == "degraded"
    assert drift_row["failed_attempt_count"] == 0


def test_master_supervisor_degrades_active_storage_pressure_clearance_in_self_audit(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setenv("ONE_NUMBERS_ORIGINAL_START_DAY", "20260422")
    _write_ready_fixture(project_root)
    _write_json(
        project_root / "governance" / "health" / "storage_pressure_clearance_latest.json",
        {
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
            "overall_status": "blocked",
            "ok": False,
            "repair_plan": [{"name": "observe_existing_storage_autopilot"}],
            "attempts": [],
            "metrics": {"active_storage_pressure": True, "autopilot_active": True},
        },
    )

    payload = supervisor.build_payload(project_root)
    self_check = next(row for row in payload["checks"] if row["name"] == "self_auditing_infra_bots")
    storage_row = next(row for row in self_check["evidence"]["bots"] if row["name"] == "storage_pressure_clearance")

    assert payload["overall_status"] == "degraded"
    assert self_check["status"] == "degraded"
    assert storage_row["status"] == "degraded"


def test_master_supervisor_degrades_reconciled_legacy_split_brain_route(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    monkeypatch.setenv("ONE_NUMBERS_ORIGINAL_START_DAY", "20260422")
    _write_ready_fixture(project_root)
    _write_json(
        health / "health_fast_latest.json",
        {
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only"},
            },
        },
    )
    _write_json(
        health / "storage_route_status_latest.json",
        {
            "mode": "local_fallback_split_brain",
            "active_root": str(project_root / "local_fallback_storage"),
            "split_brain_conflicts": 6,
            "route_verification": {"verification_state": "ready"},
        },
    )
    _write_json(health / "storage_resilience_control_latest.json", {"overall_status": "ready", "ok": True, "unresolved_split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})

    payload = supervisor.build_payload(project_root)
    route_check = next(row for row in payload["checks"] if row["name"] == "external_drive_route_health")

    assert payload["overall_status"] == "degraded"
    assert route_check["status"] == "degraded"
    assert route_check["evidence"]["reconciled_legacy_split_brain"] is True


def test_master_supervisor_degrades_bounded_drift_safe_repairs_in_self_audit(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    monkeypatch.setenv("ONE_NUMBERS_ORIGINAL_START_DAY", "20260422")
    _write_ready_fixture(project_root)
    _write_json(
        health / "health_fast_latest.json",
        {
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only"},
            },
        },
    )
    _write_json(
        health / "system_drift_autopilot_latest.json",
        {
            "timestamp_utc": "2099-04-23T20:00:00+00:00",
            "overall_status": "blocked",
            "ok": False,
            "final_guard": {"overall_status": "blocked", "blocked_surface_count": 2, "degraded_surface_count": 12},
            "repair_plan": [{"surface": "architecture_upgrade_scoreboard"}],
            "attempts": [
                {"surface": "architecture_upgrade_scoreboard", "rc": 0},
                {"surface": "master_infrastructure_supervisor", "rc": 2},
            ],
            "operator_followups": [],
        },
    )

    payload = supervisor.build_payload(project_root)
    self_check = next(row for row in payload["checks"] if row["name"] == "self_auditing_infra_bots")
    drift_row = next(row for row in self_check["evidence"]["bots"] if row["name"] == "system_drift_autopilot")

    assert payload["overall_status"] == "degraded"
    assert self_check["status"] == "degraded"
    assert drift_row["status"] == "degraded"
    assert drift_row["failed_attempt_count"] == 0


def test_master_supervisor_child_env_marks_repair_call_stack(monkeypatch) -> None:
    monkeypatch.delenv(supervisor.REPAIR_CALL_STACK_ENV, raising=False)

    env = supervisor._child_env("master_infrastructure_supervisor")

    assert env[supervisor.REPAIR_CALL_STACK_ENV] == "master_infrastructure_supervisor"


def test_master_supervisor_blocks_when_coinbase_public_api_is_down(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setenv("ONE_NUMBERS_ORIGINAL_START_DAY", "20260422")
    _write_ready_fixture(project_root)
    _write_json(
        project_root / "governance" / "health" / "coinbase_api_health_latest.json",
        {
            "overall_status": "blocked",
            "ok": False,
            "public_market_data": {"ok": False, "symbol": "BTC-USD"},
            "errors": [{"step": "get_ticker", "reason": "network_unreachable"}],
        },
    )

    payload = supervisor.build_payload(project_root)
    coinbase_check = next(row for row in payload["checks"] if row["name"] == "coinbase_api_health")

    assert payload["overall_status"] == "blocked"
    assert coinbase_check["status"] == "blocked"


def test_master_supervisor_degrades_duplicate_lane_owners(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setenv("ONE_NUMBERS_ORIGINAL_START_DAY", "20260422")
    _write_ready_fixture(project_root)

    def _fake_ps_rows(root: Path) -> list[dict]:
        return [
            {
                "pid": 101,
                "command": f"{root}/scripts/run_shadow_training_loop.py --broker coinbase --symbols BTC-USD",
            },
            {
                "pid": 102,
                "command": f"{root}/scripts/run_shadow_training_loop.py --broker coinbase --symbols ETH-USD",
            },
        ]

    monkeypatch.setattr(supervisor, "_ps_rows", _fake_ps_rows)

    payload = supervisor.build_payload(project_root)
    process_check = next(row for row in payload["checks"] if row["name"] == "process_lane_ownership")

    assert payload["overall_status"] == "degraded"
    assert process_check["status"] == "degraded"
    assert process_check["evidence"]["duplicate_lanes"] == ["coinbase_shadow"]
    assert payload["hardening_scorecard"]["process_ownership_canonical"] is False


def test_master_supervisor_ignores_watchdog_embedded_start_commands(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setenv("ONE_NUMBERS_ORIGINAL_START_DAY", "20260422")
    _write_ready_fixture(project_root)

    def _fake_ps_rows(root: Path) -> list[dict]:
        return [
            {
                "pid": 101,
                "command": (
                    f"python {root}/scripts/shadow_watchdog.py "
                    f"--schwab-start-cmd '{root}/scripts/run_all_sleeves.py --with-aggressive-modes'"
                ),
            },
            {
                "pid": 102,
                "command": f"{root}/.venv312/bin/python {root}/scripts/run_all_sleeves.py --with-aggressive-modes",
            },
        ]

    monkeypatch.setattr(supervisor, "_ps_rows", _fake_ps_rows)

    payload = supervisor.build_payload(project_root)
    process_check = next(row for row in payload["checks"] if row["name"] == "process_lane_ownership")
    schwab_row = next(row for row in process_check["evidence"]["lanes"] if row["lane"] == "schwab_all_sleeves")

    assert payload["overall_status"] == "ready"
    assert process_check["status"] == "ready"
    assert schwab_row["owner_count"] == 1


def test_master_supervisor_treats_wrapped_fx_child_as_non_owner(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setenv("ONE_NUMBERS_ORIGINAL_START_DAY", "20260422")
    _write_ready_fixture(project_root)

    def _fake_ps_rows(root: Path) -> list[dict]:
        return [
            {"pid": 201, "command": f"{root}/.venv312/bin/python {root}/scripts/run_fx_shadow.py"},
            {
                "pid": 202,
                "command": (
                    f"{root}/.venv312/bin/python {root}/scripts/run_shadow_training_loop.py "
                    "--broker schwab --profile fx --symbols UUP,FXE"
                ),
            },
        ]

    monkeypatch.setattr(supervisor, "_ps_rows", _fake_ps_rows)

    payload = supervisor.build_payload(project_root)
    process_check = next(row for row in payload["checks"] if row["name"] == "process_lane_ownership")
    fx_row = next(row for row in process_check["evidence"]["lanes"] if row["lane"] == "fx_shadow")

    assert payload["overall_status"] == "ready"
    assert process_check["status"] == "ready"
    assert fx_row["owner_count"] == 1


def test_master_supervisor_apply_treats_rc2_as_degraded_not_hard_failed(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.delenv("ONE_NUMBERS_ORIGINAL_START_DAY", raising=False)
    monkeypatch.delenv("ONE_NUMBERS_EXPECTED_START_DAY", raising=False)
    monkeypatch.delenv("INFRA_SUPERVISOR_ONE_NUMBERS_START_DAY", raising=False)
    _write_ready_fixture(project_root)

    def _fake_run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict:
        return {"cmd": list(cmd), "rc": 2, "timed_out": False, "payload": {"overall_status": "degraded"}}

    monkeypatch.setattr(supervisor, "_run_json", _fake_run_json)

    payload = supervisor.build_payload(project_root, apply=True, timeout_sec=30)

    assert payload["overall_status"] == "degraded"
    assert payload["attempts"]
    assert payload["metrics"]["hard_failed_attempt_count"] == 0
    assert payload["metrics"]["degraded_attempt_count"] == len(payload["attempts"])
