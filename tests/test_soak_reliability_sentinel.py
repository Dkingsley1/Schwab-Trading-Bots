import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import soak_reliability_sentinel as src  # noqa: E402


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_action_audit_remains_append_only_after_advisory_limit(tmp_path: Path) -> None:
    audit = tmp_path / "actions.jsonl"
    src._append_audit(audit, [{"action": "one"}, {"action": "two"}], max_lines=2)
    inode = os.stat(audit).st_ino

    src._append_audit(audit, [{"action": "three"}], max_lines=2)

    assert os.stat(audit).st_ino == inode
    assert len(audit.read_text(encoding="utf-8").splitlines()) == 3


def _healthy_surfaces(project_root: Path) -> None:
    for name, cfg in src._surface_contract(project_root).items():
        status = "" if name in {"session_ready", "health_gates"} else "ready"
        payload = {
            "timestamp_utc": src.iso_now(),
            "ok": True,
            "overall_status": status,
        }
        if name == "health_gates":
            payload["hard_gate_triggered"] = False
        if name == "ingestion_storage_control":
            payload["backpressure"] = {"effective_pressure_clear": True}
            payload["data_integrity"] = {
                "sql_invalid_lines": 0,
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_ops_write_failures": 0,
            }
        _write(
            Path(cfg["path"]),
            payload,
        )


def test_idle_sentinel_is_ready_even_when_heavy_controller_is_dormant(
    tmp_path: Path,
) -> None:
    _healthy_surfaces(tmp_path)

    payload = src.build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["heavy_controller"]["freshness_required"] is False
    assert payload["repair_request"]["active"] is False


def test_livefeed_runtime_surfaces_refresh_before_the_ten_minute_display_ttl(
    tmp_path: Path,
) -> None:
    contract = src._surface_contract(tmp_path)

    assert contract["coordination_state"]["max_age_minutes"] < 5.0
    assert contract["process_watchdog"]["max_age_minutes"] < 5.0
    assert contract["coordination_state"]["refresh_lead_minutes"] >= 1.0
    assert contract["process_watchdog"]["refresh_lead_minutes"] >= 1.0


def test_short_ttl_surface_refreshes_before_it_becomes_stale(tmp_path: Path) -> None:
    _healthy_surfaces(tmp_path)
    watchdog = tmp_path / "governance" / "health" / "process_watchdog_latest.json"
    _write(
        watchdog,
        {
            "timestamp_utc": (
                datetime.now(timezone.utc) - timedelta(minutes=3.5)
            ).isoformat(),
            "ok": True,
            "overall_status": "ready",
        },
    )
    calls: list[str] = []

    def runner(
        command: list[str], project_root: Path, timeout: int, env: dict[str, str]
    ) -> dict:
        calls.append(" ".join(command))
        _write(
            watchdog,
            {
                "timestamp_utc": src.iso_now(),
                "ok": True,
                "overall_status": "ready",
            },
        )
        return {"ok": True, "rc": 0, "duration_seconds": 0.01, "stderr_tail": ""}

    payload = src.build_payload(
        tmp_path,
        apply=True,
        state_path=tmp_path / "state.json",
        request_path=tmp_path / "request.json",
        trigger_path=tmp_path / "trigger.json",
        audit_path=tmp_path / "audit.jsonl",
        max_actions=1,
        runner=runner,
    )

    assert payload["surfaces_before"]["process_watchdog"]["stale"] is False
    assert payload["surfaces_before"]["process_watchdog"]["refresh_due"] is True
    assert calls == [
        " ".join(
            payload["surfaces_before"]["process_watchdog"]["refresh_command"]
        )
    ]
    assert payload["surfaces"]["process_watchdog"]["stale"] is False
    assert payload["overall_status"] == "ready"


def test_ingestion_pressure_runs_bounded_autopilot_with_extended_timeout(
    tmp_path: Path,
) -> None:
    _healthy_surfaces(tmp_path)
    health = tmp_path / "governance" / "health"
    storage = health / "ingestion_storage_control_latest.json"
    _write(
        storage,
        {
            "timestamp_utc": src.iso_now(),
            "ok": False,
            "overall_status": "needs_work",
            "severity": "high",
            "backpressure": {"effective_pressure_clear": False},
            "data_integrity": {
                "sql_invalid_lines": 0,
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_ops_write_failures": 0,
            },
        },
    )
    calls: list[tuple[list[str], int]] = []

    def runner(
        command: list[str], project_root: Path, timeout: int, env: dict[str, str]
    ) -> dict:
        calls.append((command, timeout))
        _write(
            storage,
            {
                "timestamp_utc": src.iso_now(),
                "ok": True,
                "overall_status": "ready",
                "severity": "stable",
                "backpressure": {"effective_pressure_clear": False},
                "data_integrity": {
                    "sql_invalid_lines": 0,
                    "sql_overlay_invalid_lines": 0,
                    "sql_overlay_ops_write_failures": 0,
                },
            },
        )
        return {"ok": True, "rc": 0, "duration_seconds": 0.01, "stderr_tail": ""}

    payload = src.build_payload(
        tmp_path,
        apply=True,
        state_path=tmp_path / "state.json",
        request_path=tmp_path / "request.json",
        audit_path=tmp_path / "audit.jsonl",
        max_actions=1,
        runner=runner,
    )

    assert len(calls) == 1
    assert "storage-backpressure-autopilot" in calls[0][0]
    assert "--quick-bounded" in calls[0][0]
    assert calls[0][1] == 180
    assert payload["surfaces"]["ingestion_storage_control"]["ready"] is True
    assert payload["blockers"] == []


def test_paper_only_live_lock_is_a_managed_protective_halt_state(
    tmp_path: Path,
) -> None:
    _healthy_surfaces(tmp_path)
    halt = (
        tmp_path
        / "governance"
        / "health"
        / "halt_trigger_control_plane_latest.json"
    )
    _write(
        halt,
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "blocked",
            "effective_state": "live_read_only",
            "execution_policy": {
                "paper_trade_lock_active": True,
                "operator_stop_active": False,
                "global_halt_active": False,
                "effective_live_order_execution_allowed": False,
            },
        },
    )

    payload = src.build_payload(tmp_path)

    row = payload["surfaces"]["halt_trigger_control_plane"]
    assert row["ready"] is True
    assert row["managed_protective_state"] is True
    assert payload["blockers"] == []


def test_actual_global_halt_is_not_managed_as_a_healthy_live_lock(
    tmp_path: Path,
) -> None:
    _healthy_surfaces(tmp_path)
    halt = (
        tmp_path
        / "governance"
        / "health"
        / "halt_trigger_control_plane_latest.json"
    )
    _write(
        halt,
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "blocked",
            "effective_state": "global_halt",
            "execution_policy": {
                "paper_trade_lock_active": True,
                "operator_stop_active": False,
                "global_halt_active": True,
                "effective_live_order_execution_allowed": False,
            },
        },
    )

    payload = src.build_payload(tmp_path)

    row = payload["surfaces"]["halt_trigger_control_plane"]
    assert row["ready"] is False
    assert row["managed_protective_state"] is False
    assert "halt_trigger_control_plane_not_ready" in payload["blockers"]


def test_health_gate_surface_requires_no_active_hard_gate(tmp_path: Path) -> None:
    _healthy_surfaces(tmp_path)
    health_gates = tmp_path / "governance" / "health" / "health_gates_latest.json"
    _write(
        health_gates,
        {
            "timestamp_utc": src.iso_now(),
            "hard_gate_triggered": True,
            "hard_gates": {"sql_progress_stall": True},
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["surfaces"]["health_gates"]["ready"] is False
    assert "health_gates_not_ready" in payload["blockers"]


def test_sentinel_refreshes_health_halt_and_coordination_in_one_bounded_cycle(
    tmp_path: Path,
) -> None:
    _healthy_surfaces(tmp_path)
    health = tmp_path / "governance" / "health"
    (health / "health_gates_latest.json").unlink()
    (health / "halt_trigger_control_plane_latest.json").unlink()
    (health / "coordination_state_latest.json").unlink()
    calls: list[str] = []

    def runner(
        command: list[str], project_root: Path, timeout: int, env: dict[str, str]
    ) -> dict:
        script = Path(command[1]).name
        calls.append(script)
        if script == "health_gates.py":
            _write(
                health / "health_gates_latest.json",
                {
                    "timestamp_utc": src.iso_now(),
                    "hard_gate_triggered": False,
                },
            )
        elif script == "halt_trigger_control_plane.py":
            _write(
                health / "halt_trigger_control_plane_latest.json",
                {
                    "timestamp_utc": src.iso_now(),
                    "overall_status": "blocked",
                    "effective_state": "live_read_only",
                    "execution_policy": {
                        "paper_trade_lock_active": True,
                        "operator_stop_active": False,
                        "global_halt_active": False,
                        "effective_live_order_execution_allowed": False,
                    },
                },
            )
        else:
            _write(
                health / "coordination_state_latest.json",
                {
                    "timestamp_utc": src.iso_now(),
                    "overall_status": "guarded",
                },
            )
        return {
            "ok": True,
            "rc": 0,
            "duration_seconds": 0.01,
            "parsed": {},
            "stderr_tail": "",
        }

    payload = src.build_payload(
        tmp_path,
        apply=True,
        state_path=tmp_path / "state.json",
        request_path=tmp_path / "request.json",
        audit_path=tmp_path / "audit.jsonl",
        max_actions=3,
        runner=runner,
    )

    assert calls == [
        "health_gates.py",
        "halt_trigger_control_plane.py",
        "coordination_state_control.py",
    ]
    assert payload["blockers"] == []


def test_sentinel_refreshes_livefeed_evidence_before_display_ttl_without_blocking(
    tmp_path: Path,
) -> None:
    _healthy_surfaces(tmp_path)
    health = tmp_path / "governance" / "health"
    target = health / "research_data_platform_control_latest.json"
    _write(
        target,
        {
            "timestamp_utc": (
                datetime.now(timezone.utc) - timedelta(minutes=181)
            ).isoformat(),
            "ok": True,
            "overall_status": "ready_with_evidence_debt",
        },
    )
    calls: list[str] = []

    def runner(
        command: list[str], project_root: Path, timeout: int, env: dict[str, str]
    ) -> dict:
        calls.append(Path(command[1]).name)
        _write(
            target,
            {
                "timestamp_utc": src.iso_now(),
                "ok": True,
                "overall_status": "ready_with_evidence_debt",
            },
        )
        return {
            "ok": True,
            "rc": 0,
            "duration_seconds": 0.01,
            "parsed": {},
            "stderr_tail": "",
        }

    payload = src.build_payload(
        tmp_path,
        apply=True,
        state_path=tmp_path / "state.json",
        request_path=tmp_path / "request.json",
        audit_path=tmp_path / "audit.jsonl",
        max_actions=1,
        runner=runner,
    )

    assert calls == ["research_data_platform_control.py"]
    before = payload["surfaces_before"]["research_data_platform_control"]
    after = payload["surfaces"]["research_data_platform_control"]
    assert before["stale"] is True
    assert after["stale"] is False
    assert after["impact_scope"] == "evidence_only"
    assert payload["evidence_advisories"] == []
    assert payload["blockers"] == []
    assert payload["overall_status"] == "ready"


def test_failed_runtime_surface_requires_fresh_heavy_recovery(tmp_path: Path) -> None:
    _healthy_surfaces(tmp_path)
    watchdog = tmp_path / "governance" / "health" / "process_watchdog_latest.json"
    _write(
        watchdog,
        {"timestamp_utc": src.iso_now(), "ok": False, "overall_status": "degraded"},
    )

    payload = src.build_payload(tmp_path, repair_grace_seconds=0)

    assert payload["ok"] is False
    assert "process_watchdog_not_ready" in payload["blockers"]
    assert "heavy_self_healing_starved_while_repair_required" in payload["blockers"]
    assert payload["repair_request"]["heavy_repair_required"] is True


def test_new_repair_request_uses_wakeup_grace_before_starvation_blocker(
    tmp_path: Path,
) -> None:
    _healthy_surfaces(tmp_path)
    watchdog = tmp_path / "governance" / "health" / "process_watchdog_latest.json"
    _write(
        watchdog,
        {"timestamp_utc": src.iso_now(), "ok": False, "overall_status": "degraded"},
    )

    payload = src.build_payload(tmp_path)

    assert "process_watchdog_not_ready" in payload["blockers"]
    assert "heavy_self_healing_starved_while_repair_required" not in payload["blockers"]
    assert "heavy_self_healing_wakeup_pending" in payload["warnings"]


def test_ready_stale_surface_waiting_for_bounded_slot_is_watch_not_failure(
    tmp_path: Path,
) -> None:
    _healthy_surfaces(tmp_path)
    watchdog = tmp_path / "governance" / "health" / "process_watchdog_latest.json"
    _write(
        watchdog,
        {
            "timestamp_utc": (
                datetime.now(timezone.utc) - timedelta(minutes=6)
            ).isoformat(),
            "ok": True,
            "overall_status": "ready",
        },
    )

    payload = src.build_payload(
        tmp_path,
        apply=True,
        state_path=tmp_path / "state.json",
        request_path=tmp_path / "request.json",
        trigger_path=tmp_path / "trigger.json",
        audit_path=tmp_path / "audit.jsonl",
        max_actions=0,
        repair_grace_seconds=600.0,
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "watch"
    assert payload["grade"] == "A"
    assert payload["blockers"] == []
    assert "process_watchdog_refresh_queued" in payload["warnings"]
    assert payload["bounded_repair"]["deferred_refresh_reasons"] == [
        "process_watchdog_refresh_due"
    ]
    assert payload["repair_request"]["active"] is True


def test_safe_storage_watch_is_managed_while_proactive_recovery_runs(
    tmp_path: Path,
) -> None:
    _healthy_surfaces(tmp_path)
    storage = (
        tmp_path / "governance" / "health" / "local_storage_reserve_guard_latest.json"
    )
    _write(
        storage,
        {
            "timestamp_utc": src.iso_now(),
            "ok": True,
            "overall_status": "watch",
            "local_storage_reserve": {
                "disk": {"known": True},
                "pressure_active": False,
                "hard_block": False,
                "emergency_active": False,
            },
            "recovery_request": {
                "active": True,
                "paper_pause_required": False,
                "collection_may_continue": True,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        apply=True,
        state_path=tmp_path / "state.json",
        request_path=tmp_path / "request.json",
        trigger_path=tmp_path / "trigger.json",
        audit_path=tmp_path / "audit.jsonl",
        max_actions=0,
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "watch"
    assert payload["grade"] == "A"
    assert payload["blockers"] == []
    assert "local_storage_reserve_guard_managed_recovery_pending" in payload["warnings"]
    assert payload["repair_request"]["active"] is True
    assert payload["repair_request"]["severity"] == "proactive"
    assert Path(payload["heavy_controller"]["wakeup_trigger_path"]).exists()


def test_apply_uses_bounded_allowlist_and_recovers_surface(tmp_path: Path) -> None:
    _healthy_surfaces(tmp_path)
    storage = (
        tmp_path / "governance" / "health" / "local_storage_reserve_guard_latest.json"
    )
    storage.unlink()
    calls: list[list[str]] = []

    def runner(
        command: list[str], project_root: Path, timeout: int, env: dict[str, str]
    ) -> dict:
        calls.append(command)
        _write(
            storage,
            {"timestamp_utc": src.iso_now(), "ok": True, "overall_status": "ready"},
        )
        return {"ok": True, "rc": 0, "duration_seconds": 0.01, "stderr_tail": ""}

    payload = src.build_payload(
        tmp_path,
        apply=True,
        state_path=tmp_path / "state.json",
        request_path=tmp_path / "request.json",
        audit_path=tmp_path / "audit.jsonl",
        max_actions=1,
        runner=runner,
    )

    assert payload["ok"] is True
    assert len(calls) == 1
    assert "local_storage_reserve_guard.py" in " ".join(calls[0])
    assert payload["repair_actions"][0]["ok"] is True


def test_repeated_failure_opens_repair_circuit() -> None:
    state = {"actions": {}}
    first = src._record_action(
        state,
        "surface",
        result={"ok": False, "rc": 2, "stderr_tail": "failed"},
        now_epoch=100.0,
        max_failures=2,
        circuit_open_seconds=600.0,
    )
    second = src._record_action(
        state,
        "surface",
        result={"ok": False, "rc": 2, "stderr_tail": "failed again"},
        now_epoch=200.0,
        max_failures=2,
        circuit_open_seconds=600.0,
    )

    assert first["circuit_until_epoch"] == 0.0
    assert second["consecutive_failures"] == 2
    assert second["circuit_until_epoch"] == 800.0
    assert (
        src._action_gate(state, "surface", now_epoch=300.0, cooldown_seconds=0)[
            "reason"
        ]
        == "repair_circuit_open"
    )


def test_session_ready_refresh_uses_resolved_runtime_python(tmp_path: Path) -> None:
    contract = src._surface_contract(tmp_path)

    command = contract["session_ready"]["command"]
    assert Path(command[0]).name.startswith("python")
    assert command[1] == str(tmp_path / "scripts" / "session_ready_check.py")


def test_fresh_recovery_closes_only_the_recovered_surface_circuit(
    tmp_path: Path,
) -> None:
    _healthy_surfaces(tmp_path)
    state_path = tmp_path / "state.json"
    _write(
        state_path,
        {
            "actions": {
                "session_ready": {
                    "consecutive_failures": 3,
                    "circuit_until_epoch": 9999999999.0,
                },
                "schwab_auth_supervisor": {
                    "consecutive_failures": 2,
                    "circuit_until_epoch": 9999999999.0,
                },
            }
        },
    )
    auth_path = (
        tmp_path / "governance" / "health" / "schwab_auth_supervisor_latest.json"
    )
    _write(
        auth_path,
        {"timestamp_utc": src.iso_now(), "ok": False, "overall_status": "blocked"},
    )

    payload = src.build_payload(
        tmp_path,
        apply=True,
        state_path=state_path,
        request_path=tmp_path / "request.json",
        audit_path=tmp_path / "audit.jsonl",
        max_actions=0,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["bounded_repair"]["recovered_circuits"] == ["session_ready"]
    assert payload["bounded_repair"]["open_circuits"] == ["schwab_auth_supervisor"]
    assert state["actions"]["session_ready"]["consecutive_failures"] == 0
    assert state["actions"]["schwab_auth_supervisor"]["consecutive_failures"] == 2


def test_interactive_auth_blocker_does_not_claim_heavy_repair_starvation(
    tmp_path: Path,
) -> None:
    _healthy_surfaces(tmp_path)
    auth_path = (
        tmp_path / "governance" / "health" / "schwab_auth_supervisor_latest.json"
    )
    _write(
        auth_path,
        {
            "timestamp_utc": src.iso_now(),
            "ok": False,
            "overall_status": "blocked",
            "operator_followups": [
                "./scripts/ops/opsctl.sh token-refresh-interactive --force --json"
            ],
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["blockers"] == ["schwab_auth_supervisor_not_ready"]
    assert payload["heavy_controller"]["freshness_required"] is False
    assert payload["repair_request"]["operator_intervention_required"] is True
    assert payload["repair_request"]["machine_repairable_reasons"] == []
