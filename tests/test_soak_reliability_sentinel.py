import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import soak_reliability_sentinel as src  # noqa: E402


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _healthy_surfaces(project_root: Path) -> None:
    for name, cfg in src._surface_contract(project_root).items():
        status = "" if name == "session_ready" else "ready"
        _write(Path(cfg["path"]), {"timestamp_utc": src.iso_now(), "ok": True, "overall_status": status})


def test_idle_sentinel_is_ready_even_when_heavy_controller_is_dormant(tmp_path: Path) -> None:
    _healthy_surfaces(tmp_path)

    payload = src.build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["heavy_controller"]["freshness_required"] is False
    assert payload["repair_request"]["active"] is False


def test_failed_runtime_surface_requires_fresh_heavy_recovery(tmp_path: Path) -> None:
    _healthy_surfaces(tmp_path)
    watchdog = tmp_path / "governance" / "health" / "process_watchdog_latest.json"
    _write(watchdog, {"timestamp_utc": src.iso_now(), "ok": False, "overall_status": "degraded"})

    payload = src.build_payload(tmp_path, repair_grace_seconds=0)

    assert payload["ok"] is False
    assert "process_watchdog_not_ready" in payload["blockers"]
    assert "heavy_self_healing_starved_while_repair_required" in payload["blockers"]
    assert payload["repair_request"]["heavy_repair_required"] is True


def test_new_repair_request_uses_wakeup_grace_before_starvation_blocker(tmp_path: Path) -> None:
    _healthy_surfaces(tmp_path)
    watchdog = tmp_path / "governance" / "health" / "process_watchdog_latest.json"
    _write(watchdog, {"timestamp_utc": src.iso_now(), "ok": False, "overall_status": "degraded"})

    payload = src.build_payload(tmp_path)

    assert "process_watchdog_not_ready" in payload["blockers"]
    assert "heavy_self_healing_starved_while_repair_required" not in payload["blockers"]
    assert "heavy_self_healing_wakeup_pending" in payload["warnings"]


def test_safe_storage_watch_is_managed_while_proactive_recovery_runs(tmp_path: Path) -> None:
    _healthy_surfaces(tmp_path)
    storage = tmp_path / "governance" / "health" / "local_storage_reserve_guard_latest.json"
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
    storage = tmp_path / "governance" / "health" / "local_storage_reserve_guard_latest.json"
    storage.unlink()
    calls: list[list[str]] = []

    def runner(command: list[str], project_root: Path, timeout: int, env: dict[str, str]) -> dict:
        calls.append(command)
        _write(storage, {"timestamp_utc": src.iso_now(), "ok": True, "overall_status": "ready"})
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
    assert src._action_gate(state, "surface", now_epoch=300.0, cooldown_seconds=0)["reason"] == "repair_circuit_open"


def test_session_ready_refresh_uses_resolved_runtime_python(tmp_path: Path) -> None:
    contract = src._surface_contract(tmp_path)

    command = contract["session_ready"]["command"]
    assert Path(command[0]).name.startswith("python")
    assert command[1] == str(tmp_path / "scripts" / "session_ready_check.py")


def test_fresh_recovery_closes_only_the_recovered_surface_circuit(tmp_path: Path) -> None:
    _healthy_surfaces(tmp_path)
    state_path = tmp_path / "state.json"
    _write(
        state_path,
        {
            "actions": {
                "session_ready": {"consecutive_failures": 3, "circuit_until_epoch": 9999999999.0},
                "schwab_auth_supervisor": {"consecutive_failures": 2, "circuit_until_epoch": 9999999999.0},
            }
        },
    )
    auth_path = tmp_path / "governance" / "health" / "schwab_auth_supervisor_latest.json"
    _write(auth_path, {"timestamp_utc": src.iso_now(), "ok": False, "overall_status": "blocked"})

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


def test_interactive_auth_blocker_does_not_claim_heavy_repair_starvation(tmp_path: Path) -> None:
    _healthy_surfaces(tmp_path)
    auth_path = tmp_path / "governance" / "health" / "schwab_auth_supervisor_latest.json"
    _write(
        auth_path,
        {
            "timestamp_utc": src.iso_now(),
            "ok": False,
            "overall_status": "blocked",
            "operator_followups": ["./scripts/ops/opsctl.sh token-refresh-interactive --force --json"],
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["blockers"] == ["schwab_auth_supervisor_not_ready"]
    assert payload["heavy_controller"]["freshness_required"] is False
    assert payload["repair_request"]["operator_intervention_required"] is True
    assert payload["repair_request"]["machine_repairable_reasons"] == []
