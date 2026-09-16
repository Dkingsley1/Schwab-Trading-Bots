import json
import signal
from datetime import datetime, timedelta, timezone

import pytest

from scripts.ops import runtime_throttle_control as control
from scripts import resource_guard


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    monkeypatch.setattr(
        resource_guard, "evaluate_refresh_job", lambda _: (True, [], {})
    )
    health = tmp_path / "governance/health"
    health.mkdir(parents=True)
    (health / "resource_guard_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "memory_pressure_state": "green",
                "load1_per_core": 0.8,
            }
        )
    )
    row = {
        "pid": 4321,
        "nice": 20,
        "cpu_percent": 70,
        "elapsed": "02:00",
        "category": "support_maintenance",
        "command": f"python {tmp_path}/scripts/build_one_numbers_report.py",
    }
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "throttle_profile": "soft_cap",
        "memory_pressure_level": "normal",
        "compute_pressure_level": "elevated",
        "host_saturation_score": 54,
        "mac_fluidity_contract": {"support_pause_recommended": False},
        "runtime_snapshot": {
            "thermal": {
                "thermal_warning_active": False,
                "performance_warning_active": False,
                "cpu_power_warning_active": False,
            }
        },
        "host_pressure_attribution": {"support_jobs_hot": True},
    }
    return tmp_path, row, payload


def test_overdue_risk_report_can_finish_with_bounded_resource_admission(admitted):
    root, row, payload = admitted
    assert control._risk_refresh_pause_exempt(root, row, payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("memory_pressure_level", "high"),
        ("compute_pressure_level", "high"),
        ("throttle_profile", "protect_live"),
        ("host_saturation_score", 61),
        ("host_saturation_score", float("nan")),
        ("mac_fluidity_contract", {"support_pause_recommended": True}),
        ("runtime_snapshot", {}),
    ],
)
def test_risk_refresh_preserves_host_pressure_boundaries(admitted, field, value):
    root, row, payload = admitted
    payload[field] = value
    assert not control._risk_refresh_pause_exempt(root, row, payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("nice", 0),
        ("cpu_percent", 101),
        ("elapsed", "15:00"),
        ("cpu_percent", float("nan")),
        ("cpu_percent", -1),
        ("elapsed", "1-00:00:00"),
        ("elapsed", "unknown"),
        ("command", "python unrelated_report.py"),
    ],
)
def test_risk_refresh_is_narrow_and_deadline_bounded(admitted, field, value):
    root, row, payload = admitted
    row[field] = value
    assert not control._risk_refresh_pause_exempt(root, row, payload)


@pytest.mark.parametrize("age", [121, -60])
def test_risk_refresh_rejects_stale_or_future_resource_evidence(admitted, age):
    root, row, payload = admitted
    path = root / "governance/health/resource_guard_latest.json"
    evidence = json.loads(path.read_text())
    evidence["timestamp_utc"] = (
        datetime.now(timezone.utc) - timedelta(seconds=age)
    ).isoformat()
    path.write_text(json.dumps(evidence))
    assert not control._risk_refresh_pause_exempt(root, row, payload)


@pytest.mark.parametrize("age", [121, -60])
def test_risk_refresh_rejects_stale_or_future_runtime_evidence(admitted, age):
    root, row, payload = admitted
    payload["timestamp_utc"] = (datetime.now(timezone.utc) - timedelta(seconds=age)).isoformat()
    assert not control._risk_refresh_pause_exempt(root, row, payload)


def test_risk_refresh_respects_resource_guard_and_fresh_report(admitted, monkeypatch):
    root, row, payload = admitted
    monkeypatch.setattr(
        resource_guard, "evaluate_refresh_job", lambda _: (False, ["guarded"], {})
    )
    assert not control._risk_refresh_pause_exempt(root, row, payload)
    monkeypatch.setattr(
        resource_guard, "evaluate_refresh_job", lambda _: (True, [], {})
    )
    summary = root / "exports/one_numbers/one_numbers_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "data_quality_session_local_timestamp": datetime.now(
                    timezone.utc
                ).isoformat()
            }
        )
    )
    assert not control._risk_refresh_pause_exempt(root, row, payload)


def test_owner_resumes_only_verified_overdue_risk_report(admitted, monkeypatch):
    root, row, payload = admitted
    state = root / "governance/health/runtime_support_pause_state.json"
    state.write_text(
        json.dumps(
            {
                "pause_requested": True,
                "paused_processes": [
                    {"pid": row["pid"]},
                    {"pid": 9876},
                ],
            }
        )
    )
    monkeypatch.setattr(
        control, "_support_live_process", lambda pid: row if pid == row["pid"] else {}
    )
    signals = []
    monkeypatch.setattr(control.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    result = control._apply_support_maintenance_pause(
        root, [row], payload, state_path=state
    )
    assert signals == [(4321, signal.SIGCONT)]
    assert result["pause_requested"] is True
    assert result["resume_successful_count"] == 1
    assert json.loads(state.read_text())["paused_processes"] == [{"pid": 9876}]


@pytest.mark.parametrize("age,expected", [(270, False), (330, True)])
def test_risk_refresh_admission_uses_scheduler_freshness_budget(admitted, monkeypatch, age, expected):
    root, row, payload = admitted
    monkeypatch.setenv("ONE_NUMBERS_REFRESH_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("ONE_NUMBERS_BREAKER_MAX_AGE_SECONDS", "600")
    summary = root / "exports/one_numbers/one_numbers_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({
        "data_quality_session_local_timestamp": (
            datetime.now(timezone.utc) - timedelta(seconds=age)
        ).isoformat(),
    }))
    assert control._risk_refresh_pause_exempt(root, row, payload) is expected


def test_risk_refresh_admission_rejects_invalid_cadence(admitted, monkeypatch):
    root, row, payload = admitted
    monkeypatch.setenv("ONE_NUMBERS_BREAKER_MAX_AGE_SECONDS", "invalid")
    assert not control._risk_refresh_pause_exempt(root, row, payload)
