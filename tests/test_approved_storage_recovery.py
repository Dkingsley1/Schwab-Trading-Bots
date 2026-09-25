from datetime import datetime, timedelta, timezone
import gzip
import json
import time
from types import SimpleNamespace

import pytest

from scripts.ops import approved_storage_recovery as admission
from scripts.ops import raw_training_compaction_intelligence as compaction
from scripts.ops import runtime_throttle_control as runtime
from scripts import resource_guard


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    health = tmp_path / "governance/health"
    health.mkdir(parents=True)
    now = datetime.now(timezone.utc).isoformat()
    resource = {
        "timestamp_utc": now,
        "memory_pressure_state": "green",
        "load1_per_core": 0.4,
    }
    payload = {
        "timestamp_utc": now,
        "throttle_profile": "soft_cap",
        "compute_pressure_level": "normal",
        "memory_pressure_level": "normal",
        "host_saturation_score": 35,
        "mac_fluidity_contract": {
            "support_pause_recommended": False,
            "fluidity_band": "guarded_smooth",
            "measurements": {
                "foreground_app_cpu_percent": 30,
                "macos_system_cpu_percent": 0,
            },
        },
        "runtime_snapshot": {
            "thermal": {
                "thermal_warning_active": False,
                "performance_warning_active": False,
                "cpu_power_warning_active": False,
            }
        },
    }
    (health / "runtime_throttle_control_latest.json").write_text(json.dumps(payload))
    (health / "resource_guard_latest.json").write_text(json.dumps(resource))
    monkeypatch.setattr(
        resource_guard, "evaluate_refresh_job", lambda data: (True, [], {})
    )
    row = {
        "pid": 1234,
        "cpu_percent": 25,
        "elapsed": "01:00",
        "category": "support_maintenance",
        "command": f"python {tmp_path}/scripts/ops/raw_training_compaction_intelligence.py --operator-approved-recovery --apply",
    }
    return tmp_path, payload, resource, row


def test_only_explicit_bounded_owner_is_admitted(admitted):
    root, payload, _, row = admitted
    assert admission.resources_admitted(root)
    assert admission.process_exempt(root, row, payload)


def test_lower_pressure_observe_profile_does_not_withdraw_recovery(admitted):
    root, payload, _, row = admitted
    payload["throttle_profile"] = "observe"
    payload["host_saturation_score"] = 20
    assert admission.process_exempt(root, row, payload)
    payload["mac_fluidity_contract"]["measurements"]["foreground_app_cpu_percent"] = 90
    assert not admission.process_exempt(root, row, payload)


def test_explicit_approval_covers_housekeeping_latch_but_not_foreground_pressure(
    admitted,
):
    root, payload, _, row = admitted
    fluidity = payload["mac_fluidity_contract"]
    fluidity["support_pause_recommended"] = True
    assert admission.process_exempt(root, row, payload)
    fluidity["measurements"]["foreground_app_cpu_percent"] = 90
    assert not admission.process_exempt(root, row, payload)
    fluidity["measurements"]["foreground_app_cpu_percent"] = 30
    fluidity["fluidity_band"] = "strained"
    assert not admission.process_exempt(root, row, payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cpu_percent", 36),
        ("cpu_percent", float("nan")),
        ("elapsed", "30:00"),
        ("elapsed", "1-00:00:00"),
        ("elapsed", "unknown"),
        ("command", "python unrelated.py --operator-approved-recovery"),
    ],
)
def test_process_exception_is_narrow(admitted, field, value):
    root, payload, _, row = admitted
    assert not admission.process_exempt(root, {**row, field: value}, payload)


def test_approval_flag_and_real_script_invocation_required(admitted):
    root, payload, _, row = admitted
    assert not admission.process_exempt(
        root,
        {**row, "command": row["command"].replace("--operator-approved-recovery", "")},
        payload,
    )
    assert not admission.process_exempt(
        root, {**row, "command": "python -c " + row["command"]}, payload
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("throttle_profile", "protect_live"),
        ("throttle_profile", "sustain"),
        ("compute_pressure_level", "high"),
        ("host_saturation_score", float("nan")),
        ("host_saturation_score", 61),
        ("mac_fluidity_contract", {"support_pause_recommended": True}),
        ("runtime_snapshot", {}),
    ],
)
def test_exception_does_not_bypass_hard_runtime_checks(admitted, field, value):
    root, payload, _, row = admitted
    assert not admission.process_exempt(root, row, {**payload, field: value})


@pytest.mark.parametrize("age", [-1, 121])
@pytest.mark.parametrize(
    "name", ["runtime_throttle_control_latest.json", "resource_guard_latest.json"]
)
def test_stale_or_future_resources_fail_closed(admitted, age, name):
    root, _, _, _ = admitted
    path = root / "governance/health" / name
    data = json.loads(path.read_text())
    data["timestamp_utc"] = (
        datetime.now(timezone.utc) - timedelta(seconds=age)
    ).isoformat()
    path.write_text(json.dumps(data))
    assert not admission.resources_admitted(root)


def test_holds_and_resource_guard_still_block(admitted, monkeypatch):
    root, _, _, _ = admitted
    stop = root / "OPERATOR_STOP.flag"
    stop.touch()
    assert not admission.resources_admitted(root)
    stop.unlink()
    monkeypatch.setattr(
        resource_guard,
        "evaluate_refresh_job",
        lambda data: (False, ["resource_pressure"], {}),
    )
    assert not admission.resources_admitted(root)


@pytest.mark.parametrize("expired", [True, False])
def test_owned_maintenance_hold_respects_expiry(admitted, expired):
    root, _, _, _ = admitted
    path = root / "governance/health/RUNTIME_MAINTENANCE_HOLD.flag"
    path.write_text(json.dumps({"expires_at_utc": (
        datetime.now(timezone.utc) + timedelta(hours=-1 if expired else 1)
    ).isoformat()}))
    assert admission.resources_admitted(root) is expired
    assert path.exists()


@pytest.mark.parametrize("content", ["broken", "{}", "[]"])
def test_unverifiable_maintenance_hold_fails_closed(admitted, content):
    root, _, _, _ = admitted
    (root / "governance/health/RUNTIME_MAINTENANCE_HOLD.flag").write_text(content)
    assert not admission.resources_admitted(root)


def test_environment_hold_blocks_approved_recovery(admitted, monkeypatch):
    root, _, _, _ = admitted
    monkeypatch.setenv("RUNTIME_MAINTENANCE_HOLD", "1")
    assert not admission.resources_admitted(root)


def test_runtime_owner_does_not_pause_approved_worker(admitted, monkeypatch):
    root, payload, _, row = admitted
    monkeypatch.setattr(
        runtime,
        "_support_maintenance_pause_requested",
        lambda *a, **k: (True, "runtime_soft_cap_support_pressure"),
    )
    signals = []
    monkeypatch.setattr(runtime.os, "kill", lambda *args: signals.append(args))
    result = runtime._apply_support_maintenance_pause(root, [row], payload)
    assert result["pause_requested"]
    assert not signals


class Guard:
    reserve = 64 * 1024**3
    deadline = time.monotonic() + 60

    def check(self):
        pass


def test_paced_compaction_preserves_full_content_and_reserve(tmp_path, monkeypatch):
    path = tmp_path / "old.jsonl"
    data = b'{"old":"data"}\n' * 1000
    path.write_bytes(data)
    target = tmp_path / "old.jsonl.gz"
    monkeypatch.setattr(
        compaction.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        compaction.shutil, "disk_usage", lambda *a: SimpleNamespace(free=65 * 1024**3)
    )
    result = compaction._compress_and_clear(
        path, target, compress_level=1, keep_raw=False, guard=Guard()
    )
    assert result["status"] == "ok"
    assert result["verified_raw_bytes"] == len(data)
    assert gzip.decompress(target.read_bytes()) == data
    assert not path.exists()


@pytest.mark.parametrize("blocker", ["reserve", "open_handle", "resource"])
def test_recovery_failures_never_remove_raw(tmp_path, monkeypatch, blocker):
    path = tmp_path / "old.jsonl"
    path.write_bytes(b"preserve me")
    target = tmp_path / "old.jsonl.gz"
    monkeypatch.setattr(
        compaction.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0 if blocker == "open_handle" else 1, stdout="", stderr=""
        ),
    )
    monkeypatch.setattr(
        compaction.shutil,
        "disk_usage",
        lambda *a: SimpleNamespace(free=(64 if blocker == "reserve" else 65) * 1024**3),
    )
    guard = Guard()
    if blocker == "resource":

        def fail():
            raise RuntimeError("resource pressure")

        guard.check = fail
    result = compaction._compress_and_clear(
        path, target, compress_level=1, keep_raw=False, guard=guard
    )
    assert result["status"] == "failed"
    assert path.read_bytes() == b"preserve me"
    assert not target.exists()
