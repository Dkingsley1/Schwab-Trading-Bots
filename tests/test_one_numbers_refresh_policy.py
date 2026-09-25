import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.ops.one_numbers_refresh_policy import refresh_policy
from scripts.ops.one_numbers_refresh_policy import bounded_refresh_admitted

NOW = datetime(2026, 9, 8, 21, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("profile", ["soft_cap", "protect_live", "hard_cap", "unknown"])
def test_overdue_risk_refresh_uses_headroom_not_backlog_profile_label(tmp_path, monkeypatch, profile):
    from scripts import resource_guard
    monkeypatch.setattr(resource_guard, "evaluate_refresh_job", lambda resource: (True, [], {}))
    now = datetime.now(timezone.utc).isoformat()
    path = tmp_path / "governance/health/resource_guard_latest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"timestamp_utc": now, "memory_pressure_state": "green", "load1_per_core": 0.8}))
    payload = {
        "timestamp_utc": now, "throttle_profile": profile,
        "memory_pressure_level": "normal", "compute_pressure_level": "elevated",
        "host_saturation_score": 45,
        "mac_fluidity_contract": {"support_pause_recommended": False},
        "runtime_snapshot": {"thermal": {
            "thermal_warning_active": False, "performance_warning_active": False,
            "cpu_power_warning_active": False}},
    }
    assert bounded_refresh_admitted(tmp_path, payload) == (profile in {"soft_cap", "protect_live"})
    if profile != "protect_live":
        return
    for key, value in (("memory_pressure_level", "critical"),
                       ("compute_pressure_level", "critical"),
                       ("host_saturation_score", 61),
                       ("timestamp_utc", "2000-01-01T00:00:00+00:00")):
        assert not bounded_refresh_admitted(tmp_path, {**payload, key: value})
    assert not bounded_refresh_admitted(tmp_path, {**payload, "mac_fluidity_contract": {"support_pause_recommended": True}})
    assert not bounded_refresh_admitted(tmp_path, {**payload, "runtime_snapshot": {"thermal": {"thermal_warning_active": True}}})
    monkeypatch.setattr(resource_guard, "evaluate_refresh_job", lambda resource: (False, ["disk_reserve"], {}))
    assert not bounded_refresh_admitted(tmp_path, payload)


def policy(tmp_path, timestamp, *, interval=3600, deadline=600, auth_age=120):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"data_quality_session_local_timestamp": timestamp}))
    token = tmp_path / "token.json"
    token.touch()
    epoch = NOW.timestamp() - auth_age
    os.utime(token, (epoch, epoch))
    return refresh_policy(
        summary,
        token,
        target_interval_seconds=interval,
        breaker_max_age_seconds=deadline,
        now=NOW,
    )


def test_off_hours_cadence_leaves_room_before_breaker_deadline(tmp_path):
    result = policy(tmp_path, "2026-09-08T16:55:00-04:00", auth_age=600)
    assert result["target_interval_seconds"] == 300
    assert result["measurement_age_seconds"] == 300
    assert result["refresh_due"] is True
    assert result["auth_epoch_refresh_required"] is False
    assert result["resource_admission_required"] is True
    assert result["maintenance_admission_required"] is True


def test_fresh_file_mtime_cannot_hide_old_measurement(tmp_path):
    result = policy(tmp_path, "2026-09-08T20:00:00Z")
    assert result["measurement_age_seconds"] == 3600
    assert result["refresh_due"] is True
    assert result["auth_epoch_refresh_required"] is True


@pytest.mark.parametrize(
    "timestamp", [None, "broken", "2026-09-08T20:59:00", "2026-09-08T22:00:00Z"]
)
def test_missing_invalid_naive_or_future_measurements_are_due(tmp_path, timestamp):
    result = policy(tmp_path, timestamp)
    assert result["measurement_age_seconds"] is None
    assert result["refresh_due"] is True
    assert result["auth_epoch_refresh_required"] is True


@pytest.mark.parametrize("content", ["[]", "null", "{bad"])
def test_malformed_summary_is_not_fresh(tmp_path, content):
    summary = tmp_path / "summary.json"
    summary.write_text(content)
    result = refresh_policy(
        summary,
        tmp_path / "missing_token",
        target_interval_seconds=300,
        breaker_max_age_seconds=600,
        now=NOW,
    )
    assert result["refresh_due"] is True
    assert result["auth_epoch_refresh_required"] is False


def test_fresh_post_auth_measurement_does_not_trigger(tmp_path):
    result = policy(tmp_path, "2026-09-08T20:59:00Z", interval=180)
    assert result["target_interval_seconds"] == 180
    assert result["refresh_due"] is False


def test_recent_measurement_from_prior_auth_epoch_is_due(tmp_path):
    assert policy(tmp_path, "2026-09-08T20:59:00Z", auth_age=10)["refresh_due"] is True


def test_disabled_deadline_preserves_requested_cadence(tmp_path):
    assert (
        policy(tmp_path, "2026-09-08T20:59:00Z", deadline=0)["target_interval_seconds"]
        == 3600
    )


@pytest.mark.parametrize("blocker", ["resource", "maintenance"])
def test_wrapper_keeps_real_admission_guards_when_evidence_is_due(tmp_path, blocker):
    ops = tmp_path / "scripts" / "ops"
    ops.mkdir(parents=True)
    python = tmp_path / ".venv314" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    for name in ("run_one_numbers_refresh_launchd.sh", "one_numbers_refresh_policy.py"):
        shutil.copyfile(ROOT / "scripts" / "ops" / name, ops / name)
    (ops.parent / "resource_guard.py").write_text(
        "import sys\nprint('real_resource_guard')\nsys.exit(%d)\n"
        % (4 if blocker == "resource" else 0)
    )
    (ops / "maintenance_slot_guard.py").write_text(
        "import sys\nprint('runtime_maintenance_hold')\nsys.exit(75)\n"
    )
    result = subprocess.run(
        ["/bin/zsh", str(ops / "run_one_numbers_refresh_launchd.sh")],
        text=True,
        capture_output=True,
        timeout=15,
        env={"PATH": os.environ["PATH"]},
    )
    assert result.returncode == 0, result.stderr
    assert "measurement_missing_or_invalid" in result.stdout
    assert "target_interval=300" in result.stdout
    assert (
        "real_resource_guard" if blocker == "resource" else "runtime_maintenance_hold"
    ) in result.stdout
    assert "build_one_numbers_report" not in result.stderr
