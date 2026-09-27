from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pytest

from scripts.ops import autonomic_resource_governor as autonomic
from scripts.ops import governor_refresh as refresh
from scripts.ops import memory_pressure_intelligence as memory
from scripts.ops import runtime_throttle_control as runtime
from scripts.ops.long_runtime_common import governor_recovery_observation


def test_fast_polls_preserve_minute_spaced_recovery_credits():
    baseline = datetime.now(timezone.utc)
    previous = {}
    counts = []
    gates = []
    for elapsed in (0, 20, 40, 60, 80, 120):
        now = baseline + timedelta(seconds=elapsed)
        recovery = governor_recovery_observation(previous, now.isoformat(), now=now)
        gate = memory._reopen_gate(
            {"status": "clear"},
            {
                "recovery_sample_due": recovery["credit_due"],
                "previous_clear_samples": previous.get("count", 0),
            },
            {},
            {},
            {},
        )
        counts.append(gate["consecutive_memory_clear_samples"])
        gates.append(gate)
        previous = {
            "timestamp_utc": now.isoformat(),
            "recovery_observation": recovery,
            "count": counts[-1],
        }
    assert counts == [1, 1, 1, 2, 2, 3]
    assert not gates[2]["safe_to_widen_p_core_workers"]
    assert gates[3]["safe_to_widen_p_core_workers"]
    assert not gates[4]["safe_for_training"]
    assert gates[5]["safe_for_training"]


def test_pressure_resets_recovery_between_healthy_credit_slots():
    trend = {
        "new_observation": True,
        "recovery_sample_due": False,
        "previous_clear_samples": 8,
        "heating": True,
        "status": "regressing",
    }
    gate = memory._reopen_gate({"status": "hard_relief"}, trend, {}, {}, {})
    assert gate["consecutive_memory_clear_samples"] == 0
    assert not gate["safe_to_widen_p_core_workers"]
    stability = autonomic._stability_state(
        {"green": False},
        {"overall_status": "degraded", "memory_pressure_level": "high"},
        {},
        trend,
        {"stability_state": {"consecutive_green_samples": 8}},
    )
    assert stability["consecutive_green_samples"] == 0
    assert not stability["training_reentry_ready"]


def test_autonomic_stability_does_not_count_every_fast_poll():
    prior = {"stability_state": {"consecutive_green_samples": 1}}
    result = autonomic._stability_state(
        {"green": True},
        {"overall_status": "ready"},
        {},
        {"new_observation": True, "recovery_sample_due": False},
        prior,
    )
    assert result["consecutive_green_samples"] == 1
    assert not result["collector_reopen_ready"]


@pytest.mark.parametrize("seconds,ready", [(0, True), (20, True), (60, False)])
def test_repeated_or_unready_evidence_cannot_earn_credit(seconds, ready):
    now = datetime.now(timezone.utc)
    prior = {"timestamp_utc": now.isoformat(), "source_timestamp_utc": now.isoformat()}
    current = now + timedelta(seconds=seconds)
    source = now if seconds == 0 else current
    result = governor_recovery_observation(
        prior, source.isoformat(), inputs_ready=ready, now=current
    )
    assert not result["credit_due"]


@pytest.mark.parametrize("previous_age", [301, -1])
def test_expired_or_future_recovery_history_is_reset(previous_age):
    now = datetime.now(timezone.utc)
    stamp = (now - timedelta(seconds=previous_age)).isoformat()
    result = governor_recovery_observation(
        {"timestamp_utc": stamp, "source_timestamp_utc": stamp},
        now.isoformat(),
        now=now,
    )
    assert result["reset_history"]
    assert result["credit_due"]  # One baseline, never inherited accumulated credit.


@pytest.mark.parametrize("source", [None, "invalid", "2999-01-01T00:00:00+00:00"])
def test_invalid_current_recovery_source_fails_closed(source):
    result = governor_recovery_observation({}, source)
    assert result["reset_history"]
    assert not result["credit_due"]
    assert result["last_credit_source_timestamp_utc"] is None


def test_runtime_publishes_the_applied_decision_without_resampling(
    tmp_path, monkeypatch
):
    calls = []
    applied = []
    out = tmp_path / "runtime.json"

    def build(root):
        calls.append(root)
        return {
            "timestamp_utc": "2026-09-12T15:00:00+00:00",
            "overall_status": "ready",
            "throttle_profile": "protect",
            "controller_contract": {},
        }

    def apply(root, payload, **kwargs):
        applied.append(payload["throttle_profile"])
        return {"applied": True, "support_maintenance_pause": {"pause_requested": True}}

    monkeypatch.setattr(runtime, "build_payload", build)
    monkeypatch.setattr(runtime, "apply_runtime_guard", apply)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runtime-throttle",
            "--project-root",
            str(tmp_path),
            "--out-file",
            str(out),
            "--apply",
            "--json",
        ],
    )
    assert runtime.main() == 0
    payload = json.loads(out.read_text())
    assert calls == [tmp_path]
    assert applied == [payload["throttle_profile"]]
    assert payload["controller_contract"]["observation_phase"] == "pre_apply"
    assert payload["apply_result"]["support_maintenance_pause"]["pause_requested"]


def test_optional_work_yields_to_fast_cadence_without_partial_attempt(
    tmp_path, monkeypatch
):
    elapsed = [0.0]
    calls = []

    def run(root, step, deadline):
        calls.append(step[0])
        elapsed[0] += 2
        return {
            "owner": step[0],
            "status": "complete",
            "attempted": True,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }

    monkeypatch.setattr(refresh.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(refresh, "_run_step", run)
    result = refresh.run_cycle(tmp_path)
    assert result["fast_control_complete"]
    assert result["fast_control_elapsed_seconds"] == 12
    assert result["cadence_seconds"] == 20
    assert result["deadline_seconds"] == 18
    assert calls == [step[0] for step in refresh.FAST_STEPS]
    assert result["steps"][-1]["reason"] == "fast_cadence_budget_reserved"
    assert result["slow_attempts"] == {}
    assert not result["ok"]


def test_installer_and_catalog_match_fast_owner_cadence():
    from scripts.ops import ops_scheduled_job_catalog as catalog

    installer = (
        Path(refresh.PROJECT_ROOT) / "scripts/ops/install_ops_automation_launchd.sh"
    ).read_text()
    assert "RUNTIME_SMOOTH_MODE_INTERVAL_SECONDS:-20" in installer
    assert '"$RUNTIME_SMOOTH_MODE_INTERVAL" 60 /bin/zsh' in installer
    spec = next(
        job for job in catalog.DEFAULT_JOB_SPECS if job.job_id == "runtime_smooth_mode"
    )
    assert spec.cadence_seconds == refresh.CADENCE_SECONDS
    assert spec.deadline_seconds == 60
