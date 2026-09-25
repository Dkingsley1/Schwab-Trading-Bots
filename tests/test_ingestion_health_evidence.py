import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from core.ingestion_health_evidence import ingestion_observation_ready
from scripts import promotion_quality_gate as quality
from scripts.ops import daily_verify_auto_remediation_bot as remediation

NOW = datetime(2026, 9, 14, 14, 0, tzinfo=timezone.utc)


def _observation(now=NOW):
    return {
        "timestamp_utc": now.isoformat(),
        "overload": False,
        "line_pressure": False,
        "file_pressure": False,
        "age_pressure": False,
        "ema_pressure": False,
        "trend_up": False,
        "pending_lines": 900,
        "pending_files": 12,
        "files_scanned": 200,
        "pending_lines_threshold": 15000,
        "pending_files_threshold": 45,
        "oldest_age_threshold_seconds": 240,
        "oldest_pending_age_seconds": 1.0,
        "ema_pending_lines": 1100.0,
        "scan_selection": {
            "selected_files": 200,
            "discovered_relevant_files": 441,
            "max_files": 200,
        },
    }


def test_current_bounded_hot_lane_observation_is_accepted():
    assert ingestion_observation_ready(
        _observation(), after_utc=(NOW - timedelta(hours=1)).isoformat(), now=NOW
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("timestamp_utc", None),
        ("timestamp_utc", "bad-date"),
        ("timestamp_utc", "2026-09-14T14:00:00"),
        ("timestamp_utc", (NOW - timedelta(seconds=301)).isoformat()),
        ("timestamp_utc", (NOW + timedelta(seconds=1)).isoformat()),
        ("overload", True),
        ("overload", "false"),
        ("line_pressure", None),
        ("age_pressure", True),
        ("ema_pressure", True),
        ("file_pressure", 0),
        ("trend_up", "false"),
        ("pending_lines", -1),
        ("pending_lines", 15000),
        ("pending_lines", "900"),
        ("pending_files", False),
        ("files_scanned", 199),
        ("pending_lines_threshold", 0),
        ("pending_files_threshold", None),
        ("oldest_age_threshold_seconds", -1),
        ("oldest_pending_age_seconds", float("nan")),
        ("ema_pending_lines", float("inf")),
        ("ema_pending_lines", 10**400),
        ("ema_pending_lines", -1),
        ("scan_selection", {}),
        (
            "scan_selection",
            {"selected_files": 200, "discovered_relevant_files": 199, "max_files": 200},
        ),
    ],
)
def test_invalid_or_unhealthy_observations_fail_closed(field, value):
    payload = _observation()
    payload[field] = value
    assert not ingestion_observation_ready(
        payload, after_utc=(NOW - timedelta(hours=1)).isoformat(), now=NOW
    )


@pytest.mark.parametrize(
    "after", [None, "", NOW.isoformat(), (NOW + timedelta(seconds=1)).isoformat()]
)
def test_observation_must_postdate_a_known_failure(after):
    assert not ingestion_observation_ready(_observation(), after_utc=after, now=NOW)


@pytest.mark.parametrize("field", list(_observation()))
def test_partial_observation_cannot_resolve_failure(field):
    payload = _observation()
    del payload[field]
    assert not ingestion_observation_ready(
        payload, after_utc=(NOW - timedelta(hours=1)).isoformat(), now=NOW
    )


def test_file_pressure_with_rising_trend_is_rejected():
    payload = _observation()
    payload.update(file_pressure=True, trend_up=True)
    assert not ingestion_observation_ready(
        payload, after_utc=(NOW - timedelta(hours=1)).isoformat(), now=NOW
    )


def _evaluate(daily, observation):
    return quality.evaluate_quality(
        {"promote_ok": True, "considered_bots": 5, "fail_share": 0.0},
        daily,
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        reconciliation_slo={"ok": True},
        ingestion_backpressure=observation,
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=True,
    )


def test_quality_reconciles_only_named_failure_without_rewriting_history():
    now = datetime.now(timezone.utc)
    daily = {
        "ok": False,
        "running": False,
        "timestamp_utc": (now - timedelta(hours=1)).isoformat(),
        "failed_checks": ["ingestion_backpressure", "unmapped_failure"],
        "checks": {"ingestion_backpressure": {"rc": 124, "ok": False}},
    }
    before = copy.deepcopy(daily)
    ok, failed, details = _evaluate(daily, _observation(now))
    assert not ok
    assert "daily_verify_not_ok" in failed
    assert details["daily_verify_resolved_failed_checks"] == ["ingestion_backpressure"]
    assert details["daily_verify_unresolved_failed_checks"] == ["unmapped_failure"]
    assert daily == before


def test_quality_clears_recovered_timeout_but_does_not_override_promotion_floor():
    now = datetime.now(timezone.utc)
    daily = {
        "ok": False,
        "running": False,
        "timestamp_utc": (now - timedelta(hours=1)).isoformat(),
        "failed_checks": ["ingestion_backpressure"],
    }
    ok, failed, details = _evaluate(daily, _observation(now))
    assert ok and not failed
    assert details["daily_verify_ingestion_observation"]["resolved"] is True
    ok, failed, details = quality.evaluate_quality(
        {"promote_ok": False, "considered_bots": 1, "fail_share": 1.0},
        daily,
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        reconciliation_slo={"ok": True},
        ingestion_backpressure=_observation(now),
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=True,
    )
    assert not ok and failed
    assert details["daily_verify_ok"] is True
    assert details["promotion"]["considered_bots"] == 1
    assert details["promotion"]["min_considered_bots"] == 4


@pytest.mark.parametrize("running", [True, None, "false"])
def test_quality_does_not_reconcile_an_unfinished_or_unknown_daily_run(running):
    now = datetime.now(timezone.utc)
    daily = {
        "ok": False,
        "running": running,
        "timestamp_utc": (now - timedelta(hours=1)).isoformat(),
        "failed_checks": ["ingestion_backpressure"],
    }
    assert _evaluate(daily, _observation(now))[2][
        "daily_verify_unresolved_failed_checks"
    ] == ["ingestion_backpressure"]


def test_quality_new_wrapper_cannot_renew_stale_underlying_observation():
    now = datetime.now(timezone.utc)
    daily = {
        "ok": False,
        "running": False,
        "timestamp_utc": (now - timedelta(hours=1)).isoformat(),
        "failed_checks": ["ingestion_backpressure"],
    }
    observation = _observation(now - timedelta(minutes=6))
    observation["updated_utc"] = now.isoformat()
    assert _evaluate(daily, observation)[2][
        "daily_verify_unresolved_failed_checks"
    ] == ["ingestion_backpressure"]


@pytest.mark.parametrize(
    "rc,mode,expected",
    [
        (0, "healthy", True),
        (124, "healthy", False),
        (0, "overload", False),
        (0, "stale", False),
        (0, "empty", False),
        (0, "malformed", False),
    ],
)
def test_native_remediator_requires_new_healthy_evidence(
    tmp_path, monkeypatch, rc, mode, expected
):
    daily_file = tmp_path / "governance/health/daily_auto_verify_latest.json"
    daily_file.parent.mkdir(parents=True)
    original = json.dumps({"failed_checks": ["ingestion_backpressure"]})
    daily_file.write_text(original)
    commands = []

    def run(cmd, *, timeout_sec):
        commands.append(cmd)
        if not any("ingestion_backpressure_guard.py" in part for part in cmd):
            return 0, "{}", ""
        observation = _observation(datetime.now(timezone.utc))
        if mode == "overload":
            observation["overload"] = True
        if mode == "stale":
            observation["timestamp_utc"] = (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            ).isoformat()
        stdout = json.dumps(observation)
        if mode == "empty":
            stdout = "{}"
        if mode == "malformed":
            stdout = stdout[:-1]
        return rc, stdout, ""

    monkeypatch.setattr(remediation, "_run", run)
    payload = remediation.build_payload(tmp_path, apply=True)
    assert payload["attempts"][0]["ok"] is expected
    assert payload["resolved_checks"] == (
        ["ingestion_backpressure"] if expected else []
    )
    assert daily_file.read_text() == original
    assert any("ingestion_backpressure_guard.py" in " ".join(cmd) for cmd in commands)


def test_native_remediator_preview_does_not_run_commands(tmp_path, monkeypatch):
    daily_file = tmp_path / "governance/health/daily_auto_verify_latest.json"
    daily_file.parent.mkdir(parents=True)
    daily_file.write_text(json.dumps({"failed_checks": ["ingestion_backpressure"]}))
    monkeypatch.setattr(
        remediation,
        "_run",
        lambda *args, **kwargs: pytest.fail("preview executed a command"),
    )
    payload = remediation.build_payload(tmp_path, apply=False)
    assert payload["unresolved_checks"] == ["ingestion_backpressure"]
    assert payload["attempts"][0]["actionable"] is True
