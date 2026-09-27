import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from core.status_label_contract import (
    bot_definition_labels,
    evidence_label,
    paper_hold_labels,
    read_label_source,
)
from scripts.ops import data_plane_recovery_controller as recovery
from scripts.ops import runtime_gate_dashboard as dashboard
from scripts.ops import runtime_throttle_control as throttle
from scripts.ops import live_feed_status_contract as livefeed
from scripts import run_execution_lane as execution

NOW = datetime(2026, 9, 15, 23, 30, tzinfo=timezone.utc)


def label(payload):
    return evidence_label(
        payload, scope="test", source="test.json", max_age_seconds=120, now=NOW
    )


@pytest.mark.parametrize(
    "stamp,expected",
    [
        (None, "timestamp_invalid"),
        ("bad", "timestamp_invalid"),
        ("2026-09-15T23:30:00", "timestamp_invalid"),
        ("2026-09-15T23:31:00+00:00", "future"),
        ("2026-09-15T23:27:59+00:00", "stale"),
        ("2026-09-15T23:28:00+00:00", "fresh"),
    ],
)
def test_labels_require_aware_nonfuture_unexpired_evidence(stamp, expected):
    result = label({"timestamp_utc": stamp, "overall_status": "ready"})
    assert result["evidence_status"] == expected
    assert result["reported_status"] == "ready"
    assert result["status"] == (
        "ready" if expected == "fresh" else f"evidence_{expected}"
    )


def test_new_report_does_not_renew_old_observation():
    result = label(
        {
            "timestamp_utc": NOW.isoformat(),
            "source_timestamp_utc": (NOW - timedelta(hours=2)).isoformat(),
            "ok": True,
        }
    )
    assert result["evidence_status"] == "stale"
    assert result["reported_status"] == "producer_completed"


def test_invalid_source_time_does_not_fall_back_to_report_time():
    assert (
        label({"timestamp_utc": NOW.isoformat(), "source_timestamp_utc": "bad"})[
            "evidence_status"
        ]
        == "timestamp_invalid"
    )


@pytest.mark.parametrize(
    "offset,expected", [(10, "future"), (-10, "timestamp_inconsistent")]
)
def test_producer_time_cannot_contradict_observation(offset, expected):
    assert (
        label(
            {
                "source_timestamp_utc": NOW.isoformat(),
                "timestamp_utc": (NOW + timedelta(seconds=offset)).isoformat(),
            }
        )["evidence_status"]
        == expected
    )


def test_missing_time_is_unknown_even_when_producer_says_ready():
    assert label({})["status"] == "evidence_missing"
    assert label({"overall_status": "ready"})["status"] == "evidence_timestamp_missing"


def test_missing_age_budget_is_explicit_not_a_freshness_claim():
    result = evidence_label(
        {"timestamp_utc": NOW.isoformat(), "overall_status": "ready"},
        scope="test",
        source="test",
        max_age_seconds=None,
        now=NOW,
    )
    assert result["status"] == "evidence_age_budget_unspecified"
    assert result["fresh"] is False


def test_report_completion_does_not_overwrite_degradation():
    payload = {
        "ok": True,
        "overall_status": "degraded",
        "timestamp_utc": NOW.isoformat(),
    }
    assert label(payload)["status"] == "degraded"
    assert dashboard._infer_status(payload, True) == "degraded"


def test_conflicting_producer_verdicts_are_not_silently_collapsed():
    result = label(
        {
            "timestamp_utc": NOW.isoformat(),
            "overall_status": "ready",
            "status": "blocked",
        }
    )
    assert result["status"] == "conflicting_status"
    assert result["producer_status_fields"] == {
        "overall_status": "ready",
        "status": "blocked",
    }


def hold_sources():
    stamp = {"timestamp_utc": NOW.isoformat()}
    return (
        {
            **stamp,
            "execution_safety_hold": {
                "active": True,
                "reason": "paper_execution_paused_for_runtime_pressure",
            },
            "runtime_execution_breaker": {
                "active": True,
                "reasons": ["market_session_closed"],
            },
        },
        {
            **stamp,
            "paper_execution_policy": {
                "pause_paper_execution": True,
                "pressure_pause_active": False,
                "reason": "paper_ramp_blocked",
                "blockers": ["write_path_recovery_pending"],
            },
        },
        {**stamp, "local_storage_reserve": {"pressure_active": True}},
    )


def test_legacy_generic_pause_reports_actual_current_causes_without_releasing():
    sources = hold_sources()
    before = copy.deepcopy(sources)
    result = paper_hold_labels(*sources, now=NOW)
    assert sources == before
    assert result["observed_runtime_hold"] is True
    assert result["current_policy_reasons"] == [
        "local_storage_reserve_pressure",
        "paper_ramp_blocked",
        "write_path_recovery_pending",
    ]
    assert result["execution_breaker_reasons"] == ["market_session_closed"]
    assert "cpu_pressure" not in str(result["current_policy_reasons"])


def test_stale_policy_does_not_explain_current_hold():
    lane, policy, storage = hold_sources()
    policy["timestamp_utc"] = storage["timestamp_utc"] = (
        NOW - timedelta(hours=1)
    ).isoformat()
    result = paper_hold_labels(lane, policy, storage, now=NOW)
    assert result["current_policy_reasons"] == ["runtime_hold_cause_unverified"]


def test_policy_hold_does_not_invent_observed_execution_hold():
    lane, policy, storage = hold_sources()
    lane["timestamp_utc"] = (NOW - timedelta(hours=1)).isoformat()
    result = paper_hold_labels(lane, policy, storage, now=NOW)
    assert result["status"] == "execution_evidence_unavailable"
    assert not result["observed_runtime_hold"]
    assert result["current_policy_reasons"]


def test_real_cpu_pressure_is_labeled_separately():
    lane, policy, _ = hold_sources()
    policy["paper_execution_policy"].update(
        pressure_pause_active=True, pressure_pause_reason="paper_execution_cpu_pressure"
    )
    result = paper_hold_labels(lane, policy, {}, now=NOW)
    assert "paper_execution_cpu_pressure" in result["current_policy_reasons"]


def test_bot_configuration_is_not_runtime_profit_or_label_evidence():
    bot = {"active": True, "data_collection_active": True, "label_contract": {}}
    record = {
        "definition_complete": True,
        "process_definition": {
            "definition_valid": True,
            "implementation_kind": "collection_wrapper",
        },
    }
    result = bot_definition_labels(bot, record)
    assert result["registry"] == "declared_active"
    assert result["collection"] == "configured_enabled"
    assert result["process"] == "defined_not_runtime_verified"
    assert (
        result["runtime"]
        == result["economic_evidence"]
        == "not_assessed_by_definition_audit"
    )
    assert result["training_labels"] == "contract_declared_not_outcomes_verified"


def test_incomplete_bot_binding_does_not_assert_implementation():
    result = bot_definition_labels(
        {"active": True, "lifecycle_state": "retired"},
        {
            "definition_complete": False,
            "process_definition": {"implementation_kind": "live_trader"},
        },
    )
    assert result["registry"] == "declared_retired"
    assert result["implementation"] == "source_binding_unverified"


@pytest.mark.parametrize(
    "flags,expected",
    [
        (
            {"PAPER_EXECUTION_RUNTIME_PAUSED_FOR_LOCAL_STORAGE": "1"},
            "paper_execution_paused_for_local_storage",
        ),
        (
            {"PAPER_400_RAMP_BLOCKED_RUNTIME_PAUSE": "1"},
            "paper_execution_paused_for_paper_ramp",
        ),
        (
            {
                "PAPER_EXECUTION_RUNTIME_PAUSED_FOR_PRESSURE": "1",
                "PAPER_EXECUTION_RUNTIME_PAUSE_REASON": "paper_ramp_blocked",
            },
            "paper_ramp_blocked",
        ),
        (
            {"PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED": "0"},
            "paper_execution_queue_consumer_disabled",
        ),
    ],
)
def test_executor_pause_reasons_preserve_hold(flags, expected, monkeypatch):
    monkeypatch.setattr(execution, "CONTROL_ENV_FILES", ())
    for key in execution.CONTROL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in flags.items():
        monkeypatch.setenv(key, value)
    assert execution._paper_execution_paused_for_runtime()
    assert execution._paper_runtime_pause_reason() == expected


def test_governor_carries_exact_reason_and_clears_it_on_release():
    policy = {
        "artifact_present": True,
        "pause_paper_execution": True,
        "reason": "paper_ramp_blocked",
        "pressure_pause_active": False,
    }
    held = throttle._runtime_env_overrides(
        "observe", "normal", "normal", paper_execution_policy=policy
    )
    assert held["PAPER_EXECUTION_RUNTIME_PAUSE_REASON"] == "paper_ramp_blocked"
    assert held["PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED"] == "0"
    policy["pause_paper_execution"] = False
    released = throttle._runtime_env_overrides(
        "observe", "normal", "normal", paper_execution_policy=policy
    )
    assert released["PAPER_EXECUTION_RUNTIME_PAUSE_REASON"] == ""


def test_reporting_reader_rejects_external_route_without_target_probe(
    tmp_path, monkeypatch
):
    path = tmp_path / "source.json"
    path.symlink_to("/outside-status-test/source.json")
    monkeypatch.setattr(
        "core.status_label_contract.os.open", lambda *a, **k: pytest.fail("must reject before opening")
    )
    assert (
        read_label_source(tmp_path, "source.json")["_label_source_error"]
        == "route_rejected"
    )


def test_unreadable_or_malformed_source_is_not_reported_missing(tmp_path):
    (tmp_path / "bad.json").write_text("not json")
    assert (
        label(read_label_source(tmp_path, "bad.json"))["status"]
        == "evidence_invalid_payload"
    )
    assert (
        label(read_label_source(tmp_path, "missing.json"))["status"]
        == "evidence_missing"
    )


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"overall_status": "ready"}, "timestamp_missing"),
        (
            {"overall_status": "ready", "timestamp_utc": "2099-01-01T00:00:00+00:00"},
            "future",
        ),
        (
            {
                "overall_status": "ready",
                "timestamp_utc": NOW.isoformat(),
                "source_timestamp_utc": "2026-09-14T00:00:00+00:00",
            },
            "stale",
        ),
    ],
)
def test_livefeed_does_not_certify_freshness_from_mtime_or_future_report(
    tmp_path, payload, expected
):
    source = tmp_path / "governance/health/test.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(payload))
    result = livefeed._artifact(tmp_path, "test.json", 120, NOW)
    assert result["fresh"] is False
    assert result["status_label"]["evidence_status"] == expected


def test_livefeed_missing_policy_is_unknown_not_allowed():
    result = livefeed._throttle_row({"throttle": {"payload": {}, "fresh": False}}, {})
    assert result["paper_state"] == "unknown"
    assert not result["paper_allowed"]
    assert result["cause"] == "runtime_evidence_stale_or_missing"


@pytest.mark.parametrize("sql_errors", [None, -1, True, float("inf"), float("nan")])
def test_unknown_sql_errors_cannot_be_labeled_zero(tmp_path, sql_errors):
    health = tmp_path / "governance/health"
    health.mkdir(parents=True)
    (health / "ingestion_storage_control_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "data_integrity": {"sql_overlay_ops_write_failures": sql_errors},
                "sql_ingestion_pending_overlay": {
                    "fresh_source_count": 1,
                    "max_source_age_seconds": 1,
                },
            }
        )
    )
    result = recovery.build_payload(tmp_path)
    assert result["recovery_diagnostics"]["current_sql_write_failure_count"] is None


def test_historical_failure_debt_is_not_current_sql_failure_count(tmp_path):
    now = datetime.now(timezone.utc)
    events = tmp_path / "governance/events"
    health = tmp_path / "governance/health"
    events.mkdir(parents=True)
    health.mkdir(parents=True)
    row = {
        "event": "write_failure",
        "timestamp_utc": (now - timedelta(hours=1)).isoformat(),
        "source": "writer",
        "target_path": "/logical/target",
        "error": "failed",
    }
    (events / f"write_failures_{now:%Y%m%d}.jsonl").write_text(json.dumps(row) + "\n")
    (health / "ingestion_storage_control_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": now.isoformat(),
                "data_integrity": {"sql_overlay_ops_write_failures": 0},
                "sql_ingestion_pending_overlay": {
                    "fresh_source_count": 1,
                    "max_source_age_seconds": 1,
                },
            }
        )
    )
    result = recovery.build_payload(tmp_path)
    diagnostic = result["recovery_diagnostics"]
    assert result["write_failure_count"] == diagnostic["historical_failure_count"] == 1
    assert diagnostic["recent_observed_failure_count"] == 0
    assert diagnostic["current_sql_write_failure_count"] == 0
    assert (
        "historical_failure_reconciliation_not_verified"
        in diagnostic["unmet_requirements"]
    )
    assert not result["write_path_recovered_by_storage"]


@pytest.mark.parametrize(
    "overlay",
    [
        {},
        {
            "fresh_source_count": 0,
            "stale_source_count": 27,
            "max_source_age_seconds": 0,
        },
        {"fresh_source_count": 1, "max_source_age_seconds": 3600},
        {"fresh_source_count": 1, "max_source_age_seconds": -1},
    ],
)
def test_default_zero_from_absent_or_stale_overlay_is_unknown(tmp_path, overlay):
    health = tmp_path / "governance/health"
    health.mkdir(parents=True)
    (health / "ingestion_storage_control_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "data_integrity": {"sql_overlay_ops_write_failures": 0},
                "sql_ingestion_pending_overlay": overlay,
            }
        )
    )
    result = recovery.build_payload(tmp_path)["recovery_diagnostics"]
    assert result["reported_sql_overlay_failure_count"] == 0
    assert result["current_sql_write_failure_count"] is None
    assert result["sql_overlay_coverage"] == "unavailable"
