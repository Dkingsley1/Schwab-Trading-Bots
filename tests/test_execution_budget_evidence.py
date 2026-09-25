from copy import deepcopy
from datetime import datetime, timedelta, timezone
import fcntl
import json

import pytest

from scripts import execution_budgeter as budget
from scripts import sleeve_slo_guard as slo
from scripts import risk_service_boundary as risk
from scripts.ops import adaptive_ops_recovery_policy as adaptive
from scripts.ops import readiness_evidence_refresh as scheduled
from scripts.ops.long_runtime_common import evidence_freshness

NOW = datetime(2026, 9, 23, 18, tzinfo=timezone.utc)


def watchdog(now=NOW):
    return {
        "timestamp_utc": now.isoformat(),
        "status": [
            {
                "name": "all_sleeves",
                "process_live": True,
                "heartbeat_ok": True,
                "heartbeat_age_seconds": 5,
                "heartbeat_max_age_seconds": 360,
            }
        ],
        "restart_storms": [],
    }


def write(root, name, payload):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def seed(root):
    write(root, "governance/health/process_watchdog_latest.json", watchdog())
    write(
        root,
        "governance/allocator/sleeve_allocator_latest.json",
        {
            "timestamp_utc": NOW.isoformat(),
            "target_weights": {"dividend": 1.0, "crypto": 0.0},
        },
    )
    write(
        root,
        "governance/risk/portfolio_risk_latest.json",
        {
            "timestamp_utc": NOW.isoformat(),
            "ok": True,
            "overall_status": "ready",
            "risk_level": "low",
            "limits": {"sleeve_exposure_caps": {"dividend": 1.0, "crypto": 0.0}},
        },
    )
    payload, _ = slo.watchdog_payload(watchdog(), {}, now=NOW)
    write(root, "governance/watchdog/sleeve_slo_latest.json", payload)


@pytest.mark.parametrize(
    "mutation", ["empty", "missing", "stale", "future", "duplicate", "no_sleeves"]
)
def test_bad_watchdog_cannot_be_rewrapped_as_healthy(mutation):
    data = watchdog()
    if mutation == "empty":
        data["status"] = []
    elif mutation == "missing":
        data = {}
    elif mutation in {"stale", "future"}:
        data["timestamp_utc"] = (
            NOW + timedelta(minutes=1 if mutation == "future" else -7)
        ).isoformat()
    elif mutation == "duplicate":
        data["status"].append(deepcopy(data["status"][0]))
    else:
        data["status"][0]["name"] = "unknown"
    result, _ = slo.watchdog_payload(data, {}, now=NOW)
    assert not result["ok"]
    assert not result["input_freshness"]["sources_ready"]
    assert result["alerts"]


def test_sustained_breaches_require_distinct_observations_and_keep_recovery():
    data = watchdog()
    data["status"][0]["process_live"] = False
    result, state = slo.watchdog_payload(data, {}, now=NOW)
    for _ in range(4):
        result, state = slo.watchdog_payload(data, state, now=NOW)
    assert result["targets"][0]["breach_streak"] == 1
    for seconds in (30, 60):
        data["timestamp_utc"] = (NOW + timedelta(seconds=seconds)).isoformat()
        result, state = slo.watchdog_payload(
            data, state, now=NOW + timedelta(seconds=seconds)
        )
    assert result["alerts"][0]["streak"] == 3
    result, state = slo.watchdog_payload(
        watchdog(NOW + timedelta(seconds=90)), state, now=NOW + timedelta(seconds=90)
    )
    assert result["ok"] and state["streaks"]["all_sleeves"] == 0


def test_snapshot_does_not_freeze_heartbeat_age():
    data = watchdog()
    data["status"][0]["heartbeat_age_seconds"] = 350
    result, _ = slo.watchdog_payload(data, {}, now=NOW + timedelta(seconds=20))
    assert result["targets"][0]["heartbeat_ok"] is False
    assert result["targets"][0]["heartbeat_age_s"] > 360


def test_same_timestamp_with_changed_payload_is_not_new_health_proof():
    data = watchdog()
    _, state = slo.watchdog_payload(data, {}, now=NOW)
    data["status"][0]["process_live"] = False
    result, updated = slo.watchdog_payload(data, state, now=NOW)
    assert not result["input_freshness"]["sources_ready"]
    assert updated == state


def test_reported_storm_is_immediate_alert_not_invented_history():
    data = watchdog()
    data["restart_storms"] = [{"name": "all_sleeves"}]
    result, _ = slo.watchdog_payload(data, {}, now=NOW)
    assert result["alerts"][0]["name"] == "all_sleeves"
    assert "restarts_last_hour" not in result["targets"][0]


def test_budget_preserves_caps_and_zero_weight_cannot_get_minimum_orders(tmp_path):
    seed(tmp_path)
    result = budget.build_payload(tmp_path, now=NOW)
    assert result["ok"]
    assert result["sleeves"]["dividend"]["max_actions_per_hour"] == 30
    assert result["sleeves"]["crypto"]["max_actions_per_hour"] == 0
    assert result["sleeves"]["crypto"]["max_open_orders"] == 0
    assert result["live_execution_authority"] is False
    assert (
        evidence_freshness(result, now=NOW + timedelta(minutes=7), max_age_minutes=120)[
            "fresh"
        ]
        is False
    )


@pytest.mark.parametrize(
    "mutation",
    ["stale", "future", "missing", "forged_wrapper", "bad_weight", "risk_blocked"],
)
def test_bad_budget_input_produces_zero_blocked_budget(tmp_path, mutation):
    seed(tmp_path)
    path = tmp_path / "governance/watchdog/sleeve_slo_latest.json"
    data = json.loads(path.read_text())
    if mutation == "missing":
        path.unlink()
    elif mutation == "bad_weight":
        path = tmp_path / "governance/allocator/sleeve_allocator_latest.json"
        data = json.loads(path.read_text())
        data["target_weights"]["dividend"] = float("nan")
        path.write_text(json.dumps(data))
    elif mutation == "risk_blocked":
        path = tmp_path / "governance/risk/portfolio_risk_latest.json"
        data = json.loads(path.read_text())
        data["ok"] = False
        path.write_text(json.dumps(data))
    else:
        stamp = NOW + timedelta(minutes=1 if mutation == "future" else -7)
        data["source_observed_at_utc"] = stamp.isoformat()
        if mutation != "forged_wrapper":
            data["timestamp_utc"] = stamp.isoformat()
        path.write_text(json.dumps(data))
    result = budget.build_payload(tmp_path, now=NOW)
    assert not result["ok"]
    assert result["blockers"]
    assert result["global"]["max_total_actions_per_hour"] == 0


def test_observed_slo_alerts_reduce_budget_without_faking_healthy_slo(tmp_path):
    seed(tmp_path)
    path = tmp_path / "governance/watchdog/sleeve_slo_latest.json"
    data = json.loads(path.read_text())
    data.update(ok=False, overall_ok=False, alerts=[{"name": "all_sleeves"}])
    path.write_text(json.dumps(data))
    result = budget.build_payload(tmp_path, now=NOW)
    assert result["ok"] and result["global"]["multiplier"] == 0.85


def test_refresh_uses_current_watchdog_and_respects_single_writer(
    tmp_path, monkeypatch
):
    seed(tmp_path)
    monkeypatch.setattr(slo, "_now_utc", lambda: NOW)
    real_build = budget.build_payload
    monkeypatch.setattr(
        budget, "build_payload", lambda *a, **kw: real_build(*a, **kw, now=NOW)
    )
    result = budget.refresh(tmp_path, refresh_slo=True)
    assert result["ok"]
    path = tmp_path / "governance/risk/execution_budget_latest.json"
    before = path.read_bytes()
    with path.with_suffix(".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            budget.refresh(tmp_path, refresh_slo=True)
    assert path.read_bytes() == before


def test_protected_alias_is_rejected_before_payload_read(tmp_path):
    (tmp_path / "governance").symlink_to("/Volumes/VIDEO", target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe_risk_evidence_path"):
        budget.refresh(tmp_path, refresh_slo=True)


def test_risk_input_without_timestamp_cannot_use_mtime(tmp_path):
    path = write(tmp_path, "budget.json", {"global": {}})
    result = risk._input_health(
        "execution_budget", path, {"global": {}}, now=NOW, max_age_minutes=120
    )
    assert not result["ready"]
    result = risk._input_health(
        "execution_budget",
        path,
        {"global": {}, "timestamp_utc": (NOW + timedelta(minutes=1)).isoformat()},
        now=NOW,
        max_age_minutes=120,
    )
    assert not result["ready"]


def test_native_schedules_refresh_owners_without_trading():
    step = next(
        row for row in scheduled.default_steps() if row["name"] == "execution_budget"
    )
    assert "--refresh-slo" in step["args"]
    plan = adaptive.build_control_plane_refresh_plan(
        {
            "control_plane_freshness": {
                "risk_service_boundary": {"refresh_due": True},
                "health_gates": {"refresh_due": True},
            }
        }
    )
    assert plan[0]["id"] == "risk_service_boundary"
    assert plan[0]["command"] == [
        "./scripts/ops/opsctl.sh",
        "risk-service-boundary",
        "--refresh-inputs",
        "--json",
    ]
    assert plan[0]["timeout_seconds"] == 30


def test_risk_carries_shortest_upstream_expiry(tmp_path):
    seed(tmp_path)
    payload = budget.build_payload(tmp_path, now=NOW)
    path = write(tmp_path, "budget.json", payload)
    health = risk._input_health(
        "execution_budget", path, payload, now=NOW, max_age_minutes=120
    )
    assert health["ready"]
    assert health["valid_until_utc"] == (NOW + timedelta(minutes=6)).isoformat()


def test_native_refresh_does_not_wait_for_wrapper_age_when_inputs_expire(tmp_path):
    now = datetime.now(timezone.utc)
    write(tmp_path, "governance/risk/risk_service_boundary_latest.json", {
        "timestamp_utc": now.isoformat(), "valid_until_utc": (now + timedelta(seconds=20)).isoformat(),
        "ok": True,
    })
    metric = adaptive._risk_evidence_metric(tmp_path)
    assert metric["fresh"] and metric["refresh_due"] and metric["input_expiry_due"]
