import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.order_intent import canonical_payload_sha256
from scripts.ops import purchase_proposals as cli
from scripts.ops import readiness_evidence_refresh as refresh

ROOT = Path(__file__).resolve().parents[1]


def put(root, relative, payload):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture
def configured(tmp_path, monkeypatch):
    for path in (cli.POLICY_PATH, cli.broker_test.POLICY_PATH):
        put(tmp_path, path, json.loads((ROOT / path).read_text()))
    policy = cli.read(tmp_path, cli.POLICY_PATH)
    current = datetime.now(timezone.utc)
    policy.update(
        valid_from_utc=(current - timedelta(days=1)).isoformat(),
        expires_at_utc=(current + timedelta(days=29)).isoformat(),
    )
    put(tmp_path, cli.POLICY_PATH, policy)
    put(
        tmp_path,
        "governance/runtime/production_candidate_state.json",
        {"candidate_id": "candidate-test"},
    )
    monkeypatch.setattr(cli, "write", put)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli.broker_test, "current_source_blockers", lambda *args: [])
    monkeypatch.setattr(cli, "_equity_session_state", lambda **kwargs: {"ready": True})
    return tmp_path


def observed(root):
    test = cli.read(root, cli.broker_test.POLICY_PATH)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "supervised_broker_test",
        "test_id": test["test_id"],
        "state": "holding_observed",
        "blockers": [],
        "live_execution_authority": False,
        "autonomous_execution": False,
        "accounting": {"pending": ["broker_trade_posting_pending"]},
        "purchase_scope": {
            "test_policy_sha256": canonical_payload_sha256(test),
            "account_policy_key": test["account_policy_key"],
            "symbol": "O",
            "entry_attempts": 1,
            "entry_state": "filled",
            "entry_gross_usd": "284.65",
        },
    }


def test_status_is_offline_and_revocation_survives_restart(configured, monkeypatch):
    monkeypatch.setattr(
        cli,
        "capture_observation",
        lambda *args: pytest.fail("no broker call permitted"),
    )
    assert cli.run(configured, "status")["connected"] is False
    assert cli.run(configured, "revoke")["revoked"]
    assert cli.run(configured, "evaluate")["state"] == "revoked"
    assert cli.run(configured, "status")["revoked"]


def test_only_fixed_read_only_subprocess_is_used(configured, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        put(configured, cli.broker_test.REPORT_PATH, observed(configured))
        return {"rc": 0}

    monkeypatch.setattr(cli, "run_bounded_process_group", runner)
    report = cli.run(configured, "evaluate")
    assert report["state"] == "holding_only"
    assert report["proposal"] == {}
    command, kwargs = calls[0]
    assert command[-2:] == ["observe", "--json"]
    assert kwargs["timeout_seconds"] == 90
    assert kwargs["env"]["ALLOW_ORDER_EXECUTION"] == "0"
    assert kwargs["env"]["EXECUTION_LANE_LIVE_ENABLED"] == "0"
    assert cli.run(configured, "evaluate", scheduled=True)["refresh_skipped"]
    assert len(calls) == 1
    assert not (
        configured / "governance/runtime/supervised_broker_test_attestation.json"
    ).exists()


@pytest.mark.parametrize(
    "result", [{"rc": 124, "timed_out": True}, {"rc": 2}, {"rc": 0}]
)
def test_failed_or_unpublished_observation_cannot_reuse_old_report(
    configured, monkeypatch, result
):
    old = observed(configured)
    old["timestamp_utc"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    put(configured, cli.broker_test.REPORT_PATH, old)
    monkeypatch.setattr(
        cli, "run_bounded_process_group", lambda *args, **kwargs: result
    )
    with pytest.raises(ValueError):
        cli.run(configured, "evaluate")
    assert cli.main(["evaluate", "--json"]) == 2
    assert cli.read(configured, cli.REPORT_PATH)["state"] == "blocked"


def test_no_source_acceptance_or_qualification_from_dirty_candidate(
    configured, monkeypatch
):
    monkeypatch.setattr(cli, "capture_observation", lambda root: observed(root))
    monkeypatch.setattr(
        cli.broker_test,
        "current_source_blockers",
        lambda *args: ["test_source_release_not_clean"],
    )
    result = cli.run(configured, "evaluate")
    assert result["source_ready"] is False
    assert result["validation"]["qualified_observations"] == 0
    assert result["validation"]["execution_review_eligible"] is False


def test_lock_contention_does_not_start_another_broker_reader(configured, monkeypatch):
    monkeypatch.setattr(
        cli, "capture_observation", lambda *args: pytest.fail("busy lock")
    )
    with cli.lock(configured):
        assert cli.main(["evaluate", "--json"]) == 4
    assert not (configured / cli.REPORT_PATH).exists()


def test_external_report_route_is_rejected(configured, tmp_path):
    target = tmp_path.parent / (tmp_path.name + "-external")
    target.write_text("{}")
    destination = configured / cli.REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(target)
    with pytest.raises(ValueError):
        cli.read(configured, cli.REPORT_PATH)


def test_expired_policy_does_not_call_broker(configured, monkeypatch):
    policy = cli.read(configured, cli.POLICY_PATH)
    policy.update(
        valid_from_utc="2026-08-01T00:00:00+00:00",
        expires_at_utc="2026-08-31T00:00:00+00:00",
    )
    put(configured, cli.POLICY_PATH, policy)
    monkeypatch.setattr(
        cli, "capture_observation", lambda *args: pytest.fail("expired")
    )
    assert cli.run(configured, "evaluate")["state"] == "expired"


def test_no_execution_subcommand(configured):
    with pytest.raises(SystemExit) as exc:
        cli.main(["submit"])
    assert exc.value.code == 2


def test_revocation_is_not_blocked_by_observer_lock(configured):
    with cli.lock(configured):
        assert cli.run(configured, "revoke")["revoked"]
    assert cli.run(configured, "evaluate")["state"] == "revoked"


def test_revocation_during_broker_read_suppresses_evaluation(configured, monkeypatch):
    def capture(root):
        cli.run(root, "revoke")
        return observed(root)

    monkeypatch.setattr(cli, "capture_observation", capture)
    result = cli.run(configured, "evaluate")
    assert result["state"] == "revoked"
    assert result["proposal"] == {}
    assert result["validation"]["qualified_observations"] == 0


def test_status_marks_expired_proposal_separately_from_report(configured):
    now = datetime.now(timezone.utc)
    put(
        configured,
        cli.REPORT_PATH,
        {
            "timestamp_utc": now.isoformat(),
            "proposal": {"orderType": "LIMIT"},
            "proposal_expires_at_utc": (now - timedelta(seconds=1)).isoformat(),
            "policy_sha256": canonical_payload_sha256(
                cli.read(configured, cli.POLICY_PATH)
            ),
        },
    )
    result = cli.run(configured, "status")
    assert result["last_evaluation_fresh"]
    assert not result["last_proposal_fresh"]


def test_native_cadence_uses_existing_profile_and_read_only_owner():
    step = next(
        row
        for row in refresh.profile_steps("accrual")
        if row["name"] == "purchase_proposals"
    )
    assert step["args"] == ["evaluate", "--scheduled", "--json"]
    assert step["owner_timeout_seconds"] == 120
    assert step["max_age_minutes"] == 15
    assert step["depends_on"] == []
    assert "purchase_proposals" not in refresh.PROFILE_STEP_NAMES["dashboard"]
