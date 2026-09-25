import json
from types import SimpleNamespace

import pytest

from core import live_execution_switch as switch
from scripts.ops import live_execution_switch as cli

NATIVE_READINESS = switch._readiness


@pytest.mark.parametrize("purpose", switch.PURPOSES)
def test_native_readiness_preserves_technical_gates_and_production_attestation(
    tmp_path, monkeypatch, purpose
):
    from core import live_canary_preflight, live_canary_allowlist

    calls = []

    def preflight(*args, **kwargs):
        calls.append(kwargs)
        return {
            "candidate_id": "test",
            "blockers": ["risk_not_ready", "operator_attestation_expired"],
            "operator_attestation_blockers": ["operator_attestation_expired"],
        }

    monkeypatch.setattr(
        live_canary_preflight, "evaluate_live_canary_preflight", preflight
    )
    monkeypatch.setattr(
        live_canary_allowlist,
        "evaluate_live_canary_allowlist",
        lambda *a: {"blockers": ["allowlist_missing"]},
    )
    result = NATIVE_READINESS(
        tmp_path, purpose=purpose, symbol="SCHD", session="NORMAL"
    )
    assert "risk_not_ready" in result["blockers"]
    assert ("operator_attestation_expired" in result["blockers"]) == (
        purpose == "production_canary"
    )
    assert ("allowlist_missing" in result["blockers"]) == (
        purpose == "production_canary"
    )
    assert calls[0]["action"] == "BUY"


def test_state_uses_native_retention_protection(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from scripts.data_retention_policy import _stale_stage_protection_reason

    assert _stale_stage_protection_reason("governance_runtime", Path(switch.STATE))


@pytest.fixture
def root(tmp_path, monkeypatch):
    for relative, value in {
        switch.CANDIDATE: {"candidate_id": "test-candidate"},
        "config/supervised_schd_broker_test_v1.json": {"symbol": "SCHD"},
        "config/live_canary_micro_policy_v1.json": {"symbol": "SCHD"},
        "config/production_readiness_control_v1.json": {},
        "config/account_policy_registry.json": {},
    }.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    (tmp_path / "governance/runtime").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        switch,
        "_readiness",
        lambda *a, **k: {"blockers": [], "candidate_id": "test-candidate"},
    )
    return tmp_path


def arm(root):
    return switch.switch_on(
        root, purpose="supervised_broker_test", symbol="SCHD", session="NORMAL"
    )


def check(root, **overrides):
    args = {
        "broker": "schwab",
        "operation": "place_order",
        "context": {
            "purpose": "supervised_broker_test",
            "symbol": "SCHD",
            "session": "NORMAL",
        },
    }
    args.update(overrides)
    return switch.check_live_execution_switch(root, **args)


def test_default_off_persisted_off_and_on(root):
    assert not check(root)["allowed"]
    assert arm(root)["switch"] == "ON"
    assert check(root)["allowed"]
    assert switch.switch_off(root)["switch"] == "OFF"
    assert not check(root)["allowed"]


@pytest.mark.parametrize(
    "operation", ["get_accounts_snapshot", "get_order", "cancel_order"]
)
def test_reads_and_cancel_remain_available_when_off(root, operation):
    assert check(root, operation=operation)["allowed"]


@pytest.mark.parametrize("operation", ["place_order", "replace_order"])
def test_mock_paper_is_unaffected(root, operation):
    assert check(root, broker="mock", operation=operation)["allowed"]


@pytest.mark.parametrize(
    "field,value",
    [("symbol", "O"), ("purpose", "production_canary"), ("session", "PM")],
)
def test_permission_is_scoped(root, field, value):
    arm(root)
    context = {
        "purpose": "supervised_broker_test",
        "symbol": "SCHD",
        "session": "NORMAL",
        field: value,
    }
    assert not check(root, context=context)["allowed"]
    assert not check(root, broker="coinbase")["allowed"]


def test_expiry_and_backwards_clock_fail_closed(root):
    arm(root)
    state = switch._read(root)
    assert switch.switch_status(root, now=state["expires_at"])["switch"] == "OFF"
    assert switch.switch_status(root, now=state["issued_at"] - 1)["switch"] == "OFF"


@pytest.mark.parametrize(
    "relative",
    [
        switch.CANDIDATE,
        "config/account_policy_registry.json",
        "config/supervised_schd_broker_test_v1.json",
    ],
)
def test_candidate_and_policy_change_revoke_permission(root, relative):
    arm(root)
    (root / relative).write_text('{"candidate_id":"changed"}')
    assert not check(root)["allowed"]


@pytest.mark.parametrize(
    "contents", ["{", "[]", '{"requested_on":true}', '"ON"', "x" * 16385]
)
def test_corrupt_state_fails_closed(root, contents):
    (root / switch.STATE).write_text(contents)
    assert not check(root)["allowed"]


def test_state_link_is_never_followed(root):
    target = root / "outside.json"
    target.write_text('{"requested_on":true}')
    (root / switch.STATE).symlink_to(target)
    assert not check(root)["allowed"]
    with pytest.raises(ValueError, match="local_owned_paths"):
        switch.switch_off(root)
    assert target.read_text() == '{"requested_on":true}'


def test_on_readiness_failure_leaves_off(root, monkeypatch):
    monkeypatch.setattr(
        switch,
        "_readiness",
        lambda *a, **k: {
            "candidate_id": "test-candidate",
            "blockers": ["risk_service_boundary_not_ready"],
        },
    )
    result = arm(root)
    assert result["switch"] == "OFF" and not result["ok"]
    assert result["activation_blockers"] == ["risk_service_boundary_not_ready"]


def test_off_supersedes_in_progress_on(root, monkeypatch):
    def readiness(*a, **k):
        switch.switch_off(root)
        return {"blockers": [], "candidate_id": "test-candidate"}

    monkeypatch.setattr(switch, "_readiness", readiness)
    assert arm(root)["activation_blockers"] == ["live_switch_request_superseded"]
    assert not check(root)["allowed"]


def test_candidate_change_during_activation_leaves_off(root, monkeypatch):
    def readiness(*a, **k):
        (root / switch.CANDIDATE).write_text('{"candidate_id":"new"}')
        return {"blockers": [], "candidate_id": "test-candidate"}

    monkeypatch.setattr(switch, "_readiness", readiness)
    with pytest.raises(ValueError, match="candidate_changed"):
        arm(root)
    assert not check(root)["allowed"]


def test_write_failure_does_not_claim_off(root, monkeypatch):
    arm(root)
    monkeypatch.setattr(switch, "safe_write_json_atomic", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="not_persisted"):
        switch.switch_off(root)
    assert check(root)["allowed"]


def fake_trader(root):
    from core.base_trader import BaseTrader

    trader = BaseTrader.__new__(BaseTrader)
    trader.project_root = str(root)
    trader.broker_name = "schwab"
    trader.live_api_retry_attempts = 1
    trader.live_guard = SimpleNamespace(
        allow_api_call=lambda *_: True, record_api_success=lambda *_: None
    )
    trader._log_live_guard_event = lambda **_: None
    trader.client = SimpleNamespace(
        place_order=lambda: SimpleNamespace(status_code=201, headers={})
    )
    return trader


@pytest.mark.parametrize("operation", ["place_order", "replace_order"])
def test_dispatch_off_never_calls_client(root, operation):
    trader = fake_trader(root)
    trader.client.place_order = lambda: pytest.fail("broker must not be called")
    result = trader._invoke_client_candidates(
        operation=operation, candidates=[("place_order", (), {})]
    )
    assert result["broker_mutation_attempted"] is False
    assert result["attempts_made"] == 0


def test_off_after_rate_admission_still_blocks_dispatch(root, monkeypatch):
    from core import base_trader

    arm(root)
    trader = fake_trader(root)
    trader.client.place_order = lambda: pytest.fail("broker must not be called")

    def rate(**kwargs):
        switch.switch_off(root)
        return {"allowed": True}

    monkeypatch.setattr(base_trader, "acquire_broker_rate_limit", rate)
    result = trader._invoke_client_candidates(
        operation="place_order",
        candidates=[("place_order", (), {})],
        context={"purpose": "supervised_broker_test", "symbol": "SCHD"},
    )
    assert result["error"] == "live_execution_switch_blocked"
    assert result["broker_mutation_attempted"] is False


def test_valid_permission_reaches_fake_client_once(root, monkeypatch):
    from core import base_trader

    arm(root)
    trader = fake_trader(root)
    calls = []
    trader.client.place_order = lambda: calls.append(1) or SimpleNamespace(
        status_code=201, headers={}
    )
    monkeypatch.setattr(
        base_trader, "acquire_broker_rate_limit", lambda **_: {"allowed": True}
    )
    result = trader._invoke_client_candidates(
        operation="place_order",
        candidates=[("place_order", (), {})],
        context={"purpose": "supervised_broker_test", "symbol": "SCHD"},
    )
    assert result["ok"] and calls == [1]


def test_noninteractive_on_is_rejected(root, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", root)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(cli, "notify_transition", lambda *a: {"returncode": 0})
    assert cli.main(["on", "--symbol", "SCHD"]) == 2
    assert "interactive_operator_terminal_required" in capsys.readouterr().out
    assert not check(root)["allowed"]


@pytest.mark.parametrize(
    "command,result,title",
    [
        ("off", {"ok": True, "switch": "OFF"}, "Live Execution OFF"),
        ("on", {"ok": True, "switch": "ON"}, "Live Execution ON"),
        (
            "on",
            {"ok": False, "switch": "OFF", "activation_blockers": ["risk"]},
            "Live Execution ON not confirmed",
        ),
        (
            "off",
            {"ok": False, "error": "disk_full"},
            "Live Execution OFF not confirmed",
        ),
    ],
)
def test_notification_uses_verified_outcome(root, monkeypatch, command, result, title):
    from scripts.ops import mac_notification_watch

    calls = []
    monkeypatch.setattr(cli, "ROOT", root)
    monkeypatch.setattr(
        mac_notification_watch,
        "_notify_mac",
        lambda *a, **kw: calls.append((a, kw)) or {"returncode": 0},
    )
    assert cli.notify_transition(command, result)["returncode"] == 0
    assert calls[0][0][0] == title
    assert calls[0][1]["open_target"] == (root / "COMMANDS.md").as_uri()
    assert "execute_target" not in calls[0][1]


def test_notification_failure_does_not_undo_off(root, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", root)
    monkeypatch.setattr(cli, "notify_transition", lambda *a: {"returncode": 1})
    assert cli.main(["off"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["switch"] == "OFF" and result["notification"]["returncode"] == 1


def test_status_has_no_notification_or_write(root, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", root)
    monkeypatch.setattr(
        cli, "notify_transition", lambda *a: pytest.fail("status must be quiet")
    )
    assert cli.main(["status"]) == 0
    assert json.loads(capsys.readouterr().out)["switch"] == "OFF"
    assert not (root / switch.STATE).exists()
