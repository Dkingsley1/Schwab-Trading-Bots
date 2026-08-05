from __future__ import annotations

import subprocess
import types
from pathlib import Path

from scripts.ops import schwab_auth_post_refresh as src


def _ready_payload(name: str) -> tuple[int, dict]:
    if name == "token_guard":
        return 0, {"ok": True, "token_ready_after": True, "broker_readiness": {"ready_for_open": True}}
    if name == "auth_lease":
        return 0, {"ok": True, "overall_status": "ready", "lease_state": "healthy"}
    if name == "account_snapshot":
        return 0, {
            "ok": True,
            "broker_truth_ok": True,
            "account_count": 3,
            "position_rows": 9,
            "broker_truth_v2_grade": "A",
        }
    if name == "covered_call_watch":
        return 0, {"timestamp_utc": "2026-08-04T02:00:00+00:00", "overall_status": "critical"}
    if name == "account_position_study":
        return 0, {"ok": True, "account_count": 3, "position_count": 9, "underlying_count": 4}
    if name == "position_opportunity_watch":
        return 0, {"ok": True, "observed_underlying_count": 4, "candidate_count": 1, "abstention_count": 3}
    if name == "sleeve_allocator":
        return 0, {"broker": "schwab", "gross_risk_budget": 0.0, "target_weights": {"core": 1.0}}
    if name == "portfolio_risk_ledger":
        return 0, {
            "timestamp_utc": "2026-08-04T02:00:00+00:00",
            "risk_level": "medium",
            "risk_score": 35.0,
            "limits": {"gross_exposure_cap": 0.17, "max_single_symbol_share": 0.15},
        }
    if name == "portfolio_allocator":
        return 0, {"ok": True, "overall_status": "ready", "summary": {"approved_intent_count": 0, "gross_budget": 0.0}}
    if name == "account_buildout_plan":
        return 0, {
            "ok": True,
            "overall_status": "ready",
            "plan_state": "observe_only",
            "buildout_ready": False,
            "account_count": 3,
            "action_count": 0,
            "review_count": 1,
        }
    return 0, {"ok": True, "overall_status": "ready", "grade": "A+", "score": 97.0, "failed_checks": []}


def test_post_refresh_runs_auth_and_paper_truth_dependencies_in_order(tmp_path: Path, monkeypatch) -> None:
    seen: list[str] = []

    def fake_runner(name: str, cmd: list[str], **kwargs) -> dict:
        seen.append(name)
        rc, payload = _ready_payload(name)
        return {"name": name, "cmd": cmd, "rc": rc, "timed_out": False, "payload": payload}

    monkeypatch.setattr(src, "mark_provider_recovered", lambda *args, **kwargs: {"state": "ready", "active": False})
    payload = src.build_payload(tmp_path, runner=fake_runner)

    assert seen == [
        "token_guard",
        "auth_lease",
        "account_snapshot",
        "covered_call_watch",
        "account_position_study",
        "position_opportunity_watch",
        "sleeve_allocator",
        "portfolio_risk_ledger",
        "portfolio_allocator",
        "account_buildout_plan",
        "paper_truth",
    ]
    assert payload["overall_status"] == "ready"
    assert payload["refresh_completed"] is True
    assert payload["paper_truth_ready"] is True
    assert payload["safety_contract"]["live_execution_allowed"] is False
    assert payload["safety_contract"]["opens_browser"] is False
    assert payload["safety_contract"]["account_buildout_does_not_publish_execution_intents"] is True


def test_post_refresh_stops_before_account_access_when_auth_is_not_ready(tmp_path: Path) -> None:
    seen: list[str] = []

    def fake_runner(name: str, cmd: list[str], **kwargs) -> dict:
        seen.append(name)
        return {
            "name": name,
            "cmd": cmd,
            "rc": 2,
            "timed_out": False,
            "payload": {"ok": False, "broker_readiness": {"ready_for_open": False}},
        }

    payload = src.build_payload(tmp_path, runner=fake_runner)

    assert seen == ["token_guard"]
    assert payload["overall_status"] == "blocked"
    assert payload["refresh_completed"] is False
    assert payload["hard_failure"].startswith("token_guard:")


def test_post_refresh_reports_real_paper_truth_attention_without_misclassifying_auth(tmp_path: Path, monkeypatch) -> None:
    def fake_runner(name: str, cmd: list[str], **kwargs) -> dict:
        rc, payload = _ready_payload(name)
        if name == "paper_truth":
            rc = 2
            payload = {"ok": False, "overall_status": "blocked", "grade": "C", "score": 81.0, "failed_checks": ["real_blocker"]}
        return {"name": name, "cmd": cmd, "rc": rc, "timed_out": False, "payload": payload}

    monkeypatch.setattr(src, "mark_provider_recovered", lambda *args, **kwargs: {})
    payload = src.build_payload(tmp_path, runner=fake_runner)

    assert payload["overall_status"] == "ready"
    assert payload["refresh_completed"] is True
    assert payload["paper_truth_ready"] is False
    assert payload["downstream_attention"] == ["paper_truth_blocked", "real_blocker"]
    assert payload["steps"][-1]["failed_checks"] == ["real_blocker"]


def test_post_refresh_subprocess_forces_paper_only_environment(tmp_path: Path, monkeypatch) -> None:
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(returncode=0, stdout='{"ok": true}\n', stderr="")

    monkeypatch.setattr(src.subprocess, "run", fake_run)
    result = src._run_step("test", ["command"], cwd=tmp_path, timeout_sec=10)

    assert result["rc"] == 0
    assert seen["env"]["MARKET_DATA_ONLY"] == "1"
    assert seen["env"]["ALLOW_ORDER_EXECUTION"] == "0"
    assert seen["env"]["TOP_BOT_ENABLE_LIVE_EXECUTION"] == "0"
    assert seen["env"]["EXECUTION_LANE_LIVE_ENABLED"] == "0"
