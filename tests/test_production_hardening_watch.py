from __future__ import annotations

from pathlib import Path

from scripts.ops import production_hardening_watch as src


def _quality(active_lane_count: int = 1) -> dict:
    return {
        "overall_status": "blocked" if active_lane_count else "ready",
        "active_lane_count": active_lane_count,
        "active_lanes": [{"lane_id": "auth_token_continuity"}] if active_lane_count else [],
    }


def _slo(*, active: int = 1, warnings: int = 0, breaches: int = 0) -> dict:
    return {
        "overall_status": "blocked" if breaches else "degraded" if warnings else "watch" if active else "ready",
        "active_lane_count": active,
        "warning_count": warnings,
        "breach_count": breaches,
        "warning_lanes": [{"lane_id": "auth_token_continuity"}] if warnings else [],
        "breached_lanes": [{"lane_id": "auth_token_continuity"}] if breaches else [],
    }


def test_production_hardening_watch_publish_only_before_slo_warning(tmp_path: Path, monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_quality(*_args, **kwargs):
        calls["quality_refresh_contract"] = kwargs.get("refresh_contract")
        return _quality()

    monkeypatch.setattr(src.production_quality_control, "build_payload", fake_quality)
    monkeypatch.setattr(src.production_quality_slo_guard, "build_payload", lambda *_args, **_kwargs: _slo(active=1))

    def fake_governor(*_args, **kwargs):
        calls.update(kwargs)
        return {"overall_status": "guarded", "adaptive_policy_router": {"action_counts": {}, "recommended_commands": []}, "safety_guard": {}}

    monkeypatch.setattr(src.infrabot_adaptive_governor, "build_payload", fake_governor)

    payload = src.build_payload(tmp_path, apply=True, execute_safe_repairs=True)

    assert payload["overall_status"] == "watch"
    assert payload["repair_execution_triggered"] is False
    assert calls["quality_refresh_contract"] is False
    assert calls["execute_safe_repairs"] is False
    assert calls["refresh_needs"] is False
    assert payload["execution_policy"]["governor_refresh_needs"] is False
    assert payload["execution_policy"]["quality_refresh_contract"] is False
    assert payload["control_contract"]["no_source_registry_refresh"] is True
    assert payload["control_contract"]["uses_published_contracts_for_scheduled_watch"] is True
    assert payload["live_execution_authority"] is False
    assert (tmp_path / "governance" / "health" / "production_hardening_watch_latest.json").exists()


def test_production_hardening_watch_delegates_safe_repairs_on_warning(tmp_path: Path, monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_quality(*_args, **kwargs):
        calls["quality_refresh_contract"] = kwargs.get("refresh_contract")
        return _quality()

    monkeypatch.setattr(src.production_quality_control, "build_payload", fake_quality)
    monkeypatch.setattr(src.production_quality_slo_guard, "build_payload", lambda *_args, **_kwargs: _slo(active=1, warnings=1))

    def fake_governor(*_args, **kwargs):
        calls.update(kwargs)
        return {
            "overall_status": "guarded",
            "adaptive_policy_router": {"action_counts": {"run_now": 1}, "recommended_commands": []},
            "safety_guard": {"live_execution_authority": False},
            "apply_result": {"safe_repair_execution": {"executed_count": 1, "live_execution_authority": False}},
        }

    monkeypatch.setattr(src.infrabot_adaptive_governor, "build_payload", fake_governor)

    payload = src.build_payload(tmp_path, execute_safe_repairs=True, max_execute_actions=2, command_timeout_seconds=60)

    assert payload["overall_status"] == "repairing"
    assert payload["repair_execution_triggered"] is True
    assert payload["repair_execution_attempted_count"] == 1
    assert calls["quality_refresh_contract"] is False
    assert calls["execute_safe_repairs"] is True
    assert calls["refresh_needs"] is False
    assert calls["max_execute_actions"] == 2
    assert calls["command_timeout_seconds"] == 60


def test_production_hardening_watch_can_opt_into_watch_execution(tmp_path: Path, monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_quality(*_args, **kwargs):
        calls["quality_refresh_contract"] = kwargs.get("refresh_contract")
        return _quality()

    monkeypatch.setattr(src.production_quality_control, "build_payload", fake_quality)
    monkeypatch.setattr(src.production_quality_slo_guard, "build_payload", lambda *_args, **_kwargs: _slo(active=1))

    def fake_governor(*_args, **kwargs):
        calls.update(kwargs)
        return {
            "overall_status": "guarded",
            "adaptive_policy_router": {"action_counts": {}, "recommended_commands": []},
            "safety_guard": {},
            "apply_result": {"safe_repair_execution": {"executed_count": 1}},
        }

    monkeypatch.setattr(src.infrabot_adaptive_governor, "build_payload", fake_governor)

    payload = src.build_payload(tmp_path, execute_safe_repairs=True, execute_on_watch=True)

    assert payload["repair_execution_triggered"] is True
    assert payload["execution_policy"]["execute_trigger"] == "watch"
    assert calls["quality_refresh_contract"] is False
    assert calls["execute_safe_repairs"] is True
    assert calls["refresh_needs"] is False
