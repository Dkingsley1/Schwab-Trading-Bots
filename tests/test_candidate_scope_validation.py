import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.ops import candidate_scope_validation as control


def _write_policy(project_root: Path) -> dict:
    policy = {
        "schema_version": 1,
        "policy_id": "test-scope-policy",
        "calendar": {
            "calendar_id": "XNYS",
            "library": "exchange-calendars",
            "minimum_version": "4.13.2",
        },
        "tiers": {
            "advisory": {
                "required_hours": 0,
                "required_completed_sessions": 0,
                "blocks_promotion": False,
            },
            "operations": {
                "required_hours": 72,
                "required_completed_sessions": 3,
                "blocks_promotion": True,
            },
            "material": {
                "required_hours": 720,
                "required_completed_sessions": 20,
                "blocks_promotion": True,
            },
        },
        "scope_tiers": {
            "operations": "operations",
            "research_advisory": "advisory",
            "strategy": "material",
        },
        "unknown_scope_tier": "material",
        "authority": {
            "live_execution_authority": False,
            "policy_changes_order_permissions": False,
        },
    }
    path = project_root / "config" / "candidate_scope_validation_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy), encoding="utf-8")
    return {
        "candidate": {
            "scope_validation_policy_path": (
                "config/candidate_scope_validation_v1.json"
            ),
            "require_scope_validation_policy": True,
        },
        "soak": {"required_hours": 720},
    }


def _row(payload: dict, scope: str) -> dict:
    return next(row for row in payload["scope_results"] if row["scope"] == scope)


def test_numeric_version_comparison_is_dependency_free() -> None:
    assert control._numeric_version("4.13.2") == (4, 13, 2)
    assert control._numeric_version("4.13.2") > control._numeric_version("4.9.0")
    assert control._numeric_version("not-a-version") is None


def test_operations_scope_requires_three_full_sessions_even_after_72_hours(
    tmp_path: Path,
) -> None:
    config = _write_policy(tmp_path)
    now = datetime(2026, 8, 10, 21, 0, tzinfo=timezone.utc)

    payload = control.evaluate_scope_validation(
        tmp_path,
        config,
        scope_windows_started_utc={"operations": "2026-08-07T21:00:00+00:00"},
        required_scopes=["operations"],
        candidate_ready=True,
        now=now,
    )

    operations = _row(payload, "operations")
    assert operations["credited_hours"] == 72.0
    assert operations["credited_completed_sessions"] == 1
    assert operations["hours_ready"] is True
    assert operations["sessions_ready"] is False
    assert payload["scope_aware_validation_complete"] is False


def test_operations_scope_clears_after_72_hours_and_three_full_sessions(
    tmp_path: Path,
) -> None:
    config = _write_policy(tmp_path)
    now = datetime(2026, 8, 10, 21, 0, tzinfo=timezone.utc)

    payload = control.evaluate_scope_validation(
        tmp_path,
        config,
        scope_windows_started_utc={
            "operations": "2026-08-05T21:00:00+00:00",
            "research_advisory": now.isoformat(),
        },
        required_scopes=["operations"],
        candidate_ready=True,
        now=now,
    )

    operations = _row(payload, "operations")
    advisory = _row(payload, "research_advisory")
    assert operations["credited_completed_sessions"] == 3
    assert operations["ready"] is True
    assert advisory["required_for_promotion"] is False
    assert payload["scope_aware_validation_complete"] is True
    assert payload["grade"] == "A+"
    assert payload["authority"]["live_execution_authority"] is False


def test_material_scope_keeps_full_720_hour_requirement(tmp_path: Path) -> None:
    config = _write_policy(tmp_path)
    now = datetime(2026, 8, 10, 21, 0, tzinfo=timezone.utc)

    payload = control.evaluate_scope_validation(
        tmp_path,
        config,
        scope_windows_started_utc={"strategy": "2026-07-12T21:00:00+00:00"},
        required_scopes=["strategy"],
        candidate_ready=True,
        now=now,
    )

    strategy = _row(payload, "strategy")
    assert strategy["required_hours"] == 720.0
    assert strategy["required_completed_sessions"] == 20
    assert strategy["hours_ready"] is False
    assert payload["scope_aware_validation_complete"] is False


def test_unknown_scope_fails_closed_at_material_tier(tmp_path: Path) -> None:
    config = _write_policy(tmp_path)
    now = datetime(2026, 8, 10, 21, 0, tzinfo=timezone.utc)

    payload = control.evaluate_scope_validation(
        tmp_path,
        config,
        scope_windows_started_utc={"unmapped_scope": "2026-08-01T21:00:00+00:00"},
        required_scopes=["unmapped_scope"],
        candidate_ready=True,
        now=now,
    )

    row = _row(payload, "unmapped_scope")
    assert row["unknown_scope_fail_closed"] is True
    assert row["tier"] == "material"
    assert row["required_hours"] == 720.0
    assert row["ready"] is False


def test_planned_maintenance_excludes_hours_and_interrupted_session(
    tmp_path: Path,
) -> None:
    config = _write_policy(tmp_path)
    now = datetime(2026, 8, 10, 21, 0, tzinfo=timezone.utc)
    maintenance = [
        {
            "offline_start_utc": "2026-08-07T14:00:00+00:00",
            "offline_end_utc": "2026-08-07T15:00:00+00:00",
        }
    ]

    payload = control.evaluate_scope_validation(
        tmp_path,
        config,
        scope_windows_started_utc={"operations": "2026-08-05T21:00:00+00:00"},
        required_scopes=["operations"],
        candidate_ready=True,
        now=now,
        maintenance_windows=maintenance,
    )

    operations = _row(payload, "operations")
    assert operations["planned_maintenance_excluded_hours"] == 1.0
    assert operations["interrupted_sessions_excluded"] == 1
    assert operations["credited_completed_sessions"] == 2
    assert operations["ready"] is False


def test_calendar_failure_zeroes_session_credit_and_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    config = _write_policy(tmp_path)
    monkeypatch.setattr(
        control,
        "_calendar_schedule",
        lambda **_kwargs: {
            "ready": False,
            "library": "exchange-calendars",
            "library_version": "",
            "calendar_id": "XNYS",
            "sessions": [],
            "errors": ["calendar_test_failure"],
        },
    )

    payload = control.evaluate_scope_validation(
        tmp_path,
        config,
        scope_windows_started_utc={"operations": "2026-08-01T21:00:00+00:00"},
        required_scopes=["operations"],
        candidate_ready=True,
        now=datetime(2026, 8, 10, 21, 0, tzinfo=timezone.utc),
    )

    assert payload["calendar"]["ready"] is False
    assert _row(payload, "operations")["credited_completed_sessions"] == 0
    assert "market_calendar_not_ready" in _row(payload, "operations")["blockers"]
    assert payload["scope_aware_validation_complete"] is False
