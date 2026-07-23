from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import storage_quota_guard as src


GIB = 1024**3


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_health(
    project_root: Path,
    *,
    governance_gb: float = 30.0,
    explanations_gb: float = 0.0,
    hot_path_over_budget_bytes: int = 0,
    hot_lane_mode: str = "full_decision_evidence",
    hot_lane_status: str = "ready",
) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        health / "storage_tier_policy_latest.json",
        {
            "overall_status": "ready",
            "pressure": {
                "hot_path_over_budget_bytes": hot_path_over_budget_bytes,
                "live_hot_path_bytes": 4 * GIB,
                "hot_budget_bytes": 25 * GIB,
            },
            "by_family": {
                "decision_explanations": {"bytes": int(explanations_gb * GIB)},
            },
            "by_service_role": {
                "governance_telemetry": {"bytes": int(governance_gb * GIB)},
            },
        },
    )
    _write_json(
        health / "hot_lane_retention_control_latest.json",
        {
            "ok": True,
            "overall_status": hot_lane_status,
            "mode": hot_lane_mode,
        },
    )
    _write_json(health / "data_collection_storage_guard_latest.json", {})


def test_quota_guard_excludes_bounded_current_day_governance_when_full_evidence_hot_path_is_green(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_health(tmp_path, governance_gb=30.0)
    monkeypatch.setattr(src, "_active_current_day_decision_bytes", lambda _project_root: 0)
    monkeypatch.setattr(src, "_active_current_day_governance_channel_bytes", lambda _project_root: 24 * GIB)
    monkeypatch.setattr(src, "_active_current_day_explanation_bytes", lambda _project_root: 0)

    payload = src.build_payload(tmp_path)

    governance = next(row for row in payload["lanes"] if row["family"] == "governance_telemetry")
    assert payload["overall_status"] == "ready"
    assert governance["status"] == "ready"
    assert governance["used_gb"] == 6.0
    assert governance["adjustments"][0]["reason"] == (
        "exclude_bounded_current_day_active_governance_channels_under_green_full_evidence_hot_lane"
    )
    assert payload["active_hot_buffer_containment"]["hot_lane_full_evidence_current_day_governance_relief"] is True


def test_quota_guard_keeps_governance_blocked_when_hot_path_is_over_budget(tmp_path: Path, monkeypatch) -> None:
    _seed_health(tmp_path, governance_gb=30.0, hot_path_over_budget_bytes=1)
    monkeypatch.setattr(src, "_active_current_day_decision_bytes", lambda _project_root: 0)
    monkeypatch.setattr(src, "_active_current_day_governance_channel_bytes", lambda _project_root: 24 * GIB)
    monkeypatch.setattr(src, "_active_current_day_explanation_bytes", lambda _project_root: 0)

    payload = src.build_payload(tmp_path)

    governance = next(row for row in payload["lanes"] if row["family"] == "governance_telemetry")
    assert payload["overall_status"] == "blocked"
    assert governance["status"] == "blocked"
    assert governance["used_gb"] == 30.0
    assert governance["adjustments"] == []
    assert payload["active_hot_buffer_containment"]["hot_lane_full_evidence_current_day_governance_relief"] is False


def test_quota_guard_excludes_current_day_explanations_when_hot_lane_retention_active(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_health(
        tmp_path,
        governance_gb=0.0,
        explanations_gb=30.0,
        hot_lane_mode="emergency_hot_thin",
        hot_lane_status="critical",
    )
    monkeypatch.setattr(src, "_active_current_day_decision_bytes", lambda _project_root: 0)
    monkeypatch.setattr(src, "_active_current_day_governance_channel_bytes", lambda _project_root: 0)
    monkeypatch.setattr(src, "_active_current_day_explanation_bytes", lambda _project_root: 20 * GIB)

    payload = src.build_payload(tmp_path)

    explanations = next(row for row in payload["lanes"] if row["family"] == "decision_explanations")
    assert payload["overall_status"] == "ready"
    assert explanations["status"] == "ready"
    assert explanations["used_gb"] == 14.0
    assert explanations["adjustments"][0]["reason"] == (
        "exclude_bounded_current_day_explanation_buffer_under_hot_lane_retention"
    )
    assert explanations["adjustments"][0]["hard_quota_protected"] is True
    assert payload["active_hot_buffer_containment"]["active_current_day_explanation_gb"] == 20.0
    assert payload["active_hot_buffer_containment"]["active_explanation_buffer_allowance_gb"] == 16.0


def test_quota_guard_does_not_hide_hard_explanation_breach(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_health(
        tmp_path,
        governance_gb=0.0,
        explanations_gb=60.0,
        hot_lane_mode="emergency_hot_thin",
        hot_lane_status="critical",
    )
    monkeypatch.setattr(src, "_active_current_day_decision_bytes", lambda _project_root: 0)
    monkeypatch.setattr(src, "_active_current_day_governance_channel_bytes", lambda _project_root: 0)
    monkeypatch.setattr(src, "_active_current_day_explanation_bytes", lambda _project_root: 40 * GIB)

    payload = src.build_payload(tmp_path)

    explanations = next(row for row in payload["lanes"] if row["family"] == "decision_explanations")
    assert payload["overall_status"] == "blocked"
    assert explanations["status"] == "blocked"
    assert explanations["used_gb"] == 60.0
    assert explanations["adjustments"] == []
