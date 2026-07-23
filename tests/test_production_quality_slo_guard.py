from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops import production_quality_slo_guard as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_quality(project_root: Path, active_lanes: list[dict], *, status: str = "blocked") -> None:
    _write_json(
        project_root / "governance" / "health" / "production_quality_control_latest.json",
        {
            "overall_status": status,
            "live_canary_readiness": {"live_canary_money_ready": status == "ready"},
            "active_lane_count": len(active_lanes),
            "active_lanes": active_lanes,
            "governor_safe_execution_command": [
                "./scripts/ops/opsctl.sh",
                "infrabot-adaptive-governor",
                "--apply",
                "--execute-safe-repairs",
                "--json",
            ],
        },
    )


def _lane(lane_id: str, *, severity: str = "critical") -> dict:
    return {
        "lane_id": lane_id,
        "title": lane_id.replace("_", " ").title(),
        "severity": severity,
        "blocking_reasons": [f"{lane_id}_blocked"],
        "owner_capabilities": ["production_quality_control"],
        "commands": [["./scripts/ops/opsctl.sh", "production-quality", "--apply", "--refresh-contract", "--json"]],
        "stop_when": f"{lane_id} clears",
        "expected_impact": "bounded production repair",
    }


def test_production_quality_slo_guard_marks_old_critical_lane_breached(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    first_seen = (datetime.now(timezone.utc) - timedelta(minutes=130)).isoformat()
    _write_quality(project_root, [_lane("raw_profitability_recovery")])
    _write_json(
        project_root / "governance" / "health" / "production_quality_slo_guard_state.json",
        {"lanes": {"raw_profitability_recovery": {"first_seen_utc": first_seen, "hit_count": 3}}},
    )

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert payload["breach_count"] == 1
    assert payload["breached_lanes"][0]["lane_id"] == "raw_profitability_recovery"
    assert payload["breached_lanes"][0]["active_minutes"] >= 120
    assert payload["control_contract"]["live_orders_remain_disabled_while_active_or_breached"] is True
    assert payload["live_execution_authority"] is False


def test_production_quality_slo_guard_warns_before_high_lane_breach(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    first_seen = (datetime.now(timezone.utc) - timedelta(minutes=180)).isoformat()
    _write_quality(project_root, [_lane("promotion_paper_freshness", severity="high")])
    _write_json(
        project_root / "governance" / "health" / "production_quality_slo_guard_state.json",
        {"lanes": {"promotion_paper_freshness": {"first_seen_utc": first_seen, "hit_count": 2}}},
    )

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert payload["warning_count"] == 1
    assert payload["breach_count"] == 0
    assert payload["warning_lanes"][0]["lane_id"] == "promotion_paper_freshness"


def test_production_quality_slo_guard_apply_clears_resolved_lane_state(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_quality(project_root, [], status="ready")
    _write_json(
        health / "production_quality_slo_guard_state.json",
        {"lanes": {"storage_pressure_clean": {"first_seen_utc": "2026-07-23T12:00:00+00:00", "hit_count": 4}}},
    )

    payload = src.build_payload(project_root, apply=True)
    state = json.loads((health / "production_quality_slo_guard_state.json").read_text(encoding="utf-8"))

    assert payload["overall_status"] == "ready"
    assert payload["active_lane_count"] == 0
    assert state["lanes"] == {}
    assert state["last_resolved_lanes"][0]["lane_id"] == "storage_pressure_clean"
    assert (health / "production_quality_slo_events.jsonl").exists()
