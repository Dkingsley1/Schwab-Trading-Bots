from datetime import datetime, timedelta, timezone

import pytest

from scripts.ops.long_runtime_common import evidence_freshness
from scripts.ops.health_fast import _platform_repair_contract
from scripts.ops.system_architecture_hardening import _platform_watch_semantics

NOW = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "payload,status",
    [
        ({}, "missing"),
        ({"overall_status": "ready"}, "timestamp_missing"),
        (
            {"timestamp_utc": "bad", "updated_at_utc": NOW.isoformat()},
            "timestamp_invalid",
        ),
        (
            {"timestamp_utc": (NOW + timedelta(minutes=2)).isoformat()},
            "future_timestamp",
        ),
        ({"timestamp_utc": (NOW - timedelta(minutes=61)).isoformat()}, "stale"),
        ({"timestamp_utc": (NOW - timedelta(minutes=60)).isoformat()}, "fresh"),
        ({"generated_utc": NOW.isoformat()}, "fresh"),
        ({"generated_utc": (NOW - timedelta(minutes=61)).isoformat()}, "stale"),
        (
            {"generated_utc": (NOW + timedelta(minutes=2)).isoformat()},
            "future_timestamp",
        ),
        ({"generated_utc": "bad"}, "timestamp_invalid"),
        (
            {"timestamp_utc": "bad", "generated_utc": NOW.isoformat()},
            "timestamp_invalid",
        ),
    ],
)
def test_producer_timestamp_is_required(payload, status):
    result = evidence_freshness(payload, now=NOW)
    assert result["status"] == status
    assert result["fresh"] is (status == "fresh")


def test_fast_health_retains_stale_status_without_claiming_current_failure():
    stale = {
        "timestamp_utc": "2000-01-01T00:00:00+00:00",
        "overall_status": "needs_work",
    }
    fresh = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "ready",
    }
    result = _platform_repair_contract(
        platform=stale, brain_v5=fresh, stabilizer=fresh, settlement=fresh
    )
    assert not result["ok"]
    assert result["issue_count"] == 1
    assert result["issues"][0]["overall_status"] == "evidence_unavailable"
    assert result["issues"][0]["reported_status"] == "needs_work"
    assert result["issues"][0]["reason"] == "stale"
    assert not result["blocks_guarded_paper"]


def test_missing_and_stale_green_sources_cannot_be_managed_to_ready():
    context = {
        name: {}
        for name in (
            "platform_intelligence",
            "platform_brain_v5",
            "platform_stabilization_quality",
            "platform_settlement_stabilization",
        )
    }
    context["platform_intelligence"] = {
        "timestamp_utc": "2000-01-01T00:00:00+00:00",
        "overall_status": "ready",
    }
    context["health_fast"] = {"strict_all_clear": True}
    context["global_killswitch"] = {}
    context["global_halt_auto_clear"] = {}
    result = _platform_watch_semantics(context)
    assert result["overall_status"] == "needs_work"
    assert "platform_intelligence_evidence=stale" in result["findings"]
    assert not result["evidence"]["managed_watch_contract"]["active"]
