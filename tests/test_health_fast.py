import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import health_fast as src


def test_health_fast_accepts_bounded_elevated_storage_drain() -> None:
    ok, blockers = src._storage_ready(
        {
            "overall_status": "ready",
            "severity": "elevated",
            "pressure_index": 0.979,
            "backpressure": {
                "core_pending_lines": 4938,
                "total_pending_lines": 7751,
                "pending_lines_threshold": 15000,
                "raw_live": {
                    "core_pending_lines": 4938,
                    "total_pending_lines": 7751,
                    "oldest_pending_age_seconds": 234.997,
                },
            },
            "bounded_recovery_contract": {
                "active_drain_progress": True,
                "drain_delta_signal_observed": False,
                "hard_gate_active": False,
                "effective_hard_gate_active": False,
            },
            "external_route_verification": {"verification_state": "ready"},
            "storage_resilience": {"overall_status": "ready"},
            "storage_efficiency_contract": {"overall_status": "ready", "grade": "A+"},
            "data_integrity": {
                "sql_invalid_lines": 0,
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_oversize_payloads": 0,
                "sql_overlay_ops_write_failures": 0,
            },
            "writer_shedding": {"hard_breaches": [], "elevated_breaches": []},
        }
    )

    assert ok is True
    assert blockers == []


def test_health_fast_blocks_bounded_storage_without_drain_progress() -> None:
    ok, blockers = src._storage_ready(
        {
            "overall_status": "ready",
            "severity": "elevated",
            "pressure_index": 0.979,
            "backpressure": {
                "core_pending_lines": 4938,
                "total_pending_lines": 7751,
                "pending_lines_threshold": 15000,
                "raw_live": {
                    "core_pending_lines": 4938,
                    "total_pending_lines": 7751,
                    "oldest_pending_age_seconds": 234.997,
                },
            },
            "bounded_recovery_contract": {
                "active_drain_progress": False,
                "drain_delta_signal_observed": False,
                "hard_gate_active": False,
                "effective_hard_gate_active": False,
            },
            "external_route_verification": {"verification_state": "ready"},
            "storage_resilience": {"overall_status": "ready"},
            "storage_efficiency_contract": {"overall_status": "ready", "grade": "A+"},
            "data_integrity": {
                "sql_invalid_lines": 0,
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_oversize_payloads": 0,
                "sql_overlay_ops_write_failures": 0,
            },
            "writer_shedding": {"hard_breaches": [], "elevated_breaches": []},
        }
    )

    assert ok is False
    assert blockers == ["storage_pressure_index_high"]


def test_health_fast_treats_deferred_off_hours_backlog_as_managed_for_paper() -> None:
    ok, blockers = src._storage_ready(
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 53.028,
            "backpressure": {
                "core_pending_lines": 809,
                "support_pending_lines": 8246,
                "deferred_pending_lines": 15899259,
                "total_pending_lines": 15908314,
                "oldest_pending_age_seconds": 2582.833,
                "pending_lines_threshold": 15000,
            },
            "storage": {"backlog_drain_status": "waiting_for_off_hours"},
            "external_route_verification": {"verification_state": "ready"},
            "data_integrity": {
                "sql_invalid_lines": 0,
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_oversize_payloads": 0,
                "sql_overlay_ops_write_failures": 0,
            },
            "writer_shedding": {"hard_breaches": ["deferred"], "elevated_breaches": ["core", "deferred"]},
        },
        {
            "ok": True,
            "overall_status": "ready",
            "root_cause": {
                "raw_live": {
                    "ok": True,
                    "status": "ready",
                    "core_pending_lines": 765,
                    "total_pending_lines": 1814,
                    "oldest_pending_age_seconds": 0.0,
                    "max_core_pending_lines": 5000,
                    "max_total_pending_lines": 15000,
                    "max_oldest_pending_age_seconds": 900,
                }
            },
        },
    )

    assert ok is True
    assert blockers == []
