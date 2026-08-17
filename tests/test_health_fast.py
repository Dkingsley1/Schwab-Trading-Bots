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


def test_health_fast_accepts_bounded_drain_on_verified_local_hot_route() -> None:
    ok, blockers = src._storage_ready(
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.697,
            "backpressure": {
                "total_pending_lines": 3479,
                "pending_lines_threshold": 15000,
                "raw_live": {
                    "core_pending_lines": 2056,
                    "total_pending_lines": 3479,
                    "oldest_pending_age_seconds": 167.287,
                },
            },
            "bounded_recovery_contract": {
                "active_drain_progress": True,
                "hard_gate_active": False,
                "effective_hard_gate_active": False,
            },
            "external_route_verification": {"verification_state": "active_local_ready"},
            "storage_resilience": {"overall_status": "ready"},
            "storage_efficiency_contract": {"overall_status": "ready", "grade": "A+"},
            "data_integrity": {},
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


def test_health_fast_accepts_clean_bounded_storage_steady_state_without_drain() -> None:
    ok, blockers = src._storage_ready(
        {
            "overall_status": "ready",
            "severity": "elevated",
            "pressure_index": 0.763,
            "backpressure": {
                "total_pending_lines": 771,
                "pending_lines_threshold": 15000,
                "raw_live": {
                    "core_pending_lines": 771,
                    "total_pending_lines": 771,
                    "oldest_pending_age_seconds": 183.0,
                },
            },
            "bounded_recovery_contract": {
                "active_drain_progress": False,
                "drain_delta_signal_observed": False,
                "hard_gate_active": False,
                "effective_hard_gate_active": False,
            },
            "external_route_verification": {"verification_state": "active_local_ready"},
            "storage_resilience": {"overall_status": "ready"},
            "storage_efficiency_contract": {"overall_status": "ready", "grade": "A"},
            "data_integrity": {},
            "writer_shedding": {"hard_breaches": [], "elevated_breaches": []},
        }
    )

    assert ok is True
    assert blockers == []


def test_health_fast_accepts_raw_collection_debt_only_when_operational_projection_is_ready() -> None:
    rollup = {
        "overall_status": "degraded",
        "operational_status": "ready",
        "operational_ok": True,
        "operational_collection": {"status": "ready", "ok": True},
    }

    assert src._collection_rollup_advisory_ready(rollup, "degraded") is True
    rollup["operational_ok"] = False
    rollup["operational_status"] = "degraded"
    rollup["operational_collection"] = {"status": "degraded", "ok": False}
    assert src._collection_rollup_advisory_ready(rollup, "degraded") is False


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


def test_health_fast_treats_small_residual_drain_active_as_managed_for_paper() -> None:
    storage = {
        "overall_status": "ready",
        "severity": "stable",
        "pressure_index": 0.534,
        "backpressure": {
            "core_pending_lines": 9,
            "support_pending_lines": 1,
            "deferred_pending_lines": 118,
            "total_pending_lines": 128,
            "oldest_pending_age_seconds": 128.22,
            "pending_lines_threshold": 15000,
        },
        "storage": {"backlog_drain_status": "drain_active"},
        "external_route_verification": {"verification_state": "ready"},
        "storage_resilience": {"overall_status": "ready"},
        "storage_efficiency_contract": {"overall_status": "ready", "grade": "A+"},
        "data_integrity": {
            "sql_invalid_lines": 0,
            "sql_overlay_invalid_lines": 0,
            "sql_overlay_oversize_payloads": 0,
            "sql_overlay_ops_write_failures": 0,
        },
        "writer_shedding": {"hard_breaches": ["support_telemetry"], "elevated_breaches": ["support_telemetry"]},
    }

    ok, blockers = src._storage_ready(storage)
    relief = src._paper_hot_path_storage_relief(storage)

    assert ok is True
    assert blockers == []
    assert relief["active"] is True
    assert relief["status"] == "managed_small_residual_drain"
