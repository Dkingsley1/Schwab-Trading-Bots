from pathlib import Path

from core.authoritative_systems import (
    EXPECTED_CONTROL_COUNT,
    EXPECTED_REFERENCE_COUNT,
    load_registry,
    validate_registry,
)
from scripts.ops.authoritative_systems_control import build_payload
from scripts.ops.live_feed_status_contract import _authoritative_systems_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_registry_has_exactly_thirty_nine_primary_references_and_eighteen_owned_controls() -> (
    None
):
    registry = load_registry()
    report = validate_registry(registry, project_root=PROJECT_ROOT)

    assert report["ok"] is True
    assert report["reference_count"] == EXPECTED_REFERENCE_COUNT == 39
    assert report["control_count"] == EXPECTED_CONTROL_COUNT == 18
    assert all(count > 0 for count in report["control_adoption"].values())


def test_authoritative_control_passes_all_local_structural_probes() -> None:
    payload = build_payload(PROJECT_ROOT)

    assert payload["ok"] is True
    assert payload["ready_control_count"] == payload["control_count"] == 18
    assert payload["grade_scope"] == "local structural implementation only"
    assert payload["live_execution_authority"] is False
    assert payload["soak_acceptance"]["reset_soak_clock"] is False


def test_livefeed_keeps_structural_and_external_evidence_counts_separate() -> None:
    row = _authoritative_systems_row(
        {
            "authoritative_systems": {
                "present": True,
                "fresh": True,
                "age_seconds": 2.0,
                "payload": {
                    "ok": True,
                    "grade": "A+",
                    "reference_count": 39,
                    "reference_target": 39,
                    "ready_control_count": 18,
                    "control_count": 18,
                    "external_evidence": {"ready_count": 0, "item_count": 10},
                    "live_execution_authority": False,
                },
            }
        }
    )

    assert row["status"] == "ready"
    assert row["references"] == row["reference_target"] == 39
    assert row["ready_controls"] == row["control_count"] == 18
    assert row["external_evidence_ready"] == 0
    assert row["external_evidence_count"] == 10
    assert row["live_authority"] is False
