from pathlib import Path

from core.authoritative_systems import (
    EXPECTED_REFERENCE_COUNT,
    load_registry,
    validate_registry,
)
from scripts.ops.authoritative_systems_control import build_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_registry_has_exactly_twenty_primary_references_and_eight_owned_controls() -> (
    None
):
    registry = load_registry()
    report = validate_registry(registry, project_root=PROJECT_ROOT)

    assert report["ok"] is True
    assert report["reference_count"] == EXPECTED_REFERENCE_COUNT == 20
    assert report["control_count"] == 8
    assert all(count > 0 for count in report["control_adoption"].values())


def test_authoritative_control_passes_all_local_structural_probes() -> None:
    payload = build_payload(PROJECT_ROOT)

    assert payload["ok"] is True
    assert payload["ready_control_count"] == payload["control_count"] == 8
    assert payload["grade_scope"] == "local structural implementation only"
    assert payload["live_execution_authority"] is False
    assert payload["soak_acceptance"]["reset_soak_clock"] is False
