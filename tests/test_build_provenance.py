from pathlib import Path

from core.build_provenance import build_local_provenance, verify_local_provenance

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_local_provenance_verifies_subjects_without_claiming_signed_ci_evidence() -> (
    None
):
    statement = build_local_provenance(
        project_root=PROJECT_ROOT,
        subject_paths=["core/build_provenance.py"],
        material_paths=["config/authoritative_systems_v1.json"],
        git_commit="probe-commit",
    )
    verification = verify_local_provenance(statement, project_root=PROJECT_ROOT)

    assert verification["ok"] is True
    assert verification["signed"] is False
    assert verification["trusted_builder"] is False
    assert verification["promotion_evidence_eligible"] is False
