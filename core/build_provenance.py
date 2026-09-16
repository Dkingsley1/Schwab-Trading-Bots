from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
PREDICATE_TYPE = "https://slsa.dev/provenance/v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(root: Path, value: str | Path) -> tuple[Path, str]:
    path = (
        (root / value).resolve()
        if not Path(value).is_absolute()
        else Path(value).resolve()
    )
    try:
        relative = path.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"provenance path escapes project root: {value}") from exc
    if not path.is_file():
        raise ValueError(f"provenance subject is not a file: {relative}")
    return path, relative


def build_local_provenance(
    *,
    project_root: str | Path,
    subject_paths: Sequence[str | Path],
    material_paths: Sequence[str | Path],
    git_commit: str,
    builder_id: str = "local://schwab-trading-bot/authoritative-control",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    subjects = []
    for value in sorted(subject_paths, key=str):
        path, relative = _safe_path(root, value)
        subjects.append({"name": relative, "digest": {"sha256": _file_sha256(path)}})
    materials = []
    for value in sorted(material_paths, key=str):
        path, relative = _safe_path(root, value)
        materials.append(
            {
                "uri": f"file://project/{relative}",
                "digest": {"sha256": _file_sha256(path)},
            }
        )
    return {
        "_type": STATEMENT_TYPE,
        "subject": subjects,
        "predicateType": PREDICATE_TYPE,
        "predicate": {
            "buildDefinition": {
                "buildType": "https://github.com/Dkingsley1/Schwab-Trading-Bots/local-control@v1",
                "externalParameters": {"git_commit": git_commit},
                "internalParameters": {},
                "resolvedDependencies": materials,
            },
            "runDetails": {
                "builder": {"id": builder_id},
                "metadata": {"invocationId": f"local-{git_commit[:16]}"},
                "byproducts": [],
            },
        },
        "attestation": {
            "signed": False,
            "trusted_builder": False,
            "ci_generated": False,
            "promotion_evidence_eligible": False,
        },
        "execution_authority": False,
    }


def verify_local_provenance(
    statement: Mapping[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    errors: list[str] = []
    if statement.get("_type") != STATEMENT_TYPE:
        errors.append("statement_type_invalid")
    if statement.get("predicateType") != PREDICATE_TYPE:
        errors.append("predicate_type_invalid")
    root = Path(project_root).resolve()
    for subject in statement.get("subject") or []:
        row = dict(subject)
        try:
            path, _ = _safe_path(root, str(row.get("name") or ""))
        except ValueError:
            errors.append(f"subject_missing:{row.get('name') or 'unknown'}")
            continue
        expected = str(dict(row.get("digest") or {}).get("sha256") or "")
        if _file_sha256(path) != expected:
            errors.append(f"subject_digest_mismatch:{row.get('name')}")
    attestation = dict(statement.get("attestation") or {})
    return {
        "ok": not errors,
        "errors": errors,
        "subject_count": len(statement.get("subject") or []),
        "signed": bool(attestation.get("signed")),
        "trusted_builder": bool(attestation.get("trusted_builder")),
        "promotion_evidence_eligible": bool(
            attestation.get("promotion_evidence_eligible")
        ),
        "external_evidence_debt": ["signed_ci_attestation", "trusted_builder_identity"],
    }
