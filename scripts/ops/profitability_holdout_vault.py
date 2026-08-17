#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, parse_iso_utc, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, parse_iso_utc, write_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "profitability_evidence_firewall_v1.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "research" / "profitability_holdout_vault_latest.json"


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _resolve(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or "")).expanduser()
    return path if path.is_absolute() else project_root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _access_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _candidate(project_root: Path) -> dict[str, Any]:
    state = load_json(project_root / "governance" / "runtime" / "production_candidate_state.json")
    return {
        "candidate_id": str(state.get("candidate_id") or "").strip(),
        "generation": _safe_int(state.get("generation"), 0),
    }


def seal_dataset(project_root: Path, config: dict[str, Any], dataset: Path) -> dict[str, Any]:
    policy = _as_dict(config.get("holdout_vault"))
    candidate = _candidate(project_root)
    path = dataset.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not candidate["candidate_id"]:
        raise RuntimeError("production candidate is not initialized")
    manifest_path = _resolve(project_root, policy.get("manifest"))
    if manifest_path.exists():
        raise RuntimeError("holdout manifest already exists; sealed holdouts are immutable")
    line_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            line_count += chunk.count(b"\n")
    manifest = {
        "schema_version": 1,
        "vault_id": hashlib.sha256(f"{candidate['candidate_id']}:{path}:{_sha256(path)}".encode("utf-8")).hexdigest()[:24],
        "candidate_id": candidate["candidate_id"],
        "candidate_generation": candidate["generation"],
        "dataset_path": str(path),
        "dataset_sha256": _sha256(path),
        "dataset_size_bytes": path.stat().st_size,
        "dataset_line_count": line_count,
        "sealed_at_utc": iso_now(),
        "purpose": "final_evaluation_only",
        "training_access_forbidden": True,
        "maximum_evaluation_accesses": _safe_int(policy.get("maximum_evaluation_accesses_per_candidate"), 1),
    }
    write_payload(manifest_path, manifest)
    return manifest


def record_evaluation_access(project_root: Path, config: dict[str, Any], *, evidence: str) -> dict[str, Any]:
    policy = _as_dict(config.get("holdout_vault"))
    manifest_path = _resolve(project_root, policy.get("manifest"))
    access_path = _resolve(project_root, policy.get("access_log"))
    manifest = load_json(manifest_path)
    candidate = _candidate(project_root)
    if not manifest or str(manifest.get("candidate_id") or "") != candidate["candidate_id"]:
        raise RuntimeError("candidate-bound sealed holdout manifest is not ready")
    existing = [
        row
        for row in _access_rows(access_path)
        if str(row.get("candidate_id") or "") == candidate["candidate_id"]
        and str(row.get("purpose") or "") == "evaluation"
    ]
    maximum = _safe_int(policy.get("maximum_evaluation_accesses_per_candidate"), 1)
    if len(existing) >= maximum:
        raise RuntimeError("holdout evaluation access limit reached")
    evidence_text = str(evidence or "").strip()
    if not evidence_text:
        raise ValueError("evaluation access requires a non-empty evidence reference")
    event = {
        "schema_version": 1,
        "timestamp_utc": iso_now(),
        "candidate_id": candidate["candidate_id"],
        "vault_id": str(manifest.get("vault_id") or ""),
        "purpose": "evaluation",
        "evidence": evidence_text,
    }
    access_path.parent.mkdir(parents=True, exist_ok=True)
    with access_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
    return event


def build_payload(project_root: Path = PROJECT_ROOT, *, config_path: Path | None = None) -> dict[str, Any]:
    config = load_json(config_path or project_root / "config" / DEFAULT_CONFIG_PATH.name)
    policy = _as_dict(config.get("holdout_vault"))
    candidate = _candidate(project_root)
    manifest_path = _resolve(project_root, policy.get("manifest"))
    access_path = _resolve(project_root, policy.get("access_log"))
    manifest = load_json(manifest_path)
    dataset_path = _resolve(project_root, manifest.get("dataset_path")) if manifest.get("dataset_path") else Path()
    manifest_present = bool(manifest)
    dataset_present = bool(manifest_present and dataset_path.is_file())
    digest_current = _sha256(dataset_path) if dataset_present else ""
    digest_matches = bool(dataset_present and digest_current == str(manifest.get("dataset_sha256") or ""))
    candidate_matches = bool(
        candidate["candidate_id"]
        and str(manifest.get("candidate_id") or "") == candidate["candidate_id"]
    )
    sealed_at = parse_iso_utc(manifest.get("sealed_at_utc"))
    accesses = [
        row
        for row in _access_rows(access_path)
        if str(row.get("candidate_id") or "") == candidate["candidate_id"]
    ]
    training_accesses = [row for row in accesses if str(row.get("purpose") or "") == "training"]
    evaluation_accesses = [row for row in accesses if str(row.get("purpose") or "") == "evaluation"]
    maximum = _safe_int(policy.get("maximum_evaluation_accesses_per_candidate"), 1)
    evidence_ready = bool(
        manifest_present
        and dataset_present
        and digest_matches
        and candidate_matches
        and sealed_at is not None
        and bool(manifest.get("training_access_forbidden", False))
        and not training_accesses
        and len(evaluation_accesses) <= maximum
    )
    blockers = []
    if not manifest_present:
        blockers.append("sealed_holdout_manifest_pending")
    elif not dataset_present:
        blockers.append("sealed_holdout_dataset_missing")
    if manifest_present and not digest_matches:
        blockers.append("sealed_holdout_digest_mismatch")
    if manifest_present and not candidate_matches:
        blockers.append("sealed_holdout_candidate_mismatch")
    if training_accesses:
        blockers.append("forbidden_training_access_detected")
    if len(evaluation_accesses) > maximum:
        blockers.append("evaluation_access_limit_exceeded")
    implementation_ready = bool(
        policy
        and policy.get("manifest")
        and policy.get("access_log")
        and policy.get("forbid_training_access", False)
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": implementation_ready,
        "overall_status": "ready" if evidence_ready else "evidence_pending",
        "implementation_ready": implementation_ready,
        "evidence_ready": evidence_ready,
        "candidate_binding": candidate,
        "manifest_path": str(manifest_path),
        "access_log_path": str(access_path),
        "manifest": manifest,
        "dataset_present": dataset_present,
        "digest_matches": digest_matches,
        "evaluation_access_count": len(evaluation_accesses),
        "training_access_count": len(training_accesses),
        "maximum_evaluation_accesses": maximum,
        "blockers": blockers,
        "control_contract": {
            "seal_is_immutable": True,
            "candidate_binding_required": True,
            "dataset_digest_verified_on_every_status_check": True,
            "training_access_forbidden": True,
            "evaluation_access_is_append_only_and_bounded": True,
            "missing_or_tampered_holdout_fails_closed": True,
            "live_execution_authority": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage the candidate-bound profitability holdout vault.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=Path("config") / DEFAULT_CONFIG_PATH.name)
    parser.add_argument("--out-file", type=Path, default=Path("governance/research") / DEFAULT_OUT_PATH.name)
    parser.add_argument("--seal-dataset", type=Path)
    parser.add_argument("--record-evaluation-access", action="store_true")
    parser.add_argument("--evidence", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_json(config_path)
    operation: dict[str, Any] = {}
    if args.seal_dataset:
        operation = {"sealed_manifest": seal_dataset(project_root, config, args.seal_dataset)}
    if args.record_evaluation_access:
        operation = {"access_event": record_evaluation_access(project_root, config, evidence=args.evidence)}
    payload = build_payload(project_root, config_path=config_path)
    if operation:
        payload["operation"] = operation
    out_path = args.out_file if args.out_file.is_absolute() else project_root / args.out_file
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "profitability_holdout_vault "
            f"status={payload['overall_status']} evidence_ready={int(bool(payload['evidence_ready']))}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
