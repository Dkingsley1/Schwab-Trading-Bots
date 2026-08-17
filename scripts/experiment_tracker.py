import argparse
import hmac
import hashlib
import json
import os
import secrets
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = (
    "governance/feature_versions/latest.json",
    "governance/walk_forward/promotion_gate_latest.json",
    "exports/one_numbers/one_numbers_summary.json",
)
DEFAULT_LEDGER_PATH = PROJECT_ROOT / "governance" / "experiments" / "immutable_experiment_ledger.jsonl"
DEFAULT_LEDGER_SUMMARY_PATH = PROJECT_ROOT / "governance" / "experiments" / "immutable_experiment_ledger_latest.json"
DEFAULT_SIGNING_KEY_PATH = PROJECT_ROOT / "governance" / "experiments" / "immutable_ledger_signing_key.txt"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_path(project_root: Path, raw: str) -> Path:
    candidate = Path(str(raw or "").strip()).expanduser()
    if not candidate.is_absolute():
        candidate = (project_root / candidate).resolve()
    return candidate


def _relative_label(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except Exception:
        return str(path.resolve())


def _bundle_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _sign_payload(payload: dict[str, Any], secret: str) -> str:
    if not str(secret or "").strip():
        return ""
    blob = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(str(secret).encode("utf-8"), blob, hashlib.sha256).hexdigest()


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return rows
    return rows


def _ensure_signing_key(path: Path) -> str:
    if path.exists():
        os.chmod(path, 0o600)
        return str(path.read_text(encoding="utf-8").strip())
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_hex(32)
    path.write_text(secret, encoding="utf-8")
    os.chmod(path, 0o600)
    return secret


def build_experiment_row(
    project_root: Path,
    *,
    name: str,
    status: str,
    notes: str,
    artifacts: list[str] | None = None,
    dataset_file: str = "",
    model_file: str = "",
    replay_file: str = "",
    tags: list[str] | None = None,
    event_type: str = "experiment",
    approval_file: str = "",
    rollback_file: str = "",
    deploy_file: str = "",
    signing_secret: str = "",
    signing_key_id: str = "",
) -> dict[str, Any]:
    resolved_paths: list[Path] = []
    seen: set[str] = set()
    for raw in list(DEFAULT_ARTIFACTS) + list(artifacts or []) + [
        dataset_file,
        model_file,
        replay_file,
        approval_file,
        rollback_file,
        deploy_file,
    ]:
        if not str(raw or "").strip():
            continue
        path = _resolve_path(project_root, str(raw))
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        resolved_paths.append(path)

    artifact_hashes: dict[str, str] = {}
    for path in resolved_paths:
        if path.exists() and path.is_file():
            artifact_hashes[_relative_label(project_root, path)] = _sha(path)

    feature_versions = _load_json(project_root / "governance" / "feature_versions" / "latest.json")
    content_store = _load_json(project_root / "governance" / "content_store" / "latest.json")
    replay_path = _resolve_path(project_root, replay_file) if str(replay_file or "").strip() else None
    replay_payload = _load_json(replay_path) if replay_path is not None and replay_path.exists() else {}
    dataset_path = _resolve_path(project_root, dataset_file) if str(dataset_file or "").strip() else None
    model_path = _resolve_path(project_root, model_file) if str(model_file or "").strip() else None
    approval_path = _resolve_path(project_root, approval_file) if str(approval_file or "").strip() else None
    rollback_path = _resolve_path(project_root, rollback_file) if str(rollback_file or "").strip() else None
    deploy_path = _resolve_path(project_root, deploy_file) if str(deploy_file or "").strip() else None

    dataset_hash = _sha(dataset_path) if dataset_path is not None and dataset_path.exists() and dataset_path.is_file() else ""
    model_hash = _sha(model_path) if model_path is not None and model_path.exists() and model_path.is_file() else ""
    replay_hash = ""
    if replay_payload:
        replay_hash = str(replay_payload.get("replay_hash") or replay_payload.get("paper_replay_hash") or "").strip()
    if not replay_hash and replay_path is not None and replay_path.exists() and replay_path.is_file():
        replay_hash = _sha(replay_path)

    attestation_contract = {
        "approval_hash": _sha(approval_path) if approval_path is not None and approval_path.exists() and approval_path.is_file() else "",
        "rollback_hash": _sha(rollback_path) if rollback_path is not None and rollback_path.exists() and rollback_path.is_file() else "",
        "deploy_hash": _sha(deploy_path) if deploy_path is not None and deploy_path.exists() and deploy_path.is_file() else "",
        "approval_file": _relative_label(project_root, approval_path) if approval_path is not None and approval_path.exists() else "",
        "rollback_file": _relative_label(project_root, rollback_path) if rollback_path is not None and rollback_path.exists() else "",
        "deploy_file": _relative_label(project_root, deploy_path) if deploy_path is not None and deploy_path.exists() else "",
    }
    attestation_contract["attestation_ready"] = bool(
        attestation_contract["approval_hash"] and attestation_contract["rollback_hash"] and attestation_contract["deploy_hash"]
    )
    normalized_tags = sorted({str(tag).strip() for tag in (tags or []) if str(tag).strip()})
    replayability = {
        "dataset_hash": dataset_hash,
        "model_hash": model_hash,
        "replay_hash": replay_hash,
        "feature_env_hash": str(feature_versions.get("env_hash") or "").strip(),
        "content_store_manifest_hash": str(content_store.get("manifest_hash") or "").strip(),
        "artifact_hash_count": len(artifact_hashes),
        "exact_replay_ready": bool(dataset_hash and model_hash and replay_hash),
    }
    replayability["bundle_hash"] = _bundle_hash(
        {
            "artifact_hashes": artifact_hashes,
            "dataset_hash": replayability["dataset_hash"],
            "model_hash": replayability["model_hash"],
            "replay_hash": replayability["replay_hash"],
            "feature_env_hash": replayability["feature_env_hash"],
            "content_store_manifest_hash": replayability["content_store_manifest_hash"],
            "event_type": str(event_type or "experiment").strip().lower(),
            "attestations": attestation_contract,
            "tags": normalized_tags,
        }
    )
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    experiment_id = f"exp_{timestamp_utc[:19].replace(':', '').replace('-', '').replace('T', '_')}_{replayability['bundle_hash'][:12]}"
    ledger_contract = {
        "append_only_path": str(DEFAULT_LEDGER_PATH),
        "latest_summary_path": str(DEFAULT_LEDGER_SUMMARY_PATH),
        "signature_key_id": str(signing_key_id or "local-immutable-ledger").strip(),
    }
    signature_digest = _sign_payload(
        {
            "experiment_id": experiment_id,
            "event_type": str(event_type or "experiment").strip().lower(),
            "replayability": replayability,
            "attestations": attestation_contract,
            "tags": normalized_tags,
        },
        str(signing_secret or ""),
    )
    ledger_contract["signature_digest"] = signature_digest
    ledger_contract["signature_ready"] = bool(signature_digest)
    ledger_contract["append_only_ready"] = True

    return {
        "timestamp_utc": timestamp_utc,
        "experiment_id": experiment_id,
        "event_type": str(event_type or "experiment").strip().lower(),
        "name": str(name or "runtime_session"),
        "status": str(status or "started"),
        "notes": str(notes or ""),
        "tags": normalized_tags,
        "artifact_hashes": artifact_hashes,
        "replayability": replayability,
        "attestations": attestation_contract,
        "ledger_contract": ledger_contract,
    }


def write_experiment_artifacts(
    project_root: Path,
    row: dict[str, Any],
    *,
    registry_path: Path | None = None,
    ledger_path: Path | None = None,
    ledger_summary_path: Path | None = None,
) -> dict[str, Any]:
    resolved_registry = registry_path or (project_root / "governance" / "experiments" / "experiment_registry.jsonl")
    resolved_ledger = ledger_path or (project_root / "governance" / "experiments" / "immutable_experiment_ledger.jsonl")
    resolved_summary = ledger_summary_path or (project_root / "governance" / "experiments" / "immutable_experiment_ledger_latest.json")

    for path in (resolved_registry, resolved_ledger):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    ledger_rows = _load_jsonl_rows(resolved_ledger)
    signed_row_count = sum(1 for item in ledger_rows if bool(((item.get("ledger_contract") or {}).get("signature_ready", False))))
    attested_row_count = sum(1 for item in ledger_rows if bool(((item.get("attestations") or {}).get("attestation_ready", False))))
    event_counts = Counter(str(item.get("event_type") or "experiment").strip().lower() for item in ledger_rows if isinstance(item, dict))
    latest_row = ledger_rows[-1] if ledger_rows else dict(row)
    latest_replayability = latest_row.get("replayability") if isinstance(latest_row.get("replayability"), dict) else {}
    latest_attestations = latest_row.get("attestations") if isinstance(latest_row.get("attestations"), dict) else {}
    latest_ledger_contract = latest_row.get("ledger_contract") if isinstance(latest_row.get("ledger_contract"), dict) else {}
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "overall_status": (
            "ready"
            if bool(latest_ledger_contract.get("signature_ready", False))
            and bool(latest_replayability.get("exact_replay_ready", False))
            else "degraded"
        ),
        "append_only_ready": True,
        "ledger_row_count": len(ledger_rows),
        "signed_row_count": signed_row_count,
        "attested_row_count": attested_row_count,
        "latest_experiment_id": str(latest_row.get("experiment_id") or ""),
        "latest_event_type": str(latest_row.get("event_type") or ""),
        "latest_exact_replay_ready": bool(latest_replayability.get("exact_replay_ready", False)),
        "latest_attestation_ready": bool(latest_attestations.get("attestation_ready", False)),
        "latest_signature_ready": bool(latest_ledger_contract.get("signature_ready", False)),
        "event_type_counts": dict(event_counts),
        "ledger_paths": {
            "registry": str(resolved_registry),
            "append_only": str(resolved_ledger),
            "latest_summary": str(resolved_summary),
        },
    }
    resolved_summary.parent.mkdir(parents=True, exist_ok=True)
    resolved_summary.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Track experiment metadata and outcomes.")
    parser.add_argument("--name", default="runtime_session")
    parser.add_argument("--status", default="started")
    parser.add_argument("--notes", default="")
    parser.add_argument("--artifact", action="append", default=[], help="Additional artifact path to hash and track.")
    parser.add_argument("--dataset-file", default="", help="Dataset artifact to bind into the replayability bundle.")
    parser.add_argument("--model-file", default="", help="Model artifact to bind into the replayability bundle.")
    parser.add_argument("--replay-file", default="", help="Replay or drill artifact containing a replay hash.")
    parser.add_argument("--tag", action="append", default=[], help="Optional experiment tag; may be provided multiple times.")
    parser.add_argument("--event-type", default="experiment")
    parser.add_argument("--approval-file", default="")
    parser.add_argument("--rollback-file", default="")
    parser.add_argument("--deploy-file", default="")
    parser.add_argument("--signing-key-file", default=str(DEFAULT_SIGNING_KEY_PATH))
    args = parser.parse_args()

    signing_key_path = Path(args.signing_key_file).expanduser()
    signing_secret = _ensure_signing_key(signing_key_path)
    row = build_experiment_row(
        PROJECT_ROOT,
        name=str(args.name),
        status=str(args.status),
        notes=str(args.notes),
        artifacts=list(args.artifact or []),
        dataset_file=str(args.dataset_file or ""),
        model_file=str(args.model_file or ""),
        replay_file=str(args.replay_file or ""),
        tags=list(args.tag or []),
        event_type=str(args.event_type or "experiment"),
        approval_file=str(args.approval_file or ""),
        rollback_file=str(args.rollback_file or ""),
        deploy_file=str(args.deploy_file or ""),
        signing_secret=signing_secret,
        signing_key_id=str(signing_key_path.name or "immutable_ledger_signing_key.txt"),
    )
    write_experiment_artifacts(PROJECT_ROOT, row)
    print(json.dumps(row, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
