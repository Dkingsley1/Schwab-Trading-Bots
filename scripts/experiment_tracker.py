import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = (
    "governance/feature_versions/latest.json",
    "governance/walk_forward/promotion_gate_latest.json",
    "exports/one_numbers/one_numbers_summary.json",
)


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
) -> dict[str, Any]:
    resolved_paths: list[Path] = []
    seen: set[str] = set()
    for raw in list(DEFAULT_ARTIFACTS) + list(artifacts or []) + [dataset_file, model_file, replay_file]:
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

    dataset_hash = _sha(dataset_path) if dataset_path is not None and dataset_path.exists() and dataset_path.is_file() else ""
    model_hash = _sha(model_path) if model_path is not None and model_path.exists() and model_path.is_file() else ""
    replay_hash = ""
    if replay_payload:
        replay_hash = str(replay_payload.get("replay_hash") or replay_payload.get("paper_replay_hash") or "").strip()
    if not replay_hash and replay_path is not None and replay_path.exists() and replay_path.is_file():
        replay_hash = _sha(replay_path)

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
            "tags": normalized_tags,
        }
    )
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    experiment_id = f"exp_{timestamp_utc[:19].replace(':', '').replace('-', '').replace('T', '_')}_{replayability['bundle_hash'][:12]}"

    return {
        "timestamp_utc": timestamp_utc,
        "experiment_id": experiment_id,
        "name": str(name or "runtime_session"),
        "status": str(status or "started"),
        "notes": str(notes or ""),
        "tags": normalized_tags,
        "artifact_hashes": artifact_hashes,
        "replayability": replayability,
    }


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
    args = parser.parse_args()

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
    )

    out = PROJECT_ROOT / "governance" / "experiments" / "experiment_registry.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps(row, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
