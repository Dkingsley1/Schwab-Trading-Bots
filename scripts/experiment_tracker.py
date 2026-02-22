import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Track experiment metadata and outcomes.")
    parser.add_argument("--name", default="runtime_session")
    parser.add_argument("--status", default="started")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    files = [
        PROJECT_ROOT / "governance" / "feature_versions" / "latest.json",
        PROJECT_ROOT / "governance" / "walk_forward" / "promotion_gate_latest.json",
        PROJECT_ROOT / "exports" / "one_numbers" / "one_numbers_summary.json",
    ]
    artifacts = {}
    for p in files:
        if p.exists():
            artifacts[str(p.relative_to(PROJECT_ROOT))] = _sha(p)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "name": args.name,
        "status": args.status,
        "notes": args.notes,
        "artifact_hashes": artifacts,
    }

    out = PROJECT_ROOT / "governance" / "experiments" / "experiment_registry.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps(row, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
