import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Version snapshot for feature/data/runtime inputs.")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "governance" / "feature_versions"))
    args = parser.parse_args()

    key_files = [
        PROJECT_ROOT / "config" / "trade_learning_policy.json",
        PROJECT_ROOT / "master_bot_registry.json",
        PROJECT_ROOT / "scripts" / "run_shadow_training_loop.py",
        PROJECT_ROOT / "config" / "requirements.lock.txt",
    ]
    env_keys = [
        "FEATURE_TIMEFRAMES",
        "FEATURE_WINDOWS",
        "SHADOW_SYMBOLS_CORE",
        "SHADOW_SYMBOLS_VOLATILE",
        "SHADOW_SYMBOLS_DEFENSIVE",
        "SHADOW_SYMBOLS_COMMOD_FX_INTL",
    ]

    file_hashes = {}
    for p in key_files:
        if p.exists():
            try:
                file_hashes[str(p.relative_to(PROJECT_ROOT))] = _sha256_file(p)
            except Exception:
                file_hashes[str(p.relative_to(PROJECT_ROOT))] = "error"

    env_snapshot = {k: os.getenv(k, "") for k in env_keys}
    env_hash = _sha256_text(json.dumps(env_snapshot, sort_keys=True, ensure_ascii=True))

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "file_hashes": file_hashes,
        "env": env_snapshot,
        "env_hash": env_hash,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"feature_version_{stamp}.json"
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    (out_dir / "latest.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
