import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _collect_backups(root: Path) -> list[Path]:
    out: list[Path] = []
    out.extend(sorted((root / "governance").glob("master_bot_registry.backup_*.json")))
    out.extend(sorted((root / "governance").glob("registry_backup_before_retrain*.json")))
    # de-dup while preserving order
    seen = set()
    uniq: list[Path] = []
    for p in out:
        if str(p) in seen:
            continue
        seen.add(str(p))
        uniq.append(p)
    return uniq


def main() -> int:
    parser = argparse.ArgumentParser(description="Model lifecycle hygiene checks, manifest, and backup maintenance.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--keep-backups", type=int, default=25)
    parser.add_argument("--apply-prune", action="store_true")
    parser.add_argument("--update-last-known-good", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument("--max-missing-active-artifacts", type=int, default=2)
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "lifecycle" / "model_lifecycle_latest.json"))
    parser.add_argument("--manifest-file", default=str(PROJECT_ROOT / "governance" / "lifecycle" / "model_manifest_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    reg = _load(registry_path)
    sub_bots = reg.get("sub_bots") if isinstance(reg.get("sub_bots"), list) else []

    active_rows = [r for r in sub_bots if isinstance(r, dict) and bool(r.get("active", False))]
    missing_active = []
    manifest_rows = []

    for row in active_rows:
        bot_id = str(row.get("bot_id", "")).strip()
        model_path = Path(str(row.get("model_path") or ""))
        log_path = Path(str(row.get("log_file") or ""))
        model_ok = model_path.exists()
        log_ok = log_path.exists()

        if not model_ok or not log_ok:
            missing_active.append(
                {
                    "bot_id": bot_id,
                    "model_exists": bool(model_ok),
                    "log_exists": bool(log_ok),
                    "model_path": str(model_path),
                    "log_path": str(log_path),
                }
            )

        manifest_rows.append(
            {
                "bot_id": bot_id,
                "model_path": str(model_path),
                "model_mtime_utc": _mtime_iso(model_path) if model_ok else None,
                "log_path": str(log_path),
                "log_mtime_utc": _mtime_iso(log_path) if log_ok else None,
                "quality_score": float(row.get("quality_score", 0.0) or 0.0),
                "test_accuracy": row.get("test_accuracy"),
            }
        )

    usage = shutil.disk_usage(str(PROJECT_ROOT))
    free_gb = usage.free / (1024.0 ** 3)

    backups = _collect_backups(PROJECT_ROOT)
    pruned = []
    if args.apply_prune and len(backups) > int(args.keep_backups):
        stale = backups[: len(backups) - int(args.keep_backups)]
        for p in stale:
            try:
                p.unlink(missing_ok=True)
                pruned.append(str(p))
            except Exception:
                continue

    lifecycle_dir = PROJECT_ROOT / "governance" / "lifecycle"
    lifecycle_dir.mkdir(parents=True, exist_ok=True)

    if args.update_last_known_good and registry_path.exists():
        lk = lifecycle_dir / "registry_last_known_good.json"
        lk.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")

    manifest_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "active_bots": len(active_rows),
        "rows": manifest_rows,
    }
    Path(args.manifest_file).write_text(json.dumps(manifest_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    ok = (len(missing_active) <= int(args.max_missing_active_artifacts)) and (free_gb >= float(args.min_free_gb))

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": bool(ok),
        "thresholds": {
            "min_free_gb": float(args.min_free_gb),
            "max_missing_active_artifacts": int(args.max_missing_active_artifacts),
            "keep_backups": int(args.keep_backups),
        },
        "disk": {"free_gb": round(free_gb, 2)},
        "active_bots": len(active_rows),
        "missing_active_artifacts": len(missing_active),
        "missing_active_examples": missing_active[:40],
        "backup_files": len(backups),
        "pruned_backups": pruned,
        "manifest_file": str(Path(args.manifest_file)),
    }

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "model_lifecycle_hygiene "
            f"ok={str(payload['ok']).lower()} free_gb={payload['disk']['free_gb']:.2f} "
            f"missing_active={payload['missing_active_artifacts']} pruned={len(pruned)}"
        )

    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
