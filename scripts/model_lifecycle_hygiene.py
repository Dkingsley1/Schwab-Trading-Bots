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
    out.extend(sorted(root.glob("master_bot_registry.backup*.json")))
    out.extend(sorted(root.glob("registry_backup_before_retrain*.json")))
    out.extend(sorted((root / "governance").glob("master_bot_registry.backup*.json")))
    out.extend(sorted((root / "governance").glob("registry_backup_before_retrain*.json")))
    out.extend(sorted((root / "governance" / "lifecycle").glob("master_bot_registry.repair_backup_*.json")))
    # de-dup while preserving order
    seen = set()
    uniq: list[Path] = []
    for p in out:
        if str(p) in seen:
            continue
        seen.add(str(p))
        uniq.append(p)
    return uniq


def _as_path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    return Path(text)


def _mtime_value(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _latest_artifact_for_bot(base: Path, bot_id: str, suffix: str) -> Path | None:
    if not bot_id or (not base.exists()):
        return None
    rows = [p for p in base.glob(f"{bot_id}_*{suffix}") if p.is_file()]
    if not rows:
        return None
    rows.sort(key=_mtime_value, reverse=True)
    return rows[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Model lifecycle hygiene checks, manifest, and backup maintenance.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--keep-backups", type=int, default=25)
    parser.add_argument("--apply-prune", action="store_true")
    parser.add_argument("--update-last-known-good", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument("--max-missing-active-artifacts", type=int, default=2)
    parser.add_argument("--repair-stale-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-repair", action=argparse.BooleanOptionalAction, default=False)
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
    repaired_rows = []

    for row in active_rows:
        bot_id = str(row.get("bot_id", "")).strip()
        model_path = _as_path(row.get("model_path"))
        log_path = _as_path(row.get("log_file"))
        model_ok = bool(model_path and model_path.exists())
        log_ok = bool(log_path and log_path.exists())

        original_model_path = str(model_path) if model_path is not None else ""
        original_log_path = str(log_path) if log_path is not None else ""
        row_repairs: dict[str, str] = {}

        if args.repair_stale_artifacts:
            if not model_ok:
                repaired_model = _latest_artifact_for_bot(PROJECT_ROOT / "models", bot_id, ".npz")
                if repaired_model is not None:
                    row["model_path"] = str(repaired_model)
                    model_path = repaired_model
                    model_ok = True
                    row_repairs["model_path"] = str(repaired_model)
            if not log_ok:
                repaired_log = _latest_artifact_for_bot(PROJECT_ROOT / "logs", bot_id, ".json")
                if repaired_log is not None:
                    row["log_file"] = str(repaired_log)
                    log_path = repaired_log
                    log_ok = True
                    row_repairs["log_file"] = str(repaired_log)

        if row_repairs:
            repaired_rows.append(
                {
                    "bot_id": bot_id,
                    "from_model_path": original_model_path,
                    "from_log_path": original_log_path,
                    "to_model_path": row_repairs.get("model_path", original_model_path),
                    "to_log_path": row_repairs.get("log_file", original_log_path),
                }
            )

        if not model_ok or not log_ok:
            missing_active.append(
                {
                    "bot_id": bot_id,
                    "model_exists": bool(model_ok),
                    "log_exists": bool(log_ok),
                    "model_path": str(model_path) if model_path is not None else "",
                    "log_path": str(log_path) if log_path is not None else "",
                }
            )

        manifest_rows.append(
            {
                "bot_id": bot_id,
                "model_path": str(model_path) if model_path is not None else "",
                "model_mtime_utc": _mtime_iso(model_path) if model_ok and model_path is not None else None,
                "log_path": str(log_path) if log_path is not None else "",
                "log_mtime_utc": _mtime_iso(log_path) if log_ok and log_path is not None else None,
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

    repair_backup_file = None
    repair_error = None
    registry_updated = False
    if args.repair_stale_artifacts and args.apply_repair and repaired_rows and isinstance(reg, dict):
        try:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            if registry_path.exists():
                backup_path = lifecycle_dir / f"master_bot_registry.repair_backup_{stamp}.json"
                backup_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
                repair_backup_file = str(backup_path)
            registry_path.write_text(json.dumps(reg, ensure_ascii=True, indent=2), encoding="utf-8")
            registry_updated = True
        except Exception as exc:
            repair_error = str(exc)

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
        "repair": {
            "enabled": bool(args.repair_stale_artifacts),
            "apply": bool(args.apply_repair),
            "fixed_count": len(repaired_rows),
            "registry_updated": bool(registry_updated),
            "backup_file": repair_backup_file,
            "error": repair_error,
            "examples": repaired_rows[:40],
        },
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
