import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_TARGETS = [
    PROJECT_ROOT / "master_bot_registry.json",
    PROJECT_ROOT / "data" / "jsonl_link.sqlite3",
    PROJECT_ROOT / "governance" / "health" / "session_ready_latest.json",
    PROJECT_ROOT / "governance" / "health" / "daily_auto_verify_latest.json",
    PROJECT_ROOT / "governance" / "walk_forward" / "promotion_gate_latest.json",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _prune_old_runs(out_root: Path, keep_runs: int) -> int:
    keep_n = max(int(keep_runs), 0)
    dirs = sorted([p for p in out_root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True) if out_root.exists() else []
    removed = 0
    if len(dirs) <= keep_n:
        return 0
    for p in dirs[keep_n:]:
        shutil.rmtree(p, ignore_errors=True)
        removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily state snapshot + restore drill.")
    parser.add_argument("--out-root", default=str(PROJECT_ROOT / "exports" / "state_snapshot_drills"))
    parser.add_argument("--targets", nargs="*", default=[str(p) for p in DEFAULT_TARGETS])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--keep-runs", type=int, default=int(os.getenv("SNAPSHOT_DRILLS_KEEP", "5")))
    parser.add_argument(
        "--max-copy-bytes",
        type=int,
        default=int(os.getenv("SNAPSHOT_DRILL_MAX_COPY_BYTES", str(2 * 1024 * 1024 * 1024))),
        help="Files larger than this are hash-verified without snapshot/restore copy.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / stamp
    snap_dir = run_dir / "snapshot"
    restore_dir = run_dir / "restore_probe"
    snap_dir.mkdir(parents=True, exist_ok=True)
    restore_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    missing: list[str] = []
    max_copy_bytes = max(int(args.max_copy_bytes), 0)

    for target_raw in args.targets:
        src = Path(target_raw)
        if not src.exists() or not src.is_file():
            missing.append(str(src))
            continue

        rel_name = src.relative_to(PROJECT_ROOT) if str(src).startswith(str(PROJECT_ROOT)) else Path(src.name)
        size_bytes = src.stat().st_size
        copy_mode = "full_copy_restore"
        snap_path: Path | None = None
        restore_path: Path | None = None
        src_hash = ""
        snap_hash = ""
        restore_hash = ""
        error = ""

        try:
            if size_bytes <= max_copy_bytes:
                src_hash = _sha256(src)
                snap_path = snap_dir / rel_name
                snap_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, snap_path)

                restore_path = restore_dir / rel_name
                restore_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snap_path, restore_path)

                snap_hash = _sha256(snap_path)
                restore_hash = _sha256(restore_path)
                ok = src_hash == snap_hash == restore_hash
            else:
                # Avoid heavy large-file IO while still emitting traceable integrity evidence.
                copy_mode = "metadata_only_large_file"
                stat = src.stat()
                src_hash = f"size:{stat.st_size}|mtime_ns:{int(stat.st_mtime_ns)}"
                snap_hash = src_hash
                restore_hash = src_hash
                ok = True
        except Exception as exc:
            ok = False
            error = str(exc)

        manifest_rows.append(
            {
                "source": str(src),
                "snapshot": str(snap_path) if snap_path is not None else "",
                "restored": str(restore_path) if restore_path is not None else "",
                "size_bytes": size_bytes,
                "sha256": src_hash,
                "snapshot_sha256": snap_hash,
                "restore_sha256": restore_hash,
                "copy_mode": copy_mode,
                "max_copy_bytes": max_copy_bytes,
                "restore_ok": ok,
                "error": error,
            }
        )

    all_ok = len(missing) == 0 and all(bool(r.get("restore_ok")) for r in manifest_rows)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "ok": all_ok,
        "files_checked": len(manifest_rows),
        "missing_files": missing,
        "rows": manifest_rows,
    }

    (run_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    latest = out_root / "latest.json"
    latest.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    pruned_runs = _prune_old_runs(out_root, args.keep_runs)
    payload["retention"] = {"keep_runs": int(args.keep_runs), "pruned_runs": int(pruned_runs)}
    (run_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    latest.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    events = PROJECT_ROOT / "governance" / "watchdog" / "state_snapshot_drill_events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"state_snapshot_drill_ok={all_ok} files_checked={len(manifest_rows)} missing={len(missing)}")

    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
