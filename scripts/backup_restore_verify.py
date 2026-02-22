import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup + restore verification drill with RTO/RPO metrics.")
    parser.add_argument("--source", default=str(PROJECT_ROOT / "governance" / "health" / "session_ready_latest.json"))
    parser.add_argument("--backup-dir", default=str(PROJECT_ROOT / "exports" / "backup_drills"))
    args = parser.parse_args()

    start = time.time()
    src = Path(args.source)
    out_dir = Path(args.backup_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(src),
        "ok": False,
        "rto_seconds": None,
        "rpo_seconds": None,
    }

    if not src.exists():
        payload["error"] = "source_missing"
    else:
        bak = out_dir / f"drill_{stamp}.bak.json"
        restored = out_dir / f"drill_{stamp}.restored.json"
        shutil.copy2(src, bak)
        shutil.copy2(bak, restored)
        src_bytes = src.read_bytes()
        restored_bytes = restored.read_bytes()
        payload.update(
            {
                "backup": str(bak),
                "restored": str(restored),
                "ok": src_bytes == restored_bytes,
                "size_bytes": len(src_bytes),
                "rpo_seconds": round(max(time.time() - src.stat().st_mtime, 0.0), 3),
            }
        )

    payload["rto_seconds"] = round(max(time.time() - start, 0.0), 3)

    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    events = PROJECT_ROOT / "governance" / "watchdog" / "backup_restore_events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    print(json.dumps(payload, ensure_ascii=True))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
