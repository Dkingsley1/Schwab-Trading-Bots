import argparse
import os
import re
import shutil
import time
from pathlib import Path

MODEL_RE = re.compile(r"^(?P<bot>.+)_\d{8}_\d{6}(?:_quantized)?\.npz$")


def _bot_id_from_name(name: str) -> str:
    m = MODEL_RE.match(name)
    if m:
        return m.group("bot")
    stem = name[:-4] if name.endswith(".npz") else name
    return stem.rsplit("_", 2)[0] if "_" in stem else stem


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive old model artifacts, keeping latest N per bot.")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--archive-dir", default=None)
    parser.add_argument("--keep-per-bot", type=int, default=int(os.getenv("MODEL_ARCHIVE_KEEP_PER_BOT", "8")))
    parser.add_argument("--min-age-hours", type=float, default=float(os.getenv("MODEL_ARCHIVE_MIN_AGE_HOURS", "24")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    models_dir = Path(args.models_dir) if args.models_dir else (project_root / "models")
    archive_dir = Path(args.archive_dir) if args.archive_dir else (models_dir / "archive")

    if not models_dir.exists():
        print(f"Models dir missing: {models_dir}")
        return 0

    files = []
    for p in models_dir.glob("*.npz"):
        if p.is_file():
            files.append(p)

    groups = {}
    for p in files:
        bot = _bot_id_from_name(p.name)
        groups.setdefault(bot, []).append(p)

    now = time.time()
    moved = 0
    kept = 0

    for bot, items in groups.items():
        items.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        keep_n = max(args.keep_per_bot, 1)
        keep = items[:keep_n]
        drop = items[keep_n:]
        kept += len(keep)

        for p in drop:
            age_h = (now - p.stat().st_mtime) / 3600.0
            if age_h < args.min_age_hours:
                kept += 1
                continue

            target = archive_dir / bot / p.name
            if args.dry_run:
                print(f"DRYRUN move {p} -> {target}")
                moved += 1
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(target))
            moved += 1

    print(
        f"Model archive complete | bots={len(groups)} kept={kept} moved={moved} "
        f"keep_per_bot={max(args.keep_per_bot, 1)} min_age_hours={args.min_age_hours:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
