import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SourceDef:
    name: str
    path: Path
    category: str
    kind: str  # dir|file


def _human_bytes(n: int) -> str:
    size = float(max(n, 0))
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size >= 1024.0 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.1f}{units[i]}"


def _dir_stats(path: Path) -> tuple[int, int, str | None]:
    if not path.exists():
        return 0, 0, None

    files = 0
    total = 0
    latest_mtime = 0.0

    if path.is_file():
        st = path.stat()
        return 1, int(st.st_size), datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()

    for root, _, names in os.walk(path):
        for n in names:
            fp = Path(root) / n
            try:
                st = fp.stat()
            except OSError:
                continue
            files += 1
            total += int(st.st_size)
            latest_mtime = max(latest_mtime, float(st.st_mtime))

    latest_iso = datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat() if latest_mtime > 0 else None
    return files, total, latest_iso


def _make_link(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a single organized data center folder of runtime logs.")
    parser.add_argument("--out-root", default=str(PROJECT_ROOT / "exports" / "data_center"))
    parser.add_argument("--stamp", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    bundle_dir = out_root / args.stamp
    links_dir = bundle_dir / "links"
    docs_dir = bundle_dir / "docs"

    sources = [
        SourceDef("decision_explanations", PROJECT_ROOT / "decision_explanations", "runtime", "dir"),
        SourceDef("decisions", PROJECT_ROOT / "decisions", "runtime", "dir"),
        SourceDef("governance", PROJECT_ROOT / "governance", "runtime", "dir"),
        SourceDef("watchdog_events", PROJECT_ROOT / "governance" / "watchdog", "watchdog", "dir"),
        SourceDef("heartbeat_health", PROJECT_ROOT / "governance" / "health", "watchdog", "dir"),
        SourceDef("logs", PROJECT_ROOT / "logs", "runtime", "dir"),
        SourceDef("sql_db", PROJECT_ROOT / "data" / "jsonl_link.sqlite3", "sql", "file"),
        SourceDef("sql_reports", PROJECT_ROOT / "exports" / "sql_reports", "sql", "dir"),
        SourceDef("sql_reports_latest", PROJECT_ROOT / "exports" / "sql_reports" / "latest", "sql", "dir"),
        SourceDef("trade_history", PROJECT_ROOT / "data" / "trade_history", "data", "dir"),
        SourceDef("bot_stack_status", PROJECT_ROOT / "exports" / "bot_stack_status", "runtime", "dir"),
        SourceDef("bot_stack_latest_json", PROJECT_ROOT / "exports" / "bot_stack_status" / "latest.json", "runtime", "file"),
        SourceDef("bot_stack_latest_md", PROJECT_ROOT / "exports" / "bot_stack_status" / "latest.md", "runtime", "file"),
    ]

    bundle_dir.mkdir(parents=True, exist_ok=True)
    links_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "bundle_dir": str(bundle_dir),
        "sources": [],
    }

    for s in sources:
        exists = s.path.exists()
        files, bytes_total, latest = _dir_stats(s.path) if exists else (0, 0, None)
        link_path = links_dir / f"{s.category}__{s.name}"
        if exists:
            _make_link(link_path, s.path)

        manifest["sources"].append(
            {
                "name": s.name,
                "category": s.category,
                "kind": s.kind,
                "path": str(s.path),
                "exists": exists,
                "link": str(link_path) if exists else None,
                "file_count": files,
                "bytes": bytes_total,
                "bytes_human": _human_bytes(bytes_total),
                "latest_mtime_utc": latest,
            }
        )

    manifest_path = docs_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    readme_lines = [
        "# Data Center",
        "",
        f"Generated (UTC): {manifest['generated_utc']}",
        f"Project Root: `{PROJECT_ROOT}`",
        f"Bundle: `{bundle_dir}`",
        "",
        "## What This Is",
        "A single organized hub of symlinked log/data sources plus inventory metadata.",
        "",
        "## Quick Start",
        f"- Open links: `{links_dir}`",
        f"- Open manifest: `{manifest_path}`",
        "",
        "## Sources",
    ]

    for row in manifest["sources"]:
        status = "OK" if row["exists"] else "MISSING"
        readme_lines.append(
            f"- [{status}] `{row['category']}/{row['name']}` -> `{row['path']}` "
            f"(files={row['file_count']}, size={row['bytes_human']}, latest={row['latest_mtime_utc']})"
        )

    (docs_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    latest_link = out_root / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(bundle_dir)

    print(f"Built data center: {bundle_dir}")
    print(f"Latest link: {latest_link}")
    print(f"Links dir: {links_dir}")
    print(f"Docs: {docs_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
