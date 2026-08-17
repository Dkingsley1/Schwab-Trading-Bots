#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path("/Users/dankingsley/PycharmProjects/schwab_trading_bot")
DEFAULT_SOURCE_ROOT = Path("/Users/dankingsley/Documents/New project")
DEFAULT_MERGED_ROOT = PROJECT_ROOT / "docs" / "documents_merge" / "new_project_schwab_workspace"
DEFAULT_ARCHIVE_PREFIX = "_schwab_prepurge_"

WORKSPACE_ITEMS = (
    "balloon_fix",
    "commands_alpha",
    "commands_most_used",
    "commands_remove_macro",
    "commands_reports_pdf",
    "commands_reports_timeline",
    "fx_patch",
    "stage",
    "stage_commands_md",
    "stage_dividend_paper",
    "stage_export_alias",
    "stage_external_csv",
    "stage_futures_tail",
    "stage_fx_context_launchd",
    "stage_fx_gate",
    "stage_fx_quotes",
    "stage_fx_session",
    "stage_fx_tail",
    "stage_fx_twelve",
    "stage_paper_report_all_sleeves",
    "stage_reports",
    "stage_retrain_note",
    "tmp_final3",
    "tmp_fix",
    "tmp_fix2",
    "tmp_fix_scripts",
    "tmp_memory_guard_patch",
    "tmp_reconcile_patch",
    "tmp_repo_patch",
    "tmp_resource_guard_patch",
    "tmp_storage_patch",
)


@dataclass
class ItemStatus:
    name: str
    source_path: str
    merged_path: str
    source_exists: bool
    merged_exists: bool
    eligible_for_archive: bool


def _item_statuses(source_root: Path, merged_root: Path) -> list[ItemStatus]:
    rows: list[ItemStatus] = []
    for name in WORKSPACE_ITEMS:
        source_path = source_root / name
        merged_path = merged_root / name
        source_exists = source_path.exists()
        merged_exists = merged_path.exists()
        rows.append(
            ItemStatus(
                name=name,
                source_path=str(source_path),
                merged_path=str(merged_path),
                source_exists=source_exists,
                merged_exists=merged_exists,
                eligible_for_archive=source_exists and merged_exists,
            )
        )
    return rows


def _archive_dirs(source_root: Path, prefix: str) -> list[Path]:
    return sorted(
        [
            path
            for path in source_root.iterdir()
            if path.is_dir() and path.name.startswith(prefix)
        ],
        key=lambda path: path.name,
    )


def _top_level_unrelated(source_root: Path, prefix: str) -> list[str]:
    excluded = {".pytest_cache", "__pycache__"}
    workspace_names = set(WORKSPACE_ITEMS)
    unrelated: list[str] = []
    for path in sorted(source_root.iterdir(), key=lambda entry: entry.name.lower()):
        if path.name in excluded or path.name.startswith(prefix):
            continue
        if path.name not in workspace_names:
            unrelated.append(path.name)
    return unrelated


def _archive_target(source_root: Path, prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return source_root / f"{prefix}{stamp}"


def _validate_source_root(source_root: Path) -> None:
    if not source_root.exists():
        raise SystemExit(f"missing_source_root: {source_root}")
    if not source_root.is_dir():
        raise SystemExit(f"source_root_not_directory: {source_root}")


def _validate_archive_dir(source_root: Path, archive_dir: Path, prefix: str) -> None:
    resolved_source = source_root.resolve()
    resolved_archive = archive_dir.resolve()
    if resolved_archive.parent != resolved_source:
        raise SystemExit(f"archive_outside_source_root: {archive_dir}")
    if not archive_dir.name.startswith(prefix):
        raise SystemExit(f"archive_name_missing_prefix: {archive_dir}")


def _print_payload(payload: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return

    print(payload["action"])
    for key, value in payload.items():
        if key == "action":
            continue
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  - {item}")
        else:
            print(f"{key}: {value}")


def _rows_to_dicts(rows: Iterable[ItemStatus]) -> list[dict]:
    return [asdict(row) for row in rows]


def plan_action(source_root: Path, merged_root: Path, prefix: str) -> dict:
    rows = _item_statuses(source_root, merged_root)
    archives = _archive_dirs(source_root, prefix)
    return {
        "action": "plan",
        "source_root": str(source_root),
        "merged_root": str(merged_root),
        "eligible_for_archive": [row.name for row in rows if row.eligible_for_archive],
        "missing_from_source": [row.name for row in rows if not row.source_exists],
        "missing_from_merged_archive": [row.name for row in rows if row.source_exists and not row.merged_exists],
        "existing_archives": [str(path) for path in archives],
        "unrelated_top_level_entries": _top_level_unrelated(source_root, prefix),
        "items": _rows_to_dicts(rows),
    }


def archive_action(source_root: Path, merged_root: Path, prefix: str, execute: bool) -> dict:
    rows = _item_statuses(source_root, merged_root)
    eligible = [row for row in rows if row.eligible_for_archive]
    skipped_missing_in_merged = [row.name for row in rows if row.source_exists and not row.merged_exists]
    archive_dir = _archive_target(source_root, prefix)

    if execute:
        archive_dir.mkdir(parents=True, exist_ok=False)
        for row in eligible:
            shutil.move(str(source_root / row.name), str(archive_dir / row.name))

    return {
        "action": "archive",
        "execute": execute,
        "source_root": str(source_root),
        "merged_root": str(merged_root),
        "archive_dir": str(archive_dir),
        "moved_items": [row.name for row in eligible],
        "skipped_missing_in_merged_archive": skipped_missing_in_merged,
        "missing_from_source": [row.name for row in rows if not row.source_exists],
    }


def purge_action(source_root: Path, prefix: str, execute: bool, latest_only: bool, archive_dir_arg: str) -> dict:
    if archive_dir_arg:
        archives = [Path(archive_dir_arg).expanduser()]
    else:
        archives = _archive_dirs(source_root, prefix)

    if latest_only and archives:
        archives = [sorted(archives, key=lambda path: path.name)[-1]]

    for archive_dir in archives:
        _validate_archive_dir(source_root, archive_dir, prefix)

    existing_archives = [path for path in archives if path.exists()]

    if execute:
        for archive_dir in existing_archives:
            shutil.rmtree(archive_dir)

    return {
        "action": "purge",
        "execute": execute,
        "source_root": str(source_root),
        "purged_archives": [str(path) for path in existing_archives],
        "missing_archives": [str(path) for path in archives if not path.exists()],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview, archive, and purge the old Schwab-specific folders in Documents/New project."
    )
    parser.add_argument("action", choices=("plan", "archive", "purge"))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--merged-root", default=str(DEFAULT_MERGED_ROOT))
    parser.add_argument("--archive-prefix", default=DEFAULT_ARCHIVE_PREFIX)
    parser.add_argument("--archive-dir", default="")
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser()
    merged_root = Path(args.merged_root).expanduser()
    prefix = str(args.archive_prefix)

    _validate_source_root(source_root)

    if args.action == "plan":
        payload = plan_action(source_root, merged_root, prefix)
    elif args.action == "archive":
        payload = archive_action(source_root, merged_root, prefix, args.execute)
    else:
        payload = purge_action(source_root, prefix, args.execute, args.latest_only, args.archive_dir)

    _print_payload(payload, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
