#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "bot_logs_finder_sync_latest.json"
DEFAULT_ALIAS_PATH = Path.home() / "bot_logs"
DEFAULT_DESKTOP_PATH = Path.home() / "Desktop" / "Bot Logs"


def _remove_existing_path(path: Path) -> tuple[bool, str]:
    if not path.exists() and not path.is_symlink():
        return True, ""
    if path.is_symlink() or path.is_file():
        path.unlink()
        return True, ""
    return False, "existing_directory_blocks_shortcut"


def _ensure_symlink(path: Path, target: Path) -> dict[str, Any]:
    path = path.expanduser()
    target = target.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, error = _remove_existing_path(path)
    if not ok:
        return {
            "path": str(path),
            "target": str(target),
            "ok": False,
            "error": error,
        }
    try:
        path.symlink_to(target)
    except Exception as exc:
        return {
            "path": str(path),
            "target": str(target),
            "ok": False,
            "error": f"{type(exc).__name__}:{exc}",
        }
    return {
        "path": str(path),
        "target": str(target),
        "ok": True,
        "error": "",
    }


def sync_shortcuts(
    project_root: Path = PROJECT_ROOT,
    *,
    alias_path: Path = DEFAULT_ALIAS_PATH,
    desktop_path: Path = DEFAULT_DESKTOP_PATH,
    create_desktop_shortcut: bool = True,
) -> dict[str, Any]:
    logs_path = project_root / "logs"
    alias = _ensure_symlink(alias_path, logs_path)

    desktop: dict[str, Any] = {
        "path": str(desktop_path.expanduser()),
        "target": str(logs_path),
        "ok": False,
        "error": "",
        "enabled": bool(create_desktop_shortcut),
    }
    if create_desktop_shortcut:
        desktop = {
            **desktop,
            **_ensure_symlink(desktop_path, logs_path),
            "enabled": True,
        }

    ok = bool(alias.get("ok", False)) and (not create_desktop_shortcut or bool(desktop.get("ok", False)))
    return {
        "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": ok,
        "project_root": str(project_root),
        "logs_path": str(logs_path),
        "alias": alias,
        "desktop_shortcut": desktop,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Republish stable Finder-facing bot_logs shortcuts for the active runtime log root.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--alias-path", default=str(DEFAULT_ALIAS_PATH))
    parser.add_argument("--desktop-path", default=str(DEFAULT_DESKTOP_PATH))
    parser.add_argument(
        "--desktop-shortcut",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the Desktop Bot Logs shortcut in addition to the home-directory alias.",
    )
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = sync_shortcuts(
        Path(args.project_root).resolve(),
        alias_path=Path(args.alias_path),
        desktop_path=Path(args.desktop_path),
        create_desktop_shortcut=bool(args.desktop_shortcut),
    )
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "bot_logs_finder_sync "
            f"ok={int(bool(payload.get('ok', False)))} "
            f"alias_ok={int(bool(((payload.get('alias') or {}).get('ok', False))))} "
            f"desktop_ok={int(bool(((payload.get('desktop_shortcut') or {}).get('ok', False))))}"
        )
    return 0 if bool(payload.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
