#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "pycharm_active_bot_highlights_latest.json"
DEFAULT_SCOPE_PATH = PROJECT_ROOT / ".idea" / "scopes" / "Active_Bots.xml"
DEFAULT_FILE_COLORS_PATH = PROJECT_ROOT / ".idea" / "fileColors.xml"
DEFAULT_WORKSPACE_PATH = PROJECT_ROOT / ".idea" / "workspace.xml"
SCOPE_NAME = "Active Bots"
COLOR_NAME = "Blue"
PROJECT_VIEW_STYLE = "scope_background_color"
FOREGROUND_BLUE_SOURCE = "pycharm_vcs_modified_file_status"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _version_key(path: Path) -> tuple[int, str]:
    match = re.search(r"brain_refinery_v(\d+)", path.name)
    version = int(match.group(1)) if match else 10**9
    return version, path.name


def _scope_term(project_root: Path, path: Path) -> str:
    project_name = project_root.name
    rel = path.relative_to(project_root).as_posix()
    return f"file[{project_name}]:{rel}"


def _brain_refinery_family_term(project_root: Path) -> str:
    return f"file[{project_root.name}]:core/brain_refinery_v*.py"


def _scope_pattern(project_root: Path, active_paths: list[Path], inactive_paths: list[Path]) -> str:
    brain_refinery_active = [path for path in active_paths if path.name.startswith("brain_refinery_v")]
    other_active = [path for path in active_paths if path not in brain_refinery_active]
    brain_refinery_inactive = [path for path in inactive_paths if path.name.startswith("brain_refinery_v")]

    parts: list[str] = []
    if brain_refinery_active:
        family = _brain_refinery_family_term(project_root)
        exclusions = [f"!{_scope_term(project_root, path)}" for path in sorted(brain_refinery_inactive, key=_version_key)]
        parts.append("&&".join([family, *exclusions]) if exclusions else family)
    parts.extend(_scope_term(project_root, path) for path in sorted(other_active, key=_version_key))
    return "||".join(parts)


def active_core_bot_paths(project_root: Path) -> tuple[list[Path], list[str]]:
    registry = _load_json(project_root / "master_bot_registry.json")
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    active_ids = [
        str(row.get("bot_id") or "").strip()
        for row in rows
        if bool(row.get("active", False)) and str(row.get("bot_id") or "").strip()
    ]
    paths: list[Path] = []
    missing: list[str] = []
    for bot_id in active_ids:
        path = project_root / "core" / f"{bot_id}.py"
        if path.exists() and path.is_file():
            paths.append(path)
        else:
            missing.append(bot_id)
    return sorted(paths, key=_version_key), missing


def inactive_core_bot_paths(project_root: Path) -> list[Path]:
    registry = _load_json(project_root / "master_bot_registry.json")
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    paths: list[Path] = []
    for row in rows:
        bot_id = str(row.get("bot_id") or "").strip()
        if not bot_id or bool(row.get("active", False)):
            continue
        path = project_root / "core" / f"{bot_id}.py"
        if path.exists() and path.is_file():
            paths.append(path)
    return sorted(paths, key=_version_key)


def _write_active_scope(scope_path: Path, *, pattern: str) -> None:
    scope_path.parent.mkdir(parents=True, exist_ok=True)
    component = ET.Element("component", {"name": "DependencyValidationManager"})
    ET.SubElement(component, "scope", {"name": SCOPE_NAME, "pattern": pattern})
    tree = ET.ElementTree(component)
    ET.indent(tree, space="  ")
    tree.write(scope_path, encoding="UTF-8", xml_declaration=True)


def _ensure_file_color(root: ET.Element, *, include_enable_options: bool) -> None:
    component = None
    for child in root.findall("component"):
        if child.get("name") == "FileColors":
            component = child
            break
    if component is None:
        component = ET.SubElement(root, "component", {"name": "FileColors"})

    if include_enable_options:
        for option_name, option_value in (
            ("enabled", "true"),
            ("enabledForProjectView", "true"),
            ("enabledForTabs", "true"),
            ("fileColorsEnabled", "true"),
            ("fileColorsEnabledForProjectView", "true"),
            ("fileColorsEnabledForTabs", "true"),
        ):
            for child in list(component.findall("option")):
                if child.get("name") == option_name:
                    component.remove(child)
            ET.SubElement(component, "option", {"name": option_name, "value": option_value})

    for child in list(component.findall("fileColor")):
        if child.get("scope") == SCOPE_NAME:
            component.remove(child)
    ET.SubElement(component, "fileColor", {"scope": SCOPE_NAME, "color": COLOR_NAME})


def _load_project_xml(path: Path) -> tuple[ET.ElementTree, ET.Element]:
    if path.exists():
        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except Exception:
            root = ET.Element("project", {"version": "4"})
            tree = ET.ElementTree(root)
    else:
        root = ET.Element("project", {"version": "4"})
        tree = ET.ElementTree(root)

    if root.tag != "project":
        root = ET.Element("project", {"version": "4"})
        tree = ET.ElementTree(root)
    return tree, root


def _write_file_color(file_colors_path: Path) -> None:
    file_colors_path.parent.mkdir(parents=True, exist_ok=True)
    tree, root = _load_project_xml(file_colors_path)
    _ensure_file_color(root, include_enable_options=True)
    ET.indent(tree, space="  ")
    tree.write(file_colors_path, encoding="UTF-8", xml_declaration=True)


def _write_workspace_file_color(workspace_path: Path) -> None:
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    tree, root = _load_project_xml(workspace_path)
    _ensure_file_color(root, include_enable_options=False)
    ET.indent(tree, space="  ")
    tree.write(workspace_path, encoding="UTF-8", xml_declaration=True)


def build_payload(project_root: Path = PROJECT_ROOT, *, apply: bool = False) -> dict[str, Any]:
    project_root = Path(project_root)
    active_paths, missing_ids = active_core_bot_paths(project_root)
    inactive_paths = inactive_core_bot_paths(project_root)
    pattern = _scope_pattern(project_root, active_paths, inactive_paths)
    scope_path = project_root / ".idea" / "scopes" / "Active_Bots.xml"
    file_colors_path = project_root / ".idea" / "fileColors.xml"
    workspace_path = project_root / ".idea" / "workspace.xml"
    if apply:
        _write_active_scope(scope_path, pattern=pattern)
        _write_file_color(file_colors_path)
        _write_workspace_file_color(workspace_path)

    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": bool(active_paths),
        "overall_status": "ready" if active_paths else "blocked",
        "apply": bool(apply),
        "scope_name": SCOPE_NAME,
        "file_color": COLOR_NAME,
        "scope_path": str(scope_path),
        "file_colors_path": str(file_colors_path),
        "workspace_path": str(workspace_path),
        "active_core_bot_file_count": len(active_paths),
        "inactive_core_bot_file_count": len(inactive_paths),
        "missing_active_core_file_count": len(missing_ids),
        "scope_strategy": "brain_refinery_family_with_inactive_exclusions",
        "scope_pattern_bytes": len(pattern.encode("utf-8")),
        "project_view_style": PROJECT_VIEW_STYLE,
        "foreground_blue_source": FOREGROUND_BLUE_SOURCE,
        "foreground_blue_supported_without_dirtying_files": False,
        "vcs_blue_requires_file_modification": True,
        "vcs_blue_dirty_file_count_to_match": len(active_paths),
        "missing_active_bot_ids_sample": missing_ids[:25],
        "active_core_bot_files_sample": [str(path.relative_to(project_root)) for path in active_paths[:25]],
        "inactive_core_bot_files_sample": [str(path.relative_to(project_root)) for path in inactive_paths[:25]],
        "pycharm_note": "The command writes both shared file colors and PyCharm workspace file colors. PyCharm's bright blue filename text is reserved for VCS-modified files, so active bots use the durable scope background instead of dirtying files to imitate Git status.",
        "contract": {
            "does_not_trade": True,
            "does_not_start_bots": True,
            "writes": [
                ".idea/scopes/Active_Bots.xml",
                ".idea/fileColors.xml",
                ".idea/workspace.xml",
                "governance/health/pycharm_active_bot_highlights_latest.json",
            ],
        },
    }
    write_payload(project_root / DEFAULT_OUT_PATH.relative_to(PROJECT_ROOT), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Write PyCharm Project-view blue highlights for active core bot files.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build_payload(Path(args.project_root).expanduser().resolve(), apply=bool(args.apply))
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "pycharm_active_bot_highlights "
            f"status={payload.get('overall_status', '')} "
            f"active_core_files={payload.get('active_core_bot_file_count', 0)} "
            f"color={payload.get('file_color', '')}"
        )
    return 0 if bool(payload.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
