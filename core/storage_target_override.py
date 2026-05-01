from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORAGE_TARGET_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.storage_target_override"


def build_storage_target_override_text(
    *,
    mount_root: str,
    project_dir: str = "schwab_trading_bot",
    mount_candidates: tuple[str, ...] | list[str] | None = None,
    volume_name: str = "",
    volume_uuid: str = "",
    disk_identifier: str = "",
) -> str:
    mount_text = str(mount_root or "").strip() or "/Volumes/BOT_LOGS"
    project_dir_text = str(project_dir or "").strip() or "schwab_trading_bot"
    candidate_values = [str(item or "").strip() for item in (mount_candidates or (mount_text,))]
    candidate_values = [item for item in candidate_values if item]
    if not candidate_values:
        candidate_values = [mount_text]
    volume_name_text = str(volume_name or "").strip() or Path(mount_text).name or "BOT_LOGS"
    lines = [
        "# Auto-managed by storage_disaster_recovery.py",
        f"BOT_LOGS_EXTERNAL_MOUNT={mount_text}",
        f"BOT_LOGS_EXTERNAL_MOUNT_CANDIDATES={','.join(candidate_values)}",
        f"BOT_LOGS_EXTERNAL_VOLUME_NAME={volume_name_text}",
        f"BOT_LOGS_EXTERNAL_PROJECT_DIR={project_dir_text}",
        f"BOT_LOGS_EXTERNAL_PROJECT_ROOT={mount_text.rstrip('/')}/{project_dir_text}",
    ]
    volume_uuid_text = str(volume_uuid or "").strip()
    if volume_uuid_text:
        lines.append(f"BOT_LOGS_EXTERNAL_VOLUME_UUID={volume_uuid_text}")
    disk_identifier_text = str(disk_identifier or "").strip()
    if disk_identifier_text:
        lines.append(f"BOT_LOGS_EXTERNAL_DISK_IDENTIFIER={disk_identifier_text}")
    return "\n".join(lines) + "\n"


def write_storage_target_override(
    *,
    mount_root: str,
    project_dir: str = "schwab_trading_bot",
    mount_candidates: tuple[str, ...] | list[str] | None = None,
    volume_name: str = "",
    volume_uuid: str = "",
    disk_identifier: str = "",
    override_path: Path = DEFAULT_STORAGE_TARGET_OVERRIDE_PATH,
) -> dict[str, object]:
    override_path.parent.mkdir(parents=True, exist_ok=True)
    text = build_storage_target_override_text(
        mount_root=mount_root,
        project_dir=project_dir,
        mount_candidates=mount_candidates,
        volume_name=volume_name,
        volume_uuid=volume_uuid,
        disk_identifier=disk_identifier,
    )
    current = ""
    if override_path.exists():
        try:
            current = override_path.read_text(encoding="utf-8")
        except Exception:
            current = ""
    changed = current != text
    if changed:
        override_path.write_text(text, encoding="utf-8")
    return {
        "path": str(override_path),
        "changed": bool(changed),
        "mount_root": str(mount_root or "").strip(),
        "project_dir": str(project_dir or "").strip() or "schwab_trading_bot",
        "volume_name": str(volume_name or "").strip(),
        "volume_uuid": str(volume_uuid or "").strip(),
        "disk_identifier": str(disk_identifier or "").strip(),
    }
