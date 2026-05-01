from __future__ import annotations

import os
import plistlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXTERNAL_MOUNT = "/Volumes/BOT_LOGS"
DEFAULT_EXTERNAL_PROJECT = "schwab_trading_bot"
DEFAULT_EXTERNAL_MOUNT_CANDIDATES: tuple[str, ...] = (
    DEFAULT_EXTERNAL_MOUNT,
)


@dataclass(frozen=True)
class ExternalStorageResolution:
    mount_root: Path
    external_root: Path
    configured_mount_root: Path
    configured_project_root: Path | None
    candidate_mount_roots: tuple[Path, ...]
    matched_mount_root: Path | None
    match_reason: str


@dataclass(frozen=True)
class ExternalVolumeInfo:
    device_identifier: str
    volume_name: str
    volume_uuid: str
    mount_point: str

    @property
    def is_mounted(self) -> bool:
        return bool(self.mount_point)


def external_project_dir() -> str:
    return os.getenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", DEFAULT_EXTERNAL_PROJECT).strip() or DEFAULT_EXTERNAL_PROJECT


def configured_external_mount_root() -> Path:
    return Path(os.getenv("BOT_LOGS_EXTERNAL_MOUNT", DEFAULT_EXTERNAL_MOUNT)).expanduser()


def configured_external_project_root() -> Path | None:
    configured = os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "").strip()
    if not configured:
        return None
    return Path(configured).expanduser()


def infer_mount_root_from_project_root(project_root: Path) -> Path:
    expanded = project_root.expanduser()
    parent = expanded.parent
    if parent != expanded and str(parent).startswith("/Volumes/"):
        return parent
    return configured_external_mount_root()


def external_mount_candidates() -> tuple[Path, ...]:
    raw = os.getenv("BOT_LOGS_EXTERNAL_MOUNT_CANDIDATES", "").strip()
    seen: set[str] = set()
    out: list[Path] = []

    def _add(candidate: str | Path | None) -> None:
        if candidate is None:
            return
        text = str(candidate).strip()
        if not text:
            return
        path = str(Path(text).expanduser())
        if path in seen:
            return
        seen.add(path)
        out.append(Path(path))

    _add(configured_external_mount_root())
    configured_root = configured_external_project_root()
    if configured_root is not None:
        _add(infer_mount_root_from_project_root(configured_root))

    if raw:
        for token in raw.split(","):
            _add(token)
    else:
        for candidate in DEFAULT_EXTERNAL_MOUNT_CANDIDATES:
            _add(candidate)

    return tuple(out)


def resolve_external_storage() -> ExternalStorageResolution:
    configured_mount_root = configured_external_mount_root()
    configured_root = configured_external_project_root()
    project_dir = external_project_dir()
    candidate_mount_roots = external_mount_candidates()

    if configured_root is not None and configured_root.exists():
        mount_root = infer_mount_root_from_project_root(configured_root)
        return ExternalStorageResolution(
            mount_root=mount_root,
            external_root=configured_root,
            configured_mount_root=configured_mount_root,
            configured_project_root=configured_root,
            candidate_mount_roots=candidate_mount_roots,
            matched_mount_root=mount_root,
            match_reason="configured_project_root_exists",
        )

    for mount_root in candidate_mount_roots:
        candidate_root = mount_root / project_dir
        if candidate_root.exists():
            return ExternalStorageResolution(
                mount_root=mount_root,
                external_root=candidate_root,
                configured_mount_root=configured_mount_root,
                configured_project_root=configured_root,
                candidate_mount_roots=candidate_mount_roots,
                matched_mount_root=mount_root,
                match_reason="candidate_project_root_exists",
            )

    if configured_root is not None:
        mount_root = infer_mount_root_from_project_root(configured_root)
        return ExternalStorageResolution(
            mount_root=mount_root,
            external_root=configured_root,
            configured_mount_root=configured_mount_root,
            configured_project_root=configured_root,
            candidate_mount_roots=candidate_mount_roots,
            matched_mount_root=None,
            match_reason="configured_project_root_missing",
        )

    mount_root = candidate_mount_roots[0] if candidate_mount_roots else configured_mount_root
    return ExternalStorageResolution(
        mount_root=mount_root,
        external_root=mount_root / project_dir,
        configured_mount_root=configured_mount_root,
        configured_project_root=None,
        candidate_mount_roots=candidate_mount_roots,
        matched_mount_root=None,
        match_reason="no_matching_candidate",
    )


def resolve_external_storage_paths() -> tuple[Path, Path]:
    resolution = resolve_external_storage()
    return resolution.mount_root, resolution.external_root


def _target_volume_name() -> str:
    configured = os.getenv("BOT_LOGS_EXTERNAL_VOLUME_NAME", "").strip()
    if configured:
        return configured
    return configured_external_mount_root().name or "BOT_LOGS"


def _target_volume_uuid() -> str:
    return os.getenv("BOT_LOGS_EXTERNAL_VOLUME_UUID", "").strip()


def _target_disk_identifier() -> str:
    return os.getenv("BOT_LOGS_EXTERNAL_DISK_IDENTIFIER", "").strip()


def list_external_volumes() -> tuple[ExternalVolumeInfo, ...]:
    try:
        proc = subprocess.run(
            ["/usr/sbin/diskutil", "list", "-plist", "external"],
            capture_output=True,
            text=False,
            check=False,
            timeout=20,
        )
    except Exception:
        return ()
    if proc.returncode != 0 or not proc.stdout:
        return ()
    try:
        payload = plistlib.loads(proc.stdout)
    except Exception:
        return ()
    rows = payload.get("AllDisksAndPartitions")
    if not isinstance(rows, list):
        return ()

    volumes: list[ExternalVolumeInfo] = []

    def _add(row: object) -> None:
        if not isinstance(row, dict):
            return
        identifier = str(row.get("DeviceIdentifier") or "").strip()
        if not identifier:
            return
        volume_name = str(row.get("VolumeName") or "").strip()
        volume_uuid = str(row.get("VolumeUUID") or row.get("DiskUUID") or "").strip()
        mount_point = str(row.get("MountPoint") or "").strip()
        if not (volume_name or volume_uuid):
            return
        volumes.append(
            ExternalVolumeInfo(
                device_identifier=identifier,
                volume_name=volume_name,
                volume_uuid=volume_uuid,
                mount_point=mount_point,
            )
        )

    for row in rows:
        _add(row)
        if isinstance(row, dict):
            for key in ("Partitions", "APFSVolumes"):
                children = row.get(key)
                if not isinstance(children, list):
                    continue
                for child in children:
                    _add(child)

    return tuple(volumes)


def find_target_external_volume() -> ExternalVolumeInfo | None:
    target_disk_identifier = _target_disk_identifier()
    target_volume_uuid = _target_volume_uuid()
    target_volume_name = _target_volume_name()
    best_match: ExternalVolumeInfo | None = None
    best_score = -1

    for volume in list_external_volumes():
        score = 0
        if target_disk_identifier and volume.device_identifier == target_disk_identifier:
            score += 100
        if target_volume_uuid and volume.volume_uuid.lower() == target_volume_uuid.lower():
            score += 80
        if target_volume_name and volume.volume_name == target_volume_name:
            score += 40
        if score > best_score and score > 0:
            best_match = volume
            best_score = score

    return best_match
