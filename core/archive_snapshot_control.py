from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for raw in entries:
        path = str(raw.get("path") or "").strip()
        digest = str(raw.get("sha256") or "").strip().lower()
        size_bytes = int(raw.get("size_bytes") or 0)
        row_count = int(raw.get("row_count") or 0)
        if not path or path in seen_paths:
            raise ValueError("manifest paths must be non-empty and unique")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"invalid sha256 for {path}")
        if size_bytes < 0 or row_count < 0:
            raise ValueError(f"negative manifest metric for {path}")
        seen_paths.add(path)
        normalized.append(
            {
                "path": path,
                "sha256": digest,
                "size_bytes": size_bytes,
                "row_count": row_count,
                "format": str(raw.get("format") or "jsonl"),
            }
        )
    normalized.sort(key=lambda row: row["path"])
    return {
        "entries": normalized,
        "entry_count": len(normalized),
        "total_bytes": sum(row["size_bytes"] for row in normalized),
        "total_rows": sum(row["row_count"] for row in normalized),
        "manifest_sha256": _sha256(normalized),
    }


def build_snapshot(
    manifest: Mapping[str, Any],
    *,
    schema_id: str,
    parent_snapshot_id: str | None = None,
    committed_at_utc: str,
) -> dict[str, Any]:
    if not schema_id or not committed_at_utc:
        raise ValueError("schema_id and committed_at_utc are required")
    semantic = {
        "schema_id": schema_id,
        "parent_snapshot_id": parent_snapshot_id,
        "committed_at_utc": committed_at_utc,
        "manifest_sha256": str(manifest.get("manifest_sha256") or ""),
        "entry_count": int(manifest.get("entry_count") or 0),
        "total_rows": int(manifest.get("total_rows") or 0),
    }
    if len(semantic["manifest_sha256"]) != 64:
        raise ValueError("snapshot requires a valid manifest digest")
    return {
        **semantic,
        "snapshot_id": f"snap-{_sha256(semantic)[:24]}",
        "commit_protocol": "compare_and_swap",
        "source_delete_authority": False,
    }


@dataclass
class SnapshotCatalog:
    snapshots: list[dict[str, Any]] = field(default_factory=list)

    @property
    def current_snapshot_id(self) -> str | None:
        return self.snapshots[-1]["snapshot_id"] if self.snapshots else None

    def commit(
        self, snapshot: Mapping[str, Any], *, expected_parent_snapshot_id: str | None
    ) -> dict[str, Any]:
        current = self.current_snapshot_id
        if current != expected_parent_snapshot_id:
            return {
                "committed": False,
                "reason": "compare_and_swap_conflict",
                "current_snapshot_id": current,
            }
        if snapshot.get("parent_snapshot_id") != expected_parent_snapshot_id:
            return {
                "committed": False,
                "reason": "snapshot_parent_mismatch",
                "current_snapshot_id": current,
            }
        snapshot_id = str(snapshot.get("snapshot_id") or "")
        if not snapshot_id or any(
            row["snapshot_id"] == snapshot_id for row in self.snapshots
        ):
            return {
                "committed": False,
                "reason": "snapshot_identity_invalid_or_duplicate",
                "current_snapshot_id": current,
            }
        self.snapshots.append(dict(snapshot))
        return {"committed": True, "snapshot_id": snapshot_id}


def plan_compaction(
    entries: Sequence[Mapping[str, Any]], *, target_bytes: int, max_inputs: int = 32
) -> dict[str, Any]:
    if target_bytes <= 0 or max_inputs < 2:
        raise ValueError("target_bytes must be positive and max_inputs at least two")
    rows = sorted(
        (dict(row) for row in entries), key=lambda row: str(row.get("path") or "")
    )
    groups: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    pending_bytes = 0

    def flush() -> None:
        nonlocal pending, pending_bytes
        if len(pending) >= 2:
            identity = [
                {"path": row["path"], "sha256": row["sha256"]} for row in pending
            ]
            digest = _sha256(identity)
            groups.append(
                {
                    "input_paths": [row["path"] for row in pending],
                    "input_sha256": [row["sha256"] for row in pending],
                    "total_bytes": pending_bytes,
                    "total_rows": sum(
                        int(row.get("row_count") or 0) for row in pending
                    ),
                    "planned_output": f"compacted/{digest[:24]}.jsonl.zst",
                    "verify_before_retire": True,
                    "source_delete_authority": False,
                }
            )
        pending = []
        pending_bytes = 0

    for row in rows:
        size = int(row.get("size_bytes") or 0)
        if pending and (
            pending_bytes + size > target_bytes or len(pending) >= max_inputs
        ):
            flush()
        pending.append(row)
        pending_bytes += size
    flush()
    return {
        "groups": groups,
        "group_count": len(groups),
        "input_count": len(rows),
        "atomic_snapshot_required": True,
        "human_readable_manifest_retained": True,
        "source_delete_authority": False,
    }
