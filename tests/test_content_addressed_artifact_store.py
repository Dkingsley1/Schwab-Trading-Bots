import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import content_addressed_artifact_store as src


def test_content_addressed_artifact_store_materializes_blob(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    artifact = project_root / "governance" / "health" / "runtime_gate_dashboard_latest.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"ok": true}', encoding="utf-8")

    payload = src.build_payload(
        project_root,
        store_root=project_root / "governance" / "content_store" / "sha256",
        tracked_paths=["governance/health/runtime_gate_dashboard_latest.json"],
        materialize=True,
    )

    assert payload["artifact_count"] == 1
    blob_path = Path(payload["artifacts"][0]["blob_path"])
    assert blob_path.exists()
    assert len(payload["manifest_hash"]) == 64


def test_content_addressed_artifact_store_skips_oversized_blob(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    artifact = project_root / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("ab", encoding="utf-8")
    store_root = project_root / "governance" / "content_store" / "sha256"

    payload = src.build_payload(
        project_root,
        store_root=store_root,
        tracked_paths=["exports/training/runtime_training_snapshot_latest.jsonl"],
        materialize=True,
        max_blob_bytes=1,
    )

    assert payload["artifact_count"] == 1
    assert payload["skipped_blob_count"] == 1
    assert payload["skipped_blob_bytes"] == 2
    row = payload["artifacts"][0]
    assert row["skipped_reason"] == "size_over_limit"
    assert row["sha256"] == ""
    assert row["blob_path"] == ""
    assert not any(path.is_file() for path in store_root.rglob("*"))


def test_content_addressed_artifact_store_gcs_unreferenced_blob(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    store_root = project_root / "governance" / "content_store" / "sha256"
    orphan = store_root / "aa" / "deadbeef"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_text("orphan", encoding="utf-8")
    old_epoch = 1_735_689_600
    os.utime(orphan, (old_epoch, old_epoch))

    payload = src.build_payload(
        project_root,
        store_root=store_root,
        tracked_paths=[],
        materialize=False,
        gc=True,
        gc_grace_days=0,
    )

    assert payload["gc"]["candidate_count"] == 1
    assert payload["gc"]["deleted_blob_count"] == 1
    assert payload["gc"]["deleted_bytes"] == len("orphan")
    assert not orphan.exists()
