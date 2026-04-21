import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import storage_split_brain_reconciler as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_storage_split_brain_reconciler_classifies_hash_match_ready(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "external"
    local_root = project_root / "local_fallback_storage"
    (external_root / "logs").mkdir(parents=True, exist_ok=True)
    (local_root / "logs").mkdir(parents=True, exist_ok=True)
    (external_root / "logs" / "state.json").write_text("same", encoding="utf-8")
    (local_root / "logs" / "state.json").write_text("same", encoding="utf-8")
    (external_root / "logs" / "state.json.local_fallback").write_text("same", encoding="utf-8")
    _write_json(project_root / "governance" / "health" / "storage_failback_sync_latest.json", {"split_brain_conflicts": 0})
    _write_json(project_root / "governance" / "health" / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})

    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", str(external_root))
    monkeypatch.setenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(local_root))

    payload = src.build_payload(project_root, full_scan=True)

    assert payload["summary"]["hash_match_ready"] == 1
    assert payload["summary"]["unresolved_conflicts"] == 0
    assert payload["conflicts"][0]["classification"] in {"ready_to_prune_local", "duplicate_conflict_copy"}


def test_storage_split_brain_reconciler_fast_path_skips_full_tree_scan_when_manifest_is_clean(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "external"
    local_root = project_root / "local_fallback_storage"
    external_root.mkdir(parents=True, exist_ok=True)
    local_root.mkdir(parents=True, exist_ok=True)
    _write_json(project_root / "governance" / "health" / "storage_failback_sync_latest.json", {"split_brain_conflicts": 0})
    _write_json(project_root / "governance" / "health" / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})

    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", str(external_root))
    monkeypatch.setenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(local_root))
    monkeypatch.setattr(src, "_iter_conflict_files", lambda _external_root: (_ for _ in ()).throw(AssertionError("fast path should not scan")))

    payload = src.build_payload(project_root)

    assert payload["scan_mode"] == "manifest_fast_path"
    assert payload["summary"]["conflict_files"] == 0
    assert payload["summary"]["reported_split_brain_conflicts"] == 0
