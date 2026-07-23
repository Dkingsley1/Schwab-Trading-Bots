import json
import sys
from types import SimpleNamespace
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


def test_storage_split_brain_reconciler_counts_router_conflicts_for_failback(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "external"
    local_root = project_root / "local_fallback_storage"
    (external_root / "logs").mkdir(parents=True, exist_ok=True)
    (local_root / "logs").mkdir(parents=True, exist_ok=True)
    (external_root / "logs" / "watchdog.log").write_text("external\n", encoding="utf-8")
    (local_root / "logs" / "watchdog.log").write_text("local\n", encoding="utf-8")
    _write_json(project_root / "governance" / "health" / "storage_failback_sync_latest.json", {"split_brain_conflicts": 0})
    _write_json(project_root / "governance" / "health" / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})

    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", str(external_root))
    monkeypatch.setenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(local_root))

    payload = src.build_payload(project_root)

    assert payload["summary"]["router_conflicts"] == 1
    assert payload["summary"]["reported_split_brain_conflicts"] == 1
    assert payload["summary"]["force_failback_eligible"] is False


def test_storage_split_brain_reconciler_repairs_log_router_conflicts(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "external"
    local_root = project_root / "local_fallback_storage"
    external_log = external_root / "logs" / "watchdog.log"
    local_log = local_root / "logs" / "watchdog.log"
    external_gz = external_root / "decision_explanations" / "paper" / "latest_decisions.log.gz"
    local_gz = local_root / "decision_explanations" / "paper" / "latest_decisions.log.gz"
    external_log.parent.mkdir(parents=True, exist_ok=True)
    local_log.parent.mkdir(parents=True, exist_ok=True)
    external_gz.parent.mkdir(parents=True, exist_ok=True)
    local_gz.parent.mkdir(parents=True, exist_ok=True)
    external_log.write_text("external\n", encoding="utf-8")
    local_log.write_text("local\n", encoding="utf-8")
    external_gz.write_bytes(b"external-gz")
    local_gz.write_bytes(b"local-gz")

    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", str(external_root))
    monkeypatch.setenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(local_root))

    result = src._repair_router_log_conflicts(project_root, archive_base=project_root / "archive", apply=True)

    assert result["candidate_count"] == 2
    assert result["allowed_count"] == 2
    assert result["repaired_count"] == 2
    assert not local_log.exists()
    assert not local_gz.exists()
    assert "local" in external_log.read_text(encoding="utf-8")
    archived = sorted((project_root / "archive" / "local_router_conflicts").rglob("*"))
    assert any(path.name == "watchdog.log" for path in archived)
    assert any(path.name == "latest_decisions.log.gz" for path in archived)


def test_storage_split_brain_reconciler_merges_decision_jsonl_with_dedupe(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "external"
    local_root = project_root / "local_fallback_storage"
    external_decisions = external_root / "decisions" / "paper" / "trade_decisions_20260630.jsonl"
    local_decisions = local_root / "decisions" / "paper" / "trade_decisions_20260630.jsonl"
    external_decisions.parent.mkdir(parents=True, exist_ok=True)
    local_decisions.parent.mkdir(parents=True, exist_ok=True)
    external_decisions.write_text('{"id": 2}\n', encoding="utf-8")
    local_decisions.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")

    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", str(external_root))
    monkeypatch.setenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(local_root))

    result = src._repair_router_log_conflicts(project_root, archive_base=project_root / "archive", apply=True)

    assert result["candidate_count"] == 1
    assert result["allowed_count"] == 1
    assert result["repaired_count"] == 1
    assert not local_decisions.exists()
    merged_lines = external_decisions.read_text(encoding="utf-8").splitlines()
    assert merged_lines.count('{"id": 1}') == 1
    assert merged_lines.count('{"id": 2}') == 1


def test_storage_split_brain_reconciler_records_bounded_failback_timeout(tmp_path: Path, monkeypatch) -> None:
    def _timeout(_project_root: Path):
        raise src.FailbackTimeout("storage route failback timed out after 1s")

    monkeypatch.setattr(src.storage_router, "route_runtime_storage", _timeout)

    payload = src._force_failback(tmp_path, timeout_sec=1, fallback_conflicts=0)

    assert payload["attempted"] is True
    assert payload["ok"] is False
    assert payload["timed_out"] is True
    assert payload["timeout_sec"] == 1
    assert "timed out" in payload["error"]


def test_storage_split_brain_reconciler_records_successful_bounded_failback(tmp_path: Path, monkeypatch) -> None:
    routing = SimpleNamespace(mode="external", active_root=tmp_path / "external", split_brain_conflicts=0)
    monkeypatch.setattr(src.storage_router, "route_runtime_storage", lambda _project_root: routing)

    payload = src._force_failback(tmp_path, timeout_sec=5, fallback_conflicts=0)

    assert payload["attempted"] is True
    assert payload["ok"] is True
    assert payload["timed_out"] is False
    assert payload["mode"] == "external"
