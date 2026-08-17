import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import runtime_throttle_control, stale_artifact_reaper_bot, stale_artifact_sweeper_bot


def test_stale_artifact_sweeper_build_payload_summarizes_stage_results(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path) -> dict:
        payload = {
            "stale_stage": {
                "enabled": True,
                "stage_only": True,
                "root": str(project_root / "data" / "stale_stage"),
                "manifest_path": str(project_root / "data" / "stale_stage" / "stale_manifest.jsonl"),
                "sections": ["all"],
                "candidate_files": 4,
                "candidate_bytes": 4096,
                "staged_files": 3,
                "staged_bytes": 3072,
                "delete_errors": 0,
                "staged_by_label": {
                    "logs": {"staged_files": 2, "staged_bytes": 2048, "candidate_files": 2},
                    "governance_health": {"staged_files": 1, "staged_bytes": 1024, "candidate_files": 2},
                },
            }
        }
        return {"cmd": cmd, "rc": 0, "duration_ms": 12.5, "payload": payload, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(stale_artifact_sweeper_bot, "_run_json_command", _fake_run)

    payload = stale_artifact_sweeper_bot.build_payload(
        project_root,
        stale_stage_sections="all",
        stale_stage_root=project_root / "data" / "stale_stage",
        stale_stage_manifest="",
    )

    assert payload["ok"] is True
    assert payload["summary"]["staged_files"] == 3
    assert payload["top_labels"][0]["label"] == "logs"
    assert payload["stale_stage"]["stage_only"] is True


def test_stale_artifact_reaper_build_payload_purges_old_staged_files(tmp_path) -> None:
    project_root = tmp_path / "project"
    stale_root = project_root / "data" / "stale_stage"
    stale_file = stale_root / "logs" / "project" / "logs" / "old.log"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("old", encoding="utf-8")
    manifest_path = stale_root / "stale_manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "staged",
                        "staged_path": str(stale_file),
                        "sha256": stale_artifact_reaper_bot.retention._path_sha256(stale_file),
                        "integrity_verified": True,
                        "economic_value": "low",
                        "protected_evidence": False,
                    }
                ),
                json.dumps({"event": "purged", "staged_path": str(stale_root / "logs" / "project" / "logs" / "older.log")}),
                json.dumps({"event": "purged", "staged_path": str(stale_root / "logs" / "project" / "logs" / "oldest.log")}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    old_epoch = 1_735_689_600
    os.utime(stale_file, (old_epoch, old_epoch))

    payload = stale_artifact_reaper_bot.build_payload(
        project_root,
        stale_stage_root=stale_root,
        stale_stage_manifest="",
        stale_purge_days=1,
    )

    manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert payload["ok"] is True
    assert payload["summary"]["deleted_files"] == 1
    assert payload["summary"]["manifest_lines_after"] == 3
    assert payload["summary"]["purge_policy"]["low_value_days"] >= 0
    assert payload["summary"]["purge_policy"]["manifest_backed_only"] is True
    assert payload["summary"]["budget_limited"] is False
    assert stale_file.exists() is False
    assert any(row.get("event") == "purged" for row in manifest_rows)
    assert len(manifest_rows) == 3


def test_stale_artifact_reaper_reindexes_then_holds_unmanifested_file(tmp_path) -> None:
    project_root = tmp_path / "project"
    stale_root = project_root / "data" / "stale_stage"
    stale_file = stale_root / "logs" / "unmanifested.log"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("keep", encoding="utf-8")
    old_epoch = 1_735_689_600
    os.utime(stale_file, (old_epoch, old_epoch))

    payload = stale_artifact_reaper_bot.build_payload(
        project_root,
        stale_stage_root=stale_root,
        stale_stage_manifest="",
        stale_purge_days=1,
    )

    assert stale_file.exists() is True
    assert payload["summary"]["deleted_files"] == 0
    assert payload["summary"]["legacy_reindexed_files"] == 1
    assert payload["summary"]["skipped_legacy_reindex_hold_files"] == 1
    assert payload["legacy_manifest_reindex"]["hold_hours"] == 24


def test_stale_artifact_reaper_merges_all_root_health_and_work() -> None:
    primary = {
        "ok": True,
        "reason": "ok",
        "summary": {
            "deleted_files": 2,
            "deleted_bytes": 20,
            "legacy_reindex_remaining_files": 3,
            "legacy_reindex_errors": 0,
            "budget_limited": False,
        },
        "artifacts": {"stale_root": "/primary"},
    }
    external = {
        "ok": False,
        "reason": "reindex_or_purge_errors",
        "summary": {
            "deleted_files": 4,
            "deleted_bytes": 40,
            "legacy_reindex_remaining_files": 5,
            "legacy_reindex_errors": 1,
            "budget_limited": True,
        },
        "artifacts": {"stale_root": "/external"},
    }

    merged = stale_artifact_reaper_bot._merge_additional_root(primary, external)

    assert merged["ok"] is False
    assert merged["reason"] == "one_or_more_stale_roots_failed"
    assert merged["summary"]["deleted_files"] == 6
    assert merged["summary"]["deleted_bytes"] == 60
    assert merged["summary"]["legacy_reindex_remaining_files"] == 8
    assert merged["summary"]["legacy_reindex_errors"] == 1
    assert merged["summary"]["budget_limited"] is True
    assert merged["summary"]["root_count"] == 2
    assert merged["summary"]["all_roots_ok"] is False
    assert [row["stale_root"] for row in merged["root_results"]] == ["/primary", "/external"]


def test_stale_artifact_reaper_uses_shared_retention_lock() -> None:
    assert stale_artifact_reaper_bot.DEFAULT_LOCK_PATH.name == "data_retention.lock"


def test_stale_artifact_reaper_applies_pressure_gated_darwin_qos(monkeypatch) -> None:
    monkeypatch.setenv("RETENTION_STALE_PCORE_ENABLED", "1")
    monkeypatch.setenv("RETENTION_STALE_PCORE_GUARD_PASSED", "1")
    monkeypatch.setenv("RETENTION_STALE_PCORE_TASKPOLICY_APPLIED", "1")
    monkeypatch.setattr(
        stale_artifact_reaper_bot,
        "_set_darwin_thread_qos",
        lambda qos_class: {"ok": qos_class == 0x19, "return_code": 0, "errno": 0},
    )

    contract = stale_artifact_reaper_bot._apply_scheduler_intent(platform_name="darwin")

    assert contract["applied"] is True
    assert contract["effective_policy"] == "darwin_user_initiated_qos_application_taskpolicy"
    assert contract["resource_guard_confirmed"] is True
    assert contract["hard_affinity_supported"] is False


def test_stale_artifact_reaper_holds_pcore_qos_when_guard_is_not_clear(monkeypatch) -> None:
    monkeypatch.setenv("RETENTION_STALE_PCORE_ENABLED", "1")
    monkeypatch.setenv("RETENTION_STALE_PCORE_GUARD_PASSED", "0")
    called = []
    monkeypatch.setattr(stale_artifact_reaper_bot, "_set_darwin_thread_qos", lambda qos_class: called.append(qos_class))

    contract = stale_artifact_reaper_bot._apply_scheduler_intent(platform_name="darwin")

    assert contract["applied"] is False
    assert contract["reason"] == "resource_guard_not_clear"
    assert called == []


def test_hourly_stale_reaper_uses_application_taskpolicy_after_resource_guard() -> None:
    runner = (PROJECT_ROOT / "scripts" / "ops" / "run_data_retention_launchd.sh").read_text(encoding="utf-8")

    assert "RETENTION_STALE_PCORE_GUARD_PASSED=1" in runner
    assert "/usr/sbin/taskpolicy -a" in runner
    assert "BOT_WORKLOAD_CLASS=maintenance_accelerated" in runner
    assert "RETENTION_STALE_REINDEX_OVERSIZED_MAX_GB:-64" in runner
    assert "RETENTION_STALE_PURGE_OVERSIZED_MAX_GB:-64" in runner


def test_runtime_throttle_can_downshift_accelerated_stale_reaper() -> None:
    classification = runtime_throttle_control._classify_process(
        "python scripts/ops/stale_artifact_reaper_bot.py --include-external-stale-root"
    )

    assert classification == {
        "category": "support_maintenance",
        "priority_tier": "throttle_first",
        "throttle_candidate": True,
    }
