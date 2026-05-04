import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import stale_artifact_reaper_bot, stale_artifact_sweeper_bot


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
                json.dumps({"event": "staged", "staged_path": str(stale_file)}),
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
    assert payload["summary"]["budget_limited"] is False
    assert stale_file.exists() is False
    assert any(row.get("event") == "purged" for row in manifest_rows)
    assert len(manifest_rows) == 3
