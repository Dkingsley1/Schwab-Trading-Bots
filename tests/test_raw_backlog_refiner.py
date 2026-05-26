import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import raw_backlog_refiner as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_backlog_health(project_root: Path) -> Path:
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 42000,
            "pending_lines_total": 52000,
            "pending_lines_deferred": 9000,
            "pending_lines_cold": 1000,
            "oldest_pending_age_seconds": 8100.0,
            "line_estimation": {
                "sparse_large_line_active": True,
                "sparse_large_line_pending_bytes": 120_000_000,
                "sparse_large_line_files": 2,
            },
            "top_pending_files": [
                {
                    "source_rel": "governance/events/write_failures_20260525.jsonl",
                    "pending_lines": 32000,
                    "oldest_pending_age_seconds": 7200.0,
                    "estimated_pending_bytes": 2_000_000,
                },
                {
                    "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260525.jsonl",
                    "pending_lines": 500,
                    "oldest_pending_age_seconds": 60.0,
                    "sparse_large_line": True,
                    "estimated_pending_bytes": 90_000_000,
                },
            ],
            "top_deferred_pending_files": [
                {
                    "source_rel": "governance/channels/runtime/intraday_aggressive_equities_schwab/runtime_20260525.jsonl",
                    "pending_lines": 139,
                    "oldest_pending_age_seconds": 7600.0,
                }
            ],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "degraded",
            "backpressure": {
                "core_pending_lines": 600000,
                "deferred_pending_lines": 9000,
                "cold_pending_lines": 1000,
                "support_pending_lines": 20,
                "total_pending_lines": 610020,
                "oldest_pending_age_seconds": 8300.0,
                "overlay_adjusted": True,
                "raw_live": {
                    "core_pending_lines": 42000,
                    "deferred_pending_lines": 9000,
                    "cold_pending_lines": 1000,
                    "support_pending_lines": 20,
                    "stale_stage_pending_lines": 0,
                    "total_pending_lines": 52020,
                    "oldest_pending_age_seconds": 8100.0,
                    "line_estimation": {
                        "sparse_large_line_active": True,
                        "sparse_large_line_pending_bytes": 120_000_000,
                        "sparse_large_line_files": 2,
                    },
                },
            },
            "stale_pending_locator": {
                "top_pending_sources": [
                    {
                        "source_rel": "governance/events/signal_generation_20260525.jsonl",
                        "shard": "governance",
                        "pressure_lane": "core",
                        "pending_lines": 500000,
                        "oldest_pending_age_seconds": 120.0,
                    }
                ]
            },
            "storage": {"retention_debt_gb": 1.2},
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 75})
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "drain_overrides": {
                "preferred_shards": ["governance", "crypto_trading", "health_fast"],
                "governance_path_focus": ["governance/events/signal_generation_20260525.jsonl"],
                "crypto_trading_path_focus": ["decisions/shadow_crypto/trade_decisions_20260525.jsonl"],
            }
        },
    )
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "writer_state_before": {
                "active": True,
                "current_step": "merge_primary",
                "completed_shard_count": 3,
                "planned_shard_count": 5,
                "merged_rows_this_cycle": 12000,
            },
            "summary": {"stale_writer_detected": False},
        },
    )
    _write_json(health / "pressure_relief_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "data_retention_latest.json", {"deleted": 0})
    _write_json(health / "stale_artifact_sweeper_bot_latest.json", {"summary": {"candidate_files": 4}})
    _write_json(health / "stale_artifact_reaper_bot_latest.json", {"summary": {"candidate_files": 2}})
    return health


def test_raw_backlog_refiner_expands_all_five_sections(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _seed_backlog_health(project_root)

    payload = src.build_payload(project_root, apply=False)

    assert payload["summary"]["raw_total_pending_lines"] == 52020
    assert payload["summary"]["overlay_gap_lines"] == 558000
    assert payload["policy"]["protected_volumes"] == ["/Volumes/VIDEO"]
    assert set(payload["sections"]) == {
        "measure_raw_backlog",
        "find_raw_hot_files",
        "drain_refine_raw_files",
        "reduce_intake_while_draining",
        "cleanup_stale_sparse_old_files",
    }
    assert payload["sections"]["find_raw_hot_files"]["metrics"]["top_hot_files"][0]["source_rel"] == "governance/events/signal_generation_20260525.jsonl"
    assert "sql_overlay_gap" in payload["sections"]["find_raw_hot_files"]["blockers"]
    assert "sparse_huge_jsonl_cleanup_needed" in payload["sections"]["cleanup_stale_sparse_old_files"]["blockers"]
    assert ["./scripts/ops/opsctl.sh", "external-backlog-drain", "--apply", "--follow-through", "--wait-timeout-seconds", "900", "--json"] in payload["recommended_commands"]
    assert any("SQL overlay catch-up" in action for action in payload["top_actions"])


def test_raw_backlog_refiner_apply_runs_safe_sequence_without_reaper(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _seed_backlog_health(project_root)
    commands: list[list[str]] = []

    def _fake_run(cmd: list[str], *, project_root: Path, timeout_seconds: float) -> dict:
        commands.append(cmd)
        return {
            "cmd": cmd,
            "rc": 0,
            "ok": True,
            "timed_out": False,
            "duration_ms": 1.0,
            "payload_summary": {"ok": True},
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(src, "_run_command", _fake_run)

    payload = src.build_payload(project_root, apply=True, wait_timeout_seconds=123, command_timeout_seconds=5)

    joined = [" ".join(cmd) for cmd in commands]
    assert any("ingestion_backpressure_guard.py --json" in item for item in joined)
    assert any("pressure_relief_control.py --apply --json" in item for item in joined)
    assert any("external_backlog_drain.py --apply --follow-through --wait-timeout-seconds 123 --json" in item for item in joined)
    assert any("stale_artifact_sweeper_bot.py --json" in item for item in joined)
    assert any("data_retention_policy.py --apply --no-stale-purge --skip-sqlite-vacuum --json" in item for item in joined)
    assert not any("stale_artifact_reaper_bot.py" in item for item in joined)
    assert payload["steps"]["apply_focused_drain"]["ok"] is True
