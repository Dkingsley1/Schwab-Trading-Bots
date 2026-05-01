from __future__ import annotations

import plistlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.execution_lane_pipeline import execution_lane_daily_path
from scripts.ops import infrastructure_autofix_bot as infra_src
from scripts.ops import stateful_storage_regression_guard as guard_src


def test_execution_lane_daily_path_prefers_external_project_root(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", str(external_root))
    monkeypatch.setenv("BOT_LOGS_PREFER_EXTERNAL", "1")
    monkeypatch.delenv("EXECUTION_LANE_ROOT", raising=False)

    path = Path(execution_lane_daily_path(project_root, "execution_results", day="20260430"))

    assert path == external_root / "governance" / "execution_lanes" / "execution_results_20260430.jsonl"


def test_stateful_storage_guard_repairs_local_dirs_and_launchd_logs(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    sql_local = project_root / "data" / "sql_link_shards"
    lane_local = project_root / "governance" / "execution_lanes"
    sql_local.mkdir(parents=True)
    lane_local.mkdir(parents=True)
    (sql_local / "jsonl_link_trading.sqlite3").write_text("sqlite", encoding="utf-8")
    (lane_local / "execution_results_20260430.jsonl").write_text('{"ok": true}\n', encoding="utf-8")
    plist_path = tmp_path / "LaunchAgents" / "com.dankingsley.ops.sql_link_writer.plist"
    plist_path.parent.mkdir(parents=True)
    with plist_path.open("wb") as handle:
        plistlib.dump(
            {
                "Label": "com.dankingsley.ops.sql_link_writer",
                "StandardOutPath": str(project_root / "logs" / "launchd_ops" / "ops_sql_link_writer.out.log"),
                "StandardErrorPath": str(project_root / "logs" / "launchd_ops" / "ops_sql_link_writer.err.log"),
            },
            handle,
        )
    monkeypatch.setenv("STATEFUL_STORAGE_REGRESSION_CHECK_OPEN_HANDLES", "0")
    monkeypatch.setattr(guard_src, "SQL_WRITER_PLIST", plist_path)
    monkeypatch.setattr(guard_src, "_active_process", lambda patterns: False)

    payload = guard_src.build_payload(project_root, external_root=str(external_root), apply=True)

    assert payload["overall_status"] == "ready"
    assert sql_local.is_symlink()
    assert lane_local.is_symlink()
    assert (external_root / "data" / "sql_link_shards" / "jsonl_link_trading.sqlite3").exists()
    assert (external_root / "governance" / "execution_lanes" / "execution_results_20260430.jsonl").exists()
    with plist_path.open("rb") as handle:
        plist = plistlib.load(handle)
    assert str(plist["StandardOutPath"]).startswith("/tmp/schwab_trading_bot/launchd_ops/")
    assert str(plist["StandardErrorPath"]).startswith("/tmp/schwab_trading_bot/launchd_ops/")


def test_infrastructure_autofix_assigns_stateful_storage_guard(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True)
    (health / "stateful_storage_regression_guard_latest.json").write_text(
        '{"overall_status": "degraded", "metrics": {"local_stateful_gb": 1.25}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        infra_src,
        "_run_json",
        lambda cmd, *, cwd, timeout_sec: {
            "cmd": list(cmd),
            "rc": 0,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "",
            "payload": {"overall_status": "ready", "ok": True, "metrics": {}},
        },
    )

    payload = infra_src.build_payload(project_root, apply=False)

    names = [row["name"] for row in payload["repair_plan"]]
    assert "stateful_storage_regression_guard" in names
    assert "stateful_storage_regression_guard" in payload["infra_bots"]
    assert payload["metrics"]["stateful_storage_local_gb"] == 1.25
