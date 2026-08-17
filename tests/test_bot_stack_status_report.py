import json
import plistlib
import sys
from datetime import datetime, timezone

from scripts import bot_stack_status_report as report


def test_classify_lane_prefers_role_specific_lanes() -> None:
    assert report._classify_lane({"bot_role": "infrastructure_sub_bot", "bot_id": "brain_refinery_v56_meta_ranker"}) == "infrastructure"
    assert report._classify_lane({"bot_role": "options_sub_bot", "bot_id": "brain_refinery_v27_term_structure_vol"}) == "options"
    assert report._classify_lane({"bot_role": "signal_sub_bot", "bot_id": "brain_refinery_v48_position_1m_3m"}) == "swing"
    assert report._classify_lane({"bot_role": "signal_sub_bot", "bot_id": "brain_refinery_v93_dividend_quality_compounder"}) == "long_term"


def test_registry_summary_groups_active_bots_by_lane() -> None:
    registry = {
        "sub_bots": [
            {"bot_id": "brain_refinery_v56_meta_ranker", "bot_role": "infrastructure_sub_bot", "active": True, "weight": 0.02, "quality_score": 0.87, "test_accuracy": 0.96, "reason": "canary"},
            {"bot_id": "brain_refinery_v27_term_structure_vol", "bot_role": "options_sub_bot", "active": True, "weight": 0.008, "quality_score": 0.22, "test_accuracy": 0.51, "reason": "options_floor"},
            {"bot_id": "brain_refinery_v48_position_1m_3m", "bot_role": "signal_sub_bot", "active": True, "weight": 0.01, "quality_score": 0.40, "test_accuracy": 0.53, "reason": "swing"},
            {"bot_id": "brain_refinery_v10_seasonal", "bot_role": "signal_sub_bot", "active": True, "weight": 0.013, "quality_score": 0.99, "test_accuracy": 0.94, "reason": "equities"},
        ]
    }

    summary = report._registry_summary(registry, top_n=5)

    assert summary["lanes"]["infrastructure"]["active_count"] == 1
    assert summary["lanes"]["options"]["active_count"] == 1
    assert summary["lanes"]["swing"]["active_count"] == 1
    assert summary["lanes"]["equities"]["active_count"] == 1


def test_overall_health_treats_schwab_loops_as_off_session_on_weekends() -> None:
    now = datetime(2026, 3, 29, 16, 0, tzinfo=timezone.utc)
    payload = report._overall_health(
        {"active": 35},
        {
            "schwab_conservative": {"latest_loop": None, "decision_lines": 0},
            "schwab_aggressive": {"latest_loop": None, "decision_lines": 0},
            "coinbase_crypto": {"latest_loop": None, "decision_lines": 50},
        },
        {"targets": [], "exists": False, "path": "/tmp/watchdog", "latest_timestamp_utc": None},
        now=now,
    )

    checks = {row["name"]: row for row in payload["checks"]}
    assert checks["live_shadow_loops"]["ok"] is True
    assert "schwab_conservative=off_session" in checks["live_shadow_loops"]["note"]
    assert checks["watchdog_schwab_live"]["ok"] is True
    assert checks["watchdog_schwab_live"]["note"] == "off_session"


def test_overall_health_falls_back_to_coinbase_activity_when_watchdog_missing() -> None:
    now = datetime(2026, 3, 30, 14, 0, tzinfo=timezone.utc)
    payload = report._overall_health(
        {"active": 35},
        {
            "schwab_conservative": {"latest_loop": {"iter": 1}, "decision_lines": 10},
            "schwab_aggressive": {"latest_loop": {"iter": 2}, "decision_lines": 8},
            "coinbase_crypto": {"latest_loop": None, "decision_lines": 125},
        },
        {"targets": [], "exists": False, "path": "/tmp/watchdog", "latest_timestamp_utc": None},
        now=now,
    )

    checks = {row["name"]: row for row in payload["checks"]}
    assert checks["watchdog_coinbase_live"]["ok"] is True
    assert "fallback_activity=True" in checks["watchdog_coinbase_live"]["note"]


def test_collect_infrastructure_bots_reads_launch_agents(tmp_path, monkeypatch) -> None:
    plist_path = tmp_path / "com.dankingsley.ops.options_flow_context.plist"
    plist_path.write_bytes(
        plistlib.dumps(
            {
                "Label": "com.dankingsley.ops.options_flow_context",
                "ProgramArguments": ["/bin/zsh", "/tmp/run_options_flow_context_launchd.sh"],
                "StartInterval": 600,
            }
        )
    )
    monkeypatch.setattr(
        report,
        "_run",
        lambda cmd: (0, "PID\tStatus\tLabel\n123\t0\tcom.dankingsley.ops.options_flow_context\n", ""),
    )

    payload = report._collect_infrastructure_bots(tmp_path)

    assert payload["count"] == 1
    assert payload["loaded_count"] == 1
    assert payload["running_count"] == 1
    assert payload["bots"][0]["schedule"] == "every 600s"


def test_render_html_includes_infrastructure_and_master_sections() -> None:
    payload = {
        "generated_utc": "2026-04-15T12:00:00+00:00",
        "registry": {
            "counts": {"total": 3, "active": 2, "inactive": 1, "deleted": 0},
            "lanes": {
                "infrastructure": {
                    "active_count": 1,
                    "active_weight": 0.02,
                    "bots": [{"bot_id": "brain_refinery_v56_meta_ranker"}],
                }
            },
            "top_active": [{"bot_id": "brain_refinery_v56_meta_ranker", "bot_role": "infrastructure_sub_bot", "weight": 0.02, "quality_score": 0.87}],
        },
        "decision_logs": {
            "schwab_conservative": {
                "grand_master": {"avg_score": 0.91},
                "options_master": {"avg_score": 0.44},
                "latest_loop": {"grand_action": "BUY", "options_action": "HOLD", "active_bots": 7},
            }
        },
        "watchdog": {"targets": [{"name": "schwab_parallel", "live": True, "process_live": True, "action": "none", "note": "ok"}]},
        "infrastructure_bots": {
            "count": 1,
            "loaded_count": 1,
            "running_count": 1,
            "bots": [{"label": "com.dankingsley.ops.options_flow_context", "loaded": True, "running": True, "pid": 123, "schedule": "every 600s", "last_exit_status": 0}],
        },
        "overall_health": {"status": "healthy", "checks": [{"name": "active_sub_bots", "ok": True, "note": "ok"}]},
    }

    html = report._render_html(payload)

    assert "Grandmaster + Master Bots" in html
    assert "Infrastructure Bots" in html
    assert "com.dankingsley.ops.options_flow_context" in html


def test_main_writes_pdf_artifact_when_requested(tmp_path, monkeypatch) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    registry_path.write_text(json.dumps({"sub_bots": []}), encoding="utf-8")
    decision_log = tmp_path / "shadow.log"
    decision_log.write_text("[Decision] action=HOLD status=OK symbol=SPY grand_master_routing score=0.5\n", encoding="utf-8")

    monkeypatch.setattr(report, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(report, "WATCHDOG_DIR", tmp_path / "watchdog")
    monkeypatch.setattr(report, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(report, "DEFAULT_HTML_PATH", tmp_path / "out" / "latest.html")
    monkeypatch.setattr(report, "DEFAULT_PDF_PATH", tmp_path / "out" / "latest.pdf")
    monkeypatch.setattr(report, "DECISION_LOGS", {"schwab_conservative": decision_log})
    monkeypatch.setattr(
        report,
        "_collect_infrastructure_bots",
        lambda: {"count": 1, "loaded_count": 1, "running_count": 1, "bots": []},
    )

    def _fake_render_pdf_from_html(_html_path, pdf_path, *, allow_gui_renderer):
        pdf_path.write_bytes(b"%PDF-1.4\n%bot stack report\n")
        return True, "ok"

    monkeypatch.setattr(report, "_render_pdf_from_html", _fake_render_pdf_from_html)
    monkeypatch.setattr(sys, "argv", ["bot_stack_status_report.py", "--render-pdf"])

    rc = report.main()

    assert rc == 0
    latest_json = json.loads((tmp_path / "out" / "latest.json").read_text(encoding="utf-8"))
    assert latest_json["pdf"]["available"] is True
    assert (tmp_path / "out" / "latest.pdf").exists()
