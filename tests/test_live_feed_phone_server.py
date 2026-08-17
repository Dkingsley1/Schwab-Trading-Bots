import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.ops.live_feed_phone_server as phone_server


def test_effective_token_skips_generation_for_loopback() -> None:
    assert phone_server._effective_token("127.0.0.1", "") == ""


def test_effective_token_generates_for_non_loopback(monkeypatch) -> None:
    monkeypatch.setattr(phone_server.secrets, "token_urlsafe", lambda n: "test-token")
    assert phone_server._effective_token("0.0.0.0", "") == "test-token"


def test_build_feed_command_uses_snapshot_and_decisions() -> None:
    cmd = phone_server._build_feed_command(
        source="all",
        lines=80,
        symbol="SPY",
        include_decisions=True,
        snapshot=True,
    )

    assert cmd[0].endswith("live_feed_tail.sh")
    assert "--snapshot" in cmd
    assert "--include-decisions" in cmd
    assert cmd[cmd.index("--symbol") + 1] == "SPY"


def test_build_feed_command_stream_mode_skips_snapshot_flag() -> None:
    cmd = phone_server._build_feed_command(
        source="all",
        lines=80,
        symbol="",
        include_decisions=False,
        snapshot=False,
    )

    assert "--snapshot" not in cmd


def test_stream_profile_heavy_is_more_phone_safe() -> None:
    heavy = phone_server._stream_profile(True)
    light = phone_server._stream_profile(False)

    assert int(heavy["max_line_chars"]) < int(light["max_line_chars"])
    assert int(heavy["batch_char_limit"]) < int(light["batch_char_limit"])
    assert float(heavy["batch_interval_seconds"]) > float(light["batch_interval_seconds"])
    assert float(heavy["heartbeat_seconds"]) > 0
    assert float(light["heartbeat_seconds"]) > 0
    assert int(heavy["retry_millis"]) > 0
    assert int(light["retry_millis"]) > 0


def test_html_page_has_phone_reconnect_watchdog() -> None:
    assert 'eventSource.addEventListener("ping"' in phone_server.HTML_PAGE
    assert 'document.addEventListener("visibilitychange"' in phone_server.HTML_PAGE
    assert 'window.addEventListener("online"' in phone_server.HTML_PAGE
    assert 'stream stale for ${Math.round(idleMs / 1000)}s, reconnecting...' in phone_server.HTML_PAGE
    assert 'id="tokenInput"' in phone_server.HTML_PAGE
    assert 'Paste the phone-feed token from the terminal output, then tap Use Token.' in phone_server.HTML_PAGE
    assert 'window.localStorage.getItem("live_feed_phone_token")' in phone_server.HTML_PAGE
    assert 'fetch(`/api/feed?${params.toString()}`' in phone_server.HTML_PAGE
    assert 'function authParams(extra = {})' in phone_server.HTML_PAGE
    assert 'params.set("token", token);' in phone_server.HTML_PAGE
    assert 'loadSnapshot({ replace: true });' in phone_server.HTML_PAGE
    assert 'indexOf("\\n"' in phone_server.HTML_PAGE
    assert 'replace(/\\r\\n/g, "\\n")' in phone_server.HTML_PAGE
    assert "payload.spcx_quote_error" in phone_server.HTML_PAGE
    assert "payload.imessage_ready" in phone_server.HTML_PAGE
    assert "payload.watchdog_active_issues" in phone_server.HTML_PAGE


def test_html_page_links_latest_system_update_report() -> None:
    assert 'id="reportLink"' in phone_server.HTML_PAGE
    assert "/reports/latest-system-update.pdf" in phone_server.HTML_PAGE
    assert "function updateReportLink()" in phone_server.HTML_PAGE
    assert "updateReportLink();" in phone_server.HTML_PAGE


def test_latest_system_update_pdf_selects_newest(tmp_path: Path) -> None:
    older = tmp_path / "schwab_system_update_20260629_0900.pdf"
    newer = tmp_path / "schwab_system_update_20260629_1041.pdf"
    ignored = tmp_path / "other_report.pdf"
    older.write_bytes(b"%PDF-older")
    newer.write_bytes(b"%PDF-newer")
    ignored.write_bytes(b"%PDF-ignored")
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))
    os.utime(ignored, (3000, 3000))

    assert phone_server._latest_system_update_pdf(tmp_path) == newer


def test_shape_stream_line_trims_heavy_mode() -> None:
    raw = "x" * 600
    shaped = phone_server._shape_stream_line(raw, include_decisions=True)

    assert len(shaped) < len(raw)
    assert "[trimmed " in shaped


def test_candidate_urls_include_token_for_lan(monkeypatch) -> None:
    monkeypatch.setattr(phone_server, "_candidate_host_ips", lambda: ["192.168.1.10"])
    urls = phone_server._candidate_urls("0.0.0.0", 8787, "abc123")

    assert urls == ["http://192.168.1.10:8787/?token=abc123"]


def test_tailscale_candidate_urls_include_dns_and_ip(monkeypatch) -> None:
    monkeypatch.setattr(
        phone_server,
        "_tailscale_status",
        lambda: {
            "Self": {
                "DNSName": "dans-laptop.example.ts.net.",
                "TailscaleIPs": ["100.72.102.41", "fd7a:115c:a1e0::9d01:669b"],
            }
        },
    )
    urls = phone_server._tailscale_candidate_urls(8787, "abc123")

    assert urls == [
        "http://dans-laptop.example.ts.net:8787/?token=abc123",
        "http://100.72.102.41:8787/?token=abc123",
        "http://[fd7a:115c:a1e0::9d01:669b]:8787/?token=abc123",
    ]


def test_helper_basics(tmp_path: Path) -> None:
    assert phone_server._load_json(tmp_path / "missing.json") == {}
    assert phone_server._is_loopback_host("localhost") is True


def test_status_summary_includes_expansion_health(tmp_path: Path, monkeypatch) -> None:
    health = tmp_path / "governance" / "health"
    external = tmp_path / "data" / "external_context"
    (health / "runtime_gate_dashboard_latest.json").parent.mkdir(parents=True, exist_ok=True)
    external.mkdir(parents=True, exist_ok=True)
    (health / "runtime_gate_dashboard_latest.json").write_text('{"overall":{"status":"ok","attention":[]}}', encoding="utf-8")
    (health / "health_gates_latest.json").write_text('{"data_quality_score":0.87,"hard_gate_triggered":false}', encoding="utf-8")
    (health / "quant_model_control_latest.json").write_text('{"overall_status":"watch","features":{"quant_model_resource_pressure_norm":0.42}}', encoding="utf-8")
    (health / "memory_efficiency_control_latest.json").write_text('{"recommended_profile":"constrained"}', encoding="utf-8")
    (health / "global_killswitch_latest.json").write_text('{"halt_state":"clear_ready","operating_mode":"degraded_collection","expansion_pressure_score":0.2}', encoding="utf-8")
    (health / "spacex_ipo_downside_watch_latest.json").write_text(
        '{"overall_status":"waiting_for_first_quote","symbol":"SPCX","quote":{"error":"quote_missing_price"},"alert":{"triggered":false},"policy":"monitoring_only_no_order_instruction"}',
        encoding="utf-8",
    )
    (health / "macro_event_intelligence_latest.json").write_text(
        '{"overall_status":"ready","market_relevance":"high","calendar_verification":{"status":"unverified"}}',
        encoding="utf-8",
    )
    (external / "live_macro_latest.json").write_text(
        '{"items":[{"headline":"SpaceX IPO downside watch active"}]}',
        encoding="utf-8",
    )
    (health / "mac_notification_watch_state.json").write_text(
        '{"imessage_enabled":true,"imessage_recipient_configured":true,"imessage_min_severity":"critical"}',
        encoding="utf-8",
    )
    (health / "remote_alert_control_latest.json").write_text(
        '{"overall_status":"ready","channels":{"imessage_bridge":true},"critical_backlog":{"unsent_count":0,"unacked_count":1}}',
        encoding="utf-8",
    )
    (health / "process_watchdog_latest.json").write_text(
        '{"overall_status":"ready","watchdog_intelligence":{"grade":"A","active_issue_count":0}}',
        encoding="utf-8",
    )
    (health / "ingestion_storage_control_latest.json").write_text(
        '{"overall_status":"ready","pressure_index":0.31}',
        encoding="utf-8",
    )
    (health / "runtime_throttle_control_latest.json").write_text(
        '{"overall_status":"advisory","throttle_profile":"soft_cap"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(phone_server, "RUNTIME_DASHBOARD", health / "runtime_gate_dashboard_latest.json")
    monkeypatch.setattr(phone_server, "HEALTH_GATES", health / "health_gates_latest.json")
    monkeypatch.setattr(phone_server, "QUANT_MODEL_CONTROL", health / "quant_model_control_latest.json")
    monkeypatch.setattr(phone_server, "MEMORY_EFFICIENCY", health / "memory_efficiency_control_latest.json")
    monkeypatch.setattr(phone_server, "GLOBAL_KILLSWITCH", health / "global_killswitch_latest.json")
    monkeypatch.setattr(phone_server, "SPACEX_IPO_WATCH", health / "spacex_ipo_downside_watch_latest.json")
    monkeypatch.setattr(phone_server, "MACRO_EVENT_INTELLIGENCE", health / "macro_event_intelligence_latest.json")
    monkeypatch.setattr(phone_server, "LIVE_MACRO", external / "live_macro_latest.json")
    monkeypatch.setattr(phone_server, "MAC_NOTIFICATION_STATE", health / "mac_notification_watch_state.json")
    monkeypatch.setattr(phone_server, "REMOTE_ALERT_CONTROL", health / "remote_alert_control_latest.json")
    monkeypatch.setattr(phone_server, "PROCESS_WATCHDOG", health / "process_watchdog_latest.json")
    monkeypatch.setattr(phone_server, "INGESTION_STORAGE_CONTROL", health / "ingestion_storage_control_latest.json")
    monkeypatch.setattr(phone_server, "RUNTIME_THROTTLE_CONTROL", health / "runtime_throttle_control_latest.json")

    summary = phone_server._status_summary()

    assert summary["dashboard_status"] == "ok"
    assert summary["halt_state"] == "clear_ready"
    assert summary["operating_mode"] == "degraded_collection"
    assert summary["quant_status"] == "watch"
    assert summary["quant_resource_pressure"] == 0.42
    assert summary["memory_profile"] == "constrained"
    assert summary["spcx_status"] == "waiting_for_first_quote"
    assert summary["spcx_symbol"] == "SPCX"
    assert summary["spcx_quote_error"] == "quote_missing_price"
    assert summary["spcx_alert_triggered"] is False
    assert summary["macro_status"] == "ready"
    assert summary["macro_relevance"] == "high"
    assert summary["macro_calendar_status"] == "unverified"
    assert summary["macro_headline"] == "SpaceX IPO downside watch active"
    assert summary["imessage_ready"] is True
    assert summary["imessage_min_severity"] == "critical"
    assert summary["remote_alert_status"] == "ready"
    assert summary["remote_alert_imessage"] is True
    assert summary["remote_alert_unacked_count"] == 1
    assert summary["watchdog_status"] == "ready"
    assert summary["watchdog_grade"] == "A"
    assert summary["watchdog_active_issues"] == 0
    assert summary["storage_status"] == "ready"
    assert summary["storage_pressure"] == 0.31
    assert summary["throttle_status"] == "advisory"
    assert summary["throttle_profile"] == "soft_cap"
    assert summary["phone_mirror_profile"] == "expanded_system_safe"
