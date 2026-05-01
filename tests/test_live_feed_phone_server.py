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
    (health / "runtime_gate_dashboard_latest.json").parent.mkdir(parents=True, exist_ok=True)
    (health / "runtime_gate_dashboard_latest.json").write_text('{"overall":{"status":"ok","attention":[]}}', encoding="utf-8")
    (health / "health_gates_latest.json").write_text('{"data_quality_score":0.87,"hard_gate_triggered":false}', encoding="utf-8")
    (health / "quant_model_control_latest.json").write_text('{"overall_status":"watch","features":{"quant_model_resource_pressure_norm":0.42}}', encoding="utf-8")
    (health / "memory_efficiency_control_latest.json").write_text('{"recommended_profile":"constrained"}', encoding="utf-8")
    (health / "global_killswitch_latest.json").write_text('{"halt_state":"clear_ready","operating_mode":"degraded_collection","expansion_pressure_score":0.2}', encoding="utf-8")

    monkeypatch.setattr(phone_server, "RUNTIME_DASHBOARD", health / "runtime_gate_dashboard_latest.json")
    monkeypatch.setattr(phone_server, "HEALTH_GATES", health / "health_gates_latest.json")
    monkeypatch.setattr(phone_server, "QUANT_MODEL_CONTROL", health / "quant_model_control_latest.json")
    monkeypatch.setattr(phone_server, "MEMORY_EFFICIENCY", health / "memory_efficiency_control_latest.json")
    monkeypatch.setattr(phone_server, "GLOBAL_KILLSWITCH", health / "global_killswitch_latest.json")

    summary = phone_server._status_summary()

    assert summary["dashboard_status"] == "ok"
    assert summary["halt_state"] == "clear_ready"
    assert summary["operating_mode"] == "degraded_collection"
    assert summary["quant_status"] == "watch"
    assert summary["quant_resource_pressure"] == 0.42
    assert summary["memory_profile"] == "constrained"
    assert summary["phone_mirror_profile"] == "expanded_system_safe"
