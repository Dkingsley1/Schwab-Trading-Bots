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


def test_shape_stream_line_trims_heavy_mode() -> None:
    raw = "x" * 600
    shaped = phone_server._shape_stream_line(raw, include_decisions=True)

    assert len(shaped) < len(raw)
    assert "[trimmed " in shaped


def test_candidate_urls_include_token_for_lan(monkeypatch) -> None:
    monkeypatch.setattr(phone_server, "_candidate_host_ips", lambda: ["192.168.1.10"])
    urls = phone_server._candidate_urls("0.0.0.0", 8787, "abc123")

    assert urls == ["http://192.168.1.10:8787/?token=abc123"]


def test_helper_basics(tmp_path: Path) -> None:
    assert phone_server._load_json(tmp_path / "missing.json") == {}
    assert phone_server._is_loopback_host("localhost") is True
