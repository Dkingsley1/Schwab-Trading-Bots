import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import reboot_resilience_guard as src


def test_default_required_labels_follow_watchdog_mode(monkeypatch) -> None:
    monkeypatch.delenv("STACK_ORCHESTRATOR_MODE", raising=False)
    labels = src._default_required_labels()
    assert "com.dankingsley.shadow_watchdog" in labels
    assert "com.dankingsley.ops.watchdog" in labels
    assert "com.dankingsley.ops.sql_link_writer" in labels
    assert "com.dankingsley.observability_exporter" in labels
    assert "com.dankingsley.livefeed-local" in labels
    assert "com.dankingsley.all_sleeves" not in labels


def test_default_required_labels_include_all_sleeves_in_all_sleeves_mode(monkeypatch) -> None:
    monkeypatch.setenv("STACK_ORCHESTRATOR_MODE", "all_sleeves")
    labels = src._default_required_labels()
    assert labels[0] == "com.dankingsley.all_sleeves"


def test_enable_label_uses_persistent_launchctl_override(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str]):
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(src, "_run", fake_run)

    action = src._enable_label("gui/501", "com.example.worker")

    assert calls == [["launchctl", "enable", "gui/501/com.example.worker"]]
    assert action["rc"] == 0


def test_reboot_guard_defers_all_recovery_during_stack_restart(monkeypatch, tmp_path) -> None:
    out_path = tmp_path / "reboot.json"
    monkeypatch.setattr(src, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(src, "STACK_STOPPED_FLAG", tmp_path / "STACK_STOPPED.flag")
    monkeypatch.setattr(src, "DEFAULT_OUT_PATH", out_path)
    monkeypatch.setattr(src, "FALLBACK_OUT_PATH", tmp_path / "fallback.json")
    monkeypatch.setattr(
        src,
        "stack_restart_fence_snapshot",
        lambda _root: {
            "active": True,
            "exists": True,
            "owner_pid": os.getpid(),
            "token": "secret",
            "payload": {"token": "secret"},
            "reason": "stack_restart_in_progress",
        },
    )
    monkeypatch.setattr(src, "_pressure_relief_context", lambda: {"active": False, "skip_labels": []})
    monkeypatch.setattr(src, "_is_loaded", lambda _domain, _label: False)
    monkeypatch.setattr(sys, "argv", ["reboot_resilience_guard.py", "--required-labels", "one,two", "--json"])

    assert src.main() == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "restarting"
    assert [row["reason"] for row in payload["skipped"]] == [
        "stack_restart_in_progress",
        "stack_restart_in_progress",
    ]
    assert "token" not in payload["stack_restart_fence"]
