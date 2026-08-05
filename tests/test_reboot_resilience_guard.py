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
