import shutil
import sys
from pathlib import Path

import pytest

from scripts import observability_exporter
from scripts.ops import bot_organization_control
from scripts.ops import chaos_drill_coordinator
from scripts.ops import control_surface_ownership
from scripts.ops import live_order_ledger_control
from scripts.ops import production_recovery_drill_harness
from scripts.ops import production_resilience_control
from scripts.ops import profitability_evidence_firewall
from scripts.ops import release_freeze_guard
from scripts.ops import soak_reliability_sentinel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("module", "relative_output", "config_name"),
    [
        (
            observability_exporter,
            "governance/health/independent_runtime_monitor_latest.json",
            "",
        ),
        (
            control_surface_ownership,
            "governance/health/control_surface_ownership_latest.json",
            "control_surface_ownership_v1.json",
        ),
        (
            bot_organization_control,
            "governance/health/bot_organization_latest.json",
            "bot_organization_v1.json",
        ),
        (
            production_resilience_control,
            "governance/health/production_resilience_control_latest.json",
            "production_resilience_v1.json",
        ),
        (
            profitability_evidence_firewall,
            "governance/health/profitability_evidence_firewall_latest.json",
            "profitability_evidence_firewall_v1.json",
        ),
        (
            release_freeze_guard,
            "governance/health/release_freeze_guard_latest.json",
            "",
        ),
        (
            production_recovery_drill_harness,
            "governance/health/production_recovery_drill_harness_latest.json",
            "",
        ),
        (
            chaos_drill_coordinator,
            "governance/health/chaos_drill_coordinator_latest.json",
            "",
        ),
        (
            soak_reliability_sentinel,
            "governance/health/soak_reliability_sentinel_latest.json",
            "",
        ),
        (
            live_order_ledger_control,
            "governance/health/live_order_ledger_control_latest.json",
            "",
        ),
    ],
)
def test_default_runtime_outputs_follow_explicit_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    module: object,
    relative_output: str,
    config_name: str,
) -> None:
    if config_name:
        target = tmp_path / "config" / config_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / "config" / config_name, target)
    monkeypatch.delenv("INDEPENDENT_MONITOR_RECEIVER_URL", raising=False)
    monkeypatch.delenv("INDEPENDENT_MONITOR_RECEIVER_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", [module.__name__, "--project-root", str(tmp_path), "--json"])

    exit_code = module.main()

    capsys.readouterr()
    assert exit_code in {0, 2}
    assert (tmp_path / relative_output).is_file()


def test_observability_metrics_follow_explicit_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("INDEPENDENT_MONITOR_RECEIVER_URL", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [observability_exporter.__name__, "--project-root", str(tmp_path), "--json"],
    )

    observability_exporter.main()

    capsys.readouterr()
    assert (tmp_path / "exports/metrics/trading_system.prom").is_file()
    assert (tmp_path / "governance/health/independent_runtime_monitor_latest.json").is_file()
