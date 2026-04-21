import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import commands_hygiene_bot as commands_src
from scripts.ops import infrastructure_autofix_bot as infra_src


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_commands_hygiene_bot_authors_commands_surface_and_runbook(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_text(project_root / "COMMANDS.md", "# old\n")
    _write_text(project_root / "scripts" / "runbook.sh", "#!/bin/zsh\nold\n")

    payload = commands_src.build_payload(project_root, apply=True)

    assert payload["overall_status"] == "degraded"
    assert payload["apply_results"]["commands_md_written"] is True
    assert payload["apply_results"]["runbook_written"] is True

    commands_text = (project_root / "COMMANDS.md").read_text(encoding="utf-8")
    assert "This file is generated from the curated operator inventory" in commands_text
    assert "### Refresh the live loops without reinstalling the stack watchdog" in commands_text
    assert "### Heavy live feed view across all feeds and decisions" in commands_text
    assert "./scripts/ops/opsctl.sh feed --source all --include-decisions" in commands_text
    assert "### Light live feed tail for all feeds" in commands_text
    assert commands_text.index("### Heavy live feed view across all feeds and decisions") < commands_text.index("### Light live feed tail for all feeds")
    assert "### Active bot stack PDF" in commands_text
    assert "./scripts/ops/opsctl.sh bot-stack-report --top 25 --render-pdf --allow-gui-pdf-renderer" in commands_text
    assert "### Options flow context sync" in commands_text
    assert "./scripts/ops/opsctl.sh options-flow-sync --json" in commands_text
    assert "## Retrain" in commands_text
    assert "### Full SQL refresh pipeline" in commands_text
    assert "./scripts/daily_log_refresh.sh" in commands_text
    assert "### Data quality refresh bundle" in commands_text
    assert '"$PY" scripts/build_one_numbers_report.py' in commands_text
    assert "### Full retrain preflight" in commands_text
    assert "./scripts/ops/opsctl.sh runtime-training-snapshot --json" in commands_text
    assert "./scripts/ops/opsctl.sh coverage-seed --write-queue --json" in commands_text
    assert "./scripts/ops/opsctl.sh coverage-gap-closer --apply-stage --launch --json" in commands_text
    assert '"$PY" scripts/retrain_schema_compatibility_guard.py --json' in commands_text
    assert '"$PY" scripts/promotion_quality_gate.py --json' in commands_text
    assert "### Guarded retrain orchestrator" in commands_text
    assert "./scripts/ops/opsctl.sh retrain-orchestrate --json" in commands_text
    assert "### Force full retrain (bypass prechecks)" in commands_text
    assert "./scripts/ops/opsctl.sh retrain-force-full" in commands_text
    assert "### Open the post-trade analysis PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh posttrade" in commands_text
    assert "### Open the crash digest PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh crash" in commands_text
    assert "30-day lookback" in commands_text
    assert "### Open the project timeline PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh timeline" in commands_text
    assert "### Open the active bot stack PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh botstack" in commands_text
    assert "## Schwab Auth" in commands_text
    assert "### Schwab authorization refresh" in commands_text
    assert "./scripts/ops/opsctl.sh token-refresh --always-auth" in commands_text
    assert "### Interactive Schwab authorization re-consent" in commands_text
    assert "./scripts/ops/opsctl.sh token-refresh-interactive" in commands_text
    assert "### Schwab auth recovery plus lane restart" in commands_text
    assert "./scripts/ops/opsctl.sh feed-refresh --source schwab" in commands_text
    assert "tastytrade-sync --json" not in commands_text

    runbook_text = (project_root / "scripts" / "runbook.sh").read_text(encoding="utf-8")
    assert "list_sections()" in runbook_text
    assert "find_section_heading()" in runbook_text


def test_infrastructure_autofix_bot_surfaces_commands_hygiene_plan_for_missing_runbook(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)

    payload = infra_src.build_payload(project_root, apply=False, timeout_sec=120)

    names = [row["name"] for row in payload["repair_plan"]]
    assert "commands_hygiene" in names


def test_render_commands_markdown_places_new_entries_in_expected_sections(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    commands_text = commands_src.render_commands_markdown(project_root)

    live_views = commands_text.split("## Live Feed Views", 1)[1].split("## Schwab Auth", 1)[0]
    schwab_auth = commands_text.split("## Schwab Auth", 1)[1].split("## Status And Health", 1)[0]
    retrain = commands_text.split("## Retrain", 1)[1].split("## Reports And PDFs", 1)[0]
    reports = commands_text.split("## Reports And PDFs", 1)[1].split("## Data Context Syncs", 1)[0]
    data_context = commands_text.split("## Data Context Syncs", 1)[1].split("## Macro And Media", 1)[0]

    assert "### Heavy live feed view across all feeds and decisions" in live_views
    assert "./scripts/ops/opsctl.sh feed --source all --include-decisions" in live_views
    assert "### Light live feed tail for all feeds" in live_views
    assert "### Schwab authorization refresh" in schwab_auth
    assert "### Interactive Schwab authorization re-consent" in schwab_auth
    assert "### Schwab auth recovery plus lane restart" in schwab_auth
    assert "### Full retrain preflight" in retrain
    assert "### Guarded retrain orchestrator" in retrain
    assert "### Force full retrain (bypass prechecks)" in retrain
    assert "### Active bot stack PDF" in reports
    assert "### Open the post-trade analysis PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh posttrade" in reports
    assert "### Open the crash digest PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh crash" in reports
    assert "30-day lookback" in reports
    assert "### Open the project timeline PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh timeline" in reports
    assert "### Open the active bot stack PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh botstack" in reports
    assert "### Options flow context sync" in data_context
