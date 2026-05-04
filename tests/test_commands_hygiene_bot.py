import json
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
    assert "Command contract hash: `" in commands_text
    assert "Command contract artifact: `governance/health/commands_contract_latest.json`." in commands_text
    assert commands_text.index("## Most Used") < commands_text.index("## Data Context Syncs")
    assert "### Refresh the live loops without reinstalling the stack watchdog" in commands_text
    assert "### Heavy live feed view across all sections" in commands_text
    assert "./scripts/ops/opsctl.sh feed --source all --heavy" in commands_text
    assert "### Heavy infrastructure live feed view" in commands_text
    assert "./scripts/ops/opsctl.sh feed --source infra --heavy --lines 160" in commands_text
    assert "### Heavy FX live feed view" in commands_text
    assert "./scripts/ops/opsctl.sh feed --source fx --heavy" in commands_text
    assert "### Light live feed tail for all feeds" in commands_text
    assert "### Start the full live stack (fresh supervised restart)" in commands_text
    assert "./scripts/ops/opsctl.sh start --force-restart" in commands_text
    assert "### Emergency stop: engage operator stop and global halt" in commands_text
    assert "./scripts/ops/opsctl.sh operator-control --engage --set-global-halt" in commands_text
    assert "### Show global halt status and blockers" in commands_text
    assert "./scripts/ops/opsctl.sh global-halt-status --json" in commands_text
    assert "### Refresh clearable global halt blockers" in commands_text
    assert "./scripts/ops/opsctl.sh global-halt-refresh --json" in commands_text
    assert "### Release operator stop only" in commands_text
    assert "./scripts/ops/opsctl.sh operator-release --json" in commands_text
    assert "### Clear all halt flags now" in commands_text
    assert "./scripts/ops/opsctl.sh clear-all-halts --json" in commands_text
    assert "### Attempt a safe global halt clear" in commands_text
    assert "./scripts/ops/opsctl.sh global-halt-auto-clear --json" in commands_text
    assert "### Refresh runtime dashboard contracts" in commands_text
    assert "./scripts/ops/opsctl.sh dashboard-refresh" in commands_text
    assert "### Runtime gate dashboard" in commands_text
    assert "./scripts/ops/opsctl.sh dashboard" in commands_text
    assert "./scripts/ops/opsctl.sh dashboard --skip-refresh" in commands_text
    assert "### Review Codex project guardrails" in commands_text
    assert "./scripts/ops/opsctl.sh codex-project-guard --staged --json" in commands_text
    assert "### Review or prune eligible local standby SQLite copies after BOT_LOGS soak" in commands_text
    assert "./scripts/ops/opsctl.sh storage-prune-standby --json" in commands_text
    assert "### Safe force-clear storage pressure supervisor" in commands_text
    assert "./scripts/ops/opsctl.sh storage-pressure-clearance --apply --force-clear-stale-gate --json" in commands_text
    assert commands_text.index("### Heavy live feed view across all sections") < commands_text.index("### Light live feed tail for all feeds")
    assert "### Active bot stack PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh botstack" in commands_text
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
    assert "### Open the paper performance PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh paper" in commands_text
    assert "### Open the sentiment PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh sentiment" in commands_text
    assert "### Open the strategy inventory PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh strategy-inventory" in commands_text
    assert "### Open the crash digest PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh crash" in commands_text
    assert "30-day lookback" in commands_text
    assert "### Open the project timeline PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh timeline" in commands_text
    assert "### Incident report" in commands_text
    assert "./scripts/ops/open_report_artifact.sh incident" in commands_text
    assert "### Incident review packet PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh incident-packet" in commands_text
    assert "### Open the incident report PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh incident" in commands_text
    assert "### Open the incident review packet PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh incident-packet" in commands_text
    assert "### Open the training report PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh training" in commands_text
    assert "### Open the macro crosscheck PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh macro" in commands_text
    assert "### Open the market correlation PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh correlation" in commands_text
    assert "### Open the source verification PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh source" in commands_text
    assert "### Open the model card PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh modelcard" in commands_text
    assert "### Open the replay feature ablation PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh replay" in commands_text
    assert "### Open the unified lane scorecard PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh unified" in commands_text
    assert "### Open the bot explainability PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh explainability" in commands_text
    assert "### Open the active bot stack PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh botstack" in commands_text
    assert "### Refresh showcase, framework map, and PDFs now" in commands_text
    assert "./scripts/ops/opsctl.sh system-explainers" in commands_text
    assert "### Open the framework map PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh framework" in commands_text
    assert "./scripts/ops/opsctl.sh report-pdfs --json" in commands_text
    assert "### Reporter quality infrabot" in commands_text
    assert "./scripts/ops/opsctl.sh report-quality-guard --repair --json" in commands_text
    assert "### Install nightly showcase and PDF refresh" in commands_text
    assert "./scripts/install_daily_log_refresh_launchd.sh" in commands_text
    assert "## Schwab Auth" in commands_text
    assert "### Schwab auth supervisor" in commands_text
    assert "./scripts/ops/opsctl.sh schwab-auth-supervisor --json" in commands_text
    assert "### Schwab authorization refresh" in commands_text
    assert "./scripts/ops/opsctl.sh token-refresh --always-auth" in commands_text
    assert "### Interactive Schwab authorization re-consent" in commands_text
    assert "./scripts/ops/opsctl.sh token-refresh-interactive" in commands_text
    assert "### Schwab auth recovery plus lane restart" in commands_text
    assert "./scripts/ops/opsctl.sh feed-refresh --source schwab" in commands_text
    assert "## Platform Expansion" in commands_text
    assert "### Apply the coordination intelligence control-plane pack" in commands_text
    assert "./scripts/ops/opsctl.sh coordination-intelligence --apply --json" in commands_text
    assert "### Preview the coordination intelligence control-plane pack" in commands_text
    assert "./scripts/ops/opsctl.sh coordination-intelligence --json" in commands_text
    assert "### Apply the alpha intelligence evolution pack" in commands_text
    assert "./scripts/ops/opsctl.sh alpha-intelligence-evolution --apply --json" in commands_text
    assert "### Preview the alpha intelligence evolution pack" in commands_text
    assert "./scripts/ops/opsctl.sh alpha-intelligence-evolution --json" in commands_text
    assert "### Apply the intelligence layer advancement pack" in commands_text
    assert "./scripts/ops/opsctl.sh intelligence-layer-advancement --apply --json" in commands_text
    assert "### Preview the intelligence layer advancement pack" in commands_text
    assert "./scripts/ops/opsctl.sh intelligence-layer-advancement --json" in commands_text
    assert "### Apply the apex self-awareness intelligence pack" in commands_text
    assert "./scripts/ops/opsctl.sh apex-self-awareness-intelligence --apply --json" in commands_text
    assert "### Preview the apex self-awareness intelligence pack" in commands_text
    assert "./scripts/ops/opsctl.sh apex-self-awareness-intelligence --json" in commands_text
    assert "tastytrade-sync --json" not in commands_text

    runbook_text = (project_root / "scripts" / "runbook.sh").read_text(encoding="utf-8")
    assert "list_sections()" in runbook_text
    assert "find_section_heading()" in runbook_text
    contract_payload = json.loads((project_root / "governance" / "health" / "commands_contract_latest.json").read_text(encoding="utf-8"))
    assert contract_payload["schema_version"] == commands_src.COMMAND_CONTRACT_SCHEMA_VERSION
    assert contract_payload["entry_count"] == 126
    assert contract_payload["contract_hash"] in commands_text


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
    status_health = commands_text.split("## Status And Health", 1)[1].split("## SQL And Reports", 1)[0]
    retrain = commands_text.split("## Retrain", 1)[1].split("## Reports And PDFs", 1)[0]
    reports = commands_text.split("## Reports And PDFs", 1)[1].split("## Data Context Syncs", 1)[0]
    data_context = commands_text.split("## Data Context Syncs", 1)[1].split("## Macro And Media", 1)[0]

    assert "### Heavy live feed view across all sections" in live_views
    assert "./scripts/ops/opsctl.sh feed --source all --heavy" in live_views
    assert "### Heavy infrastructure live feed view" in live_views
    assert "./scripts/ops/opsctl.sh feed --source infra --heavy --lines 160" in live_views
    assert "### Heavy Schwab live feed view" in live_views
    assert "### Heavy Coinbase live feed view" in live_views
    assert "### Heavy futures live feed view" in live_views
    assert "### Heavy FX live feed view" in live_views
    assert "### Light live feed tail for all feeds" in live_views
    assert "### Schwab authorization refresh" in schwab_auth
    assert "### Schwab auth supervisor" in schwab_auth
    assert "./scripts/ops/opsctl.sh schwab-auth-supervisor --json" in schwab_auth
    assert "### Interactive Schwab authorization re-consent" in schwab_auth
    assert "### Schwab auth recovery plus lane restart" in schwab_auth
    assert "### Refresh runtime dashboard contracts" in status_health
    assert "./scripts/ops/opsctl.sh dashboard-refresh" in status_health
    assert "### Runtime gate dashboard" in status_health
    assert "./scripts/ops/opsctl.sh dashboard" in status_health
    assert "### Review the cross-system drift mesh" in status_health
    assert "./scripts/ops/opsctl.sh system-drift-guard --json" in status_health
    assert "### Repair safe cross-system drift surfaces" in status_health
    assert "./scripts/ops/opsctl.sh system-drift-autopilot --apply --json" in status_health
    assert "### Master infrastructure supervisor" in status_health
    assert "./scripts/ops/opsctl.sh master-infra-supervisor --json" in status_health
    assert "### Coinbase API health" in status_health
    assert "./scripts/ops/opsctl.sh coinbase-api-health --json" in status_health
    assert "### Point-in-time event store" in status_health
    assert "./scripts/ops/opsctl.sh point-in-time-event-store --json" in status_health
    assert "### Replay hash registry guard" in status_health
    assert "./scripts/ops/opsctl.sh replay-hash-registry --json" in status_health
    assert "### Golden replay regression guard" in status_health
    assert "./scripts/ops/opsctl.sh golden-replay-regression --json" in status_health
    assert "### Full retrain preflight" in retrain
    assert "### Guarded retrain orchestrator" in retrain
    assert "### Force full retrain (bypass prechecks)" in retrain
    assert "### Review or prune eligible local standby SQLite copies after BOT_LOGS soak" in commands_text
    assert "### Refresh showcase, framework map, and PDFs now" in reports
    assert "### Install nightly showcase and PDF refresh" in reports
    assert "### Active bot stack PDF" in reports
    assert "### Open the post-trade analysis PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh posttrade" in reports
    assert "### Open the paper performance PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh paper" in reports
    assert "### Repair and validate report PDFs" in reports
    assert "./scripts/ops/opsctl.sh report-quality-guard --repair --json" in reports
    assert "### Open the sentiment PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh sentiment" in reports
    assert "### Open the strategy inventory PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh strategy-inventory" in reports
    assert "### Open the crash digest PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh crash" in reports
    assert "30-day lookback" in reports
    assert "### Open the project timeline PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh timeline" in reports
    assert "### Incident report" in reports
    assert "./scripts/ops/open_report_artifact.sh incident" in reports
    assert "### Incident review packet PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh incident-packet" in reports
    assert "### Open the incident report PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh incident" in reports
    assert "### Open the incident review packet PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh incident-packet" in reports
    assert "### Open the training report PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh training" in reports
    assert "### Open the macro crosscheck PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh macro" in reports
    assert "### Open the market correlation PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh correlation" in reports
    assert "### Open the source verification PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh source" in reports
    assert "### Open the model card PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh modelcard" in reports
    assert "### Open the replay feature ablation PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh replay" in reports
    assert "### Open the unified lane scorecard PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh unified" in reports
    assert "### Open the bot explainability PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh explainability" in reports
    assert "### Open the active bot stack PDF" in reports
    assert "./scripts/ops/open_report_artifact.sh botstack" in reports
    _, sections = commands_src._parse_commands_sections(commands_text)
    assert commands_src._source_duplicate_entry_count(sections) == 0
    assert "### Options flow context sync" in data_context
