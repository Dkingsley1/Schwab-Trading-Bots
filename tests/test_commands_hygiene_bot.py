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
    contract_payload = json.loads((project_root / "governance" / "health" / "commands_contract_latest.json").read_text(encoding="utf-8"))
    assert "This file is generated from the curated operator inventory" in commands_text
    assert "Command contract hash: `" in commands_text
    assert "Command contract artifact: `governance/health/commands_contract_latest.json`." in commands_text
    assert "**Search Bar**" in commands_text
    assert '<input type="search"' in commands_text
    assert 'list="command-search-index-options"' in commands_text
    assert "PyCharm: press Command+F or Ctrl+F" in commands_text
    assert "`paper`" in commands_text
    assert "`halt`" in commands_text
    assert "`storage`" in commands_text
    assert "Useful compound searches: `paper profitability`, `global halt`, `token refresh`" in commands_text
    assert f"Search coverage: `{contract_payload['entry_count']}` generated command entries" in commands_text
    assert '<datalist id="command-search-index-options">' in commands_text
    assert f"Generated command search index ({contract_payload['entry_count']} commands; rebuilt by commands-hygiene)" in commands_text
    search_index = commands_text.split("Generated command search index", 1)[1].split("</details>", 1)[0]
    for entry in contract_payload["entries"]:
        assert f"search-entry:{entry['fingerprint']}" in search_index
        assert f"title:{entry['title']}" in search_index
    assert commands_text.index("## Most Used") < commands_text.index("## Data Context Syncs")
    assert commands_text.index("**Search Bar**") < commands_text.index("## Most Used")
    most_used = commands_text.split("## Most Used", 1)[1].split("\n## Accounts And Positions", 1)[0]
    most_used_titles = [line for line in most_used.splitlines() if line.startswith("### ")]
    assert most_used_titles[:4] == [
        "### Keep the Mac awake",
        "### Start the full live stack",
        "### Start the full live stack (fresh supervised restart)",
        "### Stop the stack",
    ]
    assert "### Refresh the livefeed mirror without restarting sleeves" in commands_text
    assert "### Repair and restart the livefeed mirror" in commands_text
    assert "./scripts/ops/opsctl.sh livefeed-refresh-guard --apply --force-restart --freshness-minutes 10 --json" in commands_text
    assert "### Heavy operator livefeed view" in commands_text
    assert "./scripts/ops/opsctl.sh feed --source main --heavy --no-heavy-ttl --color --red-actions" in commands_text
    assert "Escaped JSON fragments are hidden by default" in commands_text
    assert "### Heavy infrastructure live feed view" not in commands_text
    assert "./scripts/ops/opsctl.sh feed --source infra --heavy --lines 160" not in commands_text
    assert "### Heavy FX live feed view" not in commands_text
    assert "./scripts/ops/opsctl.sh feed --source fx --heavy" not in commands_text
    assert "### Light live feed tail for all feeds" not in commands_text
    assert "### Live feed tail for Schwab" not in commands_text
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
    assert "## Live Feed Refreshes" not in commands_text
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
    assert "### Training and labeling intelligence" in commands_text
    assert "./scripts/ops/opsctl.sh training-labeling-intelligence --apply --json" in commands_text
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
    assert "### Docs, commands, and reporting intelligence" in commands_text
    assert "./scripts/ops/opsctl.sh docs-reporting-intelligence --apply --json" in commands_text
    assert "### Adapt infrabots to current system needs" in commands_text
    assert "./scripts/ops/opsctl.sh infrabot-adaptive-governor --apply --json" in commands_text
    assert "### PyCharm active bot blue highlights" in commands_text
    assert "./scripts/ops/opsctl.sh pycharm-active-bot-highlights --apply --json" in commands_text
    assert "### Reporter quality infrabot" in commands_text
    assert "./scripts/ops/opsctl.sh report-quality-guard --repair --json" in commands_text
    assert "### Install nightly showcase and PDF refresh" not in commands_text
    assert "./scripts/install_daily_log_refresh_launchd.sh" not in commands_text
    assert "## Schwab Auth" in commands_text
    assert "### Schwab auth supervisor" in commands_text
    assert "./scripts/ops/opsctl.sh schwab-auth-supervisor --json" in commands_text
    assert "### Local Schwab credential setup" in commands_text
    assert "./scripts/ops/opsctl.sh schwab-credentials --interactive --store keychain --json" in commands_text
    assert "### Schwab authorization refresh" in commands_text
    assert "./scripts/ops/opsctl.sh token-refresh --always-auth" in commands_text
    assert "### Interactive Schwab authorization re-consent" in commands_text
    assert "./scripts/ops/opsctl.sh token-refresh-interactive" in commands_text
    assert "### Schwab auth recovery plus lane restart" in commands_text
    assert "./scripts/ops/opsctl.sh feed-refresh --source schwab" in commands_text
    assert "## Platform Expansion" not in commands_text
    assert "./scripts/ops/opsctl.sh platform-intelligence --apply --json" not in commands_text
    assert "./scripts/ops/opsctl.sh platform-brain-v4 --apply --json" not in commands_text
    assert "./scripts/ops/opsctl.sh platform-brain-v5 --apply --json" not in commands_text
    assert "### Open the expansion inventory PDF" in commands_text
    assert "./scripts/ops/open_report_artifact.sh expansions" in commands_text
    assert "## Accounts And Positions" in commands_text
    assert "./scripts/ops/opsctl.sh schwab-account-snapshot-refresh --json" in commands_text
    assert "./scripts/ops/opsctl.sh account-position-study --json" in commands_text
    assert "./scripts/ops/opsctl.sh covered-call-roll-watch --json" in commands_text
    assert "## Event Watches" in commands_text
    assert "./scripts/ops/opsctl.sh spacex-ipo-watch --json" in commands_text
    assert "./scripts/ops/opsctl.sh spacex-ipo-watch-install --poll-seconds 30 --symbol SPCX" in commands_text
    assert "## Notifications And Alerts" in commands_text
    assert "./scripts/ops/opsctl.sh notify-test --enable-imessage" in commands_text
    assert "./scripts/ops/opsctl.sh notify-start --enable-imessage" in commands_text
    assert "### Install the startup Yes/No bot start prompt" in commands_text
    assert "./scripts/ops/opsctl.sh startup-start-prompt --install --no-kickstart --no-browser" in commands_text
    assert "suppresses Schwab browser auth, GUI Chrome opens, headless Chrome PDF/render helpers" in commands_text
    assert "### Dry-run the startup Yes/No bot start prompt" in commands_text
    assert "./scripts/ops/opsctl.sh startup-start-prompt-test --dry-run --delay-seconds 0" in commands_text
    assert "./scripts/ops/opsctl.sh notify-stop" in commands_text
    assert "## Paper Trading" in commands_text
    assert "./scripts/ops/opsctl.sh paper-400-ramp --apply --promote-roster --json" in commands_text
    assert "./scripts/ops/opsctl.sh runtime-paper-regression-guard --json" in commands_text
    assert "./scripts/ops/opsctl.sh paper-live-data-standard --apply --json" in commands_text
    assert "./scripts/ops/opsctl.sh paper-profitability-control --apply --json" in commands_text
    assert "./scripts/ops/opsctl.sh profitability-evidence-firewall --json" in commands_text
    assert "./scripts/ops/opsctl.sh profitability-independent-validator --json" in commands_text
    assert "./scripts/ops/opsctl.sh profitability-holdout-vault --json" in commands_text
    assert "./scripts/ops/opsctl.sh profitability-benchmark-capture --apply --json" in commands_text
    assert "./scripts/ops/opsctl.sh profitability-benchmark-hurdle --json" in commands_text
    assert "./scripts/ops/opsctl.sh multiple-testing --json" in commands_text
    assert "./scripts/ops/opsctl.sh decay-monitor --json" in commands_text
    assert "./scripts/ops/opsctl.sh system-plumbing-control --json" in commands_text
    assert "./scripts/ops/opsctl.sh system-architecture-hardening --apply --json" in commands_text
    assert "tastytrade-sync --json" not in commands_text

    runbook_text = (project_root / "scripts" / "runbook.sh").read_text(encoding="utf-8")
    assert "list_sections()" in runbook_text
    assert "find_section_heading()" in runbook_text
    assert 'refresh) print -r -- "Most Used" ;;' in runbook_text
    assert 'refresh) print -r -- "Live Feed Refreshes" ;;' not in runbook_text
    assert contract_payload["schema_version"] == commands_src.COMMAND_CONTRACT_SCHEMA_VERSION
    assert contract_payload["entry_count"] == 170
    assert "### Review ten-pillar production excellence" in commands_text
    assert "./scripts/ops/opsctl.sh production-excellence --json" in commands_text
    assert "### Freeze or accept a production candidate" in commands_text
    assert "### Verify the durable live-order ledger" in commands_text
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

    accounts = commands_text.split("## Accounts And Positions", 1)[1].split("## Data Context Syncs", 1)[0]
    data_context = commands_text.split("## Data Context Syncs", 1)[1].split("## Event Watches", 1)[0]
    event_watches = commands_text.split("## Event Watches", 1)[1].split("## Live Feed Views", 1)[0]
    live_views = commands_text.split("## Live Feed Views", 1)[1].split("## Notifications And Alerts", 1)[0]
    notifications = commands_text.split("## Notifications And Alerts", 1)[1].split("## Paper Trading", 1)[0]
    paper_trading = commands_text.split("## Paper Trading", 1)[1].split("## Reports And PDFs", 1)[0]
    reports = commands_text.split("## Reports And PDFs", 1)[1].split("## Retrain", 1)[0]
    retrain = commands_text.split("## Retrain", 1)[1].split("## Schwab Auth", 1)[0]
    schwab_auth = commands_text.split("## Schwab Auth", 1)[1].split("## SQL And Reports", 1)[0]
    status_health = commands_text.split("## Status And Health", 1)[1].split("## Storage", 1)[0]

    assert "### Heavy operator livefeed view" in live_views
    assert "./scripts/ops/opsctl.sh feed --source main --heavy --no-heavy-ttl --color --red-actions" in live_views
    assert "red-dominant while leaving `BUY` green and `SELL` red" in live_views
    assert "decisions plus important storage, backpressure, auth, halt, and alert messages" in live_views
    assert "unreadable logs are skipped and counted" in live_views
    assert "### Heavy live feed with file diagnostics" in live_views
    assert "./scripts/ops/opsctl.sh feed --source main --heavy --show-files --no-heavy-ttl --color --red-actions" in live_views
    assert "without the pressure-relief heavy-feed TTL" in live_views
    assert "### Heavy infrastructure live feed view" not in live_views
    assert "./scripts/ops/opsctl.sh feed --source infra --heavy --lines 160" not in live_views
    assert "### Heavy Schwab live feed view" not in live_views
    assert "### Heavy Coinbase live feed view" not in live_views
    assert "### Heavy futures live feed view" not in live_views
    assert "### Heavy FX live feed view" not in live_views
    assert "### Light live feed tail for all feeds" not in live_views
    assert "main-tail --lines 80" not in commands_text
    assert "coinbase-tail --lines 80" not in commands_text
    assert "futures-tail --lines 80" not in commands_text
    assert "livefeed-refresh --force-restart" not in commands_text
    assert "jq '.' governance/health/livefeed_local_latest.json" not in commands_text
    assert "### Refresh Schwab account positions" in accounts
    assert "./scripts/ops/opsctl.sh schwab-account-snapshot-refresh --json" in accounts
    assert "### Study all visible account positions" in accounts
    assert "./scripts/ops/opsctl.sh account-position-study --json" in accounts
    assert "### Watch covered-call roll windows" in accounts
    assert "./scripts/ops/opsctl.sh covered-call-roll-watch --json" in accounts
    assert "### Review account policy context" in accounts
    assert "### Install the SpaceX/SPCX IPO downside watcher" in event_watches
    assert "./scripts/ops/opsctl.sh spacex-ipo-watch-install --poll-seconds 30 --symbol SPCX" in event_watches
    assert "### Run the SpaceX/SPCX downside watch once" in event_watches
    assert "./scripts/ops/opsctl.sh spacex-ipo-watch --json" in event_watches
    assert "### Run macro event intelligence" in event_watches
    assert "### Send a test iMessage notification" in notifications
    assert "./scripts/ops/opsctl.sh notify-test --enable-imessage" in notifications
    assert "### Start the Mac notification and iMessage watcher" in notifications
    assert "./scripts/ops/opsctl.sh notify-start --enable-imessage" in notifications
    assert "### Install the startup Yes/No bot start prompt" in notifications
    assert "./scripts/ops/opsctl.sh startup-start-prompt --install --no-kickstart --no-browser" in notifications
    assert "### Dry-run the startup Yes/No bot start prompt" in notifications
    assert "./scripts/ops/opsctl.sh startup-start-prompt-test --dry-run --delay-seconds 0" in notifications
    assert "### Review remote alert control" in notifications
    assert "### Arm or candidate-promote the guarded 400 bot paper ramp" in paper_trading
    assert "./scripts/ops/opsctl.sh paper-400-ramp --apply --promote-roster --json" in paper_trading
    assert "### Check paper runtime regression guard" in paper_trading
    assert "### Apply the paper live-data standard" in paper_trading
    assert "### Apply paper profitability controls" in paper_trading
    assert "### Schwab authorization refresh" in schwab_auth
    assert "### Schwab auth supervisor" in schwab_auth
    assert "./scripts/ops/opsctl.sh schwab-auth-supervisor --json" in schwab_auth
    assert "### Local Schwab credential setup" in schwab_auth
    assert "./scripts/ops/opsctl.sh schwab-credentials --interactive --store keychain --json" in schwab_auth
    assert "does not open Chrome or a headless browser" in schwab_auth
    assert "### Interactive Schwab authorization re-consent" in schwab_auth
    assert "### Schwab auth recovery plus lane restart" in schwab_auth
    assert "### Refresh runtime dashboard contracts" in status_health
    assert "./scripts/ops/opsctl.sh dashboard-refresh" in status_health
    assert "### Runtime gate dashboard" in status_health
    assert "./scripts/ops/opsctl.sh dashboard" in status_health
    assert "### Review the cross-system drift mesh" in status_health
    assert "./scripts/ops/opsctl.sh system-drift-guard --json" in status_health
    assert "### Review system plumbing control" in status_health
    assert "./scripts/ops/opsctl.sh system-plumbing-control --json" in status_health
    assert "### Repair safe cross-system drift surfaces" in status_health
    assert "./scripts/ops/opsctl.sh system-drift-autopilot --apply --json" in status_health
    assert "### Apply system architecture hardening" in status_health
    assert "./scripts/ops/opsctl.sh system-architecture-hardening --apply --json" in status_health
    assert "### Master infrastructure supervisor" in status_health
    assert "./scripts/ops/opsctl.sh master-infra-supervisor --json" in status_health
    assert "### Adapt infrabots to current system needs" in status_health
    assert "./scripts/ops/opsctl.sh infrabot-adaptive-governor --apply --json" in status_health
    assert "### Docs, commands, and reporting intelligence" in status_health
    assert "./scripts/ops/opsctl.sh docs-reporting-intelligence --apply --json" in status_health
    assert "### PyCharm active bot blue highlights" in status_health
    assert "./scripts/ops/opsctl.sh pycharm-active-bot-highlights --apply --json" in status_health
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
    assert "### Training and labeling intelligence" in retrain
    assert "### Force full retrain (bypass prechecks)" in retrain
    assert "### Review or prune eligible local standby SQLite copies after BOT_LOGS soak" in commands_text
    assert "### Refresh showcase, framework map, and PDFs now" in reports
    assert "### Install nightly showcase and PDF refresh" not in reports
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
