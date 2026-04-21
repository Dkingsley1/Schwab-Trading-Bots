#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "commands_hygiene_latest.json"
LEGACY_RUNBOOK_ALIASES = {
    "live": "Live Feed Views",
    "refresh": "Live Feed Refreshes",
    "health": "Status And Health",
    "retrain": "Retrain",
    "analysis": "Strategy Research",
    "reports": "Reports And PDFs",
    "halts": "Status And Health",
    "sim-paper": "Most Used",
}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _trim_blank_edges(lines: list[str]) -> list[str]:
    out = list(lines)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def _normalize_key(raw: str) -> str:
    return " ".join(str(raw or "").strip().lower().split())


def _normalize_code(raw: str) -> str:
    return "\n".join(line.rstrip() for line in str(raw or "").strip().splitlines())


def _extract_first_code_block(lines: list[str]) -> str:
    in_block = False
    block: list[str] = []
    for line in lines:
        if line.startswith("```"):
            if in_block:
                break
            in_block = True
            continue
        if in_block:
            block.append(line)
    return "\n".join(block).strip()


def _entry_fingerprint(entry: dict[str, Any]) -> tuple[str, str]:
    title_key = _normalize_key(str(entry.get("title") or ""))
    code_key = _normalize_code(_extract_first_code_block(list(entry.get("lines") or [])))
    if code_key:
        return title_key, code_key
    body_key = _normalize_key("\n".join(str(line or "") for line in list(entry.get("lines") or [])))
    return title_key, body_key


def _parse_commands_sections(text: str) -> tuple[list[str], list[dict[str, Any]]]:
    preamble: list[str] = []
    sections: list[dict[str, Any]] = []
    current_section: dict[str, Any] | None = None
    current_entry: dict[str, Any] | None = None

    def flush_entry() -> None:
        nonlocal current_entry
        if current_entry is None or current_section is None:
            current_entry = None
            return
        current_entry["lines"] = _trim_blank_edges(list(current_entry.get("lines") or []))
        current_section["entries"].append(current_entry)
        current_entry = None

    def flush_section() -> None:
        nonlocal current_section
        if current_section is None:
            return
        current_section["intro_lines"] = _trim_blank_edges(list(current_section.get("intro_lines") or []))
        sections.append(current_section)
        current_section = None

    for raw in text.splitlines():
        if raw.startswith("## "):
            flush_entry()
            flush_section()
            current_section = {
                "heading": raw[3:].strip(),
                "intro_lines": [],
                "entries": [],
            }
            continue
        if raw.startswith("### ") and current_section is not None:
            flush_entry()
            current_entry = {
                "title": raw[4:].strip(),
                "lines": [raw],
            }
            continue
        if current_entry is not None:
            current_entry["lines"].append(raw)
        elif current_section is not None:
            current_section["intro_lines"].append(raw)
        else:
            preamble.append(raw)

    flush_entry()
    flush_section()
    return _trim_blank_edges(preamble), sections


def _section(heading: str, *entries: dict[str, Any], intro_lines: Iterable[str] = ()) -> dict[str, Any]:
    return {
        "heading": heading,
        "intro_lines": [str(line) for line in intro_lines],
        "entries": list(entries),
    }


def _command_entry(
    project_root: Path,
    title: str,
    command_lines: Iterable[str],
    *,
    notes: Iterable[str] = (),
) -> dict[str, Any]:
    lines = [
        f"### {title}",
        "```bash",
        f"cd {project_root}",
        *[str(line) for line in command_lines],
        "```",
    ]
    note_lines = [str(line) for line in notes if str(line).strip()]
    if note_lines:
        lines.extend(["", *note_lines])
    return {"title": title, "lines": lines}


def _open_path_entry(
    project_root: Path,
    title: str,
    target_path: Path,
    *,
    notes: Iterable[str] = (),
) -> dict[str, Any]:
    return _command_entry(project_root, title, [f"open {target_path}"], notes=notes)


def _commands_inventory(project_root: Path) -> list[dict[str, Any]]:
    bot_stack_pdf_path = project_root / "exports" / "bot_stack_status" / "latest.pdf"
    report_bundle_pdf_path = project_root / "exports" / "reports" / "report_pdf_bundle_latest.pdf"
    daily_ops_pdf_path = project_root / "exports" / "reports" / "daily_ops_report_latest.pdf"
    paper_performance_pdf_path = project_root / "exports" / "reports" / "paper_performance_latest.pdf"
    sentiment_pdf_path = project_root / "exports" / "reports" / "sentiment_report_latest.pdf"
    strategy_attribution_pdf_path = project_root / "exports" / "reports" / "strategy_attribution_latest.pdf"
    post_trade_analysis_pdf_path = project_root / "exports" / "reports" / "post_trade_analysis_latest.pdf"
    crash_report_pdf_path = project_root / "exports" / "reports" / "crash_reports" / "crash_report_digest_latest.pdf"
    project_timeline_pdf_path = project_root / "exports" / "reports" / "project_timeline" / "project_timeline_latest.pdf"
    training_report_pdf_path = project_root / "exports" / "reports" / "training_reports" / "training_report_latest.pdf"
    macro_crosscheck_pdf_path = project_root / "exports" / "reports" / "macro_crosscheck_latest.pdf"
    market_correlation_pdf_path = project_root / "exports" / "reports" / "market_crypto_correlation_latest.pdf"
    source_verification_pdf_path = project_root / "exports" / "reports" / "source_verification_latest.pdf"
    retrain_scorecard_pdf_path = project_root / "exports" / "sql_reports" / "retrain_scorecard_latest.pdf"
    daily_runtime_summary_pdf_path = project_root / "exports" / "sql_reports" / "daily_runtime_summary_latest.pdf"
    daily_auto_verify_pdf_path = project_root / "exports" / "sql_reports" / "daily_auto_verify_latest.pdf"
    model_card_pdf_path = project_root / "exports" / "sql_reports" / "model_card_latest.pdf"
    paper_calibration_pdf_path = project_root / "exports" / "sql_reports" / "paper_execution_calibration_latest.pdf"
    replay_ablation_pdf_path = project_root / "exports" / "sql_reports" / "replay_feature_ablation_latest.pdf"
    one_numbers_pdf_path = project_root / "exports" / "one_numbers" / "one_numbers_latest.pdf"
    one_numbers_csv_path = project_root / "exports" / "one_numbers" / "latest.csv"
    state_snapshot_pdf_path = project_root / "exports" / "state_snapshot_drills" / "state_snapshot_drills_latest.pdf"
    unified_lane_pdf_path = project_root / "exports" / "sql_reports" / "unified_lane_scorecard_latest.pdf"
    bot_explainability_pdf_path = project_root / "exports" / "sql_reports" / "bot_explainability_latest.pdf"
    report_pdf_open_entries = [
        _open_path_entry(project_root, "Open the report catalog PDF", report_bundle_pdf_path),
        _open_path_entry(project_root, "Open the daily ops PDF", daily_ops_pdf_path),
        _open_path_entry(project_root, "Open the paper performance PDF", paper_performance_pdf_path),
        _open_path_entry(project_root, "Open the sentiment PDF", sentiment_pdf_path),
        _open_path_entry(project_root, "Open the strategy attribution PDF", strategy_attribution_pdf_path),
        _command_entry(
            project_root,
            "Open the post-trade analysis PDF",
            ["./scripts/ops/open_report_artifact.sh posttrade"],
            notes=[
                "This refreshes the post-trade analysis source, renders the report PDF bundle, prefers the PDF artifact, and falls back to printable HTML or markdown if the PDF renderer is unavailable."
            ],
        ),
        _command_entry(
            project_root,
            "Open the crash digest PDF",
            ["./scripts/ops/open_report_artifact.sh crash"],
            notes=[
                "This regenerates the crash digest with a 30-day lookback by default, prefers the PDF artifact, and falls back to printable HTML if the PDF renderer is unavailable."
            ],
        ),
        _command_entry(
            project_root,
            "Open the project timeline PDF",
            ["./scripts/ops/open_report_artifact.sh timeline"],
            notes=[
                "This regenerates the timeline report, prefers the PDF artifact, and falls back to printable HTML if the PDF renderer is unavailable."
            ],
        ),
        _open_path_entry(project_root, "Open the training report PDF", training_report_pdf_path),
        _open_path_entry(project_root, "Open the macro crosscheck PDF", macro_crosscheck_pdf_path),
        _open_path_entry(project_root, "Open the market correlation PDF", market_correlation_pdf_path),
        _open_path_entry(project_root, "Open the source verification PDF", source_verification_pdf_path),
        _open_path_entry(project_root, "Open the retrain scorecard PDF", retrain_scorecard_pdf_path),
        _open_path_entry(project_root, "Open the daily runtime summary PDF", daily_runtime_summary_pdf_path),
        _open_path_entry(project_root, "Open the daily auto verify PDF", daily_auto_verify_pdf_path),
        _open_path_entry(project_root, "Open the model card PDF", model_card_pdf_path),
        _open_path_entry(project_root, "Open the paper execution calibration PDF", paper_calibration_pdf_path),
        _open_path_entry(project_root, "Open the replay feature ablation PDF", replay_ablation_pdf_path),
        _open_path_entry(project_root, "Open the one numbers PDF", one_numbers_pdf_path),
        _open_path_entry(project_root, "Open the state snapshot drills PDF", state_snapshot_pdf_path),
        _command_entry(
            project_root,
            "Open the active bot stack PDF",
            ["./scripts/ops/open_report_artifact.sh botstack"],
            notes=[
                "This refreshes the bot stack report, prefers the PDF artifact, and falls back to HTML or markdown if the PDF renderer is unavailable."
            ],
        ),
        _open_path_entry(
            project_root,
            "Open the unified lane scorecard PDF",
            unified_lane_pdf_path,
            notes=["This one is on-demand and only exists after it has been generated."],
        ),
        _open_path_entry(
            project_root,
            "Open the bot explainability PDF",
            bot_explainability_pdf_path,
            notes=["This one is on-demand and only exists after it has been generated."],
        ),
    ]
    return [
        _section(
            "Most Used",
            _command_entry(project_root, "Keep the Mac awake", ["caffeinate -dimsu"]),
            _command_entry(project_root, "Start the full live stack", ["./scripts/ops/opsctl.sh start"]),
            _command_entry(
                project_root,
                "Brain switch: launch the mode switchboard",
                ['PY="$(zsh ./scripts/ops/runtime_python.sh)"', 'SWITCHBOARD_MODES="shadow,paper" "$PY" scripts/run_mode_switchboard.py'],
                notes=[
                    "Valid modes are `shadow`, `paper`, and `live`.",
                    "This launches one `main.py` child per mode and sets `BOT_MODE` automatically.",
                ],
            ),
            _command_entry(
                project_root,
                "Phone mirror view for the live feed",
                ["./scripts/ops/opsctl.sh phone-feed --host 0.0.0.0 --source all --include-decisions"],
                notes=[
                    "This starts the phone-friendly live feed mirror and prints the local and Tailscale URLs in the terminal.",
                    "When `--host 0.0.0.0` is used without `--token`, the server auto-generates a remote-access token for you.",
                ],
            ),
            _open_path_entry(project_root, "Open the One Numbers CSV in Numbers", one_numbers_csv_path),
            _open_path_entry(project_root, "Open the One Numbers PDF", one_numbers_pdf_path),
            _command_entry(
                project_root,
                "Broker Truth Step 1: refresh Schwab auth",
                ["./scripts/ops/opsctl.sh token-refresh --always-auth"],
                notes=[
                    "Use this first when broker-truth lanes start showing transient 403s or auth churn.",
                ],
            ),
            _command_entry(
                project_root,
                "Broker Truth Step 2: restart the Schwab loops",
                ["./scripts/ops/opsctl.sh feed-refresh --source schwab"],
                notes=[
                    "This forces the Schwab sleeves to pick up the refreshed token and republish their latest broker-truth snapshots.",
                ],
            ),
            _command_entry(
                project_root,
                "Broker Truth Step 3: verify broker readiness and lane statuses",
                [
                    """/Users/dankingsley/PycharmProjects/schwab_trading_bot/.venv312/bin/python -c "from pathlib import Path; import json; root=Path('/Users/dankingsley/PycharmProjects/schwab_trading_bot/governance/health'); broker=json.loads((root/'broker_readiness_latest.json').read_text()); print(f'ready_for_open={broker.get(\\\"ready_for_open\\\")} auth_ok={broker.get(\\\"auth_ok\\\")} token_warning_level={broker.get(\\\"token_warning_level\\\")}'); print('lane,status,mismatch_count,error'); [print(f'{p.name.replace(\\\"broker_truth_\\\", \\\"\\\").replace(\\\"_latest.json\\\", \\\"\\\")},{json.loads(p.read_text()).get(\\\"status\\\", \\\"\\\")},{int(json.loads(p.read_text()).get(\\\"mismatch_count\\\", 0) or 0)},{json.loads(p.read_text()).get(\\\"error\\\") or \\\"\\\"}') for p in sorted(root.glob('broker_truth_*_latest.json')) if 'shared_snapshot' not in p.name]\""""
                ],
                notes=[
                    "Healthy target: `ready_for_open=True`, `auth_ok=True`, and all Schwab broker-truth lanes reporting `status=ok` with `mismatch_count=0`.",
                ],
            ),
            _command_entry(
                project_root,
                "Refresh the live loops without reinstalling the stack watchdog",
                ["./scripts/ops/opsctl.sh feed-refresh --source all"],
                notes=[
                    "`feed-refresh` is a live-loop restart helper, not a passive data-context sync. It kills and restarts the relevant market-data loops. If you want a full supervised stack refresh instead of a feed-loop refresh, use `./scripts/ops/opsctl.sh start --force-restart`.",
                ],
            ),
            _command_entry(project_root, "Stop the stack", ["./scripts/ops/opsctl.sh stop"]),
            _command_entry(project_root, "Validate documented commands", ["./scripts/ops/opsctl.sh command-validity --json"]),
        ),
        _section(
            "Storage",
            _command_entry(project_root, "Switch collection to the Mac's internal drive", ["./scripts/ops/opsctl.sh storage-switch-local"]),
            _command_entry(project_root, "Switch collection back to the external BOT_LOGS drive", ["./scripts/ops/opsctl.sh storage-switch-external"]),
            _command_entry(project_root, "Safe-eject the external BOT_LOGS drive", ["./scripts/ops/opsctl.sh storage-safe-eject"]),
        ),
        _section(
            "Live Feed Refreshes",
            _command_entry(project_root, "Refresh Schwab equities, Schwab futures, and FX", ["./scripts/ops/opsctl.sh feed-refresh --source schwab"]),
            _command_entry(project_root, "Refresh Coinbase spot and Coinbase futures", ["./scripts/ops/opsctl.sh feed-refresh --source coinbase"]),
            _command_entry(project_root, "Refresh FX only", ["./scripts/ops/opsctl.sh feed-refresh --source fx"]),
        ),
        _section(
            "Live Feed Views",
            _command_entry(
                project_root,
                "Heavy live feed view across all feeds and decisions",
                ["./scripts/ops/opsctl.sh feed --source all --include-decisions"],
                notes=[
                    "Use this as the primary all-feeds operator view when you want the broad multi-feed tail plus decision-stream context in one window.",
                    "If the Mac is running an `air_safe` or `constrained` memory-efficiency profile, the feed automatically trims decision fanout and uses a lower default line budget unless you pass your own `--lines` or `--no-memory-aware`.",
                ],
            ),
            _command_entry(project_root, "Light live feed tail for all feeds", ["./scripts/ops/opsctl.sh feed --source all --lines 80"]),
            _command_entry(project_root, "Live feed tail for Schwab, Coinbase, and futures", ["./scripts/ops/opsctl.sh main-tail --lines 80"]),
            _command_entry(project_root, "Live feed tail for Schwab", ["./scripts/ops/opsctl.sh schwab-tail --lines 80"]),
            _command_entry(project_root, "Live feed tail for Coinbase", ["./scripts/ops/opsctl.sh coinbase-tail --lines 80"]),
            _command_entry(project_root, "Live feed tail for all futures sleeves", ["./scripts/ops/opsctl.sh futures-tail --lines 80"]),
            _command_entry(project_root, "Live feed tail for Schwab futures", ["./scripts/ops/opsctl.sh schwab-futures-tail --lines 80"]),
            _command_entry(project_root, "Live feed tail for Coinbase futures", ["./scripts/ops/opsctl.sh coinbase-futures-tail --lines 80"]),
            _command_entry(project_root, "Live feed tail for FX", ["./scripts/ops/opsctl.sh fx-tail --lines 80"]),
        ),
        _section(
            "Schwab Auth",
            _command_entry(
                project_root,
                "Schwab authorization refresh",
                ["./scripts/ops/opsctl.sh token-refresh --always-auth"],
                notes=[
                    "Use this when the Schwab browser grant is stale or broker-truth lanes start showing auth churn.",
                ],
            ),
            _command_entry(
                project_root,
                "Interactive Schwab authorization re-consent",
                ["./scripts/ops/opsctl.sh token-refresh-interactive"],
                notes=[
                    "Run this when you want to force the browser-based Schwab authorization flow directly.",
                ],
            ),
            _command_entry(
                project_root,
                "Schwab auth recovery plus lane restart",
                [
                    "./scripts/ops/opsctl.sh token-refresh --always-auth",
                    "./scripts/ops/opsctl.sh feed-refresh --source schwab",
                ],
                notes=[
                    "This is the paste-ready recovery pair when refreshed authorization needs to be picked up by the Schwab loops immediately.",
                ],
            ),
            intro_lines=[
                "Use these exact Schwab authorization commands when tokens expire, browser consent needs renewal, or broker-truth lanes start surfacing 401/403 errors.",
            ],
        ),
        _section(
            "Status And Health",
            _command_entry(project_root, "Runtime status", ["./scripts/ops/opsctl.sh status"]),
            _command_entry(project_root, "Health snapshot", ["./scripts/ops/opsctl.sh health"]),
            _command_entry(project_root, "Doctor", ["./scripts/ops/opsctl.sh doctor"]),
        ),
        _section(
            "SQL And Reports",
            _command_entry(
                project_root,
                "Full SQL refresh pipeline",
                ["./scripts/daily_log_refresh.sh"],
                notes=[
                    "Use this when you want the full SQL/log/report refresh instead of the one-pass writer sync.",
                ],
            ),
            _command_entry(project_root, "Quick SQL sync", ["./scripts/ops/opsctl.sh sql-sync"]),
            _command_entry(
                project_root,
                "Data quality refresh bundle",
                [
                    "./scripts/ops/opsctl.sh feed-refresh --source all",
                    "./scripts/daily_log_refresh.sh",
                    'PY="$(zsh ./scripts/ops/runtime_python.sh)"',
                    '"$PY" scripts/build_one_numbers_report.py',
                ],
                notes=[
                    "Use this when One Numbers is stale or you want the latest data-quality averages and report artifacts refreshed together.",
                ],
            ),
        ),
        _section(
            "Retrain",
            _command_entry(
                project_root,
                "Full retrain preflight",
                [
                    "./scripts/daily_log_refresh.sh",
                    "./scripts/ops/opsctl.sh runtime-training-snapshot --json",
                    "./scripts/ops/opsctl.sh coverage-seed --write-queue --json",
                    "./scripts/ops/opsctl.sh coverage-gap-closer --apply-stage --launch --json",
                    'PY="$(zsh ./scripts/ops/runtime_python.sh)"',
                    '"$PY" scripts/retrain_schema_compatibility_guard.py --json',
                    '"$PY" scripts/promotion_quality_gate.py --json',
                ],
                notes=[
                    "Run this before a manual full retrain so SQL state, runtime snapshots, coverage, and promotion gates are fresh.",
                ],
            ),
            _command_entry(
                project_root,
                "Guarded retrain orchestrator",
                ["./scripts/ops/opsctl.sh retrain-orchestrate --json"],
                notes=[
                    "This is the safer manual retrain entrypoint because it refreshes stale artifacts and honors freshness checks before launching weekly retrain.",
                ],
            ),
            _command_entry(
                project_root,
                "Force full retrain (bypass prechecks)",
                ["./scripts/ops/opsctl.sh retrain-force-full"],
                notes=[
                    "Use this only when you intentionally want to bypass the normal data-quality, freshness, snapshot-sync, and sample-quota prechecks.",
                ],
            ),
            intro_lines=[
                "Use these commands when you are preparing or launching a manual retrain cycle.",
            ],
        ),
        _section(
            "Reports And PDFs",
            _command_entry(project_root, "One Numbers report", ['PY="$(zsh ./scripts/ops/runtime_python.sh)"', '"$PY" scripts/build_one_numbers_report.py']),
            _command_entry(project_root, "Paper performance report", ['./scripts/ops/opsctl.sh paper-performance --day "$(date -u +%Y%m%d)" --week-days 7 --json']),
            _command_entry(project_root, "Report catalog bundle", ["./scripts/ops/opsctl.sh report-pdfs --json"]),
            _command_entry(
                project_root,
                "Active bot stack PDF",
                ["./scripts/ops/opsctl.sh bot-stack-report --top 25 --render-pdf --allow-gui-pdf-renderer"],
                notes=[f"Latest PDF path: `{bot_stack_pdf_path}`."],
            ),
            *report_pdf_open_entries,
            intro_lines=[
                "This section includes the generate commands plus direct open commands for each report PDF.",
            ],
        ),
        _section(
            "Data Context Syncs",
            _command_entry(
                project_root,
                "Options flow context sync",
                ["./scripts/ops/opsctl.sh options-flow-sync --json"],
                notes=["`options-flow-sync` is the canonical command. `tastytrade-sync` remains a legacy alias for backward compatibility."],
            ),
            _command_entry(project_root, "Crypto market context sync", ["./scripts/ops/opsctl.sh crypto-market-sync --json"]),
            _command_entry(project_root, "Stock / crypto correlation sync", ["./scripts/ops/opsctl.sh market-correlation-sync --json"]),
            _command_entry(project_root, "FX market context sync", ["./scripts/ops/opsctl.sh fx-market-sync --json"]),
            _command_entry(project_root, "Macro context sync", ["./scripts/ops/opsctl.sh macro-context-sync --json"]),
            _command_entry(project_root, "Source verification", ["./scripts/ops/opsctl.sh source-verification --json"]),
        ),
        _section(
            "Macro And Media",
            _command_entry(
                project_root,
                "Start the macro auto-watch lane",
                ['./scripts/ops/opsctl.sh macro-auto-start --force-restart --youtube-channel-url "https://www.youtube.com/@federalreserve" --template fed --speaker "Federal Reserve" --source "Federal Reserve"'],
            ),
            _command_entry(project_root, "Show macro auto-watch status", ["./scripts/ops/opsctl.sh macro-auto-status --json"]),
        ),
    ]


def render_commands_markdown(project_root: Path = PROJECT_ROOT) -> str:
    preamble = [
        "# Commands (Canonical)",
        "",
        "Use these exact commands as the current source of truth.",
        "",
        "This file is generated from the curated operator inventory in `scripts/ops/commands_hygiene_bot.py`.",
        "Rebuild it with `./scripts/ops/opsctl.sh commands-hygiene --apply` after changing that inventory.",
        "",
        "This file is intentionally trimmed down:",
        "- paper mode is the operating default",
        "- no simulate variants are listed",
        "- no duplicate restart commands are listed when a broader command already covers them",
    ]
    parts = ["\n".join(preamble)]
    for section in _commands_inventory(project_root):
        blocks = [f"## {section['heading']}"]
        intro_text = "\n".join(_trim_blank_edges(list(section.get("intro_lines") or []))).strip()
        if intro_text:
            blocks.append(intro_text)
        for entry in list(section.get("entries") or []):
            blocks.append("\n".join(_trim_blank_edges(list(entry.get("lines") or []))).rstrip())
        parts.append("\n\n".join(block for block in blocks if block))
    rendered = "\n\n".join(part for part in parts if part).rstrip()
    return rendered + "\n"


def _source_duplicate_entry_count(sections: list[dict[str, Any]]) -> int:
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for section in sections:
        for entry in list(section.get("entries") or []):
            fingerprint = _entry_fingerprint(entry)
            if fingerprint in seen:
                duplicates += 1
            else:
                seen.add(fingerprint)
    return duplicates


def clean_commands_markdown(text: str, *, project_root: Path) -> tuple[str, dict[str, int]]:
    _, before_sections = _parse_commands_sections(text)
    desired_commands = render_commands_markdown(project_root)
    _, after_sections = _parse_commands_sections(desired_commands)
    before_entry_count = sum(len(list(section.get("entries") or [])) for section in before_sections)
    after_entry_count = sum(len(list(section.get("entries") or [])) for section in after_sections)
    return desired_commands, {
        "section_count_before": len(before_sections),
        "section_count_after": len(after_sections),
        "entry_count_before": before_entry_count,
        "entry_count_after": after_entry_count,
        "duplicate_entry_count": _source_duplicate_entry_count(before_sections),
        "empty_section_count": sum(
            1
            for section in before_sections
            if not _trim_blank_edges(list(section.get("intro_lines") or []))
            and not list(section.get("entries") or [])
        ),
        "keep_awake_canonicalized": 0,
        "options_flow_canonicalized": 0,
        "feed_refresh_canonicalized": 0,
        "report_sections_merged": 0,
        "workspace_hygiene_removed": 0,
        "report_entries_rehomed": 0,
        "authored_from_inventory": int(desired_commands != text),
    }


def render_runbook_script() -> str:
    return """#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNBOOK="$PROJECT_ROOT/COMMANDS.md"

if [[ ! -f "$RUNBOOK" ]]; then
  echo "Missing runbook: $RUNBOOK"
  exit 1
fi

slugify() {
  print -r -- "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

list_sections() {
  awk '/^## / { print substr($0, 4) }' "$RUNBOOK"
}

resolve_section() {
  local raw="${1:-}"
  case "$raw" in
    live) print -r -- "Live Feed Views" ;;
    refresh) print -r -- "Live Feed Refreshes" ;;
    health) print -r -- "Status And Health" ;;
    retrain) print -r -- "Retrain" ;;
    analysis) print -r -- "Strategy Research" ;;
    reports) print -r -- "Reports And PDFs" ;;
    halts) print -r -- "Status And Health" ;;
    sim-paper) print -r -- "Most Used" ;;
    *) print -r -- "$raw" ;;
  esac
}

find_section_heading() {
  local requested="$1"
  local resolved
  resolved="$(resolve_section "$requested")"
  while IFS= read -r heading; do
    [[ -n "$heading" ]] || continue
    if [[ "$heading" == "$resolved" || "$(slugify "$heading")" == "$requested" || "$(slugify "$heading")" == "$(slugify "$resolved")" ]]; then
      print -r -- "$heading"
      return 0
    fi
  done < <(list_sections)
  return 1
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/runbook.sh                 # show section list
  ./scripts/runbook.sh all             # show full COMMANDS.md
  ./scripts/runbook.sh <section>

Sections:
EOF
  while IFS= read -r heading; do
    [[ -n "$heading" ]] || continue
    echo "  $(slugify "$heading")"
  done < <(list_sections)
}

extract_heading() {
  local heading="$1"
  awk -v h="$heading" '
    $0 == "## " h { show=1; print; next }
    /^## / && show == 1 { exit }
    show == 1 { print }
  ' "$RUNBOOK"
}

if [[ $# -eq 0 ]]; then
  usage
  exit 0
fi

case "$1" in
  all)
    cat "$RUNBOOK"
    ;;
  *)
    if ! heading="$(find_section_heading "$1")"; then
      echo "Unknown section: $1"
      usage
      exit 2
    fi
    extract_heading "$heading"
    ;;
esac
"""


def build_payload(project_root: Path = PROJECT_ROOT, *, apply: bool = False) -> dict[str, Any]:
    commands_path = project_root / "COMMANDS.md"
    runbook_path = project_root / "scripts" / "runbook.sh"
    commands_text = _read_text(commands_path)
    runbook_text = _read_text(runbook_path)
    authored_commands_text, metrics = clean_commands_markdown(commands_text, project_root=project_root)
    desired_runbook_text = render_runbook_script()

    commands_changed = authored_commands_text != commands_text
    runbook_changed = desired_runbook_text != runbook_text

    issues: list[str] = []
    if commands_changed:
        issues.append("commands_authored_from_inventory")
    if runbook_changed:
        issues.append("runbook_heading_drift")

    apply_results = {
        "commands_md_written": False,
        "runbook_written": False,
    }

    if apply and commands_changed:
        commands_path.parent.mkdir(parents=True, exist_ok=True)
        commands_path.write_text(authored_commands_text, encoding="utf-8")
        apply_results["commands_md_written"] = True
    if apply and runbook_changed:
        runbook_path.parent.mkdir(parents=True, exist_ok=True)
        runbook_path.write_text(desired_runbook_text, encoding="utf-8")
        current_mode = 0o755
        try:
            current_mode = runbook_path.stat().st_mode
        except Exception:
            pass
        runbook_path.chmod(current_mode | 0o111)
        apply_results["runbook_written"] = True

    overall_status = "degraded" if (commands_changed or runbook_changed) else "ready"
    recommended_actions = ordered_unique(
        [
            "run commands-hygiene in apply mode when you want COMMANDS.md re-authored from the curated inventory"
            if commands_changed
            else "",
            "edit scripts/ops/commands_hygiene_bot.py instead of hand-editing COMMANDS.md directly"
            if commands_changed
            else "",
            "let runbook.sh resolve live section slugs dynamically so it follows current headings"
            if runbook_changed
            else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 2,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply": bool(apply),
        "commands_path": str(commands_path),
        "runbook_path": str(runbook_path),
        "commands_changed": commands_changed,
        "runbook_changed": runbook_changed,
        "issues": issues,
        "metrics": metrics,
        "apply_results": apply_results,
        "legacy_runbook_aliases": dict(LEGACY_RUNBOOK_ALIASES),
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Author COMMANDS.md and the runbook helper from the curated operator command inventory.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), apply=bool(args.apply))
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "commands_hygiene_bot "
            f"overall_status={payload.get('overall_status', '')} "
            f"commands_changed={int(bool(payload.get('commands_changed', False)))} "
            f"runbook_changed={int(bool(payload.get('runbook_changed', False)))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
