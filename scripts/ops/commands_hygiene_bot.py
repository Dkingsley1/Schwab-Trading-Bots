#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
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
DEFAULT_CONTRACT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "commands_contract_latest.json"
COMMAND_CONTRACT_SCHEMA_VERSION = 1
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


def _parse_tokens(line: str) -> list[str]:
    try:
        return shlex.split(line)
    except Exception:
        return []


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


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _alphabetized_inventory(sections: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_sections: list[dict[str, Any]] = []
    for section in sections:
        copied = dict(section)
        entries = [dict(entry) for entry in list(copied.get("entries") or [])]
        entries.sort(key=lambda entry: _normalize_key(str(entry.get("title") or "")))
        copied["entries"] = entries
        sorted_sections.append(copied)
    sorted_sections.sort(
        key=lambda section: (
            0 if _normalize_key(str(section.get("heading") or "")) == "most used" else 1,
            _normalize_key(str(section.get("heading") or "")),
        )
    )
    return sorted_sections


def build_command_contract(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    sections = _alphabetized_inventory(_commands_inventory(project_root))
    for section_index, section in enumerate(sections):
        section_name = str(section.get("heading") or "").strip()
        for entry_index, entry in enumerate(list(section.get("entries") or [])):
            title = str(entry.get("title") or "").strip()
            code_block = _extract_first_code_block(list(entry.get("lines") or []))
            normalized_code = _normalize_code(code_block)
            command_lines = [line for line in normalized_code.splitlines() if line.strip()]
            opsctl_subcommands: list[str] = []
            script_paths: list[str] = []
            for line in command_lines:
                tokens = _parse_tokens(line.strip())
                if len(tokens) >= 2 and tokens[0] in {"./scripts/ops/opsctl.sh", "scripts/ops/opsctl.sh"}:
                    opsctl_subcommands.append(tokens[1])
                for token in tokens:
                    if token.startswith("./scripts/") or token.startswith("scripts/"):
                        path = token[2:] if token.startswith("./") else token
                        if Path(path).suffix in {".py", ".sh"}:
                            script_paths.append(path)
            contract_row = {
                "section": section_name,
                "section_index": int(section_index),
                "entry_index": int(entry_index),
                "title": title,
                "command_block": normalized_code,
                "command_lines": command_lines,
                "opsctl_subcommands": ordered_unique(opsctl_subcommands),
                "script_paths": ordered_unique(script_paths),
            }
            contract_row["fingerprint"] = _stable_hash(
                {
                    "section": contract_row["section"],
                    "title": contract_row["title"],
                    "command_block": contract_row["command_block"],
                }
            )
            entries.append(contract_row)

    contract_base = {
        "schema_version": COMMAND_CONTRACT_SCHEMA_VERSION,
        "source": "scripts/ops/commands_hygiene_bot.py:_commands_inventory",
        "entry_count": len(entries),
        "entries": entries,
    }
    return {
        "timestamp_utc": iso_now(),
        **contract_base,
        "contract_hash": _stable_hash(contract_base),
    }


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
    strategy_inventory_pdf_path = project_root / "exports" / "reports" / "strategy_inventory" / "strategy_inventory_latest.pdf"
    expansion_inventory_pdf_path = project_root / "exports" / "reports" / "expansion_inventory" / "expansion_inventory_latest.pdf"
    quant_model_control_pdf_path = project_root / "exports" / "reports" / "quant_model_control" / "quant_model_control_latest.pdf"
    post_trade_analysis_pdf_path = project_root / "exports" / "reports" / "post_trade_analysis_latest.pdf"
    crash_report_pdf_path = project_root / "exports" / "reports" / "crash_reports" / "crash_report_digest_latest.pdf"
    project_timeline_pdf_path = project_root / "exports" / "reports" / "project_timeline" / "project_timeline_latest.pdf"
    system_overview_pdf_path = project_root / "exports" / "reports" / "system_overview" / "system_overview_weekly_platform_history_latest.pdf"
    incident_report_pdf_path = project_root / "exports" / "reports" / "incident_report_latest.pdf"
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
    special_features_pdf_path = project_root / "exports" / "reports" / "showcase" / "special_features_latest.pdf"
    report_pdf_open_entries = [
        _open_path_entry(project_root, "Open the report catalog PDF", report_bundle_pdf_path),
        _open_path_entry(project_root, "Open the daily ops PDF", daily_ops_pdf_path),
        _command_entry(
            project_root,
            "Open the paper performance PDF",
            ["./scripts/ops/open_report_artifact.sh paper"],
            notes=[
                "This refreshes paper-performance data without the GUI renderer, then opens the report-ready chart PDF with daily, weekly, window-change, and sleeve-scoreboard views."
            ],
        ),
        _command_entry(
            project_root,
            "Open the sentiment PDF",
            ["./scripts/ops/open_report_artifact.sh sentiment"],
            notes=[
                "This regenerates the current sentiment report, prefers the PDF artifact, and falls back to HTML or markdown if the PDF renderer is unavailable."
            ],
        ),
        _open_path_entry(project_root, "Open the strategy attribution PDF", strategy_attribution_pdf_path),
        _command_entry(
            project_root,
            "Open the strategy inventory PDF",
            ["./scripts/ops/open_report_artifact.sh strategy-inventory"],
            notes=[
                f"Latest PDF path: `{strategy_inventory_pdf_path}`.",
                "This regenerates the complete sleeve/strategy inventory from the system config, renders the PDF bundle, and opens the report-ready PDF.",
            ],
        ),
        _command_entry(
            project_root,
            "Open the expansion inventory PDF",
            ["./scripts/ops/open_report_artifact.sh expansions"],
            notes=[
                f"Latest PDF path: `{expansion_inventory_pdf_path}`.",
                "This regenerates the expansion list from registry-backed packs and control-plane config files, then opens the report-ready PDF.",
            ],
        ),
        _command_entry(
            project_root,
            "Open the quant model control PDF",
            ["./scripts/ops/opsctl.sh quant-model-control --json", f"open {quant_model_control_pdf_path}"],
            notes=[
                f"Latest PDF path: `{quant_model_control_pdf_path}`.",
                "This refreshes the advanced quant-model feature, MLX, resource-cap, and research-only policy report.",
            ],
        ),
        _command_entry(
            project_root,
            "Open the post-trade analysis PDF",
            ["./scripts/ops/open_report_artifact.sh posttrade"],
            notes=[
                "This refreshes post-trade data with timeout/cached-artifact fallbacks, then opens the report-ready PDF with assessment, calibration, runtime, softguard, and source notes."
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
        _open_path_entry(
            project_root,
            "Open the system overview PDF",
            system_overview_pdf_path,
            notes=[
                "This opens the week-by-week platform history and current-position overview PDF.",
            ],
        ),
        _command_entry(
            project_root,
            "Open the incident report PDF",
            ["./scripts/ops/open_report_artifact.sh incident"],
            notes=[
                "This refreshes the decision-oriented incident report, prefers the PDF artifact, and falls back to HTML or markdown if the PDF renderer is unavailable."
            ],
        ),
        _command_entry(
            project_root,
            "Open the incident review packet PDF",
            ["./scripts/ops/open_report_artifact.sh incident-packet"],
            notes=[
                "This refreshes the immutable incident review packet and opens its compact PDF companion, falling back to the JSON packet if needed."
            ],
        ),
        _command_entry(
            project_root,
            "Open the training report PDF",
            ["./scripts/ops/open_report_artifact.sh training"],
            notes=[
                "This regenerates the training report, prefers the PDF artifact, and falls back to printable HTML or markdown if the PDF renderer is unavailable."
            ],
        ),
        _command_entry(
            project_root,
            "Open the macro crosscheck PDF",
            ["./scripts/ops/open_report_artifact.sh macro"],
            notes=[
                "This regenerates the macro crosscheck source, renders the report PDF bundle, prefers the PDF artifact, and falls back to markdown if the PDF renderer is unavailable."
            ],
        ),
        _command_entry(
            project_root,
            "Open the market correlation PDF",
            ["./scripts/ops/open_report_artifact.sh correlation"],
            notes=[
                "This renders the report PDF bundle, prefers the market-correlation PDF artifact, and falls back to markdown if the PDF renderer is unavailable."
            ],
        ),
        _command_entry(
            project_root,
            "Open the source verification PDF",
            ["./scripts/ops/open_report_artifact.sh source"],
            notes=[
                "This regenerates source verification, renders the report PDF bundle, prefers the PDF artifact, and falls back to markdown if the PDF renderer is unavailable."
            ],
        ),
        _open_path_entry(project_root, "Open the retrain scorecard PDF", retrain_scorecard_pdf_path),
        _open_path_entry(project_root, "Open the daily runtime summary PDF", daily_runtime_summary_pdf_path),
        _command_entry(
            project_root,
            "Open the daily auto verify PDF",
            ["./scripts/ops/open_report_artifact.sh daily-auto-verify"],
            notes=[
                f"Latest PDF path: `{daily_auto_verify_pdf_path}`.",
                "This regenerates daily auto verify, renders the report PDF bundle, prefers the PDF artifact, and falls back to JSON evidence if the PDF is unavailable.",
            ],
        ),
        _command_entry(
            project_root,
            "Open the model card PDF",
            ["./scripts/ops/open_report_artifact.sh modelcard"],
            notes=[
                f"Latest PDF path: `{model_card_pdf_path}`.",
                "This renders the report PDF bundle, prefers the model card PDF, and falls back to JSON evidence if the PDF is unavailable.",
            ],
        ),
        _command_entry(
            project_root,
            "Open the paper execution calibration PDF",
            ["./scripts/ops/open_report_artifact.sh calibration"],
            notes=[
                f"Latest PDF path: `{paper_calibration_pdf_path}`.",
                "This renders the report PDF bundle, prefers the paper execution calibration PDF, and falls back to JSON evidence if the PDF is unavailable.",
            ],
        ),
        _command_entry(
            project_root,
            "Open the replay feature ablation PDF",
            ["./scripts/ops/open_report_artifact.sh replay"],
            notes=[
                "This regenerates the replay feature ablation evidence, renders the report PDF bundle, prefers the PDF artifact, and falls back to the latest JSON evidence if a PDF cannot be rendered."
            ],
        ),
        _open_path_entry(project_root, "Open the state snapshot drills PDF", state_snapshot_pdf_path),
        _command_entry(
            project_root,
            "Open the active bot stack PDF",
            ["./scripts/ops/open_report_artifact.sh botstack"],
            notes=[
                "This refreshes the bot stack report, prefers the PDF artifact, and falls back to HTML or markdown if the PDF renderer is unavailable."
            ],
        ),
        _command_entry(
            project_root,
            "Open the unified lane scorecard PDF",
            ["./scripts/ops/open_report_artifact.sh unified"],
            notes=[
                "This regenerates the unified lane scorecard, renders the report PDF bundle, prefers the PDF artifact, and falls back to markdown if the PDF renderer is unavailable."
            ],
        ),
        _command_entry(
            project_root,
            "Open the bot explainability PDF",
            ["./scripts/ops/open_report_artifact.sh explainability"],
            notes=[
                "This regenerates bot explainability evidence, renders the report PDF bundle, prefers the PDF artifact, and falls back to JSON evidence if the PDF renderer is unavailable."
            ],
        ),
    ]
    return [
        _section(
            "Most Used",
            _command_entry(project_root, "Keep the Mac awake", ["caffeinate -dimsu"]),
            _command_entry(
                project_root,
                "Start the full live stack",
                ["./scripts/ops/opsctl.sh start"],
                notes=[
                    "Use this for the normal supervised start path when the stack is already healthy or only lightly stale.",
                ],
            ),
            _command_entry(
                project_root,
                "Start the full live stack (fresh supervised restart)",
                ["./scripts/ops/opsctl.sh start --force-restart"],
                notes=[
                    "Use this after stale paper lanes, restart storms, or auth recovery so the running stack is rebuilt cleanly.",
                ],
            ),
            _command_entry(
                project_root,
                "Runtime mode switchboard",
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
                "Refresh the special features and framework map reports",
                [
                    "./scripts/ops/opsctl.sh showcase-refresh",
                    "./scripts/ops/opsctl.sh system-explainers",
                    "./scripts/ops/opsctl.sh report-pdfs --json",
                ],
                notes=[
                    "Use this when you want the latest special-features packet and framework-map report regenerated together.",
                ],
            ),
            _command_entry(
                project_root,
                "Open the special features PDF",
                ["./scripts/ops/open_report_artifact.sh special"],
                notes=[
                    "This refreshes the special-features PDF with the deterministic renderer, then opens it.",
                ],
            ),
            _command_entry(
                project_root,
                "Open the framework map PDF",
                ["./scripts/ops/open_report_artifact.sh framework"],
                notes=[
                    "This refreshes the framework-map source, renders a deterministic PDF, and falls back to HTML if the PDF is unavailable.",
                ],
            ),
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
                ["./scripts/ops/opsctl.sh livefeed-refresh"],
                notes=[
                    "`livefeed-refresh` is the all-feeds shortcut for the `feed-refresh` live-loop restart helper. It kills and restarts the relevant market-data loops. Use `./scripts/ops/opsctl.sh livefeed-refresh --dry-run` to validate the route without touching processes. If you want a full supervised stack refresh instead of a feed-loop refresh, use `./scripts/ops/opsctl.sh start --force-restart`.",
                ],
            ),
            _command_entry(
                project_root,
                "Stop the stack",
                ["./scripts/ops/opsctl.sh stop"],
                notes=[
                    "This is the normal supervised stop path. It does not automatically engage an emergency operator halt.",
                ],
            ),
            _command_entry(
                project_root,
                "Emergency stop: engage operator stop and global halt",
                ["./scripts/ops/opsctl.sh operator-control --engage --set-global-halt --reason operator_emergency_stop --json"],
                notes=[
                    "Use this as the red-button stop when you want both the operator stop flag and the global trading halt set immediately.",
                ],
            ),
            _command_entry(
                project_root,
                "Show global halt status and blockers",
                ["./scripts/ops/opsctl.sh global-halt-status --json"],
                notes=[
                    "This prints the current global halt posture, any active halt reasons, and the blockers that still prevent a safe clear.",
                ],
            ),
            _command_entry(
                project_root,
                "Refresh clearable global halt blockers",
                ["./scripts/ops/opsctl.sh global-halt-refresh --json"],
                notes=[
                    "This refreshes the watchdog, auth, data-plane, and runtime-clearance blocker artifacts, then re-evaluates what still prevents a safe clear. It will not release OPERATOR_STOP for you.",
                ],
            ),
            _command_entry(
                project_root,
                "Release operator stop only",
                ["./scripts/ops/opsctl.sh operator-release --json"],
                notes=[
                    "This releases the manual OPERATOR_STOP flag without bypassing the global halt safe-clear checks.",
                ],
            ),
            _command_entry(
                project_root,
                "Clear all halt flags now",
                ["./scripts/ops/opsctl.sh clear-all-halts --json"],
                notes=[
                    "This clears both OPERATOR_STOP and GLOBAL_TRADING_HALT in one command. It is a manual collection-unblock override; it does not mark auth, snapshot recovery, or backpressure gates healthy.",
                ],
            ),
            _command_entry(
                project_root,
                "Attempt a safe global halt clear",
                ["./scripts/ops/opsctl.sh global-halt-auto-clear --json"],
                notes=[
                    "This only clears the halt when the runtime, auth, watchdog, and data-plane guardrails are back inside the safe-clear envelope.",
                ],
            ),
            _command_entry(
                project_root,
                "Run post-restart settlement",
                ["./scripts/ops/opsctl.sh post-restart-settle --apply --json"],
                notes=[
                    "This rechecks restart sanity, auth lease, global halt blockers, collector contracts, process watchdog coverage, and runtime throttle after a restart.",
                ],
            ),
            _command_entry(
                project_root,
                "Fast read-only health check",
                ["./scripts/ops/opsctl.sh health-fast --json"],
                notes=[
                    "This reads the latest health artifacts without starting report refreshes, daily verification, or PDF/report jobs.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply pressure relief controls",
                ["./scripts/ops/opsctl.sh pressure-relief --apply --json"],
                notes=[
                    "This writes the pressure-relief override used by runtime loading, maintenance guards, heavy feed TTL, SQL cadence, foreground-app awareness, macro capture niceness, MLX/quant caps, report caps, and quiet-window behavior.",
                ],
            ),
            _command_entry(project_root, "Validate documented commands", ["./scripts/ops/opsctl.sh command-validity --json"]),
        ),
        _section(
            "Storage",
            _command_entry(project_root, "Switch collection to the Mac's internal drive", ["./scripts/ops/opsctl.sh storage-switch-local"]),
            _command_entry(project_root, "Switch collection back to the external BOT_LOGS drive", ["./scripts/ops/opsctl.sh storage-switch-external"]),
            _command_entry(project_root, "Run the storage disaster recovery bot", ["./scripts/ops/opsctl.sh storage-disaster-recovery --apply --json"]),
            _command_entry(
                project_root,
                "Safe force-clear storage pressure supervisor",
                ["./scripts/ops/opsctl.sh storage-pressure-clearance --apply --force-clear-stale-gate --json"],
                notes=[
                    "This is the parent storage pressure bot. It forces safe refresh/checkpoint/drain actions, but only clears stale storage gates after live WAL and backlog metrics are inside the safe envelope.",
                ],
            ),
            _command_entry(
                project_root,
                "Repair local stateful storage regressions",
                ["./scripts/ops/opsctl.sh stateful-storage-regression-guard --apply --json"],
                notes=[
                    "This guard keeps SQL shards, execution-lane telemetry, and SQL writer launchd logs routed away from the internal disk.",
                ],
            ),
            _command_entry(
                project_root,
                "Review or prune eligible local standby SQLite copies after BOT_LOGS soak",
                ["./scripts/ops/opsctl.sh storage-prune-standby --json"],
                notes=[
                    "This is a dry run by default. Add `--apply` after the external route has soaked long enough to prune only the verified standby copies, or add `--include-curated-standby` if you intentionally want to touch curated standby paths too.",
                ],
            ),
            _command_entry(project_root, "Safe-eject the external BOT_LOGS drive", ["./scripts/ops/opsctl.sh storage-safe-eject"]),
        ),
        _section(
            "Live Feed Refreshes",
            _command_entry(project_root, "Refresh all live feeds", ["./scripts/ops/opsctl.sh livefeed-refresh"]),
            _command_entry(project_root, "Refresh Schwab equities, Schwab futures, and FX", ["./scripts/ops/opsctl.sh feed-refresh --source schwab"]),
            _command_entry(project_root, "Refresh Coinbase spot and Coinbase futures", ["./scripts/ops/opsctl.sh feed-refresh --source coinbase"]),
            _command_entry(project_root, "Refresh FX only", ["./scripts/ops/opsctl.sh feed-refresh --source fx"]),
        ),
        _section(
            "Live Feed Views",
            _command_entry(
                project_root,
                "Heavy live feed view across all sections",
                ["./scripts/ops/opsctl.sh feed --source all --heavy"],
                notes=[
                    "Use this as the primary all-feeds operator view when you want sleeve logs, decision streams, highlighted health states, and infrastructure health artifacts in one window.",
                    "Heavy views use a red-only highlight by default while preserving `[ALERT]`, `[WATCH]`, `[OK]`, and `[FLOW]` labels; pass `--no-color` only when redirecting clean text to a file.",
                    "If the Mac is running an `air_safe` or `constrained` memory-efficiency profile, the feed automatically trims decision fanout and uses a lower default line budget unless you pass your own `--lines` or `--no-memory-aware`.",
                ],
            ),
            _command_entry(project_root, "Heavy infrastructure live feed view", ["./scripts/ops/opsctl.sh feed --source infra --heavy --lines 160"]),
            _command_entry(project_root, "Heavy main live feed view", ["./scripts/ops/opsctl.sh feed --source main --heavy"]),
            _command_entry(project_root, "Heavy Schwab live feed view", ["./scripts/ops/opsctl.sh feed --source schwab --heavy"]),
            _command_entry(project_root, "Heavy Coinbase live feed view", ["./scripts/ops/opsctl.sh feed --source coinbase --heavy"]),
            _command_entry(project_root, "Heavy futures live feed view", ["./scripts/ops/opsctl.sh feed --source futures --heavy"]),
            _command_entry(project_root, "Heavy FX live feed view", ["./scripts/ops/opsctl.sh feed --source fx --heavy"]),
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
                "Schwab auth supervisor",
                [
                    "./scripts/ops/opsctl.sh schwab-auth-supervisor --json",
                    "./scripts/ops/opsctl.sh schwab-auth-supervisor --apply --json",
                ],
                notes=[
                    "Use this first when Schwab auth looks freshly authorized but the system still reports token, callback-port, or browser OAuth drift.",
                    "The apply form cleans up stale Schwab auth helper processes and refreshes the token/lease artifacts without opening a browser loop.",
                ],
            ),
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
                ["./scripts/ops/opsctl.sh token-refresh-interactive --force --prompt-before-browser --json"],
                notes=[
                    "Run this when you need to update the browser handshake after changing credentials, renewing consent, or clearing stale callback/token state.",
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
                "Use these exact Schwab authorization commands when tokens expire, browser consent needs renewal, callback ports get stuck, or broker-truth lanes start surfacing 401/403 errors.",
            ],
        ),
        _section(
            "Status And Health",
            _command_entry(project_root, "Runtime status", ["./scripts/ops/opsctl.sh status"]),
            _command_entry(project_root, "Health snapshot", ["./scripts/ops/opsctl.sh health"]),
            _command_entry(project_root, "Doctor", ["./scripts/ops/opsctl.sh doctor"]),
            _command_entry(
                project_root,
                "Refresh runtime dashboard contracts",
                ["./scripts/ops/opsctl.sh dashboard-refresh"],
                notes=[
                    "This hydrates the runtime gate dashboard prerequisites first so missing sections become explicit health outputs instead of silent omissions.",
                ],
            ),
            _command_entry(
                project_root,
                "Runtime gate dashboard",
                ["./scripts/ops/opsctl.sh dashboard"],
                notes=[
                    "By default this now runs a runtime-artifact refresh pass first. Use `./scripts/ops/opsctl.sh dashboard --skip-refresh` when you want a pure read of the current artifact set.",
                ],
            ),
            _command_entry(
                project_root,
                "Review the cross-system drift mesh",
                ["./scripts/ops/opsctl.sh system-drift-guard --json"],
                notes=[
                    "This rolls command drift, summary/report drift, governance drift, workstation drift, and stack-runtime drift into one registry-backed health view.",
                ],
            ),
            _command_entry(
                project_root,
                "Review Codex project guardrails",
                ["./scripts/ops/opsctl.sh codex-project-guard --staged --json"],
                notes=[
                    "Run this before Codex-authored commits or GitHub updates to catch source-of-truth drift, mixed-domain staging, and separate-domain README/docs leakage.",
                ],
            ),
            _command_entry(
                project_root,
                "Plan or apply the MLX library upgrade bundle",
                [
                    "./scripts/ops/opsctl.sh mlx-library-upgrade --json",
                    "./scripts/ops/opsctl.sh mlx-library-upgrade --apply --json",
                ],
                notes=[
                    "The dry run prints the pinned MLX package bundle from `config/requirements.lock.txt`; the apply form installs those pins, then you should run `./scripts/ops/opsctl.sh mlx-audit --json`.",
                ],
            ),
            _command_entry(
                project_root,
                "Repair safe cross-system drift surfaces",
                ["./scripts/ops/opsctl.sh system-drift-autopilot --apply --json"],
                notes=[
                    "This runs the safe drift-repair mesh. It refreshes and repairs repairable surfaces without inventing destructive operator actions.",
                ],
            ),
            _command_entry(
                project_root,
                "Master infrastructure supervisor",
                ["./scripts/ops/opsctl.sh master-infra-supervisor --json"],
                notes=[
                    "This parent check watches child infrastructure bots, command routes, storage health, report jobs, and One Numbers original-start coverage as one dependency graph.",
                ],
            ),
            _command_entry(
                project_root,
                "Docs, commands, and reporting intelligence",
                ["./scripts/ops/opsctl.sh docs-reporting-intelligence --apply --json"],
                notes=[
                    "This refreshes the README, COMMANDS.md, report-quality, and PyCharm visibility intelligence layer, including blue active-bot rows in `docs/pycharm/intelligence_layers_latest.md`.",
                ],
            ),
            _command_entry(
                project_root,
                "Deeper self-awareness intelligence layers",
                ["./scripts/ops/opsctl.sh deeper-intelligence-layers --apply --json"],
                notes=[
                    "This installs and scores the 10 deeper self-awareness layers: causal world model, belief ledger, digital twin replay, adversarial simulator, self-scientific method, resource economist, promotion court, living ontology, operator dialogue, and constitutional risk.",
                ],
            ),
            _command_entry(
                project_root,
                "PyCharm active bot blue highlights",
                ["./scripts/ops/opsctl.sh pycharm-active-bot-highlights --apply --json"],
                notes=[
                    "This writes the JetBrains `Active Bots` scope and blue file-color mapping so active `core/brain_refinery_*.py` files get a durable Project-pane scope background. PyCharm's bright blue filename text remains reserved for VCS-modified files.",
                ],
            ),
            _command_entry(
                project_root,
                "Reporter quality infrabot",
                ["./scripts/ops/opsctl.sh report-quality-guard --repair --json"],
                notes=[
                    "This repairs the sendout PDF bundle, verifies PDF integrity, and blocks regressions where paper-performance or post-trade lose their report-ready renderers.",
                ],
            ),
            _command_entry(
                project_root,
                "Coinbase API health",
                ["./scripts/ops/opsctl.sh coinbase-api-health --json"],
                notes=[
                    "This checks Coinbase public market-data endpoints and reports only credential presence booleans, never secret values.",
                ],
            ),
            _command_entry(
                project_root,
                "Point-in-time event store",
                ["./scripts/ops/opsctl.sh point-in-time-event-store --json"],
                notes=[
                    "This rebuilds the normalized event store used to prove source state at replay and report time.",
                ],
            ),
            _command_entry(
                project_root,
                "Replay hash registry guard",
                ["./scripts/ops/opsctl.sh replay-hash-registry --json"],
                notes=[
                    "This persists expected replay hashes and alerts when deterministic replay output drifts.",
                ],
            ),
            _command_entry(
                project_root,
                "Golden replay regression guard",
                ["./scripts/ops/opsctl.sh golden-replay-regression --json"],
                notes=[
                    "This compares deterministic replay against the golden replay pack or the seeded replay hash fallback.",
                ],
            ),
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
                    "./scripts/ops/opsctl.sh livefeed-refresh",
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
                "Training and labeling intelligence",
                ["./scripts/ops/opsctl.sh training-labeling-intelligence --apply --json"],
                notes=[
                    "Normalizes label contracts, writes training-process intelligence, and keeps targeted retrain candidates behind schema, feature-store, coverage, runtime, and lineage gates.",
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
            _command_entry(
                project_root,
                "Paper performance report",
                ["./scripts/ops/open_report_artifact.sh paper"],
                notes=[
                    "This refreshes the paper-performance source and opens the report-ready chart PDF.",
                ],
            ),
            _command_entry(
                project_root,
                "Incident report",
                ["./scripts/ops/open_report_artifact.sh incident"],
                notes=[
                    f"Latest PDF path: `{incident_report_pdf_path}`.",
                    "This refreshes the incident source and rebuilds the PDF through the deterministic send-out renderer.",
                ],
            ),
            _command_entry(
                project_root,
                "Incident review packet PDF",
                ["./scripts/ops/open_report_artifact.sh incident-packet"],
                notes=[
                    f"Latest PDF path: `{project_root / 'exports' / 'reports' / 'incident_review_packet_latest.pdf'}`.",
                    "This writes the immutable incident review packet JSON and rebuilds its PDF companion through the deterministic send-out renderer.",
                ],
            ),
            _command_entry(
                project_root,
                "Refresh showcase, framework map, and PDFs now",
                [
                    "./scripts/ops/opsctl.sh showcase-refresh",
                    "./scripts/ops/opsctl.sh system-explainers",
                    "./scripts/ops/opsctl.sh report-pdfs --json",
                ],
                notes=[
                    "This is the paste-ready deterministic PDF refresh path when you want the special-features PDF and the framework-map-style reports regenerated together.",
                ],
            ),
            _command_entry(
                project_root,
                "Install nightly showcase and PDF refresh",
                ["./scripts/install_daily_log_refresh_launchd.sh"],
                notes=[
                    "This installs the macOS launchd job that refreshes showcase docs, system explainers, and PDFs automatically each night.",
                ],
            ),
            _command_entry(project_root, "Report catalog bundle", ["./scripts/ops/opsctl.sh report-pdfs --json"]),
            _command_entry(
                project_root,
                "Repair and validate report PDFs",
                ["./scripts/ops/opsctl.sh report-quality-guard --repair --json"],
                notes=[
                    "This is the report infrabot pass for external sendouts: it rebuilds PDFs, checks header/EOF/size integrity, and verifies report-ready renderers for upgraded reports.",
                ],
            ),
            _command_entry(
                project_root,
                "Active bot stack PDF",
                ["./scripts/ops/open_report_artifact.sh botstack"],
                notes=[
                    f"Latest PDF path: `{bot_stack_pdf_path}`.",
                    "This refreshes the bot-stack source and rebuilds the PDF through the deterministic send-out renderer.",
                ],
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
            "Platform Expansion",
            _command_entry(
                project_root,
                "Apply the 12-layer platform intelligence control plane",
                ["./scripts/ops/opsctl.sh platform-intelligence --apply --json"],
                notes=[
                    "Adds the operational layer for bot lifecycle, data quality scoring, provider failover, backpressure prediction, duplicate-alpha detection, paper capacity, self-healing playbooks, sleeve masters, training readiness, regime routing, execution realism, and black-box recording.",
                    "This writes the platform-intelligence override while keeping the layer advisory/read-only and MLX-first.",
                ],
            ),
            _command_entry(
                project_root,
                "Preview the 12-layer platform intelligence control plane",
                ["./scripts/ops/opsctl.sh platform-intelligence --json"],
                notes=[
                    "Use this to inspect all 12 platform sections and their latest artifacts without changing runtime overrides.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply Platform Brain v4 Grande",
                ["./scripts/ops/opsctl.sh platform-brain-v4 --apply --json"],
                notes=[
                    "Adds the decision-brain layer: executive meta-orchestration, causal world modeling, experience memory v2, expansion simulation, priority ranking, self-upgrade planning, critic council, outcome verification, bot economics, data value scoring, training scheduling, and operator intent modeling.",
                    "The layer is advisory/read-only, keeps MLX as default, and preserves paper-trade lock and live-execution separation.",
                ],
            ),
            _command_entry(
                project_root,
                "Preview Platform Brain v4 Grande",
                ["./scripts/ops/opsctl.sh platform-brain-v4 --json"],
                notes=[
                    "Use this to inspect the 12 brain sections, next-best command, ranked priorities, expansion simulations, and verification plan without writing overrides.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply Platform Brain v5 Reflex Cortex",
                ["./scripts/ops/opsctl.sh platform-brain-v5 --apply --json"],
                notes=[
                    "Adds the reflex layer: temporal self-modeling, safe reflex routing, regret and outcome ledgering, scenario rehearsal, adaptive cadence, safe autonomy boundary, critic fusion, resource budgeting, data contract negotiation, bot curriculum, dependency mapping, and strategic roadmap synthesis.",
                    "The layer stays advisory/read-only, keeps MLX as default, and preserves paper-trade lock with live execution disabled.",
                ],
            ),
            _command_entry(
                project_root,
                "Preview Platform Brain v5 Reflex Cortex",
                ["./scripts/ops/opsctl.sh platform-brain-v5 --json"],
                notes=[
                    "Use this to inspect the reflex queue, safe-vs-operator-reviewed actions, scenario rehearsal, resource budgets, and roadmap without writing overrides.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply the seven-part stabilization and quality layer",
                ["./scripts/ops/opsctl.sh platform-stabilization --apply --json"],
                notes=[
                    "Adds the guarded pre-expansion layer for backlog drainage, bot data quality, duplicate-alpha compression, paper-trade realism, provider cooldown/failover, ready-only microtraining, and expansion rehearsal.",
                    "This writes the stabilization override, assigns infrastructure bots to each lane, keeps MLX as default, and keeps live execution disabled.",
                ],
            ),
            _command_entry(
                project_root,
                "Preview the seven-part stabilization and quality layer",
                ["./scripts/ops/opsctl.sh platform-stabilization --json"],
                notes=[
                    "Use this before another expansion to see the backlog, quality, duplicate-alpha, paper realism, provider, training, and rehearsal gates in one place.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply the settlement stabilization layer",
                ["./scripts/ops/opsctl.sh platform-settlement-stabilization --apply --json"],
                notes=[
                    "Adds the post-expansion settlement layer for queue decay, single-writer protection, market-hours cadence, global-halt clear readiness, paper collection floors, off-hours drain planning, and stabilization memory.",
                    "This layer keeps MLX as default, leaves live execution disabled, and records whether each stabilization pass actually reduces pressure.",
                ],
            ),
            _command_entry(
                project_root,
                "Preview the settlement stabilization layer",
                ["./scripts/ops/opsctl.sh platform-settlement-stabilization --json"],
                notes=[
                    "Use this after a large expansion or during market hours to see whether the system is settling cleanly before adding more bots or training load.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply the coordination intelligence control-plane pack",
                ["./scripts/ops/opsctl.sh coordination-intelligence --apply --json"],
                notes=[
                    "Adds the guarded coordination layer: bot genome lineage, strategy conflict resolution, capital simulation, regime memory, research-to-bot intake, feature quality, adversarial paper-trade lab, sleeve master summaries, admission committee, and explainability dashboard.",
                    "The bots are collection-only until their evidence, data-quality, and runtime thresholds clear.",
                ],
            ),
            _command_entry(
                project_root,
                "Preview the coordination intelligence control-plane pack",
                ["./scripts/ops/opsctl.sh coordination-intelligence --json"],
                notes=[
                    "Use the dry run to see planned coordination bots, storage contracts, and paper-only guardrails without changing the registry.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply the alpha intelligence evolution pack",
                ["./scripts/ops/opsctl.sh alpha-intelligence-evolution --apply --json"],
                notes=[
                    "Adds the guarded alpha advancement layer: training readiness, execution reality, portfolio exposure, source confidence, research intake, duplicate-alpha novelty control, regime memory v2, dashboard v2, adapter mesh, and cleanup governor.",
                    "The bots are collection-only with paper/live execution blocked until readiness, execution realism, source confidence, and duplicate-alpha gates clear.",
                ],
            ),
            _command_entry(
                project_root,
                "Preview the alpha intelligence evolution pack",
                ["./scripts/ops/opsctl.sh alpha-intelligence-evolution --json"],
                notes=[
                    "Use the dry run to see planned alpha intelligence bots, data intakes, storage contracts, and self-awareness upgrades without changing the registry.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply the intelligence layer advancement pack",
                ["./scripts/ops/opsctl.sh intelligence-layer-advancement --apply --json"],
                notes=[
                    "Adds the guarded meta-intelligence layer: metacognitive routing, counterfactual world models, alpha benchmarks, memory compression, critic debate, active learning, ensemble uncertainty, library routing, safety invariants, and self-improvement backlog planning.",
                    "The bots are collection-only with paper/live execution blocked until benchmark, memory-quality, safety-invariant, and runtime-pressure gates clear.",
                ],
            ),
            _command_entry(
                project_root,
                "Preview the intelligence layer advancement pack",
                ["./scripts/ops/opsctl.sh intelligence-layer-advancement --json"],
                notes=[
                    "Use the dry run to see planned intelligence-layer bots, data intakes, storage contracts, and routing/safety guardrails without changing the registry.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply the apex self-awareness intelligence pack",
                ["./scripts/ops/opsctl.sh apex-self-awareness-intelligence --apply --json"],
                notes=[
                    "Adds the 46 guarded apex bots that bring the platform to 1000 total bots: deep self-modeling, meta-reasoning, experience memory, scenario oracles, upgrade foundry, causal alpha safety, resource autonomy, operator copilot, Grand Master collective intelligence, and research frontier scouting.",
                    "The bots are collection-only with live execution, allocation, and training blocked until 120 days, 30000 observations, and safety/resource/memory gates clear.",
                ],
            ),
            _command_entry(
                project_root,
                "Preview the apex self-awareness intelligence pack",
                ["./scripts/ops/opsctl.sh apex-self-awareness-intelligence --json"],
                notes=[
                    "Use the dry run to see the 46 planned apex bots, 1000-bot target contract, storage limits, and paper-only guardrails without changing the registry.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply the deep recursive awareness pack",
                ["./scripts/ops/opsctl.sh deep-recursive-awareness --apply --json"],
                notes=[
                    "Adds the guarded recursive awareness layer: causal self-diagnosis, predictive runtime oracle, experience memory core, self-upgrade critic board, operator context governor, internal critic board, and living platform map.",
                    "The bots are collection-only with live execution, allocation, and training blocked until 150 days, 36000 observations, and causal/runtime/memory/critic/operator-context gates clear.",
                ],
            ),
            _command_entry(
                project_root,
                "Preview the deep recursive awareness pack",
                ["./scripts/ops/opsctl.sh deep-recursive-awareness --json"],
                notes=[
                    "Use the dry run to see the 28 planned recursive-awareness bots, data intakes, storage limits, and paper-only guardrails without changing the registry.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply the 24-sleeve quant strategy gap pack",
                ["./scripts/ops/opsctl.sh quant-strategy-gap --apply --json"],
                notes=[
                    "Adds 24 practical tradable-alpha strategy sleeves with five collect-only bots each: event arb, relative value, carry, ETF/NAV, auction flow, dealer expiry, rates/credit/crypto basis, and liquidity simulation.",
                    "The pack is zero-weight, training-excluded, paper-disabled, live-disabled, and thin-sampled until 120 days, 45000 observations, and strategy evidence gates clear.",
                ],
            ),
            _command_entry(
                project_root,
                "Preview the 24-sleeve quant strategy gap pack",
                ["./scripts/ops/opsctl.sh quant-strategy-gap --json"],
                notes=[
                    "Use this dry run to inspect planned strategy sleeves, bot IDs, storage guardrails, and paper-only floors without changing the registry.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply the 14-organ platform organ systems pack",
                ["./scripts/ops/opsctl.sh platform-organs --apply --json"],
                notes=[
                    "Adds 70 collect-only platform organ bots across data quality, feature registry, replay lab, execution realism, portfolio brain, alpha decay, regime routing, research assimilation, promotion court, cockpit, resource metabolism, memory lymphatics, backpressure circulation, and audit immunity.",
                    "The pack stays zero-weight, training-excluded, paper-disabled, live-disabled, and thin-sampled until 150 days, 60000 observations, and runtime/regression/operator gates clear.",
                ],
            ),
            _command_entry(
                project_root,
                "Preview the 14-organ platform organ systems pack",
                ["./scripts/ops/opsctl.sh platform-organs --json"],
                notes=[
                    "Use this dry run to inspect the organ systems, bot IDs, storage guardrails, and paper-only floors without changing the registry.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply the 14-muscle trading action systems pack",
                ["./scripts/ops/opsctl.sh trading-muscles --apply --json"],
                notes=[
                    "Adds 70 collect-only trading muscle bots across intraday momentum, mean reversion, swing trend, options convexity, options income, futures macro, crypto basis, volatility arbitrage, events, relative value, portfolio hedging, execution timing, position sizing, and exits/rebalancing.",
                    "The pack generates trade-candidate, sizing, hedge, exit, and execution-rehearsal evidence only; training, paper trading, live trading, allocation, and execution stay blocked until 180 days, 75000 observations, and quality/risk/halts gates clear.",
                ],
            ),
            _command_entry(
                project_root,
                "Preview the 14-muscle trading action systems pack",
                ["./scripts/ops/opsctl.sh trading-muscles --json"],
                notes=[
                    "Use this dry run to inspect the trading muscles, bot IDs, storage guardrails, and trade-candidate-only floor without changing the registry.",
                ],
            ),
            _command_entry(
                project_root,
                "Arm the guarded 400 bot paper ramp",
                ["./scripts/ops/opsctl.sh paper-400-ramp --apply --json"],
                notes=[
                    "Plans the 400-bot paper target now and only writes the high paper caps after Monday 2026-05-11 when global halt, memory, runtime, and ingestion gates are clean.",
                    "The controller keeps live execution blocked, paper-trade lock enabled, and explains any blocker in `governance/health/paper_400_ramp_latest.json`.",
                ],
            ),
            _command_entry(
                project_root,
                "Preview the guarded 400 bot paper ramp",
                ["./scripts/ops/opsctl.sh paper-400-ramp --json"],
                notes=[
                    "Use this to see whether the ramp is planned, armed, or blocked before changing runtime overrides.",
                ],
            ),
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
    contract = build_command_contract(project_root)
    preamble = [
        "# Commands (Canonical)",
        "",
        "Use these exact commands as the current source of truth.",
        "",
        "This file is generated from the curated operator inventory in `scripts/ops/commands_hygiene_bot.py`.",
        "Rebuild it with `./scripts/ops/opsctl.sh commands-hygiene --apply` after changing that inventory.",
        f"Command contract hash: `{contract['contract_hash']}`.",
        "Command contract artifact: `governance/health/commands_contract_latest.json`.",
        "",
        "This file is intentionally trimmed down with Most Used pinned first and the remaining sections alphabetized by section and command title:",
        "- paper mode is the operating default",
        "- no simulate variants are listed",
        "- no duplicate restart commands are listed when a broader command already covers them",
    ]
    parts = ["\n".join(preamble)]
    for section in _alphabetized_inventory(_commands_inventory(project_root)):
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
    contract_path = project_root / "governance" / "health" / "commands_contract_latest.json"
    commands_text = _read_text(commands_path)
    runbook_text = _read_text(runbook_path)
    command_contract = build_command_contract(project_root)
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
        "contract_written": False,
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
    if apply:
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(command_contract, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        apply_results["contract_written"] = True

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
        "contract_path": str(contract_path),
        "command_contract_hash": command_contract["contract_hash"],
        "command_contract_entry_count": command_contract["entry_count"],
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
