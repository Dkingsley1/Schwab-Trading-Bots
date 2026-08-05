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
MANUAL_OPERATOR_EXCLUDED_SECTIONS = {
    "Platform Expansion",
    "Macro And Media",
}
MANUAL_OPERATOR_EXCLUDED_TITLES = {
    "Install nightly showcase and PDF refresh",
}
LEGACY_RUNBOOK_ALIASES = {
    "live": "Live Feed Views",
    "refresh": "Most Used",
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


MOST_USED_PINNED_TITLES = [
    "Keep the Mac awake",
    "Start the full live stack",
    "Start the full live stack (fresh supervised restart)",
    "Stop the stack",
]

COMMAND_SEARCH_TERMS = [
    "start",
    "stop",
    "paper",
    "profitability",
    "soak",
    "halt",
    "auth",
    "schwab",
    "coinbase",
    "livefeed",
    "storage",
    "dashboard",
    "runtime",
    "watchdog",
    "backlog",
    "retrain",
    "reports",
    "startup",
    "login",
    "notification",
]


def _entry_sort_key(section_heading: str, entry: dict[str, Any]) -> tuple[int, str]:
    title = str(entry.get("title") or "")
    normalized_title = _normalize_key(title)
    if _normalize_key(section_heading) == "most used":
        pinned = {
            _normalize_key(pinned_title): index
            for index, pinned_title in enumerate(MOST_USED_PINNED_TITLES)
        }
        if normalized_title in pinned:
            return pinned[normalized_title], normalized_title
        return len(MOST_USED_PINNED_TITLES), normalized_title
    return 0, normalized_title


def _alphabetized_inventory(sections: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_sections: list[dict[str, Any]] = []
    for section in sections:
        copied = dict(section)
        entries = [dict(entry) for entry in list(copied.get("entries") or [])]
        section_heading = str(copied.get("heading") or "")
        entries.sort(key=lambda entry: _entry_sort_key(section_heading, entry))
        copied["entries"] = entries
        sorted_sections.append(copied)
    sorted_sections.sort(
        key=lambda section: (
            0 if _normalize_key(str(section.get("heading") or "")) == "most used" else 1,
            _normalize_key(str(section.get("heading") or "")),
        )
    )
    return sorted_sections


def _manual_operator_inventory(project_root: Path) -> list[dict[str, Any]]:
    """Return only commands the operator is expected to paste manually."""
    sections: list[dict[str, Any]] = []
    excluded_sections = {_normalize_key(section) for section in MANUAL_OPERATOR_EXCLUDED_SECTIONS}
    excluded_titles = {_normalize_key(title) for title in MANUAL_OPERATOR_EXCLUDED_TITLES}
    for section in _commands_inventory(project_root):
        heading = str(section.get("heading") or "")
        if _normalize_key(heading) in excluded_sections:
            continue
        copied = dict(section)
        copied["entries"] = [
            dict(entry)
            for entry in list(section.get("entries") or [])
            if _normalize_key(str(entry.get("title") or "")) not in excluded_titles
        ]
        sections.append(copied)
    return sections


def build_command_contract(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    sections = _alphabetized_inventory(_manual_operator_inventory(project_root))
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


def _slugify_search_token(raw: str) -> str:
    token = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(raw or ""))
    return "-".join(part for part in token.split("-") if part)


def _html_attr(raw: str) -> str:
    return (
        str(raw or "")
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _compact_search_text(raw: str, *, max_len: int = 180) -> str:
    text = " ".join(str(raw or "").split())
    if max_len > 0 and len(text) > max_len:
        return text[: max_len - 3].rstrip() + "..."
    return text


def _render_command_search_index(contract: dict[str, Any]) -> list[str]:
    entries = [dict(entry) for entry in list(contract.get("entries") or [])]
    lines = [
        "",
        f"Search coverage: `{len(entries)}` generated command entries from the current command contract.",
        "",
        '<datalist id="command-search-index-options">',
    ]
    for entry in entries:
        section = str(entry.get("section") or "")
        title = str(entry.get("title") or "")
        lines.append(f'  <option value="{_html_attr(title)} ({_html_attr(section)})"></option>')
    lines.extend(
        [
            "</datalist>",
            "",
            f"<details>",
            f"<summary>Generated command search index ({len(entries)} commands; rebuilt by commands-hygiene)</summary>",
            "",
            "Each row is generated from `governance/health/commands_contract_latest.json`, so added, removed, renamed, or cleaned-up commands change this index automatically.",
            "",
        ]
    )
    for entry in entries:
        section = str(entry.get("section") or "").strip()
        title = str(entry.get("title") or "").strip()
        fingerprint = str(entry.get("fingerprint") or "").strip()
        command_lines = [str(line or "") for line in list(entry.get("command_lines") or []) if str(line or "").strip()]
        opsctl = ", ".join(str(item) for item in list(entry.get("opsctl_subcommands") or []) if str(item or "").strip())
        scripts = ", ".join(str(item) for item in list(entry.get("script_paths") or []) if str(item or "").strip())
        first_command = _compact_search_text(command_lines[0] if command_lines else "")
        lines.append(
            "- "
            f"search-entry:{fingerprint} "
            f"section:`{section}` "
            f"section_key:`{_slugify_search_token(section)}` "
            f"title:{title} "
            f"title_key:`{_slugify_search_token(title)}` "
            f"opsctl:`{opsctl or 'none'}` "
            f"scripts:`{scripts or 'none'}` "
            f"first_command:`{first_command}`"
        )
    lines.append("</details>")
    return lines


def _render_pycharm_search_strip(contract: dict[str, Any]) -> list[str]:
    sections = ordered_unique(
        _slugify_search_token(str(entry.get("section") or ""))
        for entry in list(contract.get("entries") or [])
    )
    section_terms = [section for section in sections if section][:12]
    quick_terms = ordered_unique([*COMMAND_SEARCH_TERMS, *section_terms])[:28]
    return [
        "**Search Bar**",
        "",
        '<input type="search" list="command-search-index-options" placeholder="PyCharm: press Command+F or Ctrl+F, then search any command, section, opsctl alias, or script path" style="width: 100%; padding: 8px;" />',
        "",
        "PyCharm note: the field above is a visible search landing strip in Markdown preview; the reliable editor search is `Command+F` on Mac or `Ctrl+F` elsewhere.",
        "",
        "Fast search tokens: " + " ".join(f"`{term}`" for term in quick_terms) + ".",
        "",
        "Useful compound searches: `paper profitability`, `global halt`, `token refresh`, `livefeed heavy`, `storage prune`, `soak readiness`.",
        *_render_command_search_index(contract),
    ]


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


def _open_report_entry(
    project_root: Path,
    title: str,
    report_key: str,
    *,
    notes: Iterable[str] = (),
) -> dict[str, Any]:
    return _command_entry(project_root, title, [f"./scripts/ops/open_report_artifact.sh {report_key}"], notes=notes)


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
        _open_report_entry(
            project_root,
            "Open the report catalog PDF",
            "report-catalog",
            notes=[
                f"Latest PDF path: `{report_bundle_pdf_path}`.",
                "This rebuilds the documented report catalog first, then opens the report-ready bundle PDF with HTML fallback.",
            ],
        ),
        _open_report_entry(
            project_root,
            "Open the daily ops PDF",
            "daily-ops",
            notes=[
                f"Latest PDF path: `{daily_ops_pdf_path}`.",
                "This refreshes the daily ops source, rebuilds the PDF bundle, then opens the report with markdown/JSON fallback.",
            ],
        ),
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
        _open_report_entry(
            project_root,
            "Open the strategy attribution PDF",
            "strategy-attribution",
            notes=[
                f"Latest PDF path: `{strategy_attribution_pdf_path}`.",
                "This refreshes strategy attribution, rebuilds the PDF bundle, and falls back to markdown or JSON evidence.",
            ],
        ),
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
            ["./scripts/ops/open_report_artifact.sh quant"],
            notes=[
                f"Latest PDF path: `{quant_model_control_pdf_path}`.",
                "This refreshes the advanced quant-model feature, MLX, resource-cap, and research-only policy report, then opens the report-ready PDF.",
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
        _open_report_entry(
            project_root,
            "Open the system overview PDF",
            "system-overview",
            notes=[
                f"Latest PDF path: `{system_overview_pdf_path}`.",
                "This opens the week-by-week platform history and current-position overview PDF with markdown fallback.",
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
        _open_report_entry(
            project_root,
            "Open the daily runtime summary PDF",
            "daily-runtime",
            notes=[
                f"Latest PDF path: `{daily_runtime_summary_pdf_path}`.",
                "This rebuilds the PDF bundle and falls back to the runtime JSON artifact if the PDF is unavailable.",
            ],
        ),
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
        _open_report_entry(
            project_root,
            "Open the state snapshot drills PDF",
            "state-snapshot",
            notes=[
                f"Latest PDF path: `{state_snapshot_pdf_path}`.",
                "This rebuilds the state snapshot drill PDF and falls back to the latest drill JSON.",
            ],
        ),
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
                "Stop the stack",
                ["./scripts/ops/opsctl.sh stop"],
                notes=[
                    "This is the normal supervised stop path. It does not automatically engage an emergency operator halt.",
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
            _open_report_entry(
                project_root,
                "Open the One Numbers CSV in Numbers",
                "one-numbers-csv",
                notes=[
                    f"Latest CSV path: `{one_numbers_csv_path}`.",
                    "This refreshes One Numbers first so the CSV alias points at the freshest report day before opening it.",
                ],
            ),
            _open_report_entry(
                project_root,
                "Open the One Numbers PDF",
                "one-numbers",
                notes=[
                    f"Latest PDF path: `{one_numbers_pdf_path}`.",
                    "This refreshes One Numbers, rebuilds the PDF bundle, and falls back to markdown or JSON evidence if needed.",
                ],
            ),
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
                    "This ensures the supervised Schwab sleeves are running and lets them pick up the refreshed token without a hard bounce. Add `--force-restart` only when you intentionally want to restart the loops.",
                ],
            ),
            _command_entry(
                project_root,
                "Broker Truth Step 3: verify broker readiness and lane statuses",
                [
                    """/Users/dankingsley/PycharmProjects/schwab_trading_bot/.venv314/bin/python -c "from pathlib import Path; import json; root=Path('/Users/dankingsley/PycharmProjects/schwab_trading_bot/governance/health'); broker=json.loads((root/'broker_readiness_latest.json').read_text()); print(f'ready_for_open={broker.get(\\\"ready_for_open\\\")} auth_ok={broker.get(\\\"auth_ok\\\")} token_warning_level={broker.get(\\\"token_warning_level\\\")}'); print('lane,status,mismatch_count,error'); [print(f'{p.name.replace(\\\"broker_truth_\\\", \\\"\\\").replace(\\\"_latest.json\\\", \\\"\\\")},{json.loads(p.read_text()).get(\\\"status\\\", \\\"\\\")},{int(json.loads(p.read_text()).get(\\\"mismatch_count\\\", 0) or 0)},{json.loads(p.read_text()).get(\\\"error\\\") or \\\"\\\"}') for p in sorted(root.glob('broker_truth_*_latest.json')) if 'shared_snapshot' not in p.name]\""""
                ],
                notes=[
                    "Healthy target: `ready_for_open=True`, `auth_ok=True`, and all Schwab broker-truth lanes reporting `status=ok` with `mismatch_count=0`.",
                ],
            ),
            _command_entry(
                project_root,
                "Refresh the livefeed mirror without restarting sleeves",
                ["./scripts/ops/opsctl.sh livefeed-refresh"],
                notes=[
                    "`livefeed-refresh` is the operator-safe livefeed repair path. It refreshes the supervised local livefeed mirror and validates `governance/health/livefeed_local_latest.json` without restarting sleeve loops. Use `feed-refresh --source ... --stack-refresh` only when you intentionally want loop start/recovery work.",
                ],
            ),
            _command_entry(
                project_root,
                "Repair and restart the livefeed mirror",
                ["./scripts/ops/opsctl.sh livefeed-refresh-guard --apply --force-restart --freshness-minutes 10 --json"],
                notes=[
                    "Use this when the terminal livefeed starts showing stale output, escaped JSON fragments, token blobs, or mid-line storage payloads.",
                    "This validates every livefeed refresh route, restarts only the supervised local mirror, and checks `governance/health/livefeed_local_latest.json`; it does not restart sleeve loops or change paper/live execution authority.",
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
                    "Associated bots/control layers: `runtime-gate-dashboard`, `master-infrastructure-supervisor`, `system-drift-guard`.",
                ],
            ),
            _command_entry(
                project_root,
                "Ask what backlog and runtime need next",
                ["./scripts/ops/opsctl.sh system-needs --json"],
                notes=[
                    "Use this when you want the system to name the exact blocker, shard/file, next command, expected impact, risk, and stop condition.",
                    "Associated bots/control layers: `system-needs-intelligence`, `autonomic-resource-governor`, `memory-pressure-intelligence`, `writer-process-intelligence`.",
                ],
            ),
            _command_entry(
                project_root,
                "Run the architecture upgrade scoreboard",
                ["./scripts/ops/opsctl.sh architecture-upgrade-scoreboard --json"],
                notes=[
                    "This scores the current architecture expansion layers against their proof artifacts and separates bounded recovery from true blockers.",
                    "Associated bots/control layers: `architecture-upgrade-scoreboard`, `system-architecture-contract-graph`, `system-drift-guard`.",
                ],
            ),
            _command_entry(
                project_root,
                "Run adversarial system drills",
                ["./scripts/ops/opsctl.sh system-adversarial-drills --run-probes --json"],
                notes=[
                    "This runs safe read-only probes and ranks cross-layer weak points without enabling live execution or launching duplicate storage drains.",
                    "Add `--apply` when you want the drill result artifact written to `governance/drills/system_adversarial_drill_results_latest.json`.",
                    "Associated bots/control layers: `system-adversarial-drill-autopilot`, `health-fast`, `system-drift-guard`, `master-infra-supervisor`.",
                ],
            ),
            _command_entry(
                project_root,
                "Run intense system drills",
                ["./scripts/ops/opsctl.sh system-intense-drills --apply --json"],
                notes=[
                    "This executes the existing intense drill suite and writes the improvement plan, using safe improvements only when explicitly requested.",
                    "Associated bots/control layers: `system-intense-drill-autopilot`, `runtime-throttle`, `incident-closeout`, `live-canary-control`.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply raw backlog refinement",
                ["./scripts/ops/opsctl.sh raw-backlog-refiner --apply --json"],
                notes=[
                    "This expands raw backlog handling into five coordinated sections: measurement, hot-file mapping, focused drain handoff, intake relief, and safe stale/sparse cleanup.",
                    "Associated bots/control layers: `raw-backlog-refiner`, `external-backlog-drain`, `ingestion-priority-queue`, `pressure-relief-control`, `stale-artifact-sweeper`.",
                ],
            ),
            _command_entry(
                project_root,
                "Check backlog writer and drainer status",
                ["./scripts/ops/opsctl.sh writer-cycle-coordinator --json"],
                notes=[
                    "This is the read-only writer/drainer check. Use it before launching another catch-up cycle so a running single writer is not duplicated.",
                    "Associated bots/control layers: `writer-cycle-coordinator`, `writer-process-intelligence`, `backpressure-drainer-fleet`, `ingestion-storage-governor`.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply backlog writer catch-up waves",
                ["./scripts/ops/opsctl.sh writer-cycle-coordinator --apply --json"],
                notes=[
                    "This lets the single writer run bounded catch-up waves and then hands off follow-through to the active drainer lane.",
                    "Associated bots/control layers: `writer-cycle-coordinator`, `backpressure-drainer-fleet`, `storage-backpressure-autopilot`, `retention-debt-sheriff`.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply autonomic P-core resource governor",
                ["./scripts/ops/opsctl.sh autonomic-governor --apply --json"],
                notes=[
                    "This applies the host-aware budget for live loops, backlog writer, collectors, trainings, MLX/GPU jobs, reports, and foreground apps.",
                    "Associated bots/control layers: `autonomic-resource-governor`, `host-capability-contract`, `os-adapter-layer`, `workload-class-registry`, `computer-task-intelligence`.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply income operating platform controls",
                ["./scripts/ops/opsctl.sh income-operating-platform --apply --json"],
                notes=[
                    "This refreshes the 10-lane income operating platform: promotion gate, realized profit engine, drawdown governor, paper/live fill gap, live-micro lock, withdrawal simulator, account rules, sleeve ranking, failure drills, and human dashboard.",
                    "Associated bots/control layers: `income-operating-platform`, `income-readiness-control`, `paper-profitability-control`, `account-policy-context`, `chaos-drill-coordinator`.",
                ],
            ),
            _command_entry(
                project_root,
                "Check capital rotation control",
                ["./scripts/ops/opsctl.sh capital-rotation-control --json"],
                notes=[
                    "This builds the paper-only capital movement map: sleeve inflow/outflow pressure, weak-sleeve outflow, paper tilt recommendations, and live-money promotion blockers.",
                    "Associated bots/control layers: `capital-rotation-control`, `capital-growth-intelligence`, `capital-growth-awareness`, `paper-profitability-control`, `whole-system-governor`.",
                ],
            ),
            _command_entry(
                project_root,
                "Check Schwab indicator intelligence",
                ["./scripts/ops/opsctl.sh schwab-indicator-intelligence --json"],
                notes=[
                    "This builds the Schwab thinkorswim study/strategy catalog, classifies each item by market circumstance, and maps advisory usage to sleeve families.",
                    "Associated bots/control layers: `schwab-indicator-intelligence`, `indicator-bot-common`, `sleeve-strategy-coverage`, `system-self-model`.",
                ],
            ),
            _command_entry(
                project_root,
                "Check 12-lane system expansion execution",
                ["./scripts/ops/opsctl.sh system-expansion-execution --json"],
                notes=[
                    "This builds the 12-lane expansion execution layer: predictive stability, self-healing routes, stale-surface repair, Schwab feature bridge, collector utility, sleeve safe modes, deficiency repair, hot-path storage, capital simulation, promotion ledger, dependency hardening, and operator memory.",
                    "Associated bots/control layers: `system-expansion-execution`, `system-architecture-contract-graph`, `schwab-indicator-intelligence`, `capital-rotation-control`, `system-self-model`.",
                ],
            ),
            _command_entry(
                project_root,
                "Build the paper evidence packet",
                ["./scripts/ops/opsctl.sh evidence-packet --json"],
                notes=[
                    "This builds the repeatable 30/60/90-day paper evidence packet with sleeve attribution, drawdown/income controls, realized-profit conversion, ops stability, and promotion lineage.",
                    "Associated bots/control layers: `paper-performance`, `sleeve-profitability-dashboard`, `paper-profitability-control`, `income-operating-platform`, `promotion-quality-gate`.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply memory pressure and multitasking controls",
                ["./scripts/ops/opsctl.sh memory-pressure-intelligence --apply --json"],
                notes=[
                    "This refreshes unified-memory, compression, swap, observer-overhead, foreground-app, and P-core widening gates before backlog or training work expands.",
                    "Associated bots/control layers: `memory-pressure-intelligence`, `autonomic-resource-governor`, `runtime-throttle`, `creative-cotenant-guard`.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply pressure relief controls",
                ["./scripts/ops/opsctl.sh pressure-relief --apply --json"],
                notes=[
                    "This writes the pressure-relief override used by runtime loading, maintenance guards, heavy feed TTL, SQL cadence, foreground-app awareness, macro capture niceness, MLX/quant caps, report caps, and quiet-window behavior.",
                    "Associated bots/control layers: `pressure-relief-control`, `runtime-throttle`, `ingestion-storage-governor`, `mlx-intelligence-router`.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply runtime throttle and P-core priority controls",
                ["./scripts/ops/opsctl.sh runtime-throttle --apply --json"],
                notes=[
                    "This refreshes process priority, niceness, fanout limits, P-core feedback, and co-tenant headroom after the host pressure picture changes.",
                    "Canonical `master_bot_registry.json` writes are blocked by default; runtime registry adjustments publish `runtime_throttle_registry_candidate_latest.json` unless explicitly source-write authorized.",
                    "Associated bots/control layers: `runtime-throttle`, `process-fanout-guard`, `memory-pressure-intelligence`, `autonomic-resource-governor`.",
                ],
            ),
            _command_entry(
                project_root,
                "Check support maintenance yield gate",
                ["./scripts/ops/opsctl.sh support-maintenance-gate --json"],
                notes=[
                    "This reports whether support, report, media, and maintenance jobs should yield to memory pressure and Mac fluidity controls.",
                    "Associated bots/control layers: `support-maintenance-gate`, `runtime-throttle`, `memory-efficiency-control`, `swap-pressure-governor`.",
                ],
            ),
            _command_entry(
                project_root,
                "Watch P-core/E-core load with low overhead",
                ["sudo /Library/Frameworks/Python.framework/Versions/3.14/bin/asitop --interval 3 --show_cores 1"],
                notes=[
                    "Use this as the normal Apple Silicon watcher. The 3-second interval reduces observer overhead so the monitor is less likely to create the pressure it is measuring.",
                    "Associated bots/control layers: external observer for `memory-pressure-intelligence`, `autonomic-resource-governor`, and `runtime-throttle`.",
                ],
            ),
            _command_entry(
                project_root,
                "Watch P-core/E-core load live/heavy",
                ["sudo /Library/Frameworks/Python.framework/Versions/3.14/bin/asitop --interval 1 --show_cores 1"],
                notes=[
                    "Use this briefly when you need faster visual feedback. The memory intelligence layer can flag interval-1 asitop as observer overhead if it starts distorting CPU or memory pressure.",
                    "Associated bots/control layers: external observer for `memory-pressure-intelligence`, `autonomic-resource-governor`, and `runtime-throttle`.",
                ],
            ),
            _command_entry(project_root, "Validate documented commands", ["./scripts/ops/opsctl.sh command-validity --json"]),
        ),
        _section(
            "Accounts And Positions",
            _command_entry(
                project_root,
                "Refresh Schwab account positions",
                ["./scripts/ops/opsctl.sh schwab-account-snapshot-refresh --json"],
                notes=[
                    "Refreshes the shared Schwab account snapshot so Roth/cash account holdings, equities, and option legs are visible to the position-study and covered-call layers.",
                ],
            ),
            _command_entry(
                project_root,
                "Study all visible account positions",
                ["./scripts/ops/opsctl.sh account-position-study --json"],
                notes=[
                    "Builds `governance/health/account_position_study_latest.json` from all visible Schwab accounts, account aliases, recent sleeve decisions, and covered-call roll context.",
                ],
            ),
            _command_entry(
                project_root,
                "Watch covered-call roll windows",
                ["./scripts/ops/opsctl.sh covered-call-roll-watch --json"],
                notes=[
                    "Evaluates held covered calls against account aliases, DTE windows, ITM depth, hard roll targets, and per-underlying preferences before publishing roll alerts.",
                ],
            ),
            _command_entry(
                project_root,
                "Review account policy context",
                ["./scripts/ops/opsctl.sh account-policy-context --json"],
                notes=[
                    "Summarizes account-level rules and constraints so Roth/cash position logic stays separated from strategy and roll-watch interpretation.",
                ],
            ),
        ),
        _section(
            "Event Watches",
            _command_entry(
                project_root,
                "Run the SpaceX/SPCX downside watch once",
                ["./scripts/ops/opsctl.sh spacex-ipo-watch --json"],
                notes=[
                    "Reads the current SPCX/SpaceX quote context and writes the monitoring-only downside artifact without creating an order instruction.",
                ],
            ),
            _command_entry(
                project_root,
                "Install the SpaceX/SPCX IPO downside watcher",
                ["./scripts/ops/opsctl.sh spacex-ipo-watch-install --poll-seconds 30 --symbol SPCX --until-utc 2026-06-13T01:00:00+00:00"],
                notes=[
                    "Installs the launchd watcher for first-print, high-watermark, IPO-price, spread, and proxy weakness alerts; policy remains monitoring-only with automatic execution disabled.",
                ],
            ),
            _command_entry(
                project_root,
                "Run macro event intelligence",
                ["./scripts/ops/opsctl.sh macro-event-intelligence --json"],
                notes=[
                    "Checks active macro/event bulletins, calendar verification, market relevance, and event-watch context used by the livefeed status snapshot.",
                ],
            ),
        ),
        _section(
            "Notifications And Alerts",
            _command_entry(
                project_root,
                "Send a test iMessage notification",
                ['./scripts/ops/opsctl.sh notify-test --enable-imessage --imessage-recipient "you@example.com" --imessage-min-severity critical'],
                notes=[
                    "Use this after changing the recipient or iMessage allowlist; replace the recipient with the phone/email address that receives iMessage.",
                ],
            ),
            _command_entry(
                project_root,
                "Start the Mac notification and iMessage watcher",
                ['./scripts/ops/opsctl.sh notify-start --enable-imessage --imessage-recipient "you@example.com" --imessage-min-severity critical'],
                notes=[
                    "Installs and starts the macOS notification watcher with iMessage delivery enabled for critical allowed events.",
                ],
            ),
            _command_entry(
                project_root,
                "Install the startup Yes/No bot start prompt",
                ["./scripts/ops/opsctl.sh startup-start-prompt --install --no-kickstart --no-browser"],
                notes=[
                    "Arms a login-time macOS banner plus Yes/No prompt for starting `schwab_trading_bot` through the guarded `opsctl start` path.",
                    "The startup prompt path suppresses Schwab browser auth, GUI Chrome opens, headless Chrome PDF/render helpers, and timeline auto-PDF work.",
                    "The default install waits until the next login so it does not unexpectedly prompt or restart the stack right now.",
                ],
            ),
            _command_entry(
                project_root,
                "Dry-run the startup Yes/No bot start prompt",
                ["./scripts/ops/opsctl.sh startup-start-prompt-test --dry-run --delay-seconds 0"],
                notes=[
                    "Verifies the startup prompt state artifact without showing the GUI prompt or starting the trading stack.",
                ],
            ),
            _command_entry(project_root, "Stop the notification watcher", ["./scripts/ops/opsctl.sh notify-stop"]),
            _command_entry(
                project_root,
                "Review remote alert control",
                ["./scripts/ops/opsctl.sh remote-alert-control --json"],
                notes=[
                    "Summarizes critical alert backlog, iMessage bridge state, unacked alerts, and remote-notification readiness.",
                ],
            ),
        ),
        _section(
            "Paper Trading",
            _command_entry(
                project_root,
                "Review guarded 400 bot paper ramp",
                ["./scripts/ops/opsctl.sh paper-400-ramp --json"],
                notes=[
                    "Shows whether the 400-bot paper ramp is planned, armed, promoted, or blocked before writing runtime overrides.",
                ],
            ),
            _command_entry(
                project_root,
                "Arm or candidate-promote the guarded 400 bot paper ramp",
                ["./scripts/ops/opsctl.sh paper-400-ramp --apply --promote-roster --json"],
                notes=[
                    "Writes guarded paper caps and publishes a candidate registry promotion when global halt, memory, runtime, and ingestion gates are clean.",
                    "Canonical `master_bot_registry.json` writes require `--allow-source-registry-write` or `PAPER_400_RAMP_ALLOW_SOURCE_REGISTRY_WRITE=1`.",
                ],
            ),
            _command_entry(
                project_root,
                "Check paper runtime regression guard",
                ["./scripts/ops/opsctl.sh runtime-paper-regression-guard --json"],
                notes=[
                    "Verifies runtime throttle, resource guard, paper-ramp, support niceness, and paper execution pause contracts after a ramp or degradation fix.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply the paper live-data standard",
                ["./scripts/ops/opsctl.sh paper-live-data-standard --apply --json"],
                notes=[
                    "Reapplies the paper-only live-data standard so eligible sleeves can observe real market data while live execution remains blocked.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply paper profitability controls",
                ["./scripts/ops/opsctl.sh paper-profitability-control --apply --json"],
                notes=[
                    "Refreshes the profitability, weak-profile containment, and promotion-readiness controls that feed the paper evidence packet.",
                ],
            ),
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
            "Live Feed Views",
            _command_entry(
                project_root,
                "Heavy operator livefeed view",
                ["./scripts/ops/opsctl.sh feed --source main --heavy --no-heavy-ttl --color --red-actions"],
                notes=[
                    "Use this as the primary operator view when you want decisions plus important storage, backpressure, auth, halt, and alert messages in one window.",
                    "The `--red-actions` palette keeps the feed red-dominant while leaving `BUY` green and `SELL` red.",
                    "If the Mac is running an `air_safe` or `constrained` memory-efficiency profile, the feed automatically trims decision fanout and uses a lower default line budget unless you pass your own `--lines` or `--no-memory-aware`.",
                    "The feed now probes files before following them; unreadable logs are skipped and counted instead of cutting off the stream.",
                    "Escaped JSON fragments are hidden by default so byte-tail startup cannot flood the terminal with `stdout_tail`, token, or storage-route payloads; add `--show-json-fragments` only for raw formatter debugging.",
                ],
            ),
            _command_entry(
                project_root,
                "Heavy live feed with file diagnostics",
                ["./scripts/ops/opsctl.sh feed --source main --heavy --show-files --no-heavy-ttl --color --red-actions"],
                notes=[
                    "Use this when the feed looks sparse or cut off; it prints followed files plus any skipped unreadable file paths and keeps the operator tab open without the pressure-relief heavy-feed TTL.",
                ],
            ),
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
                "Local Schwab credential setup",
                [
                    "./scripts/ops/opsctl.sh schwab-credentials --check --json",
                    "./scripts/ops/opsctl.sh schwab-credentials --interactive --store keychain --json",
                ],
                notes=[
                    "Prompts locally for Schwab API credentials and stores them in macOS Keychain by default; no secret values are printed or written to tracked files.",
                    "This command does not open Chrome or a headless browser. Run the interactive token refresh after credentials are stored if OAuth consent needs renewal.",
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
                "Review system plumbing control",
                ["./scripts/ops/opsctl.sh system-plumbing-control --json"],
                notes=[
                    "Publishes the shared queue, storage, writer, data-plane, and paper/live boundary contract used to diagnose present degradation.",
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
                "Apply system architecture hardening",
                ["./scripts/ops/opsctl.sh system-architecture-hardening --apply --json"],
                notes=[
                    "Writes the cross-layer architecture hardening artifact and read-only guardrails for queue, storage, runtime, paper/live, and reporting contracts.",
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
                "Adapt infrabots to current system needs",
                ["./scripts/ops/opsctl.sh infrabot-adaptive-governor --apply --json"],
                notes=[
                    "This publishes the shared needs contract, capability registry, adaptive policy router, safety guard, and feedback ledger used to keep infrabots aligned with current degradation.",
                    "The apply form writes coordination contracts only; it does not launch repair fanout, retraining, live execution, or competing SQLite writers.",
                ],
            ),
            _command_entry(
                project_root,
                "Publish production-quality repair lanes",
                ["./scripts/ops/opsctl.sh production-quality --apply --refresh-contract --json"],
                notes=[
                    "This turns live-canary blockers into ordered safe repair lanes for raw profitability, paper continuity, auth continuity, storage pressure, and promotion/paper freshness.",
                ],
            ),
            _command_entry(
                project_root,
                "Track production-quality SLO recurrence",
                ["./scripts/ops/opsctl.sh production-quality-slo --apply --refresh-quality --json"],
                notes=[
                    "This keeps state across checks so repeated production-quality lane failures become watch, warning, or breach evidence instead of isolated snapshots.",
                ],
            ),
            _command_entry(
                project_root,
                "Run production hardening watch",
                ["./scripts/ops/opsctl.sh production-hardening-watch --apply --json"],
                notes=[
                    "This refreshes live-canary readiness, production-quality lanes, SLO state, and infrabot routing in one safe control loop. Safe repair execution remains opt-in and governor-allowlisted.",
                ],
            ),
            _command_entry(
                project_root,
                "Review ten-pillar production excellence",
                ["./scripts/ops/opsctl.sh production-excellence --json"],
                notes=[
                    "Reports the frozen candidate, clean soak, recovery drills, live execution, independent fills, promotion candidates, profitability, canary, grading integrity, and institutional evidence as ten fail-closed pillars.",
                    "Evidence debt is visible but does not interrupt healthy paper collection; live order submission stays locked until all ten pillars are ready.",
                ],
            ),
            _command_entry(
                project_root,
                "Freeze or accept a production candidate",
                [
                    "./scripts/ops/opsctl.sh production-excellence --apply --initialize-candidate --json",
                    "./scripts/ops/opsctl.sh production-excellence --apply --accept-candidate-change --change-reason \"Describe the reviewed production change\" --json",
                ],
                notes=[
                    "Initialize only after the intended production code is committed. Accepted changes reset only the affected evidence scopes and preserve historical profitability.",
                ],
            ),
            _command_entry(
                project_root,
                "Verify the durable live-order ledger",
                ["./scripts/ops/opsctl.sh live-order-ledger --json"],
                notes=[
                    "Checks the transactional order-intent ledger, hash-chained lifecycle events, and unresolved submit or cancel outcomes. Unknown broker outcomes require reconciliation and are never auto-retried.",
                    "After independently verifying broker truth, use `--resolve-intent ID --resolution STATE --evidence TEXT`; the evidence-backed resolution is appended to the ledger event chain.",
                ],
            ),
            _command_entry(
                project_root,
                "Refresh health gates",
                ["./scripts/ops/opsctl.sh health-gates --json"],
                notes=[
                    "This refreshes the health-gates artifact directly when stale health-gate state is blocking production-quality or live-canary readiness.",
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
            "Strategy Research",
            _command_entry(
                project_root,
                "Review the 10-layer deep quant advisory upgrade",
                ["./scripts/ops/opsctl.sh deep-quant-layer-upgrade --json"],
                notes=[
                    "Installs and reports the 10 deeper quant layers: residual alpha, meta-labeling, conformal abstention, execution-cost decay, crowding/cross-impact, changepoints, systematic flow, robust optimization, special situations, and research governance.",
                    "The layer pack is collection-only and advisory-only; paper, live, allocation, execution, and training intake stay blocked until promotion gates clear.",
                ],
            ),
            _command_entry(
                project_root,
                "Apply the 10-layer dual-mode library efficiency upgrade",
                ["./scripts/ops/opsctl.sh library-efficiency-deepening --apply --json"],
                notes=[
                    "Installs the 10 library-efficiency layers across both MLX and non-MLX libraries: routing, columnar data, MLX inference, incremental feature cache, pricing kernels, econometrics, tabular alpha, graph impact, path signatures, and benchmark-cost governance.",
                    "The contracts apply to both paper rehearsal and live advisory parity; paper/live execution authority remains disabled until runtime, promotion, and broker live gates clear.",
                ],
            ),
            _command_entry(
                project_root,
                "Push advancement until the safety guard pauses it",
                ["./scripts/ops/opsctl.sh safety-bounded-advancement-frontier --apply --json"],
                notes=[
                    "Applies the next 10 safe control-plane frontier stages: route assimilation, freshness DAG, cache ownership, cost ledger, paper/live parity witness, incremental feature reuse, pricing reuse, cross-impact graphing, route retirement, and soak/pause guard.",
                    "The command intentionally stops at advisory/control-plane scope when promotion evidence, active training, or live authority gates say the system needs a soak period.",
                ],
            ),
            _command_entry(
                project_root,
                "Push the 12-domain whole-system frontier",
                ["./scripts/ops/opsctl.sh whole-system-safety-frontier --apply --json"],
                notes=[
                    "Applies the safe control-plane frontier for promotion evidence, paper/live fill truth, feature cache, storage/backpressure, livefeed reliability, account positions, risk exposure graph, benchmark cost, model retirement court, A+ cockpit, notifications, and disaster-recovery replay.",
                    "This command stops before execution authority, allocation authority, training intake, new high-volume collectors, heavy replay, or automatic model deletion.",
                ],
            ),
            _command_entry(
                project_root,
                "Push system efficiency until the safety guard pauses it",
                ["./scripts/ops/opsctl.sh system-efficiency-frontier --apply --json"],
                notes=[
                    "Applies the safe system-efficiency frontier for backend routing, runtime and memory caps, storage/write pressure, feature caching, livefeed trimming, training scheduling, report rendering, alert noise, paper execution truth, model route lifecycle, replay proof, and operator command flow.",
                    "This command is low-churn control-plane work only; it stops before execution authority, allocation authority, training intake, new high-volume collectors, heavy replay, destructive cleanup, or automatic model deletion.",
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
                    "The controller keeps live execution blocked, paper-trade lock enabled, and source registry writes blocked unless the operator uses the explicit source-write override.",
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
        "- passive automation installers and expansion-pack reference commands are kept out of the operator-facing list",
        "",
        *_render_pycharm_search_strip(contract),
    ]
    parts = ["\n".join(preamble)]
    for section in _alphabetized_inventory(_manual_operator_inventory(project_root)):
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
    refresh) print -r -- "Most Used" ;;
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
