#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "exports" / "reports" / "system_summary"
LATEST_HTML_PATH = DEFAULT_OUT_DIR / "system_summary_latest.html"
LATEST_PDF_PATH = DEFAULT_OUT_DIR / "system_summary_latest.pdf"
LATEST_JSON_PATH = PROJECT_ROOT / "governance" / "health" / "system_summary_report_latest.json"
APP_BROWSER_CANDIDATES = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
)
PDF_RENDER_TIMEOUT_SECONDS = float(os.getenv("SYSTEM_SUMMARY_PDF_TIMEOUT_SECONDS", "20"))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _fmt_score(value: Any) -> str:
    return f"{_safe_float(value, 0.0):.2f}"


def _fmt_pct(value: Any) -> str:
    try:
        raw = float(value)
    except Exception:
        return "n/a"
    if raw <= 1.0:
        raw *= 100.0
    return f"{raw:.1f}%"


def _fmt_timestamp(raw: Any) -> str:
    if not raw:
        return "unknown"
    text = str(raw).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return text
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _titleize_slug(raw: Any) -> str:
    return str(raw or "").replace("_", " ").strip().title()


def _describe_watch_item(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("surface", "title", "issue", "reason", "status", "name"):
            value = str(item.get(key) or "").strip()
            if value:
                return value
        try:
            return json.dumps(item, ensure_ascii=True, sort_keys=True)
        except Exception:
            return str(item)
    return str(item).strip()


def _default_allow_gui_pdf_renderer() -> bool:
    return any(candidate.exists() for candidate in APP_BROWSER_CANDIDATES)


def _run(cmd: list[str], *, cwd: Path = PROJECT_ROOT, timeout_seconds: float | None = None) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except subprocess.TimeoutExpired as exc:
        out = str(exc.output or "").strip()
        err = str(exc.stderr or "").strip()
        detail = "\n".join([line for line in [err, f"timeout_after_seconds={timeout_seconds}"] if line])
        return 124, out, detail
    except Exception as exc:
        return 1, "", str(exc)


def _parse_json_output(text: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _pdf_renderer_binary(allow_gui_renderer: bool) -> tuple[str, str]:
    env_override = (
        os.getenv("SYSTEM_SUMMARY_PDF_BIN", "").strip()
        or os.getenv("PROJECT_TIMELINE_PDF_BIN", "").strip()
        or os.getenv("REPORT_PDF_BUNDLE_PDF_BIN", "").strip()
    )
    if env_override:
        env_bin = Path(env_override).expanduser()
        if env_bin.exists():
            kind = "wkhtmltopdf" if env_bin.name == "wkhtmltopdf" else ("browser_app" if env_bin.suffix == ".app" else "browser")
            return str(env_bin), kind

    wkhtmltopdf = shutil.which("wkhtmltopdf")
    if wkhtmltopdf:
        return wkhtmltopdf, "wkhtmltopdf"

    for candidate in (
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("microsoft-edge"),
        shutil.which("msedge"),
    ):
        if candidate:
            return candidate, "browser"

    # Browser app binaries still render headlessly via `open -na ... --args --headless`,
    # so they remain valid even when the policy disallows an interactive GUI renderer.
    for candidate in APP_BROWSER_CANDIDATES:
        if candidate.exists():
            return str(candidate), "browser_app"

    return "", ""


def _render_pdf_via_open_app(renderer: str, html_uri: str, pdf_path: Path) -> tuple[bool, str]:
    app_binary = Path(renderer)
    app_bundle = app_binary.parents[2]
    profile_dir = Path(tempfile.mkdtemp(prefix="system-summary-pdf-open-"))
    try:
        if pdf_path.exists():
            pdf_path.unlink()
        cmd = [
            "open",
            "-na",
            str(app_bundle),
            "--args",
            "--headless",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile_dir}",
            f"--print-to-pdf={pdf_path}",
            html_uri,
        ]
        rc, out, err = _run(cmd)
        if rc != 0:
            return False, err or out or f"rc={rc}"
        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                return True, "ok"
            time.sleep(0.5)
        return False, "pdf_render_timeout"
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)


def _render_pdf_from_html(html_path: Path, pdf_path: Path, *, allow_gui_renderer: bool) -> tuple[bool, str]:
    renderer, renderer_kind = _pdf_renderer_binary(allow_gui_renderer)
    if not renderer:
        return False, "pdf_renderer_not_found"
    html_uri = html_path.resolve().as_uri()
    if renderer_kind == "wkhtmltopdf":
        rc, out, err = _run([renderer, html_uri, str(pdf_path)])
    else:
        profile_dir = Path(tempfile.mkdtemp(prefix="system-summary-pdf-"))
        try:
            rc, out, err = _run(
                [
                    renderer,
                    "--headless",
                    "--disable-gpu",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--silent-launch",
                    "--no-startup-window",
                    "--disable-background-networking",
                    "--metrics-recording-only",
                    f"--user-data-dir={profile_dir}",
                    f"--print-to-pdf={pdf_path}",
                    html_uri,
                ],
                timeout_seconds=PDF_RENDER_TIMEOUT_SECONDS,
            )
        finally:
            shutil.rmtree(profile_dir, ignore_errors=True)
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return True, out or err or "ok"
    return False, err or out or f"rc={rc}"


def _extract_markdown_bullets(path: Path, limit: int = 8) -> list[str]:
    bullets: list[str] = []
    for raw in _read_text(path).splitlines():
        line = raw.strip()
        if line.startswith("- "):
            bullets.append(line[2:].strip())
        if len(bullets) >= limit:
            break
    return bullets


def _link_entry(path: Path | None, label: str) -> str:
    if path is None or not path.exists():
        return ""
    return f'<a href="{html.escape(path.resolve().as_uri())}">{html.escape(label)}</a>'


def _primary_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _report_catalog(project_root: Path) -> list[dict[str, Any]]:
    reports_root = project_root / "exports" / "reports"
    one_numbers_root = project_root / "exports" / "one_numbers" / "latest"
    bot_stack_root = project_root / "exports" / "bot_stack_status"
    entries = [
        {
            "slug": "system_summary",
            "title": "Compiled System Summary",
            "paths": [LATEST_PDF_PATH, LATEST_HTML_PATH],
        },
        {
            "slug": "framework_map",
            "title": "Framework Map",
            "paths": [
                reports_root / "system_explainers" / "framework_map_v2_latest.pdf",
                reports_root / "system_explainers" / "framework_map_v2_latest.html",
            ],
        },
        {
            "slug": "special_features",
            "title": "Special Features",
            "paths": [
                reports_root / "showcase" / "special_features_latest.pdf",
                project_root / "docs" / "showcase" / "generated" / "special_features_latest.html",
            ],
        },
        {
            "slug": "project_timeline",
            "title": "Project Timeline",
            "paths": [
                reports_root / "project_timeline" / "project_timeline_latest.pdf",
                reports_root / "project_timeline" / "project_timeline_print_latest.html",
                reports_root / "project_timeline" / "project_timeline_latest.md",
            ],
        },
        {
            "slug": "one_numbers",
            "title": "One Numbers",
            "paths": [
                one_numbers_root / "one_numbers_latest.pdf",
                one_numbers_root / "one_numbers_latest.md",
                one_numbers_root / "one_numbers_latest.xlsx",
                one_numbers_root / "one_numbers_latest.csv",
            ],
        },
        {
            "slug": "bot_stack",
            "title": "Bot Stack Status",
            "paths": [
                bot_stack_root / "latest.pdf",
                bot_stack_root / "latest.html",
                bot_stack_root / "latest.json",
            ],
        },
        {
            "slug": "training_report",
            "title": "Training Report",
            "paths": [
                reports_root / "training_reports" / "training_report_latest.pdf",
                reports_root / "training_reports" / "training_report_print_latest.html",
                reports_root / "training_reports" / "training_report_latest.md",
            ],
        },
        {
            "slug": "incident_report",
            "title": "Incident Report",
            "paths": [
                reports_root / "incident_report_latest.pdf",
                reports_root / "incident_report_latest.html",
                reports_root / "incident_report_latest.md",
            ],
        },
        {
            "slug": "report_bundle",
            "title": "Report Catalog Bundle",
            "paths": [
                reports_root / "report_pdf_bundle_latest.pdf",
                reports_root / "report_pdf_bundle_latest.html",
            ],
        },
    ]
    catalog: list[dict[str, Any]] = []
    for row in entries:
        paths = [Path(p) for p in row["paths"]]
        primary = _primary_existing(paths)
        available = [path for path in paths if path.exists()]
        catalog.append(
            {
                "slug": row["slug"],
                "title": row["title"],
                "primary_path": str(primary) if primary else "",
                "primary_format": primary.suffix.lstrip(".").lower() if primary else "",
                "available_formats": [
                    {
                        "format": path.suffix.lstrip(".").lower(),
                        "path": str(path),
                    }
                    for path in available
                ],
                "available": bool(primary),
            }
        )
    return catalog


def _refresh_supporting_artifacts(project_root: Path) -> list[dict[str, Any]]:
    py = str(project_root / ".venv314" / "bin" / "python")
    steps = [
        {
            "name": "architecture_upgrade_scoreboard",
            "cmd": [py, str(project_root / "scripts" / "ops" / "architecture_upgrade_scoreboard.py"), "--json"],
        },
        {
            "name": "showcase_highlights",
            "cmd": [py, str(project_root / "scripts" / "ops" / "update_showcase_highlights.py")],
        },
        {
            "name": "system_explainers",
            "cmd": [py, str(project_root / "scripts" / "ops" / "system_explainer_docs.py")],
        },
        {
            "name": "project_timeline",
            "cmd": [py, str(project_root / "scripts" / "ops" / "project_timeline_report.py"), "--auto", "--json"],
        },
        {
            "name": "bot_stack_status",
            "cmd": [py, str(project_root / "scripts" / "bot_stack_status_report.py"), "--top", "25"],
        },
    ]
    rows: list[dict[str, Any]] = []
    for step in steps:
        rc, out, err = _run(step["cmd"], cwd=project_root)
        rows.append(
            {
                "name": step["name"],
                "rc": rc,
                "ok": rc == 0,
                "stdout_tail": "\n".join(out.splitlines()[-8:]),
                "stderr_tail": "\n".join(err.splitlines()[-8:]),
                "payload": _parse_json_output(out),
            }
        )
    return rows


def build_payload(project_root: Path = PROJECT_ROOT, *, refresh_supporting_artifacts: bool = False) -> dict[str, Any]:
    refresh_rows = _refresh_supporting_artifacts(project_root) if refresh_supporting_artifacts else []
    health_root = project_root / "governance" / "health"
    reports_root = project_root / "exports" / "reports"
    section_guard = _load_json(health_root / "section_grade_guard_latest.json")
    platform = _load_json(health_root / "platform_control_plane_latest.json")
    highlights = _load_json(project_root / "docs" / "showcase" / "generated" / "highlights_latest.json")
    one_numbers = _load_json(project_root / "exports" / "one_numbers" / "latest" / "one_numbers_summary.json")
    cost_telemetry = _load_json(health_root / "cost_telemetry_latest.json")
    live_readiness = _load_json(health_root / "live_readiness_smoke_latest.json")
    closeout = _load_json(health_root / "incident_closeout_autopilot_latest.json")
    training_quality = _load_json(health_root / "training_quality_control_latest.json")
    training_lineage = _load_json(health_root / "training_lineage_manifest_latest.json")
    chaos_drills = _load_json(health_root / "chaos_drill_coordinator_latest.json")
    cross_host = _load_json(health_root / "cross_host_parity_report_latest.json")
    architecture = _load_json(health_root / "architecture_upgrade_scoreboard_latest.json")
    incident_timeline = _load_json(health_root / "incident_timeline_latest.json")
    bot_stack = _load_json(project_root / "exports" / "bot_stack_status" / "latest.json")
    immutable_ledger = _load_json(project_root / "governance" / "experiments" / "immutable_experiment_ledger_latest.json")
    promotion_packet = _load_json(project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json")
    timeline_md = reports_root / "project_timeline" / "project_timeline_latest.md"

    institutional = platform.get("institutional_readiness") if isinstance(platform.get("institutional_readiness"), dict) else {}
    domain_map = platform.get("institutional_domains_by_slug") if isinstance(platform.get("institutional_domains_by_slug"), dict) else {}
    grade_sections = section_guard.get("sections") if isinstance(section_guard.get("sections"), list) else []
    bot_summary = highlights.get("bot_summary") if isinstance(highlights.get("bot_summary"), dict) else {}
    architecture_summary = highlights.get("architecture_summary") if isinstance(highlights.get("architecture_summary"), dict) else {}
    special_features = highlights.get("special_features") if isinstance(highlights.get("special_features"), dict) else {}
    top_bots = bot_summary.get("top_active_bots") if isinstance(bot_summary.get("top_active_bots"), list) else []
    domain_rows = []
    for slug, value in domain_map.items():
        if not isinstance(value, dict):
            continue
        domain_rows.append(
            {
                "slug": str(slug),
                "title": str(value.get("title") or _titleize_slug(slug)),
                "score": _safe_float(value.get("score"), 0.0),
                "status": str(value.get("status") or ""),
            }
        )
    strongest_domains = sorted(domain_rows, key=lambda row: (-row["score"], row["title"]))[:6]
    frontier_domains = sorted(domain_rows, key=lambda row: (row["score"], row["title"]))[:6]
    timeline_bullets = _extract_markdown_bullets(timeline_md, limit=10)
    if not timeline_bullets:
        timeline_bullets = [
            "Timeline markdown is present but did not expose bullet highlights on this refresh.",
        ]

    catalog = _report_catalog(project_root)
    watchlist = []
    for item in (incident_timeline.get("watch_surfaces") if isinstance(incident_timeline.get("watch_surfaces"), list) else [])[:6]:
        description = _describe_watch_item(item)
        if description:
            watchlist.append(description)
    for item in (incident_timeline.get("recommended_actions") if isinstance(incident_timeline.get("recommended_actions"), list) else [])[:4]:
        description = _describe_watch_item(item)
        if description and description not in watchlist:
            watchlist.append(description)
    section_rows = []
    for row in grade_sections:
        if not isinstance(row, dict):
            continue
        section_rows.append(
            {
                "section": str(row.get("section") or ""),
                "letter_grade": str(row.get("letter_grade") or ""),
                "raw_letter_grade": str(row.get("raw_letter_grade") or ""),
                "score": _safe_float(row.get("score"), 0.0),
                "state": str(row.get("state") or ""),
            }
        )

    executive_summary = (
        f"The platform is currently graded {section_guard.get('overall_letter_grade', 'unknown')} overall, "
        f"with institutional readiness {_fmt_score(institutional.get('overall_score'))}/100, "
        f"{_safe_int(bot_summary.get('active_count'), 0)} active bots, "
        f"and {sum(1 for row in catalog if row.get('available'))} explainer/report documents ready for review. "
        f"Cross-host parity is {str(cross_host.get('overall_status') or 'unknown')}, chaos/restore discipline is "
        f"{str(chaos_drills.get('overall_status') or 'unknown')}, and the current training lineage contract is "
        f"{str(training_lineage.get('overall_status') or 'unknown')} at lineage score {_fmt_score(training_lineage.get('lineage_score'))}."
    )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "ready",
        "executive_summary": executive_summary,
        "section_grade_board": {
            "overall_letter_grade": str(section_guard.get("overall_letter_grade") or ""),
            "raw_overall_letter_grade": str(section_guard.get("raw_overall_letter_grade") or ""),
            "sections": section_rows,
        },
        "institutional_readiness": {
            "overall_score": _safe_float(institutional.get("overall_score"), 0.0),
            "overall_status": str(institutional.get("overall_status") or ""),
            "weakest_domains": institutional.get("weakest_domains") if isinstance(institutional.get("weakest_domains"), list) else [],
            "strongest_domains": strongest_domains,
            "frontier_domains": frontier_domains,
            "domain_scores": {
                slug: _safe_float((value or {}).get("score"), 0.0)
                for slug, value in domain_map.items()
                if isinstance(value, dict)
            },
        },
        "operations_snapshot": {
            "live_readiness_score": _safe_float(live_readiness.get("readiness_score"), 0.0),
            "live_mode": str(live_readiness.get("mode") or ""),
            "closeout_score": _safe_float(closeout.get("closeout_score"), 0.0),
            "training_quality_index": _safe_float(
                training_quality.get("training_quality_index", training_quality.get("training_quality_score")),
                0.0,
            ),
            "training_quality_base_score": _safe_float(training_quality.get("training_quality_base_score"), 0.0),
            "training_quality_bonus_score": _safe_float(training_quality.get("training_quality_bonus_score"), 0.0),
            "training_quality_score": _safe_float(training_quality.get("training_quality_score"), 0.0),
            "lineage_score": _safe_float(training_lineage.get("lineage_score"), 0.0),
            "tracked_sqlite_gb": _safe_float(((cost_telemetry.get("storage_cost_proxy") or {}).get("tracked_sqlite_gb")), 0.0),
            "pressure_index": _safe_float(((cost_telemetry.get("storage_cost_proxy") or {}).get("pressure_index")), 0.0),
            "decision_rows_day": _safe_int(one_numbers.get("combined_decision_total_rows"), 0),
            "governance_rows_day": _safe_int(one_numbers.get("combined_governance_total_rows"), 0),
            "blocked_rate_day": _safe_float(one_numbers.get("combined_blocked_rate"), 0.0),
            "paper_executions_day": _safe_int(one_numbers.get("paper_executed_total"), 0),
            "cross_host_proof_count": _safe_int(cross_host.get("proof_written_count"), 0),
            "restore_proof_ready": bool(((chaos_drills.get("restore_discipline") or {}).get("restore_proof_ready", False))),
        },
        "proof_stack": {
            "immutable_ledger": {
                "overall_status": str(immutable_ledger.get("overall_status") or ""),
                "append_only_ready": bool(immutable_ledger.get("append_only_ready", False)),
                "latest_exact_replay_ready": bool(immutable_ledger.get("latest_exact_replay_ready", False)),
                "latest_signature_ready": bool(immutable_ledger.get("latest_signature_ready", False)),
                "latest_attestation_ready": bool(immutable_ledger.get("latest_attestation_ready", False)),
                "ledger_row_count": _safe_int(immutable_ledger.get("ledger_row_count"), 0),
            },
            "promotion_packet": {
                "packet_complete": bool(promotion_packet.get("packet_complete", False)),
                "ready_for_committee": bool(promotion_packet.get("ready_for_committee", False)),
                "committee_packet_seed_ready": bool(promotion_packet.get("committee_packet_seed_ready", False)),
                "signature_verified": bool(((promotion_packet.get("signature") or {}).get("verified", False))),
                "exact_replay_ready": bool(((promotion_packet.get("replayability_contract") or {}).get("exact_replay_ready", False))),
                "trained_models_complete": bool(promotion_packet.get("trained_models_complete", False)),
            },
            "training_lineage": {
                "overall_status": str(training_lineage.get("overall_status") or ""),
                "lineage_score": _safe_float(training_lineage.get("lineage_score"), 0.0),
                "exact_replay_ready": bool(training_lineage.get("exact_replay_ready", False)),
                "lineage_contract_ready": bool(training_lineage.get("lineage_contract_ready", False)),
            },
            "cross_host_parity": {
                "overall_status": str(cross_host.get("overall_status") or ""),
                "proof_written_count": _safe_int(cross_host.get("proof_written_count"), 0),
                "nightly_proof_ready": bool(cross_host.get("nightly_proof_ready", False)),
            },
            "restore_discipline": {
                "overall_status": str(chaos_drills.get("overall_status") or ""),
                "restore_proof_ready": bool(((chaos_drills.get("restore_discipline") or {}).get("restore_proof_ready", False))),
                "overdue_drills": _safe_int(len(chaos_drills.get("overdue_drills") or []), 0),
            },
        },
        "active_bots": {
            "active_count": _safe_int(bot_summary.get("active_count"), 0),
            "total_registered": _safe_int(bot_summary.get("total_registered"), 0),
            "roles": bot_summary.get("active_roles") if isinstance(bot_summary.get("active_roles"), dict) else {},
            "top_active_bots": top_bots,
            "bot_stack_latest": bot_stack if isinstance(bot_stack, dict) else {},
        },
        "architecture_upgrades": {
            "upgrade_count": _safe_int(architecture.get("upgrade_count"), 0),
            "ready_count": _safe_int(architecture.get("ready_count"), 0),
            "rows": architecture.get("rows") if isinstance(architecture.get("rows"), list) else [],
            "summary": architecture_summary,
        },
        "special_features": special_features,
        "watchlist": watchlist,
        "timeline_highlights": timeline_bullets,
        "report_catalog": catalog,
        "supporting_refresh": refresh_rows,
        "recommended_commands": [
            "./scripts/ops/opsctl.sh system-summary --render-pdf --allow-gui-pdf-renderer --json",
            "./scripts/ops/open_report_artifact.sh summary",
            "./scripts/ops/opsctl.sh report-pdfs --allow-gui-pdf-renderer --json",
        ],
    }


def _render_html(payload: dict[str, Any]) -> str:
    board = payload.get("section_grade_board") if isinstance(payload.get("section_grade_board"), dict) else {}
    sections = board.get("sections") if isinstance(board.get("sections"), list) else []
    readiness = payload.get("institutional_readiness") if isinstance(payload.get("institutional_readiness"), dict) else {}
    ops = payload.get("operations_snapshot") if isinstance(payload.get("operations_snapshot"), dict) else {}
    active = payload.get("active_bots") if isinstance(payload.get("active_bots"), dict) else {}
    architecture = payload.get("architecture_upgrades") if isinstance(payload.get("architecture_upgrades"), dict) else {}
    proof_stack = payload.get("proof_stack") if isinstance(payload.get("proof_stack"), dict) else {}
    special_features = payload.get("special_features") if isinstance(payload.get("special_features"), dict) else {}
    watchlist = payload.get("watchlist") if isinstance(payload.get("watchlist"), list) else []
    timeline = payload.get("timeline_highlights") if isinstance(payload.get("timeline_highlights"), list) else []
    catalog = payload.get("report_catalog") if isinstance(payload.get("report_catalog"), list) else []
    generated = _fmt_timestamp(payload.get("timestamp_utc"))

    section_cards = []
    for row in sections:
        if not isinstance(row, dict):
            continue
        section_cards.append(
            f"""
            <article class="metric-card">
              <div class="eyebrow">{html.escape(str(row.get('section') or '').replace('_', ' ').title())}</div>
              <div class="metric">{html.escape(str(row.get('letter_grade') or ''))}</div>
              <div class="submetric">raw {html.escape(str(row.get('raw_letter_grade') or ''))} · score {_fmt_score(row.get('score'))}</div>
            </article>
            """.strip()
        )

    top_bot_rows = []
    for row in active.get("top_active_bots") or []:
        if not isinstance(row, dict):
            continue
        top_bot_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('bot_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('bot_role') or ''))}</td>"
            f"<td>{_fmt_score(_safe_float(row.get('test_accuracy'), 0.0) * 100.0 if _safe_float(row.get('test_accuracy'), 0.0) <= 1.0 else row.get('test_accuracy'))}</td>"
            f"<td>{_fmt_score(row.get('quality_score'))}</td>"
            "</tr>"
        )
    if not top_bot_rows:
        top_bot_rows.append("<tr><td colspan='4'>No top-bot snapshot was available on this refresh.</td></tr>")

    architecture_rows = []
    for row in (architecture.get("rows") or [])[:8]:
        if not isinstance(row, dict):
            continue
        architecture_rows.append(
            f"""
            <article class="upgrade-card">
              <div class="upgrade-title">{html.escape(str(row.get('title') or ''))}</div>
              <div class="status-chip status-{html.escape(str(row.get('status') or 'missing').lower())}">{html.escape(str(row.get('status') or 'missing'))}</div>
              <p>{html.escape(str(row.get('proof') or ''))}</p>
            </article>
            """.strip()
        )
    if not architecture_rows:
        architecture_rows.append("<article class='upgrade-card'><div class='upgrade-title'>Architecture upgrade scoreboard unavailable</div><p>No rows were published on this refresh.</p></article>")

    strongest_domain_rows = []
    for row in (readiness.get("strongest_domains") or [])[:6]:
        if not isinstance(row, dict):
            continue
        strongest_domain_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('title') or ''))}</td>"
            f"<td>{_fmt_score(row.get('score'))}</td>"
            f"<td>{html.escape(str(row.get('status') or '')) or 'strong'}</td>"
            "</tr>"
        )
    if not strongest_domain_rows:
        strongest_domain_rows.append("<tr><td colspan='3'>No strong-domain snapshot was available.</td></tr>")

    frontier_domain_rows = []
    for row in (readiness.get("frontier_domains") or [])[:6]:
        if not isinstance(row, dict):
            continue
        frontier_domain_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('title') or ''))}</td>"
            f"<td>{_fmt_score(row.get('score'))}</td>"
            f"<td>{html.escape(str(row.get('status') or '')) or 'watch'}</td>"
            "</tr>"
        )
    if not frontier_domain_rows:
        frontier_domain_rows.append("<tr><td colspan='3'>No frontier-domain snapshot was available.</td></tr>")

    proof_cards = []
    proof_specs = [
        ("Immutable Ledger", proof_stack.get("immutable_ledger") if isinstance(proof_stack.get("immutable_ledger"), dict) else {}),
        ("Promotion Packet", proof_stack.get("promotion_packet") if isinstance(proof_stack.get("promotion_packet"), dict) else {}),
        ("Training Lineage", proof_stack.get("training_lineage") if isinstance(proof_stack.get("training_lineage"), dict) else {}),
        ("Cross-Host Parity", proof_stack.get("cross_host_parity") if isinstance(proof_stack.get("cross_host_parity"), dict) else {}),
        ("Restore Discipline", proof_stack.get("restore_discipline") if isinstance(proof_stack.get("restore_discipline"), dict) else {}),
    ]
    for title, details in proof_specs:
        if not isinstance(details, dict):
            details = {}
        chips = []
        for key, value in details.items():
            if isinstance(value, bool):
                chips.append(f"{_titleize_slug(key)}: {'yes' if value else 'no'}")
            elif isinstance(value, (int, float)):
                chips.append(f"{_titleize_slug(key)}: {_fmt_score(value) if isinstance(value, float) else value}")
            elif value:
                chips.append(f"{_titleize_slug(key)}: {value}")
        proof_cards.append(
            f"""
            <article class="feature-card">
              <div class="feature-title">{html.escape(title)}</div>
              <p>{html.escape(' · '.join(chips[:6]) or 'No proof details were available on this refresh.')}</p>
            </article>
            """.strip()
        )

    architecture_summary_rows = []
    for key, value in (architecture.get("summary") or {}).items():
        architecture_summary_rows.append(
            f"<span class='role-pill'>{html.escape(_titleize_slug(key))}: {html.escape(str(value))}</span>"
        )

    special_feature_rows = []
    for key, value in special_features.items():
        special_feature_rows.append(
            f"<article class='feature-card'><div class='feature-title'>{html.escape(str(key).replace('_', ' ').title())}</div><p>{html.escape(str(value))}</p></article>"
        )

    report_rows = []
    for row in catalog:
        if not isinstance(row, dict):
            continue
        available_formats = row.get("available_formats") if isinstance(row.get("available_formats"), list) else []
        links = []
        for item in available_formats:
            if not isinstance(item, dict):
                continue
            path = Path(str(item.get("path") or ""))
            label = str(item.get("format") or "file").upper()
            if path.exists():
                links.append(_link_entry(path, label))
        link_html = " · ".join(link for link in links if link) or "<span class='muted'>not available</span>"
        report_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('title') or ''))}</td>"
            f"<td>{'ready' if bool(row.get('available')) else 'missing'}</td>"
            f"<td>{link_html}</td>"
            "</tr>"
        )

    timeline_rows = "".join(f"<li>{html.escape(str(item))}</li>" for item in timeline[:10]) or "<li>No timeline bullets were available.</li>"
    watchlist_rows = "".join(f"<li>{html.escape(str(item))}</li>" for item in watchlist[:8]) or "<li>No active watch items were published on this refresh.</li>"

    role_rows = []
    for role, count in sorted((active.get("roles") or {}).items(), key=lambda item: (-_safe_int(item[1], 0), str(item[0]))):
        role_rows.append(f"<span class='role-pill'>{html.escape(str(role))}: {_safe_int(count, 0)}</span>")

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>System Summary</title>
  <style>
    @page {{
      size: letter;
      margin: 0.55in;
    }}
    :root {{
      --ink: #14202a;
      --muted: #5a6876;
      --line: #d8dee6;
      --paper: #fbfaf7;
      --card: #ffffff;
      --accent: #0f766e;
      --accent-soft: #d8f1ed;
      --gold: #9a6700;
      --blue: #0b5cad;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f7f3ea 0%, #f9fbfd 42%, #ffffff 100%);
      line-height: 1.45;
    }}
    .page {{
      max-width: 8.1in;
      margin: 0 auto;
      padding: 0.08in;
    }}
    .hero {{
      background: radial-gradient(circle at top right, rgba(15,118,110,0.14), transparent 32%), linear-gradient(135deg, #fffdf9 0%, #eef6f7 100%);
      border: 1px solid rgba(20,32,42,0.08);
      border-radius: 24px;
      padding: 28px 30px 24px;
      box-shadow: 0 18px 44px rgba(20,32,42,0.08);
      margin-bottom: 18px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: -0.03em;
    }}
    .hero .sub {{
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 18px;
    }}
    .hero-grid, .metrics-grid, .upgrade-grid, .feature-grid {{
      display: grid;
      gap: 12px;
    }}
    .hero-grid {{ grid-template-columns: repeat(4, 1fr); }}
    .metrics-grid {{ grid-template-columns: repeat(3, 1fr); margin-top: 14px; }}
    .upgrade-grid, .feature-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .two-up {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }}
    .hero-card, .metric-card, .upgrade-card, .feature-card, .panel {{
      background: var(--card);
      border: 1px solid rgba(20,32,42,0.08);
      border-radius: 18px;
      padding: 14px 16px;
    }}
    .hero-card .label, .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.11em;
      font-size: 11px;
      color: var(--muted);
    }}
    .hero-card .value, .metric {{
      font-size: 26px;
      font-weight: 700;
      margin-top: 6px;
      letter-spacing: -0.03em;
    }}
    .hero-card .detail, .submetric {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
    }}
    .section {{
      margin-top: 18px;
    }}
    .section h2 {{
      margin: 0 0 8px;
      font-size: 20px;
      letter-spacing: -0.02em;
    }}
    .section .lead {{
      color: var(--muted);
      margin: 0 0 12px;
      font-size: 13px;
    }}
    .status-chip {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: var(--accent-soft);
      color: var(--accent);
      margin-top: 8px;
    }}
    .status-ready {{ background: #def5eb; color: #0f7a4e; }}
    .status-degraded {{ background: #fff2d7; color: #9a6700; }}
    .status-blocked {{ background: #ffe1e1; color: #b42318; }}
    .status-missing {{ background: #edf1f7; color: #617188; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid rgba(20,32,42,0.08);
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 12px;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      background: #f6f8fb;
    }}
    tr:last-child td {{ border-bottom: none; }}
    .role-pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #f1f5f9;
      margin: 0 8px 8px 0;
      font-size: 12px;
    }}
    .muted {{ color: var(--muted); }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    a {{
      color: var(--blue);
      text-decoration: none;
    }}
    .footer-note {{
      margin-top: 16px;
      color: var(--muted);
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">Compiled System Summary</div>
      <h1>Trading Platform Executive Packet</h1>
      <p class="sub">Compiled from live artifacts, explainers, timelines, showcase docs, and operator reports. Generated {html.escape(generated)}.</p>
      <div class="hero-grid">
        <article class="hero-card">
          <div class="label">Overall Grade</div>
          <div class="value">{html.escape(str(board.get("overall_letter_grade") or ""))}</div>
          <div class="detail">raw {html.escape(str(board.get("raw_overall_letter_grade") or ""))}</div>
        </article>
        <article class="hero-card">
          <div class="label">Institutional Score</div>
          <div class="value">{_fmt_score(readiness.get("overall_score"))}</div>
          <div class="detail">{html.escape(str(readiness.get("overall_status") or ""))}</div>
        </article>
        <article class="hero-card">
          <div class="label">Active Bots</div>
          <div class="value">{_safe_int(active.get("active_count"), 0)}</div>
          <div class="detail">{_safe_int(active.get("total_registered"), 0)} registered</div>
        </article>
        <article class="hero-card">
          <div class="label">Document Library</div>
          <div class="value">{sum(1 for row in catalog if bool(row.get("available")))}</div>
          <div class="detail">ready explainers and reports</div>
        </article>
      </div>
      <div class="panel" style="margin-top: 14px;">
        <strong>Executive summary</strong>
        <p>{html.escape(str(payload.get("executive_summary") or ""))}</p>
      </div>
    </section>

    <section class="section">
      <h2>Section Grade Board</h2>
      <p class="lead">This is the current board-level read of the system across architecture, live readiness, storage, training, ops, portability, and partner surfaces.</p>
      <div class="metrics-grid">
        {''.join(section_cards)}
      </div>
    </section>

    <section class="section">
      <h2>Operating Posture</h2>
      <p class="lead">The current runtime snapshot ties together live readiness, training proof, storage/cost pressure, and incident closeout so the system can be explained as an operator would see it.</p>
      <div class="hero-grid">
        <article class="hero-card">
          <div class="label">Live Readiness</div>
          <div class="value">{_fmt_score(ops.get("live_readiness_score"))}</div>
          <div class="detail">{html.escape(str(ops.get("live_mode") or ""))}</div>
        </article>
        <article class="hero-card">
          <div class="label">Training Quality Index</div>
          <div class="value">{_fmt_score(ops.get("training_quality_index"))}</div>
          <div class="detail">base {_fmt_score(ops.get("training_quality_base_score"))} + bonus {_fmt_score(ops.get("training_quality_bonus_score"))}</div>
        </article>
        <article class="hero-card">
          <div class="label">Storage Pressure</div>
          <div class="value">{_fmt_score(ops.get("pressure_index"))}</div>
          <div class="detail">{_fmt_score(ops.get("tracked_sqlite_gb"))} GB tracked</div>
        </article>
        <article class="hero-card">
          <div class="label">Incident Closeout</div>
          <div class="value">{_fmt_score(ops.get("closeout_score"))}</div>
          <div class="detail">cross-host proofs {_safe_int(ops.get("cross_host_proof_count"), 0)}</div>
        </article>
      </div>
      <div class="panel" style="margin-top: 12px;">
        <strong>One Numbers</strong>
        <p>
          Daily decisions: {_safe_int(ops.get("decision_rows_day"), 0)} · Governance rows: {_safe_int(ops.get("governance_rows_day"), 0)} ·
          Blocked rate: {_fmt_pct(ops.get("blocked_rate_day"))} · Paper executions: {_safe_int(ops.get("paper_executions_day"), 0)} ·
          Restore proof ready: {str(bool(ops.get("restore_proof_ready"))).lower()}
        </p>
      </div>
    </section>

    <section class="section">
      <h2>Proof Stack</h2>
      <p class="lead">These are the live proof contracts behind the platform: immutable experiment evidence, promotion packets, replayability, cross-host parity, and restore discipline.</p>
      <div class="feature-grid">
        {''.join(proof_cards)}
      </div>
    </section>

    <section class="section">
      <h2>Institutional Domain Map</h2>
      <p class="lead">A compact view of the strongest operating domains and the current frontier domains that still have the most room to grow.</p>
      <div class="two-up">
        <div>
          <table>
            <thead>
              <tr><th>Strongest Domain</th><th>Score</th><th>Status</th></tr>
            </thead>
            <tbody>
              {''.join(strongest_domain_rows)}
            </tbody>
          </table>
        </div>
        <div>
          <table>
            <thead>
              <tr><th>Frontier Domain</th><th>Score</th><th>Status</th></tr>
            </thead>
            <tbody>
              {''.join(frontier_domain_rows)}
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Active Bots</h2>
      <p class="lead">This is the current live roster snapshot for the platform, including top active bots and role mix.</p>
      <div class="panel">{''.join(role_rows) or "<span class='muted'>No role snapshot was available.</span>"}</div>
      <table style="margin-top: 12px;">
        <thead>
          <tr><th>Bot ID</th><th>Role</th><th>Test Accuracy</th><th>Quality Score</th></tr>
        </thead>
        <tbody>
          {''.join(top_bot_rows)}
        </tbody>
      </table>
    </section>

    <section class="section">
      <h2>Architectural Upgrades</h2>
      <p class="lead">These are the upgrade proof surfaces that make the platform more than a trading script: cross-platform proof, switchboard control, self-healing ops, governance packets, and restore discipline.</p>
      <div class="panel" style="margin-bottom: 12px;">{''.join(architecture_summary_rows) or "<span class='muted'>No architecture summary fields were published on this refresh.</span>"}</div>
      <div class="upgrade-grid">
        {''.join(architecture_rows)}
      </div>
    </section>

    <section class="section">
      <h2>Special Features</h2>
      <p class="lead">A concise feature story for showing the system off without losing operational honesty.</p>
      <div class="feature-grid">
        {''.join(special_feature_rows)}
      </div>
    </section>

    <section class="section">
      <h2>Timeline Highlights</h2>
      <p class="lead">Recent buildout milestones and report-facing moments pulled from the current project timeline artifact.</p>
      <div class="panel">
        <ul>{timeline_rows}</ul>
      </div>
    </section>

    <section class="section">
      <h2>Live Watchlist</h2>
      <p class="lead">A concise watch surface so the packet stays honest about current operator attention without overwhelming the executive story.</p>
      <div class="panel">
        <ul>{watchlist_rows}</ul>
      </div>
    </section>

    <section class="section">
      <h2>Individual Document Library</h2>
      <p class="lead">The compiled packet does not replace the individual explainers and PDFs. It sits above them and keeps them one click away.</p>
      <table>
        <thead>
          <tr><th>Document</th><th>Status</th><th>Formats</th></tr>
        </thead>
        <tbody>
          {''.join(report_rows)}
        </tbody>
      </table>
    </section>

    <p class="footer-note">This packet is designed to stay current without breaking the individual document workflow. The framework map, special features PDF, project timeline, one numbers outputs, training report, incident report, and report bundle are all preserved as standalone artifacts.</p>
  </div>
</body>
</html>
"""
    return html_doc


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a professional compiled system summary from the live report and explainer artifacts.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--out-file", default=str(LATEST_JSON_PATH))
    parser.add_argument("--refresh-supporting-artifacts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render-pdf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-gui-pdf-renderer", action=argparse.BooleanOptionalAction, default=_default_allow_gui_pdf_renderer())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(project_root, refresh_supporting_artifacts=bool(args.refresh_supporting_artifacts))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    timestamp_html = out_dir / f"system_summary_{timestamp}.html"
    html_doc = _render_html(payload)
    timestamp_html.write_text(html_doc, encoding="utf-8")
    LATEST_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_HTML_PATH.write_text(html_doc, encoding="utf-8")

    pdf_ok = False
    pdf_detail = "render_skipped"
    timestamp_pdf = out_dir / f"system_summary_{timestamp}.pdf"
    if bool(args.render_pdf):
        pdf_ok, pdf_detail = _render_pdf_from_html(
            LATEST_HTML_PATH,
            timestamp_pdf,
            allow_gui_renderer=bool(args.allow_gui_pdf_renderer),
        )
        if pdf_ok and timestamp_pdf.exists():
            shutil.copyfile(timestamp_pdf, LATEST_PDF_PATH)

    payload["html_paths"] = {
        "latest": str(LATEST_HTML_PATH),
        "timestamped": str(timestamp_html),
    }
    payload["pdf"] = {
        "enabled": bool(args.render_pdf),
        "ok": bool(pdf_ok),
        "detail": str(pdf_detail),
        "latest": str(LATEST_PDF_PATH),
        "timestamped": str(timestamp_pdf),
    }
    payload["report_catalog"] = _report_catalog(project_root)

    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "system_summary_report "
            f"overall_status={payload.get('overall_status', '')} "
            f"overall_grade={((payload.get('section_grade_board') or {}).get('overall_letter_grade') or '')}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
