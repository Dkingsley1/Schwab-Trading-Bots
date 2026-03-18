#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RENDER_SUPPORT_DIR = PROJECT_ROOT / "exports" / "reports" / "pdf_render_sources"
INDEX_HTML_PATH = PROJECT_ROOT / "exports" / "reports" / "report_pdf_bundle_latest.html"
INDEX_PDF_PATH = PROJECT_ROOT / "exports" / "reports" / "report_pdf_bundle_latest.pdf"
LATEST_METADATA_PATH = PROJECT_ROOT / "governance" / "health" / "report_pdf_bundle_latest.json"


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except Exception as exc:
        return 1, "", str(exc)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _pdf_renderer_binary(allow_gui_renderer: bool) -> tuple[str, str]:
    env_override = (
        os.getenv("REPORT_PDF_BUNDLE_PDF_BIN", "").strip()
        or os.getenv("TRAINING_REPORT_PDF_BIN", "").strip()
        or os.getenv("PROJECT_TIMELINE_PDF_BIN", "").strip()
    )
    if env_override:
        env_bin = Path(env_override).expanduser()
        if env_bin.exists():
            kind = "wkhtmltopdf" if env_bin.name == "wkhtmltopdf" else "browser"
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

    if allow_gui_renderer:
        for candidate in (
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
            Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        ):
            if candidate.exists():
                return str(candidate), "browser"

    return "", ""


def _render_pdf_from_html(html_path: Path, pdf_path: Path, *, allow_gui_renderer: bool) -> tuple[bool, str]:
    renderer, renderer_kind = _pdf_renderer_binary(allow_gui_renderer=allow_gui_renderer)
    if not renderer:
        return False, "pdf_renderer_not_found"
    html_uri = html_path.resolve().as_uri()
    if renderer_kind == "wkhtmltopdf":
        cmd = [renderer, html_uri, str(pdf_path)]
    else:
        cmd = [renderer, "--headless", "--disable-gpu", f"--print-to-pdf={pdf_path}", html_uri]
    rc, out, err = _run(cmd)
    if rc == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0:
        return True, out or "ok"
    return False, err or out or f"rc={rc}"


def _latest_artifact(pattern: str) -> Path | None:
    rows = []
    for value in glob.glob(pattern):
        path = Path(value)
        if not path.is_file():
            continue
        name = path.name
        if ".local_fallback" in name or name == ".DS_Store" or "latest" in name:
            continue
        rows.append(path)
    if not rows:
        return None
    rows.sort(key=lambda item: (item.stat().st_mtime, item.name))
    return rows[-1]


def _existing(path: Path) -> Path | None:
    return path if path.exists() and path.is_file() else None


def _preferred_markdown_or_json(markdown_path: Path, json_path: Path) -> tuple[str, Path | None]:
    markdown = _existing(markdown_path)
    if markdown is not None:
        return "markdown", markdown
    json_candidate = _existing(json_path)
    if json_candidate is not None:
        return "json", json_candidate
    return "markdown", None


def _build_specs(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    reports_dir = project_root / "exports" / "reports"
    sql_reports_dir = project_root / "exports" / "sql_reports"
    one_numbers_dir = project_root / "exports" / "one_numbers"
    state_snapshot_dir = project_root / "exports" / "state_snapshot_drills"
    governance_health_dir = project_root / "governance" / "health"

    crash_html = reports_dir / "crash_reports" / "crash_report_digest_print_latest.html"
    project_timeline_html = reports_dir / "project_timeline" / "project_timeline_print_latest.html"
    training_html = reports_dir / "training_reports" / "training_report_print_latest.html"
    daily_ops_md = reports_dir / "daily_ops_report_latest.md"
    retrain_md = _latest_artifact(str(sql_reports_dir / "retrain_scorecard_*.md"))
    daily_runtime_json = _latest_artifact(str(sql_reports_dir / "daily_runtime_summary_*.json"))
    replay_json = _latest_artifact(str(sql_reports_dir / "replay_feature_ablation_*.json"))
    one_numbers_md = _latest_artifact(str(one_numbers_dir / "one_numbers_*.md"))
    strategy_kind, strategy_source = _preferred_markdown_or_json(
        reports_dir / "strategy_attribution_latest.md",
        governance_health_dir / "strategy_attribution_latest.json",
    )
    post_trade_kind, post_trade_source = _preferred_markdown_or_json(
        reports_dir / "post_trade_analysis_latest.md",
        governance_health_dir / "post_trade_analysis_latest.json",
    )

    return [
        {
            "slug": "crash_report_digest",
            "title": "Crash Report Digest",
            "kind": "html",
            "source_path": crash_html,
            "pdf_path": reports_dir / "crash_reports" / "crash_report_digest_latest.pdf",
        },
        {
            "slug": "project_timeline",
            "title": "Project Timeline Report",
            "kind": "html",
            "source_path": project_timeline_html,
            "pdf_path": reports_dir / "project_timeline" / "project_timeline_latest.pdf",
        },
        {
            "slug": "training_report",
            "title": "Training Report",
            "kind": "html",
            "source_path": training_html,
            "pdf_path": reports_dir / "training_reports" / "training_report_latest.pdf",
        },
        {
            "slug": "daily_ops_report",
            "title": "Daily Ops Report",
            "kind": "markdown",
            "source_path": daily_ops_md,
            "pdf_path": reports_dir / "daily_ops_report_latest.pdf",
        },
        {
            "slug": "retrain_scorecard",
            "title": "Retrain Scorecard",
            "kind": "markdown",
            "source_path": retrain_md,
            "pdf_path": sql_reports_dir / "retrain_scorecard_latest.pdf",
        },
        {
            "slug": "unified_lane_scorecard",
            "title": "Unified Lane Scorecard",
            "kind": "markdown",
            "source_path": sql_reports_dir / "unified_lane_scorecard_latest.md",
            "pdf_path": sql_reports_dir / "unified_lane_scorecard_latest.pdf",
        },
        {
            "slug": "daily_runtime_summary",
            "title": "Daily Runtime Summary",
            "kind": "json",
            "source_path": daily_runtime_json,
            "pdf_path": sql_reports_dir / "daily_runtime_summary_latest.pdf",
        },
        {
            "slug": "daily_auto_verify",
            "title": "Daily Auto Verify",
            "kind": "json",
            "source_path": _existing(governance_health_dir / "daily_auto_verify_latest.json") or _latest_artifact(str(sql_reports_dir / "daily_auto_verify_*.json")),
            "pdf_path": sql_reports_dir / "daily_auto_verify_latest.pdf",
        },
        {
            "slug": "model_card",
            "title": "Model Card",
            "kind": "json",
            "source_path": _existing(governance_health_dir / "model_card_latest.json") or _latest_artifact(str(sql_reports_dir / "model_card_*.json")),
            "pdf_path": sql_reports_dir / "model_card_latest.pdf",
        },
        {
            "slug": "bot_explainability",
            "title": "Bot Explainability",
            "kind": "json",
            "source_path": _existing(governance_health_dir / "bot_explainability_latest.json") or _latest_artifact(str(sql_reports_dir / "bot_explainability_*.json")),
            "pdf_path": sql_reports_dir / "bot_explainability_latest.pdf",
        },
        {
            "slug": "paper_execution_calibration",
            "title": "Paper Execution Calibration",
            "kind": "json",
            "source_path": _existing(governance_health_dir / "paper_execution_calibration_latest.json"),
            "pdf_path": sql_reports_dir / "paper_execution_calibration_latest.pdf",
        },
        {
            "slug": "strategy_attribution",
            "title": "Strategy Attribution",
            "kind": strategy_kind,
            "source_path": strategy_source,
            "pdf_path": reports_dir / "strategy_attribution_latest.pdf",
        },
        {
            "slug": "post_trade_analysis",
            "title": "Post-Trade Analysis",
            "kind": post_trade_kind,
            "source_path": post_trade_source,
            "pdf_path": reports_dir / "post_trade_analysis_latest.pdf",
        },
        {
            "slug": "replay_feature_ablation",
            "title": "Replay Feature Ablation",
            "kind": "json",
            "source_path": replay_json,
            "pdf_path": sql_reports_dir / "replay_feature_ablation_latest.pdf",
        },
        {
            "slug": "one_numbers",
            "title": "One Numbers Report",
            "kind": "markdown",
            "source_path": one_numbers_md,
            "pdf_path": one_numbers_dir / "one_numbers_latest.pdf",
        },
        {
            "slug": "state_snapshot_drills",
            "title": "State Snapshot Drill",
            "kind": "json",
            "source_path": state_snapshot_dir / "latest.json",
            "pdf_path": state_snapshot_dir / "state_snapshot_drills_latest.pdf",
        },
    ]


def _markdown_to_html(text: str) -> str:
    out: list[str] = []
    in_list = False
    in_code = False
    code_lines: list[str] = []

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    def close_code() -> None:
        nonlocal in_code, code_lines
        if in_code:
            out.append("<pre class=\"content\">" + html.escape("\n".join(code_lines)) + "</pre>")
            in_code = False
            code_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if stripped.startswith("```"):
            close_list()
            if in_code:
                close_code()
            else:
                in_code = True
                code_lines = []
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not stripped:
            close_list()
            continue
        if stripped.startswith("# "):
            close_list()
            out.append(f"<h1>{html.escape(stripped[2:].strip())}</h1>")
            continue
        if stripped.startswith("## "):
            close_list()
            out.append(f"<h2>{html.escape(stripped[3:].strip())}</h2>")
            continue
        if stripped.startswith("### "):
            close_list()
            out.append(f"<h3>{html.escape(stripped[4:].strip())}</h3>")
            continue
        if stripped.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{html.escape(stripped[2:].strip())}</li>")
            continue
        close_list()
        out.append(f"<p>{html.escape(stripped)}</p>")

    close_code()
    close_list()
    return "\n".join(out) or "<p>No markdown content available.</p>"


def _json_to_html(path: Path) -> str:
    payload = _load_json(path)
    if payload:
        summary_rows = []
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                continue
            summary_rows.append(
                "<tr>"
                f"<th>{html.escape(str(key))}</th>"
                f"<td>{html.escape(str(value))}</td>"
                "</tr>"
            )
            if len(summary_rows) >= 12:
                break
        summary_html = (
            "<table><tbody>" + "".join(summary_rows) + "</tbody></table>" if summary_rows else "<p>No scalar summary fields available.</p>"
        )
        pretty = json.dumps(payload, ensure_ascii=True, indent=2)
    else:
        summary_html = "<p>JSON payload could not be parsed; raw text shown below.</p>"
        pretty = path.read_text(encoding="utf-8")
    return summary_html + "<pre class=\"content\">" + html.escape(pretty) + "</pre>"


def _wrap_html(*, title: str, source_path: Path, body_html: str, generated_utc: str) -> str:
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3efe6;
      --ink: #1f2933;
      --muted: #66737f;
      --card: #fffaf2;
      --line: #d7ccb9;
      --accent: #9a3412;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: linear-gradient(180deg, #efe8db 0%, #f7f3ec 100%); color: var(--ink); font: 15px/1.55 Georgia, 'Times New Roman', serif; }}
    .page {{ max-width: 980px; margin: 0 auto; padding: 34px 24px 48px; }}
    .hero, .section {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 10px 26px rgba(31, 41, 51, 0.08); }}
    .hero {{ padding: 24px 26px; }}
    .section {{ margin-top: 18px; padding: 18px 22px; }}
    h1, h2, h3 {{ margin: 0 0 10px; font-family: 'Avenir Next', 'Segoe UI', sans-serif; }}
    h1 {{ font-size: 30px; }}
    h2 {{ margin-top: 20px; font-size: 20px; }}
    h3 {{ margin-top: 18px; font-size: 16px; }}
    p.meta {{ margin: 0; color: var(--muted); }}
    p {{ margin: 10px 0 0; }}
    ul {{ margin: 10px 0 0 20px; padding: 0; }}
    li {{ margin: 5px 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; vertical-align: top; }}
    th {{ width: 28%; color: var(--muted); font-family: 'Avenir Next', 'Segoe UI', sans-serif; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
    pre.content {{ margin: 14px 0 0; padding: 16px; border-radius: 14px; background: #f7f1e7; border: 1px solid #eadfce; white-space: pre-wrap; word-break: break-word; font: 12px/1.5 'SF Mono', 'Menlo', monospace; }}
    .path {{ margin-top: 10px; font: 12px/1.4 'SF Mono', 'Menlo', monospace; color: var(--accent); word-break: break-all; }}
  </style>
</head>
<body>
  <div class=\"page\">
    <section class=\"hero\">
      <h1>{html.escape(title)}</h1>
      <p class=\"meta\">PDF generated {html.escape(generated_utc)}</p>
      <p class=\"path\">Source: {html.escape(str(source_path))}</p>
    </section>
    <section class=\"section\">
      {body_html}
    </section>
  </div>
</body>
</html>
"""


def _render_entry_html(spec: dict[str, Any], *, generated_utc: str) -> str:
    source_path = Path(spec["source_path"])
    kind = str(spec["kind"])
    if kind == "markdown":
        body = _markdown_to_html(source_path.read_text(encoding="utf-8"))
    elif kind == "json":
        body = _json_to_html(source_path)
    else:
        body = "<p>This PDF was rendered directly from the source HTML report.</p>"
    return _wrap_html(title=str(spec["title"]), source_path=source_path, body_html=body, generated_utc=generated_utc)


def _render_index_html(entries: list[dict[str, Any]], *, generated_utc: str) -> str:
    rows = []
    for entry in entries:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(entry.get('title', '')))}</td>"
            f"<td>{html.escape(str(entry.get('kind', '')))}</td>"
            f"<td>{html.escape(str(entry.get('status', '')))}</td>"
            f"<td>{html.escape(str(entry.get('pdf_path', '')))}</td>"
            f"<td>{html.escape(str(entry.get('source_path', '')))}</td>"
            "</tr>"
        )
    body = (
        "<h2>Generated PDF Inventory</h2>"
        "<table><thead><tr><th>Report</th><th>Kind</th><th>Status</th><th>PDF</th><th>Source</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    return _wrap_html(
        title="Trading System PDF Bundle",
        source_path=LATEST_METADATA_PATH,
        body_html=body,
        generated_utc=generated_utc,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PDF companions for the project report families.")
    parser.add_argument(
        "--allow-gui-pdf-renderer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow GUI browser app bundles when no CLI PDF renderer is available.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.allow_gui_pdf_renderer is None:
        allow_gui_pdf_renderer = _env_flag("REPORT_PDF_BUNDLE_ALLOW_GUI_PDF_RENDERER", "0")
    else:
        allow_gui_pdf_renderer = bool(args.allow_gui_pdf_renderer)

    generated_utc = datetime.now(timezone.utc).isoformat()
    RENDER_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for spec in _build_specs():
        source_path = spec.get("source_path")
        pdf_path = Path(spec["pdf_path"])
        entry = {
            "slug": spec["slug"],
            "title": spec["title"],
            "kind": spec["kind"],
            "source_path": str(source_path) if source_path else "",
            "pdf_path": str(pdf_path),
            "status": "missing_source",
            "detail": "missing_source",
            "bytes": 0,
            "support_html": "",
        }
        if source_path is None:
            entries.append(entry)
            continue
        source = Path(source_path)
        if not source.exists() or not source.is_file():
            entries.append(entry)
            continue

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        if pdf_path.exists():
            pdf_path.unlink()

        html_source = source
        if spec["kind"] != "html":
            support_html = RENDER_SUPPORT_DIR / f"{spec['slug']}_latest.html"
            support_html.write_text(_render_entry_html(spec, generated_utc=generated_utc), encoding="utf-8")
            html_source = support_html
            entry["support_html"] = str(support_html)

        ok, detail = _render_pdf_from_html(html_source, pdf_path, allow_gui_renderer=allow_gui_pdf_renderer)
        entry["status"] = "ok" if ok else "error"
        entry["detail"] = detail
        entry["bytes"] = int(pdf_path.stat().st_size) if ok and pdf_path.exists() else 0
        entries.append(entry)

    INDEX_HTML_PATH.write_text(_render_index_html(entries, generated_utc=generated_utc), encoding="utf-8")
    if INDEX_PDF_PATH.exists():
        INDEX_PDF_PATH.unlink()
    index_ok, index_detail = _render_pdf_from_html(INDEX_HTML_PATH, INDEX_PDF_PATH, allow_gui_renderer=allow_gui_pdf_renderer)

    payload = {
        "generated_utc": generated_utc,
        "allow_gui_pdf_renderer": bool(allow_gui_pdf_renderer),
        "index_html": str(INDEX_HTML_PATH),
        "index_pdf": str(INDEX_PDF_PATH) if INDEX_PDF_PATH.exists() else "",
        "index_ok": bool(index_ok),
        "index_detail": str(index_detail),
        "entries": entries,
        "ok_count": sum(1 for row in entries if row.get("status") == "ok"),
        "missing_count": sum(1 for row in entries if row.get("status") == "missing_source"),
        "error_count": sum(1 for row in entries if row.get("status") == "error"),
    }
    LATEST_METADATA_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "report_pdf_bundle "
            f"ok_count={payload['ok_count']} missing_count={payload['missing_count']} error_count={payload['error_count']}"
        )
        if INDEX_PDF_PATH.exists():
            print(f"Index PDF: {INDEX_PDF_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
