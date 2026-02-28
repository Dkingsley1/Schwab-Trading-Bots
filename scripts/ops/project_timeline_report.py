#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "exports" / "reports" / "project_timeline"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "project_timeline_state.json"


def _run(cmd: List[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except Exception as exc:  # pragma: no cover - defensive fallback
        return 1, "", str(exc)


def _git_ok() -> bool:
    rc, out, _ = _run(["git", "rev-parse", "--is-inside-work-tree"])
    return rc == 0 and out.strip().lower() == "true"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _collect_git() -> Dict[str, Any]:
    if not _git_ok():
        return {
            "available": False,
            "branch": "unknown",
            "head": "unknown",
            "commits": [],
            "commit_count": 0,
            "status_porcelain": "",
            "status_lines": [],
            "status_branch_line": "",
        }

    _, branch, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    _, head, _ = _run(["git", "rev-parse", "HEAD"])
    _, log_out, _ = _run(["git", "log", "--reverse", "--date=iso", "--pretty=format:%ad|%h|%s"])
    _, status_porcelain, _ = _run(["git", "status", "--short", "--branch"])

    commits: List[Dict[str, str]] = []
    for line in log_out.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        commits.append(
            {
                "date": parts[0].strip(),
                "sha": parts[1].strip(),
                "subject": parts[2].strip(),
            }
        )

    status_lines = status_porcelain.splitlines()
    status_branch_line = status_lines[0] if status_lines and status_lines[0].startswith("## ") else ""
    working_tree = [line for line in status_lines if not line.startswith("## ")]

    return {
        "available": True,
        "branch": branch.strip() or "unknown",
        "head": head.strip() or "unknown",
        "commits": commits,
        "commit_count": len(commits),
        "status_porcelain": status_porcelain,
        "status_lines": working_tree,
        "status_branch_line": status_branch_line,
    }


def _classify_status(lines: List[str]) -> Dict[str, int]:
    counts = {
        "modified": 0,
        "added": 0,
        "deleted": 0,
        "renamed": 0,
        "untracked": 0,
        "other": 0,
    }
    for line in lines:
        code = line[:2]
        if code == "??":
            counts["untracked"] += 1
            continue
        if "M" in code:
            counts["modified"] += 1
        elif "A" in code:
            counts["added"] += 1
        elif "D" in code:
            counts["deleted"] += 1
        elif "R" in code:
            counts["renamed"] += 1
        else:
            counts["other"] += 1
    return counts


def _latest_file_name(glob_pat: str) -> str:
    files = sorted(PROJECT_ROOT.glob(glob_pat), key=lambda p: p.name)
    return files[-1].name if files else ""


def _collect_ops_snapshot() -> Dict[str, Any]:
    promotion = _load_json(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_gate_latest.json")
    graduation = _load_json(PROJECT_ROOT / "governance" / "walk_forward" / "new_bot_graduation_latest.json")
    retrain = _load_json(PROJECT_ROOT / "governance" / "health" / "retrain_scorecard_latest.json")
    leak = _load_json(PROJECT_ROOT / "governance" / "health" / "leak_overfit_guard_latest.json")
    preflight = _load_json(PROJECT_ROOT / "governance" / "health" / "preflight_autofix_latest.json")
    storage = _load_json(PROJECT_ROOT / "governance" / "health" / "storage_failback_sync_latest.json")

    preflight_events: List[Dict[str, str]] = []
    for path in sorted(PROJECT_ROOT.glob("logs/all_sleeves_*.log"), key=lambda p: p.name):
        stamp = path.stem.replace("all_sleeves_", "")
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]
        except Exception:
            continue
        preflight_line = next((line for line in lines if line.startswith("PREFLIGHT ")), "")
        if not preflight_line:
            continue
        fail_line = next((line for line in lines if line.startswith(" - FAIL ")), "")
        preflight_events.append(
            {
                "stamp": stamp,
                "result": preflight_line.replace("PREFLIGHT ", "", 1).strip(),
                "detail": fail_line.replace(" - FAIL ", "", 1).strip() if fail_line else "",
            }
        )
    preflight_events = preflight_events[-25:]

    return {
        "promotion": promotion,
        "graduation": graduation,
        "retrain": retrain,
        "leak": leak,
        "preflight": preflight,
        "storage": storage,
        "latest_all_sleeves_log": _latest_file_name("logs/all_sleeves_*.log"),
        "latest_coinbase_log": _latest_file_name("logs/coinbase_live_*.log"),
        "preflight_events": preflight_events,
    }


def _build_signature(git_data: Dict[str, Any], ops_data: Dict[str, Any]) -> str:
    token = {
        "head": git_data.get("head", ""),
        "branch": git_data.get("branch", ""),
        "status_porcelain": git_data.get("status_porcelain", ""),
        "latest_all_sleeves_log": ops_data.get("latest_all_sleeves_log", ""),
        "latest_coinbase_log": ops_data.get("latest_coinbase_log", ""),
        "promotion_ts": (ops_data.get("promotion") or {}).get("timestamp_utc", ""),
        "graduation_ts": (ops_data.get("graduation") or {}).get("timestamp_utc", ""),
        "retrain_ts": (ops_data.get("retrain") or {}).get("timestamp_utc", ""),
    }
    raw = json.dumps(token, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _fmt(val: Any, default: str = "n/a") -> str:
    if val is None:
        return default
    if isinstance(val, float):
        return f"{val:.6f}".rstrip("0").rstrip(".")
    txt = str(val).strip()
    return txt if txt else default


def _chrome_binary() -> str:
    env_override = os.getenv("PROJECT_TIMELINE_PDF_BIN", "").strip()
    env_bin = Path(env_override).expanduser() if env_override else None
    candidates = []
    if env_bin:
        candidates.append(env_bin)
    candidates.extend(
        [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
            Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        ]
    )
    for c in candidates:
        if c.exists():
            return str(c)
    return ""


def _render_pdf_from_html(html_path: Path, pdf_path: Path) -> tuple[bool, str]:
    chrome = _chrome_binary()
    if not chrome:
        return False, "chrome_binary_not_found"
    html_uri = html_path.resolve().as_uri()
    cmd = [
        chrome,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={pdf_path}",
        html_uri,
    ]
    rc, out, err = _run(cmd)
    if rc == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0:
        return True, out or "ok"
    detail = err or out or f"rc={rc}"
    return False, detail


def _render_markdown(context: Dict[str, Any]) -> str:
    git_data = context["git"]
    ops_data = context["ops"]
    counts = _classify_status(git_data["status_lines"])

    promotion = ops_data.get("promotion") or {}
    graduation = ops_data.get("graduation") or {}
    retrain = ops_data.get("retrain") or {}
    leak = ops_data.get("leak") or {}
    preflight = ops_data.get("preflight") or {}
    storage = ops_data.get("storage") or {}

    md: List[str] = []
    md.append("# Project Timeline Report")
    md.append("")
    md.append(f"Generated (UTC): `{context['generated_utc']}`")
    md.append(f"Generated (Local): `{context['generated_local']}`")
    md.append(f"Project root: `{PROJECT_ROOT}`")
    md.append("")
    md.append("## Snapshot")
    md.append(f"- Git available: `{git_data['available']}`")
    md.append(f"- Branch: `{git_data['branch']}`")
    md.append(f"- HEAD: `{git_data['head']}`")
    md.append(f"- Total commits: `{git_data['commit_count']}`")
    if git_data["commits"]:
        md.append(
            f"- First commit: `{git_data['commits'][0]['date']}` `{git_data['commits'][0]['sha']}` "
            f"{git_data['commits'][0]['subject']}"
        )
        md.append(
            f"- Latest commit: `{git_data['commits'][-1]['date']}` `{git_data['commits'][-1]['sha']}` "
            f"{git_data['commits'][-1]['subject']}"
        )
    md.append("")
    md.append("## Working Tree")
    md.append(f"- Modified: `{counts['modified']}`")
    md.append(f"- Added: `{counts['added']}`")
    md.append(f"- Deleted: `{counts['deleted']}`")
    md.append(f"- Renamed: `{counts['renamed']}`")
    md.append(f"- Untracked: `{counts['untracked']}`")
    md.append(f"- Other: `{counts['other']}`")
    if git_data["status_lines"]:
        md.append("")
        md.append("### Files")
        for line in git_data["status_lines"]:
            md.append(f"- `{line}`")
    else:
        md.append("- Working tree is clean.")
    md.append("")
    md.append("## Runtime and Gates")
    md.append(
        f"- Promotion gate: `promote_ok={_fmt(promotion.get('promote_ok'))}` "
        f"`fail_share={_fmt(promotion.get('fail_share'))}` "
        f"`failed/considered={_fmt(promotion.get('failed_bots'))}/{_fmt(promotion.get('considered_bots'))}` "
        f"`timestamp={_fmt(promotion.get('timestamp_utc'))}`"
    )
    maturity = graduation.get("maturity") if isinstance(graduation.get("maturity"), dict) else {}
    md.append(
        f"- Graduation gate: `ok={_fmt(graduation.get('ok'))}` "
        f"`mature_pass_rate={_fmt(maturity.get('mature_pass_rate'))}` "
        f"`immature_active_count={_fmt(graduation.get('immature_active_count'))}` "
        f"`timestamp={_fmt(graduation.get('timestamp_utc'))}`"
    )
    md.append(
        f"- Retrain scorecard: `status_counts={_fmt(retrain.get('status_counts'))}` "
        f"`master_update_status={_fmt(retrain.get('master_update_status'))}` "
        f"`failure_count={_fmt(retrain.get('failure_count'))}` "
        f"`timestamp={_fmt(retrain.get('timestamp_utc'))}`"
    )
    md.append(
        f"- Leak/overfit: `ok={_fmt(leak.get('ok'))}` "
        f"`counts={_fmt(leak.get('counts'))}` "
        f"`timestamp={_fmt(leak.get('timestamp_utc'))}`"
    )
    md.append(
        f"- Preflight autofix: `preflight_ok={_fmt(preflight.get('preflight_ok'))}` "
        f"`broker={_fmt(preflight.get('broker'))}` "
        f"`simulate={_fmt(preflight.get('simulate'))}` "
        f"`timestamp={_fmt(preflight.get('timestamp_utc'))}`"
    )
    md.append(
        f"- Storage route: `mode={_fmt(storage.get('mode'))}` "
        f"`active_root={_fmt(storage.get('active_root'))}`"
    )
    md.append(f"- Latest all_sleeves log: `{_fmt(ops_data.get('latest_all_sleeves_log'))}`")
    md.append(f"- Latest coinbase log: `{_fmt(ops_data.get('latest_coinbase_log'))}`")
    md.append("")
    md.append("## Preflight Milestones (from logs)")
    if ops_data["preflight_events"]:
        for event in ops_data["preflight_events"]:
            detail = f" | fail=`{event['detail']}`" if event.get("detail") else ""
            md.append(f"- `{event['stamp']}` | `{event['result']}`{detail}")
    else:
        md.append("- No preflight events found.")
    md.append("")
    md.append("## Full Commit Timeline")
    if git_data["commits"]:
        for idx, row in enumerate(git_data["commits"], start=1):
            md.append(f"{idx}. `{row['date']}` | `{row['sha']}` | {row['subject']}")
    else:
        md.append("1. No git history available.")
    md.append("")
    md.append("## Auto-Update")
    md.append("- This file is generated by `scripts/ops/project_timeline_report.py`.")
    md.append(
        "- Auto mode compares git + runtime signatures and only refreshes when something changes "
        "(commits, working tree, key gate artifacts, or latest loop logs)."
    )
    return "\n".join(md).strip() + "\n"


def _render_html(context: Dict[str, Any]) -> str:
    git_data = context["git"]
    ops_data = context["ops"]
    counts = _classify_status(git_data["status_lines"])

    promotion = ops_data.get("promotion") or {}
    graduation = ops_data.get("graduation") or {}
    retrain = ops_data.get("retrain") or {}
    leak = ops_data.get("leak") or {}
    preflight = ops_data.get("preflight") or {}
    storage = ops_data.get("storage") or {}
    maturity = graduation.get("maturity") if isinstance(graduation.get("maturity"), dict) else {}

    commit_rows = []
    for idx, row in enumerate(git_data["commits"], start=1):
        commit_rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{html.escape(row['date'])}</td>"
            f"<td><code>{html.escape(row['sha'])}</code></td>"
            f"<td>{html.escape(row['subject'])}</td>"
            "</tr>"
        )
    if not commit_rows:
        commit_rows.append("<tr><td colspan='4'>No git history available.</td></tr>")

    status_rows = []
    for line in git_data["status_lines"]:
        status_rows.append(f"<tr><td><code>{html.escape(line)}</code></td></tr>")
    if not status_rows:
        status_rows.append("<tr><td>Working tree is clean.</td></tr>")

    preflight_rows = []
    for event in ops_data["preflight_events"]:
        preflight_rows.append(
            "<tr>"
            f"<td><code>{html.escape(event.get('stamp', ''))}</code></td>"
            f"<td>{html.escape(event.get('result', ''))}</td>"
            f"<td>{html.escape(event.get('detail', ''))}</td>"
            "</tr>"
        )
    if not preflight_rows:
        preflight_rows.append("<tr><td colspan='3'>No preflight events found.</td></tr>")

    def li(label: str, value: Any) -> str:
        return f"<li><b>{html.escape(label)}:</b> <code>{html.escape(_fmt(value))}</code></li>"

    html_doc = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Project Timeline Report</title>
  <style>
    :root {{
      --text: #111;
      --muted: #555;
      --line: #ddd;
      --bg: #fff;
    }}
    body {{
      color: var(--text);
      background: var(--bg);
      font-family: \"Georgia\", \"Times New Roman\", serif;
      margin: 24px;
      line-height: 1.4;
    }}
    h1, h2, h3 {{
      margin: 0.6em 0 0.35em;
      page-break-after: avoid;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.95rem;
      margin-bottom: 16px;
    }}
    code {{
      font-family: \"Menlo\", \"Consolas\", monospace;
      font-size: 0.92em;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 10px 0 18px;
      page-break-inside: avoid;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      vertical-align: top;
      font-size: 0.95rem;
    }}
    th {{
      text-align: left;
      background: #f7f7f7;
    }}
    ul {{
      margin-top: 8px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}
    .panel {{
      border: 1px solid var(--line);
      padding: 10px 12px;
      page-break-inside: avoid;
    }}
    @media print {{
      body {{
        margin: 0.5in;
      }}
      a {{
        color: inherit;
        text-decoration: none;
      }}
    }}
  </style>
</head>
<body>
  <h1>Project Timeline Report</h1>
  <div class=\"meta\">
    Generated (UTC): <code>{html.escape(context['generated_utc'])}</code><br>
    Generated (Local): <code>{html.escape(context['generated_local'])}</code><br>
    Project root: <code>{html.escape(str(PROJECT_ROOT))}</code>
  </div>

  <h2>Snapshot</h2>
  <ul>
    {li("Git available", git_data["available"])}
    {li("Branch", git_data["branch"])}
    {li("HEAD", git_data["head"])}
    {li("Total commits", git_data["commit_count"])}
    {li("First commit", f"{git_data['commits'][0]['date']} {git_data['commits'][0]['sha']} {git_data['commits'][0]['subject']}" if git_data["commits"] else "n/a")}
    {li("Latest commit", f"{git_data['commits'][-1]['date']} {git_data['commits'][-1]['sha']} {git_data['commits'][-1]['subject']}" if git_data["commits"] else "n/a")}
  </ul>

  <h2>Working Tree</h2>
  <div class=\"grid\">
    <div class=\"panel\">
      <ul>
        {li("Modified", counts["modified"])}
        {li("Added", counts["added"])}
        {li("Deleted", counts["deleted"])}
        {li("Renamed", counts["renamed"])}
        {li("Untracked", counts["untracked"])}
        {li("Other", counts["other"])}
      </ul>
    </div>
    <div class=\"panel\">
      <div><b>Branch status</b></div>
      <div><code>{html.escape(git_data["status_branch_line"] or "n/a")}</code></div>
    </div>
  </div>
  <table>
    <thead><tr><th>Files</th></tr></thead>
    <tbody>
      {"".join(status_rows)}
    </tbody>
  </table>

  <h2>Runtime and Gates</h2>
  <ul>
    {li("Promotion gate promote_ok", promotion.get("promote_ok"))}
    {li("Promotion fail_share", promotion.get("fail_share"))}
    {li("Promotion failed/considered", f"{_fmt(promotion.get('failed_bots'))}/{_fmt(promotion.get('considered_bots'))}")}
    {li("Graduation ok", graduation.get("ok"))}
    {li("Graduation mature_pass_rate", maturity.get("mature_pass_rate"))}
    {li("Graduation immature_active_count", graduation.get("immature_active_count"))}
    {li("Retrain status_counts", retrain.get("status_counts"))}
    {li("Retrain master_update_status", retrain.get("master_update_status"))}
    {li("Leak/overfit ok", leak.get("ok"))}
    {li("Leak/overfit counts", leak.get("counts"))}
    {li("Preflight preflight_ok", preflight.get("preflight_ok"))}
    {li("Storage mode", storage.get("mode"))}
    {li("Storage active_root", storage.get("active_root"))}
    {li("Latest all_sleeves log", ops_data.get("latest_all_sleeves_log"))}
    {li("Latest coinbase log", ops_data.get("latest_coinbase_log"))}
  </ul>

  <h2>Preflight Milestones (from logs)</h2>
  <table>
    <thead><tr><th>Stamp</th><th>Result</th><th>First Fail Detail</th></tr></thead>
    <tbody>
      {"".join(preflight_rows)}
    </tbody>
  </table>

  <h2>Full Commit Timeline</h2>
  <table>
    <thead><tr><th>#</th><th>Date</th><th>SHA</th><th>Subject</th></tr></thead>
    <tbody>
      {"".join(commit_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    return html_doc


def _load_state(path: Path) -> Dict[str, Any]:
    return _load_json(path)


def _save_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate full project timeline reports (markdown + printable HTML).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--auto", action="store_true", help="Skip regeneration when signature is unchanged.")
    parser.add_argument("--force", action="store_true", help="Regenerate even when signature is unchanged.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    generated_utc = datetime.now(timezone.utc).isoformat()
    generated_local = datetime.now().astimezone().isoformat()

    git_data = _collect_git()
    ops_data = _collect_ops_snapshot()
    signature = _build_signature(git_data, ops_data)

    context = {
        "generated_utc": generated_utc,
        "generated_local": generated_local,
        "signature": signature,
        "git": git_data,
        "ops": ops_data,
    }

    out_dir = Path(args.output_dir)
    state_file = Path(args.state_file)
    latest_md = out_dir / "project_timeline_latest.md"
    latest_html = out_dir / "project_timeline_print_latest.html"
    latest_pdf = out_dir / "project_timeline_latest.pdf"

    state = _load_state(state_file)
    unchanged = (
        args.auto
        and not args.force
        and state.get("signature") == signature
        and latest_md.exists()
        and latest_html.exists()
        and latest_pdf.exists()
    )

    if unchanged:
        payload = {
            "changed": False,
            "signature": signature,
            "latest_markdown": str(latest_md),
            "latest_printable_html": str(latest_html),
            "latest_pdf": str(latest_pdf),
            "generated_utc": generated_utc,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(
                "project_timeline_report unchanged "
                f"latest_markdown={latest_md} latest_printable_html={latest_html}"
            )
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ts_md = out_dir / f"project_timeline_{stamp}.md"
    ts_html = out_dir / f"project_timeline_print_{stamp}.html"
    ts_pdf = out_dir / f"project_timeline_{stamp}.pdf"

    md_text = _render_markdown(context)
    html_text = _render_html(context)

    latest_md.write_text(md_text, encoding="utf-8")
    latest_html.write_text(html_text, encoding="utf-8")
    ts_md.write_text(md_text, encoding="utf-8")
    ts_html.write_text(html_text, encoding="utf-8")

    pdf_ok, pdf_detail = _render_pdf_from_html(latest_html, latest_pdf)
    if pdf_ok:
        try:
            shutil.copy2(latest_pdf, ts_pdf)
        except Exception as exc:
            pdf_ok = False
            pdf_detail = f"timestamp_pdf_copy_failed:{exc}"
            ts_pdf = Path("")
    else:
        ts_pdf = Path("")

    state_payload = {
        "signature": signature,
        "generated_utc": generated_utc,
        "latest_markdown": str(latest_md),
        "latest_printable_html": str(latest_html),
        "latest_pdf": str(latest_pdf) if latest_pdf.exists() else "",
        "timestamped_markdown": str(ts_md),
        "timestamped_printable_html": str(ts_html),
        "timestamped_pdf": str(ts_pdf) if ts_pdf else "",
        "pdf_ok": bool(pdf_ok),
        "pdf_detail": str(pdf_detail),
        "head": git_data.get("head"),
        "branch": git_data.get("branch"),
        "commit_count": git_data.get("commit_count"),
    }
    _save_state(state_file, state_payload)

    payload = {
        "changed": True,
        "signature": signature,
        "latest_markdown": str(latest_md),
        "latest_printable_html": str(latest_html),
        "latest_pdf": str(latest_pdf) if latest_pdf.exists() else "",
        "timestamped_markdown": str(ts_md),
        "timestamped_printable_html": str(ts_html),
        "timestamped_pdf": str(ts_pdf) if ts_pdf else "",
        "pdf_ok": bool(pdf_ok),
        "pdf_detail": str(pdf_detail),
        "generated_utc": generated_utc,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"Wrote: {ts_md}")
        print(f"Wrote: {ts_html}")
        if ts_pdf:
            print(f"Wrote: {ts_pdf}")
        print(f"Latest MD: {latest_md}")
        print(f"Latest HTML: {latest_html}")
        if latest_pdf.exists():
            print(f"Latest PDF: {latest_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
