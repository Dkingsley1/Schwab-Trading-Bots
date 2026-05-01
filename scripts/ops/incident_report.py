#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops import incident_review_packet, incident_timeline
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from . import incident_review_packet, incident_timeline
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_JSON_PATH = PROJECT_ROOT / "governance" / "health" / "incident_report_latest.json"
DEFAULT_MD_PATH = PROJECT_ROOT / "exports" / "reports" / "incident_report_latest.md"
DEFAULT_HTML_PATH = PROJECT_ROOT / "exports" / "reports" / "incident_report_latest.html"
DEFAULT_PDF_PATH = PROJECT_ROOT / "exports" / "reports" / "incident_report_latest.pdf"
APP_BROWSER_CANDIDATES = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _run(cmd: list[str], *, project_root: Path) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except Exception as exc:
        return 1, "", str(exc)


def _default_allow_gui_pdf_renderer() -> bool:
    return any(candidate.exists() for candidate in APP_BROWSER_CANDIDATES)


def _pdf_renderer_binary(allow_gui_renderer: bool) -> tuple[str, str]:
    env_override = (
        os.getenv("INCIDENT_REPORT_PDF_BIN", "").strip()
        or os.getenv("REPORT_PDF_BUNDLE_PDF_BIN", "").strip()
        or os.getenv("TRAINING_REPORT_PDF_BIN", "").strip()
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

    if allow_gui_renderer:
        for candidate in APP_BROWSER_CANDIDATES:
            if candidate.exists():
                return str(candidate), "browser_app"

    return "", ""


def _render_pdf_via_open_app(renderer: str, html_uri: str, pdf_path: Path, *, project_root: Path) -> tuple[bool, str]:
    app_binary = Path(renderer)
    app_bundle = app_binary.parents[2]
    profile_dir = Path(tempfile.mkdtemp(prefix="incident-report-open-"))
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
        rc, out, err = _run(cmd, project_root=project_root)
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


def _render_pdf_from_html(html_path: Path, pdf_path: Path, *, allow_gui_renderer: bool, project_root: Path) -> tuple[bool, str]:
    renderer, renderer_kind = _pdf_renderer_binary(allow_gui_renderer=allow_gui_renderer)
    if not renderer:
        return False, "pdf_renderer_not_found"
    html_uri = html_path.resolve().as_uri()
    if renderer_kind == "wkhtmltopdf":
        cmd = [renderer, html_uri, str(pdf_path)]
        rc, out, err = _run(cmd, project_root=project_root)
    elif renderer_kind == "browser_app":
        return _render_pdf_via_open_app(renderer, html_uri, pdf_path, project_root=project_root)
    else:
        profile_dir = Path(tempfile.mkdtemp(prefix="incident-report-pdf-"))
        try:
            cmd = [
                renderer,
                "--headless=new",
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
            ]
            rc, out, err = _run(cmd, project_root=project_root)
        finally:
            shutil.rmtree(profile_dir, ignore_errors=True)
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return True, out or err or "ok"
    return False, err or out or f"rc={rc}"


def _list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _source_paths(project_root: Path) -> dict[str, Path]:
    health_root = project_root / "governance" / "health"
    return {
        "incident_timeline": health_root / "incident_timeline_latest.json",
        "incident_review_packet": health_root / "incident_review_packet_latest.json",
        "incident_closeout_autopilot": health_root / "incident_closeout_autopilot_latest.json",
        "live_runtime_separation": health_root / "live_runtime_separation_control_latest.json",
        "auth_lease": health_root / "auth_lease_manager_latest.json",
        "remote_alert_control": health_root / "remote_alert_control_latest.json",
        "lane_thaw": health_root / "lane_thaw_controller_latest.json",
        "data_plane_recovery": health_root / "data_plane_recovery_controller_latest.json",
        "process_watchdog": health_root / "process_watchdog_latest.json",
    }


def _source_contract(name: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    exists = path.exists()
    status = str(payload.get("overall_status") or payload.get("status") or "unknown").strip() or "unknown"
    reason = "ok"
    if not exists:
        status = "unknown"
        reason = "artifact_missing"
    elif not payload:
        status = "unknown"
        reason = "artifact_unreadable_or_empty"
    elif status == "unknown":
        reason = "status_missing"
    return {
        "name": name,
        "path": str(path),
        "exists": exists,
        "status": status,
        "reason": reason,
    }


def _remediation_packs(open_surfaces: list[dict[str, Any]], paths: dict[str, Path]) -> list[dict[str, Any]]:
    surface_catalog = {
        "runtime_separation": {
            "owner": "runtime_ops",
            "expected_artifacts": ["live_runtime_separation", "incident_review_packet"],
            "actions": [
                "clear runtime separation blockers before resuming halted or write-heavy lanes",
                "re-run the runtime clearance artifact after the cold-lane prerequisites are satisfied",
            ],
        },
        "auth_lease": {
            "owner": "broker_auth",
            "expected_artifacts": ["auth_lease", "incident_review_packet"],
            "actions": [
                "refresh or pre-stage the broker lease before expiry turns into execution failure",
                "re-publish the auth lease manager artifact with a healthy lease window",
            ],
        },
        "process_watchdog": {
            "owner": "ops_watchdog",
            "expected_artifacts": ["process_watchdog", "incident_review_packet"],
            "actions": [
                "stabilize the watchdog surface and confirm restart storms or alert bursts are no longer active",
            ],
        },
        "data_plane_recovery": {
            "owner": "data_plane",
            "expected_artifacts": ["data_plane_recovery", "incident_review_packet"],
            "actions": [
                "drain write failures and confirm the recovery controller is back in a ready state",
            ],
        },
    }
    packs: list[dict[str, Any]] = []
    for row in open_surfaces:
        surface = str(row.get("surface") or row.get("category") or "surface").strip()
        profile = surface_catalog.get(surface, {})
        severity = str(row.get("severity") or row.get("status") or "warning").strip().lower()
        expected = [
            str(paths.get(name))
            for name in profile.get("expected_artifacts", ["incident_review_packet"])
            if isinstance(profile, dict) and paths.get(name) is not None
        ]
        packs.append(
            {
                "surface": surface,
                "owner": str(profile.get("owner") or "operations"),
                "severity": severity,
                "summary": str(row.get("summary") or "").strip(),
                "actions": list(profile.get("actions") or ["clear the blocking condition and refresh the supporting artifact"]),
                "expected_artifacts": expected,
                "closeout_when": f"{surface} is no longer open and the supporting artifact returns ready or healthy",
            }
        )
    return packs


def refresh_supporting_artifacts(project_root: Path) -> None:
    paths = _source_paths(project_root)
    timeline_payload = incident_timeline.build_payload(project_root)
    write_payload(paths["incident_timeline"], timeline_payload)
    review_payload = incident_review_packet.build_payload(project_root)
    write_payload(paths["incident_review_packet"], review_payload)


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    recent_limit: int = 10,
    surface_limit: int = 10,
) -> dict[str, Any]:
    paths = _source_paths(project_root)
    timeline = load_json(paths["incident_timeline"])
    review = load_json(paths["incident_review_packet"])
    closeout = load_json(paths["incident_closeout_autopilot"])
    runtime = load_json(paths["live_runtime_separation"])
    auth = load_json(paths["auth_lease"])
    alerts = load_json(paths["remote_alert_control"])
    thaw = load_json(paths["lane_thaw"])
    data_plane = load_json(paths["data_plane_recovery"])
    watchdog = load_json(paths["process_watchdog"])

    recent_incidents = _list_dicts(timeline.get("recent_incidents"))[: max(int(recent_limit), 1)]
    open_surfaces = _list_dicts(timeline.get("open_surfaces"))[: max(int(surface_limit), 1)]

    overall_status = str(review.get("overall_status") or timeline.get("overall_status") or "ready")
    review_required = bool(review.get("review_required")) or _safe_int(timeline.get("open_incident_count"), len(open_surfaces)) > 0
    review_state = str(review.get("review_state") or ("awaiting_remediation" if review_required else "ready_to_archive"))
    packet_sha256 = str(review.get("packet_sha256") or "")

    open_count = _safe_int(timeline.get("open_incident_count"), len(open_surfaces))
    recent_count = _safe_int(timeline.get("recent_incident_count"), len(recent_incidents))
    critical_open_count = sum(
        1
        for row in open_surfaces
        if str(row.get("severity") or row.get("status") or "").strip().lower() in {"critical", "blocked"}
    )
    warning_open_count = max(open_count - critical_open_count, 0)

    recent_categories = ordered_unique(
        str(row.get("category") or "").strip().lower()
        for row in recent_incidents
        if str(row.get("category") or "").strip()
    )
    open_surface_names = ordered_unique(
        str(row.get("surface") or row.get("category") or "").strip()
        for row in open_surfaces
        if str(row.get("surface") or row.get("category") or "").strip()
    )

    runtime_clearance_state = str(((runtime.get("clearance_plan") or {}).get("clearance_state") or "")).strip()
    auth_lease_state = str(auth.get("lease_state") or "").strip()
    critical_backlog = alerts.get("critical_backlog") if isinstance(alerts.get("critical_backlog"), dict) else {}
    unacked_critical = _safe_int(critical_backlog.get("unacked_count"))
    unsent_critical = _safe_int(critical_backlog.get("unsent_count"))
    paused_lane_count = _safe_int(thaw.get("paused_lane_count"))
    candidate_lane_count = _safe_int(thaw.get("candidate_count"))
    write_failure_count = _safe_int(data_plane.get("write_failure_count"))
    account_snapshot_failure_count = _safe_int(data_plane.get("account_snapshot_failure_count"))
    restart_storm_count = len(watchdog.get("restart_storms")) if isinstance(watchdog.get("restart_storms"), list) else 0
    watchdog_alert_count = len(watchdog.get("alerts")) if isinstance(watchdog.get("alerts"), list) else 0

    narrative = (
        "This report is separate from the timeline: the timeline preserves chronology, while the incident report "
        "compresses current impact, evidence, and recommended operator actions into a decision-ready view."
    )
    executive_summary = ordered_unique(
        [
            f"Overall incident status is {overall_status}; review state is {review_state.replace('_', ' ')}.",
            (
                f"{open_count} open surfaces remain across {', '.join(open_surface_names[:4])}."
                if open_count > 0 and open_surface_names
                else "No open surfaces are currently reported by the incident timeline."
            ),
            (
                f"Recent incident volume is {recent_count} with categories {', '.join(recent_categories[:4])}."
                if recent_count > 0 and recent_categories
                else (f"Recent incident volume is {recent_count}." if recent_count > 0 else "No recent incidents were found in the current timeline window.")
            ),
            (
                f"Auth lease is {auth_lease_state or 'unknown'}, runtime clearance is {runtime_clearance_state or 'unknown'}, "
                f"and critical alert backlog is unacked={unacked_critical} unsent={unsent_critical}."
            ),
            (
                f"Process watchdog reports restart_storms={restart_storm_count} and alerts={watchdog_alert_count}."
                if restart_storm_count or watchdog_alert_count
                else ""
            ),
        ]
    )

    closure_contract = (
        closeout
        if closeout
        else (
            review.get("closure_contract")
            if isinstance(review.get("closure_contract"), dict)
            else (timeline.get("auto_close_contract") if isinstance(timeline.get("auto_close_contract"), dict) else {})
        )
    )
    recommended_actions = ordered_unique(
        list(review.get("recommended_actions") or [])
        + list(timeline.get("recommended_actions") or [])
        + [
            "clear runtime separation and auth lease blockers before re-enabling halted or write-heavy lanes" if overall_status == "blocked" else "",
            "archive the review packet hash with the closure contract once open surfaces remain clear" if bool(closure_contract.get("closure_ready", False)) else "",
            "use the timeline for chronology and this report for remediation approvals, escalation, and closeout decisions",
        ]
    )

    source_statuses = {
        "incident_timeline": str(timeline.get("overall_status") or ""),
        "incident_review_packet": str(review.get("overall_status") or ""),
        "incident_closeout_autopilot": str(closeout.get("overall_status") or ""),
        "live_runtime_separation": str(runtime.get("overall_status") or ""),
        "auth_lease": str(auth.get("overall_status") or ""),
        "remote_alert_control": str(alerts.get("overall_status") or ""),
        "lane_thaw": str(thaw.get("overall_status") or ""),
        "data_plane_recovery": str(data_plane.get("overall_status") or ""),
        "process_watchdog": str(watchdog.get("overall_status") or ""),
    }
    source_contracts = {
        name: _source_contract(name, path, load_json(path))
        for name, path in paths.items()
    }
    remediation_packs = _remediation_packs(open_surfaces, paths)
    expected_artifacts = ordered_unique(
        artifact
        for pack in remediation_packs
        for artifact in (pack.get("expected_artifacts") or [])
        if str(artifact).strip()
    )
    closeout_contract = {
        **closure_contract,
        "blocking_surface_count": open_count,
        "required_artifacts": ordered_unique(list(closure_contract.get("required_artifacts") or []) + expected_artifacts),
        "source_contract_failures": [
            name for name, row in source_contracts.items() if str((row or {}).get("reason") or "") != "ok"
        ],
        "closeout_ready": bool(
            closure_contract.get("closeout_ready", closure_contract.get("closure_ready", False))
        ) and open_count == 0,
    }

    review_snapshot_summary = review.get("source_snapshot") if isinstance(review.get("source_snapshot"), dict) else {}

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "report_generated": True,
        "overall_status": overall_status,
        "review_required": review_required,
        "review_state": review_state,
        "packet_sha256": packet_sha256,
        "incident_scope": {
            "report_kind": "decision_oriented_incident_report",
            "timeline_companion": str(paths["incident_timeline"]),
            "review_packet_companion": str(paths["incident_review_packet"]),
            "separation_from_timeline": narrative,
        },
        "incident_counts": {
            "recent_incident_count": recent_count,
            "open_incident_count": open_count,
            "critical_open_surface_count": critical_open_count,
            "warning_open_surface_count": warning_open_count,
        },
        "recent_categories": recent_categories,
        "open_surface_names": open_surface_names,
        "executive_summary": executive_summary,
        "incident_narrative": narrative,
        "control_plane_snapshot": {
            "runtime_clearance_state": runtime_clearance_state,
            "auth_lease_state": auth_lease_state,
            "critical_alert_backlog": {
                "unacked_count": unacked_critical,
                "unsent_count": unsent_critical,
            },
            "paused_lane_count": paused_lane_count,
            "candidate_lane_count": candidate_lane_count,
            "write_failure_count": write_failure_count,
            "account_snapshot_failure_count": account_snapshot_failure_count,
            "watchdog_restart_storm_count": restart_storm_count,
            "watchdog_alert_count": watchdog_alert_count,
        },
        "review_snapshot_summary": review_snapshot_summary,
        "closure_contract": closure_contract,
        "source_contracts": source_contracts,
        "remediation_packs": remediation_packs,
        "expected_artifacts": expected_artifacts,
        "closeout_contract": closeout_contract,
        "source_statuses": source_statuses,
        "evidence_paths": {name: str(path) for name, path in paths.items()},
        "open_surfaces": open_surfaces,
        "recent_incidents": recent_incidents,
        "recommended_actions": recommended_actions,
        "artifacts": {
            "mode": "pending",
            "json": str(DEFAULT_JSON_PATH),
            "markdown": "",
            "html": "",
            "pdf": "",
            "pdf_available": False,
            "pdf_detail": "",
        },
    }


def _kv_lines(rows: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in rows.items():
        if isinstance(value, dict):
            if not value:
                out.append(f"- {key}: {{}}")
                continue
            bits = ", ".join(f"{name}={value[name]}" for name in sorted(value))
            out.append(f"- {key}: {bits}")
            continue
        out.append(f"- {key}: {value}")
    return out


def _surface_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- No open surfaces are currently active."]
    out: list[str] = []
    for row in rows:
        surface = str(row.get("surface") or row.get("category") or "surface")
        status = str(row.get("status") or "")
        severity = str(row.get("severity") or "")
        age_minutes = _safe_float(row.get("age_minutes"))
        summary = str(row.get("summary") or "").strip()
        parts = [surface]
        if status:
            parts.append(f"status={status}")
        if severity:
            parts.append(f"severity={severity}")
        if age_minutes is not None:
            parts.append(f"age_minutes={round(age_minutes, 3)}")
        if summary:
            parts.append(summary)
        out.append("- " + " | ".join(parts))
    return out


def _incident_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- No recent incident highlights are available."]
    out: list[str] = []
    for row in rows:
        parts = [str(row.get("timestamp_utc") or "")]
        category = str(row.get("category") or "").strip()
        severity = str(row.get("severity") or "").strip()
        summary = str(row.get("summary") or "").strip()
        source_rel = str(row.get("source_rel") or "").strip()
        if category:
            parts.append(f"category={category}")
        if severity:
            parts.append(f"severity={severity}")
        if summary:
            parts.append(summary)
        if source_rel:
            parts.append(f"source={source_rel}")
        out.append("- " + " | ".join(part for part in parts if part))
    return out


def _remediation_pack_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- No remediation packs are currently open."]
    out: list[str] = []
    for row in rows:
        actions = ", ".join(str(item).strip() for item in (row.get("actions") or []) if str(item).strip())
        artifacts = ", ".join(str(item).strip() for item in (row.get("expected_artifacts") or []) if str(item).strip())
        out.append(
            "- "
            + " | ".join(
                part
                for part in [
                    str(row.get("surface") or "surface"),
                    f"owner={row.get('owner')}",
                    f"severity={row.get('severity')}",
                    f"actions={actions}" if actions else "",
                    f"expected_artifacts={artifacts}" if artifacts else "",
                ]
                if part
            )
        )
    return out


def render_markdown(payload: dict[str, Any]) -> str:
    counts = payload.get("incident_counts") if isinstance(payload.get("incident_counts"), dict) else {}
    control = payload.get("control_plane_snapshot") if isinstance(payload.get("control_plane_snapshot"), dict) else {}
    closure_contract = payload.get("closure_contract") if isinstance(payload.get("closure_contract"), dict) else {}
    evidence_paths = payload.get("evidence_paths") if isinstance(payload.get("evidence_paths"), dict) else {}
    status_lines = _kv_lines(payload.get("source_statuses") if isinstance(payload.get("source_statuses"), dict) else {})
    control_lines = _kv_lines(control)
    evidence_block = "\n".join(f"{name}: {value}" for name, value in evidence_paths.items())

    parts = [
        "# Incident Report",
        "",
        f"Generated: {payload.get('timestamp_utc', '')}",
        f"Overall status: {payload.get('overall_status', '')}",
        f"Review state: {payload.get('review_state', '')}",
        "",
        "## Executive Summary",
        *[f"- {row}" for row in list(payload.get("executive_summary") or [])],
        "",
        "## Why This Is Separate From The Timeline",
        str(payload.get("incident_narrative") or ""),
        "",
        "## Decision View",
        f"- review_required: {bool(payload.get('review_required', False))}",
        f"- packet_sha256: {payload.get('packet_sha256', '') or 'missing'}",
        *[f"- {name}: {value}" for name, value in counts.items()],
        "",
        "## Open Surfaces",
        *_surface_lines(_list_dicts(payload.get("open_surfaces"))),
        "",
        "## Recent Incident Highlights",
        *_incident_lines(_list_dicts(payload.get("recent_incidents"))),
        "",
        "## Remediation Packs",
        *_remediation_pack_lines(_list_dicts(payload.get("remediation_packs"))),
        "",
        "## Control Plane Snapshot",
        *control_lines,
        "",
        "## Source Statuses",
        *status_lines,
        "",
        "## Recommended Actions",
        *[f"- {row}" for row in list(payload.get("recommended_actions") or [])],
        "",
        "## Closeout Contract",
        *[f"- {name}: {value}" for name, value in (payload.get("closeout_contract") if isinstance(payload.get("closeout_contract"), dict) else {}).items()],
        "",
        "## Evidence And Closeout",
        *[f"- {name}: {value}" for name, value in closure_contract.items()],
        "",
        "```text",
        evidence_block,
        "```",
    ]
    return "\n".join(parts).strip() + "\n"


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
    return "\n".join(out) or "<p>No incident report content available.</p>"


def render_html(payload: dict[str, Any], *, source_path: Path) -> str:
    status = str(payload.get("overall_status") or "ready").strip().lower()
    badge_class = {
        "ready": "badge-ready",
        "degraded": "badge-degraded",
        "blocked": "badge-blocked",
        "critical": "badge-blocked",
    }.get(status, "badge-degraded")
    markdown_body = _markdown_to_html(render_markdown(payload))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Incident Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5efe7;
      --ink: #1d2935;
      --muted: #64748b;
      --card: #fffaf4;
      --line: #dbcfc2;
      --accent: #9f1239;
      --ready: #166534;
      --degraded: #a16207;
      --blocked: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: linear-gradient(180deg, #ece3d7 0%, #f7f2ea 100%); color: var(--ink); font: 15px/1.6 Georgia, 'Times New Roman', serif; }}
    .page {{ max-width: 1020px; margin: 0 auto; padding: 34px 24px 56px; }}
    .hero, .section {{ background: var(--card); border: 1px solid var(--line); border-radius: 20px; box-shadow: 0 12px 32px rgba(29, 41, 53, 0.08); }}
    .hero {{ padding: 28px; }}
    .section {{ margin-top: 18px; padding: 20px 24px; }}
    h1, h2, h3 {{ margin: 0 0 10px; font-family: 'Avenir Next', 'Segoe UI', sans-serif; }}
    h1 {{ font-size: 32px; }}
    h2 {{ margin-top: 18px; font-size: 21px; }}
    h3 {{ margin-top: 16px; font-size: 16px; }}
    p {{ margin: 10px 0 0; }}
    p.meta {{ margin: 0; color: var(--muted); }}
    ul {{ margin: 10px 0 0 20px; padding: 0; }}
    li {{ margin: 6px 0; }}
    .hero-grid {{ display: grid; grid-template-columns: 1.3fr 0.7fr; gap: 18px; }}
    .badge {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 6px 12px; font: 600 12px/1 'Avenir Next', 'Segoe UI', sans-serif; letter-spacing: 0.05em; text-transform: uppercase; }}
    .badge-ready {{ background: #dcfce7; color: var(--ready); }}
    .badge-degraded {{ background: #fef3c7; color: var(--degraded); }}
    .badge-blocked {{ background: #fee2e2; color: var(--blocked); }}
    .summary-card {{ border: 1px solid var(--line); border-radius: 16px; background: rgba(255, 255, 255, 0.55); padding: 14px 16px; }}
    .summary-card h2 {{ margin-top: 0; font-size: 14px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
    .summary-card .metric {{ font: 600 28px/1 'Avenir Next', 'Segoe UI', sans-serif; }}
    .summary-card .label {{ margin-top: 6px; color: var(--muted); font: 12px/1.4 'Avenir Next', 'Segoe UI', sans-serif; text-transform: uppercase; letter-spacing: 0.04em; }}
    pre.content {{ margin: 14px 0 0; padding: 16px; border-radius: 14px; background: #f7f0e6; border: 1px solid #eadfce; white-space: pre-wrap; word-break: break-word; font: 12px/1.5 'SF Mono', 'Menlo', monospace; }}
    .path {{ margin-top: 10px; color: var(--accent); font: 12px/1.45 'SF Mono', 'Menlo', monospace; word-break: break-all; }}
    @media (max-width: 760px) {{
      .hero-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <span class="badge {badge_class}">{html.escape(status or 'unknown')}</span>
          <h1>Decision-Oriented Incident Report</h1>
          <p class="meta">Generated {html.escape(str(payload.get('timestamp_utc') or ''))}</p>
          <p>{html.escape(str(payload.get('incident_narrative') or ''))}</p>
          <p class="path">Source JSON: {html.escape(str(source_path))}</p>
        </div>
        <div class="summary-card">
          <h2>Current Counts</h2>
          <div class="metric">{html.escape(str(((payload.get('incident_counts') or {}).get('open_incident_count') or 0)))}</div>
          <div class="label">Open Surfaces</div>
          <div class="metric" style="margin-top: 14px;">{html.escape(str(((payload.get('incident_counts') or {}).get('recent_incident_count') or 0)))}</div>
          <div class="label">Recent Incidents</div>
        </div>
      </div>
    </section>
    <section class="section">
      {markdown_body}
    </section>
  </div>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a decision-oriented incident report that is separate from the incident timeline.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--md-out-file", default=str(DEFAULT_MD_PATH))
    parser.add_argument("--html-out-file", default=str(DEFAULT_HTML_PATH))
    parser.add_argument("--pdf-out-file", default=str(DEFAULT_PDF_PATH))
    parser.add_argument("--recent-limit", type=int, default=10)
    parser.add_argument("--surface-limit", type=int, default=10)
    parser.add_argument(
        "--refresh-supporting-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refresh the incident timeline and review packet before compiling the incident report.",
    )
    parser.add_argument(
        "--allow-gui-pdf-renderer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow GUI browser app bundles when no CLI PDF renderer is available.",
    )
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_path = Path(args.out_file).expanduser()
    md_path = Path(args.md_out_file).expanduser()
    html_path = Path(args.html_out_file).expanduser()
    pdf_path = Path(args.pdf_out_file).expanduser()

    if args.refresh_supporting_artifacts:
        refresh_supporting_artifacts(project_root)

    if args.allow_gui_pdf_renderer is None:
        allow_gui_pdf_renderer = _default_allow_gui_pdf_renderer()
    else:
        allow_gui_pdf_renderer = bool(args.allow_gui_pdf_renderer)

    payload = build_payload(project_root, recent_limit=args.recent_limit, surface_limit=args.surface_limit)
    payload["artifacts"]["json"] = str(out_path)

    if args.json_only:
        payload["artifacts"].update(
            {
                "mode": "json_only",
                "pdf_available": False,
                "pdf_detail": "skipped_json_only",
            }
        )
        write_payload(out_path, payload)
    else:
        markdown = render_markdown(payload)
        html_doc = render_html(payload, source_path=out_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown, encoding="utf-8")
        html_path.write_text(html_doc, encoding="utf-8")

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        if pdf_path.exists():
            pdf_path.unlink()
        pdf_ok, pdf_detail = _render_pdf_from_html(
            html_path,
            pdf_path,
            allow_gui_renderer=allow_gui_pdf_renderer,
            project_root=project_root,
        )
        payload["artifacts"].update(
            {
                "mode": "full_bundle",
                "markdown": str(md_path),
                "html": str(html_path),
                "pdf": str(pdf_path) if pdf_ok and pdf_path.exists() else "",
                "pdf_available": bool(pdf_ok and pdf_path.exists()),
                "pdf_detail": str(pdf_detail),
            }
        )
        write_payload(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "incident_report "
            f"overall_status={payload.get('overall_status', '')} "
            f"review_required={int(bool(payload.get('review_required', False)))} "
            f"pdf_available={int(bool((payload.get('artifacts') or {}).get('pdf_available', False)))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
