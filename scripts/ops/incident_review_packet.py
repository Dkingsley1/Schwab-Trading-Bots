#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "incident_review_packet_latest.json"
DEFAULT_PDF_PATH = PROJECT_ROOT / "exports" / "reports" / "incident_review_packet_latest.pdf"


def _pdf_text_escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_simple_pdf(path: Path, *, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text_lines = [str(title or "Incident Review Packet"), ""] + [str(line or "") for line in lines]
    commands = ["BT", "/F1 11 Tf", "54 750 Td", "14 TL"]
    for idx, line in enumerate(text_lines[:48]):
        if idx:
            commands.append("T*")
        commands.append(f"({_pdf_text_escape(line[:118])}) Tj")
    commands.append("ET")
    stream = "\n".join(commands).encode("utf-8")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref_offset = sum(len(chunk) for chunk in chunks)
    xref = [f"xref\n0 {len(objects) + 1}\n".encode("ascii"), b"0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    chunks.extend(
        xref
        + [
            b"trailer\n",
            f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("ascii"),
            b"startxref\n",
            f"{xref_offset}\n".encode("ascii"),
            b"%%EOF\n",
        ]
    )
    path.write_bytes(b"".join(chunks))


def _packet_pdf_lines(payload: dict[str, Any]) -> list[str]:
    closure = payload.get("closure_contract") if isinstance(payload.get("closure_contract"), dict) else {}
    immutability = payload.get("immutability_contract") if isinstance(payload.get("immutability_contract"), dict) else {}
    source_snapshot = payload.get("source_snapshot") if isinstance(payload.get("source_snapshot"), dict) else {}
    timeline = source_snapshot.get("timeline") if isinstance(source_snapshot.get("timeline"), dict) else {}
    runtime = source_snapshot.get("runtime") if isinstance(source_snapshot.get("runtime"), dict) else {}
    auth = source_snapshot.get("auth") if isinstance(source_snapshot.get("auth"), dict) else {}
    alerts = source_snapshot.get("alerts") if isinstance(source_snapshot.get("alerts"), dict) else {}
    lines = [
        f"Generated UTC: {payload.get('timestamp_utc', '')}",
        f"Overall status: {payload.get('overall_status', '')}",
        f"Review state: {payload.get('review_state', '')}",
        f"Review required: {str(bool(payload.get('review_required', False))).lower()}",
        f"Packet SHA256: {payload.get('packet_sha256', '')}",
        f"Open incidents: {payload.get('open_incident_count', 0)}",
        f"Watch surfaces: {payload.get('watch_surface_count', 0)}",
        f"Recent categories: {', '.join(list(payload.get('recent_categories') or [])) or 'none'}",
        "",
        "Closure Contract",
        f"closure_ready={str(bool(closure.get('closure_ready', False))).lower()} candidate_count={closure.get('candidate_count', 0)}",
        f"review_required={str(bool(closure.get('review_required', False))).lower()} reason={closure.get('closure_reason', '') or 'n/a'}",
        "",
        "Source Snapshot",
        f"timeline_status={timeline.get('overall_status', '')} open={timeline.get('open_incident_count', 0)} recent={timeline.get('recent_incident_count', 0)}",
        f"runtime_status={runtime.get('overall_status', '')} clearance={runtime.get('clearance_state', '')}",
        f"auth_status={auth.get('overall_status', '')} lease={auth.get('lease_state', '')}",
        f"alerts_status={alerts.get('overall_status', '')} critical_backlog={alerts.get('critical_backlog', {})}",
        "",
        "Immutability",
        f"hash_algorithm={immutability.get('hash_algorithm', 'sha256')} snapshot_bytes={immutability.get('source_snapshot_bytes', 0)}",
        "",
        "Recommended Actions",
    ]
    for action in list(payload.get("recommended_actions") or [])[:10]:
        lines.append(f"- {action}")
    source_paths = immutability.get("source_paths") if isinstance(immutability.get("source_paths"), list) else []
    if source_paths:
        lines.extend(["", "Source Paths"])
        lines.extend(f"- {path}" for path in source_paths[:8])
    return lines


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    timeline_path = health_root / "incident_timeline_latest.json"
    runtime_path = health_root / "live_runtime_separation_control_latest.json"
    auth_path = health_root / "auth_lease_manager_latest.json"
    alerts_path = health_root / "remote_alert_control_latest.json"
    thaw_path = health_root / "lane_thaw_controller_latest.json"
    data_plane_path = health_root / "data_plane_recovery_controller_latest.json"

    timeline = load_json(timeline_path)
    runtime = load_json(runtime_path)
    auth = load_json(auth_path)
    alerts = load_json(alerts_path)
    thaw = load_json(thaw_path)
    data_plane = load_json(data_plane_path)

    source_snapshot = {
        "timeline": {
            "overall_status": str(timeline.get("overall_status") or ""),
            "recent_incident_count": int(timeline.get("recent_incident_count", 0) or 0),
            "open_incident_count": int(timeline.get("open_incident_count", 0) or 0),
            "watch_surface_count": int(timeline.get("watch_surface_count", 0) or 0),
            "review_required": bool(timeline.get("review_required", False)),
            "open_surfaces": timeline.get("open_surfaces") if isinstance(timeline.get("open_surfaces"), list) else [],
            "watch_surfaces": timeline.get("watch_surfaces") if isinstance(timeline.get("watch_surfaces"), list) else [],
            "stitched_threads": timeline.get("stitched_threads") if isinstance(timeline.get("stitched_threads"), list) else [],
            "recent_incidents": timeline.get("recent_incidents") if isinstance(timeline.get("recent_incidents"), list) else [],
        },
        "runtime": {
            "overall_status": str(runtime.get("overall_status") or ""),
            "clearance_state": str(((runtime.get("clearance_plan") or {}).get("clearance_state") or "")),
        },
        "auth": {
            "overall_status": str(auth.get("overall_status") or ""),
            "lease_state": str(auth.get("lease_state") or ""),
        },
        "alerts": {
            "overall_status": str(alerts.get("overall_status") or ""),
            "critical_backlog": alerts.get("critical_backlog") if isinstance(alerts.get("critical_backlog"), dict) else {},
        },
        "lane_thaw": {
            "overall_status": str(thaw.get("overall_status") or ""),
            "paused_lane_count": int(thaw.get("paused_lane_count", 0) or 0),
            "candidate_count": int(thaw.get("candidate_count", 0) or 0),
        },
        "data_plane": {
            "overall_status": str(data_plane.get("overall_status") or ""),
            "write_failure_count": int(data_plane.get("write_failure_count", 0) or 0),
            "account_snapshot_failure_count": int(data_plane.get("account_snapshot_failure_count", 0) or 0),
        },
    }
    packet_sha256 = _stable_hash(source_snapshot)
    open_incident_count = int(((source_snapshot.get("timeline") or {}).get("open_incident_count", 0) or 0))
    watch_surface_count = int(((source_snapshot.get("timeline") or {}).get("watch_surface_count", 0) or 0))
    recent_incidents = ((source_snapshot.get("timeline") or {}).get("recent_incidents") or [])
    recent_categories = sorted(
        {
            str((row or {}).get("category") or "").strip().lower()
            for row in recent_incidents
            if isinstance(row, dict) and str((row or {}).get("category") or "").strip()
        }
    )
    review_required = bool(timeline.get("review_required", open_incident_count > 0))
    overall_status = ("ready" if not review_required else str(timeline.get("overall_status") or "degraded"))
    auto_close = ((timeline.get("auto_close_contract") or {}) if isinstance(timeline.get("auto_close_contract"), dict) else {})
    closure_contract = {
        "closure_ready": bool(auto_close.get("closure_ready", False)) and not review_required,
        "candidate_count": int(auto_close.get("candidate_count", 0) or 0),
        "review_required": review_required,
        "closure_reason": str(auto_close.get("closure_reason") or ""),
    }

    recommended_actions = ordered_unique(
        list(timeline.get("recommended_actions") or [])[:2]
        + [
            "treat this packet hash as the immutable incident-review anchor when you discuss interventions or approvals" if review_required else "",
            "close the runtime and auth blockers before archiving the incident packet" if overall_status == "blocked" else "",
            "archive the packet hash and mark the incident closed once the auto-close contract stays green" if bool(closure_contract.get("closure_ready", False)) else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "review_required": review_required,
        "review_state": ("awaiting_remediation" if review_required else "ready_to_archive"),
        "open_incident_count": open_incident_count,
        "watch_surface_count": watch_surface_count,
        "recent_categories": recent_categories,
        "packet_sha256": packet_sha256,
        "closure_contract": closure_contract,
        "immutability_contract": {
            "hash_algorithm": "sha256",
            "source_paths": [str(timeline_path), str(runtime_path), str(auth_path), str(alerts_path), str(thaw_path), str(data_plane_path)],
            "source_snapshot_bytes": len(json.dumps(source_snapshot, ensure_ascii=True).encode("utf-8")),
        },
        "source_snapshot": source_snapshot,
        "recommended_actions": recommended_actions,
        "artifacts": {
            "json": str(DEFAULT_OUT_PATH),
            "pdf": "",
            "pdf_available": False,
            "pdf_detail": "not_rendered",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish an immutable review packet for the current incident timeline.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--pdf-out-file", default=str(DEFAULT_PDF_PATH))
    parser.add_argument("--render-pdf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    pdf_path = Path(args.pdf_out_file).expanduser()
    payload["artifacts"]["json"] = str(out_path)
    if bool(args.render_pdf):
        try:
            _write_simple_pdf(pdf_path, title="Incident Review Packet", lines=_packet_pdf_lines(payload))
            payload["artifacts"].update(
                {
                    "pdf": str(pdf_path),
                    "pdf_available": bool(pdf_path.exists() and pdf_path.stat().st_size > 0),
                    "pdf_detail": "direct_pdf",
                }
            )
        except Exception as exc:
            payload["artifacts"].update(
                {
                    "pdf": "",
                    "pdf_available": False,
                    "pdf_detail": f"pdf_render_failed:{type(exc).__name__}:{exc}",
                }
            )
    else:
        payload["artifacts"]["pdf_detail"] = "skipped"
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "incident_review_packet "
            f"overall_status={payload.get('overall_status', '')} "
            f"review_required={int(bool(payload.get('review_required', False)))} "
            f"pdf_available={int(bool((payload.get('artifacts') or {}).get('pdf_available', False)))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
