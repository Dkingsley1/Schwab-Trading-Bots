#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_python import resolve_runtime_python
    from scripts.ops.long_runtime_common import iso_now, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_python import resolve_runtime_python
    from .long_runtime_common import iso_now, write_payload


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_summary_autopilot_latest.json"
DEFAULT_STEP_TIMEOUT_SEC = int(os.environ.get("SYSTEM_SUMMARY_AUTOPILOT_STEP_TIMEOUT_SECONDS", "300"))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_json_output(text: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(text or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _run(cmd: list[str], timeout_sec: int = DEFAULT_STEP_TIMEOUT_SEC) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
        rc = int(proc.returncode)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc = 124
        stdout = str(exc.output or "")
        stderr = str(exc.stderr or "")
        stderr = "\n".join([line for line in [stderr, f"timeout_after_seconds={timeout_sec}"] if line])
    payload = _parse_json_output(stdout)
    duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
    return {
        "rc": rc,
        "payload": payload,
        "stdout_tail": "\n".join(stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-10:]),
        "duration_ms": duration_ms,
        "timeout_sec": timeout_sec,
    }


def build_payload(project_root: Path = PROJECT_ROOT, step_timeout_sec: int = DEFAULT_STEP_TIMEOUT_SEC) -> dict[str, Any]:
    chrome_guard = _load_json(project_root / "governance" / "health" / "chrome_headless_guard_latest.json")
    policy = str(chrome_guard.get("timeline_pdf_policy") or "allow").strip().lower()
    render_pdf = policy != "suppress"
    allow_gui = policy not in {"suppress", "headless_only"}

    summary_cmd = [
        str(PY),
        str(project_root / "scripts" / "ops" / "system_summary_report.py"),
        "--refresh-supporting-artifacts",
        "--json",
    ]
    if render_pdf:
        summary_cmd.append("--render-pdf")
    else:
        summary_cmd.append("--no-render-pdf")
    if allow_gui:
        summary_cmd.append("--allow-gui-pdf-renderer")
    else:
        summary_cmd.append("--no-allow-gui-pdf-renderer")

    bundle_cmd = [
        str(PY),
        str(project_root / "scripts" / "ops" / "report_pdf_bundle.py"),
        "--json",
    ]
    if allow_gui:
        bundle_cmd.append("--allow-gui-pdf-renderer")
    else:
        bundle_cmd.append("--no-allow-gui-pdf-renderer")

    summary_result = _run(summary_cmd, timeout_sec=step_timeout_sec)
    if render_pdf:
        bundle_result = _run(bundle_cmd, timeout_sec=step_timeout_sec)
    else:
        bundle_result = {
            "rc": 0,
            "payload": {"ok": True, "overall_status": "skipped", "reason": "pdf_policy_suppressed"},
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_ms": 0.0,
            "timeout_sec": step_timeout_sec,
        }
    summary_payload = summary_result.get("payload") if isinstance(summary_result.get("payload"), dict) else {}
    bundle_payload = bundle_result.get("payload") if isinstance(bundle_result.get("payload"), dict) else {}
    summary_pdf = summary_payload.get("pdf") if isinstance(summary_payload.get("pdf"), dict) else {}
    summary_pdf_enabled = bool(summary_pdf.get("enabled", False))
    summary_pdf_ok = bool(summary_pdf.get("ok", False))
    bundle_ok = bool(bundle_payload.get("ok", False))
    bundle_index_ok = bool(bundle_payload.get("index_ok", bundle_ok))

    overall_status = "ready"
    if int(summary_result.get("rc", 1)) != 0 or not summary_payload:
        overall_status = "blocked"
    elif summary_pdf_enabled and not summary_pdf_ok:
        overall_status = "degraded"
    elif int(bundle_result.get("rc", 1)) != 0 or (render_pdf and not bundle_index_ok):
        overall_status = "degraded"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status != "blocked",
        "overall_status": overall_status,
        "chrome_policy": policy,
        "render_pdf": render_pdf,
        "allow_gui_pdf_renderer": allow_gui,
        "step_timeout_sec": step_timeout_sec,
        "system_summary": {
            "rc": int(summary_result.get("rc", 1)),
            "duration_ms": float(summary_result.get("duration_ms", 0.0) or 0.0),
            "timeout_sec": int(summary_result.get("timeout_sec", step_timeout_sec) or step_timeout_sec),
            "payload_summary": {
                "overall_status": str(summary_payload.get("overall_status") or ""),
                "overall_grade": str(((summary_payload.get("section_grade_board") or {}).get("overall_letter_grade") or "")),
                "html_latest": str(((summary_payload.get("html_paths") or {}).get("latest") or "")),
                "pdf_latest": str(((summary_payload.get("pdf") or {}).get("latest") or "")),
                "pdf_ok": bool(summary_pdf_ok),
            },
            "stdout_tail": str(summary_result.get("stdout_tail") or ""),
            "stderr_tail": str(summary_result.get("stderr_tail") or ""),
        },
        "report_bundle": {
            "rc": int(bundle_result.get("rc", 1)),
            "duration_ms": float(bundle_result.get("duration_ms", 0.0) or 0.0),
            "timeout_sec": int(bundle_result.get("timeout_sec", step_timeout_sec) or step_timeout_sec),
            "payload_summary": {
                "overall_status": str(bundle_payload.get("overall_status") or ("ready" if bundle_index_ok else "")),
                "index_ok": bool(bundle_index_ok),
                "artifact_present_count_after": int(bundle_payload.get("artifact_present_count_after", bundle_payload.get("ok_count", 0)) or 0),
            },
            "stdout_tail": str(bundle_result.get("stdout_tail") or ""),
            "stderr_tail": str(bundle_result.get("stderr_tail") or ""),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the compiled system summary and downstream PDF bundle on a bounded autopilot.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--step-timeout-seconds", type=int, default=DEFAULT_STEP_TIMEOUT_SEC)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), step_timeout_sec=max(1, int(args.step_timeout_seconds)))
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "system_summary_autopilot "
            f"overall_status={payload.get('overall_status', '')} "
            f"render_pdf={str(payload.get('render_pdf', False)).lower()}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
