#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / "governance" / "health" / "report_pdf_bundle_latest.json"
OUTPUT_PATH = PROJECT_ROOT / "governance" / "health" / "report_quality_guard_latest.json"
MIN_PDF_BYTES = 10_000
REPORT_READY_RENDERERS = {
    "framework_map_v2": "report_ready_framework_map_pdf",
    "paper_performance": "report_ready_paper_performance_pdf",
    "post_trade_analysis": "report_ready_post_trade_pdf",
    "project_timeline": "report_ready_project_timeline_pdf",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _pdf_check(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "exists": bool(path.exists()),
        "bytes": 0,
        "valid_header": False,
        "valid_eof": False,
        "large_enough": False,
        "ok": False,
    }
    if not path.exists() or not path.is_file():
        return out
    data = path.read_bytes()
    out["bytes"] = int(len(data))
    out["valid_header"] = data.startswith(b"%PDF-")
    out["valid_eof"] = b"%%EOF" in data[-2048:]
    out["large_enough"] = len(data) >= MIN_PDF_BYTES
    out["ok"] = bool(out["valid_header"] and out["valid_eof"] and out["large_enough"])
    return out


def _run_repair(project_root: Path, *, timeout_seconds: int) -> dict[str, Any]:
    cmd = [sys.executable, str(project_root / "scripts" / "ops" / "sendout_pdf_refresh.py"), "--json"]
    try:
        proc = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True, check=False, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return {"attempted": True, "ok": False, "rc": 124, "error": f"timeout_after_{timeout_seconds}s"}
    stdout = (proc.stdout or "").strip()
    payload = {}
    if stdout.startswith("{") and stdout.endswith("}"):
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            payload = {}
    return {
        "attempted": True,
        "ok": int(proc.returncode) == 0 and bool(payload.get("ok", False)),
        "rc": int(proc.returncode),
        "error": (proc.stderr or "").strip()[-1000:],
        "entry_count": int(payload.get("entry_count", 0) or 0),
        "missing_count": int(payload.get("missing_count", 0) or 0),
        "small_pdf_count": int(payload.get("small_pdf_count", 0) or 0),
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, repair: bool = False, timeout_seconds: int = 120) -> dict[str, Any]:
    catalog_path = project_root / "governance" / "health" / "report_pdf_bundle_latest.json"
    output_path = project_root / "governance" / "health" / "report_quality_guard_latest.json"
    repair_result = {"attempted": False}
    if repair or not catalog_path.exists():
        repair_result = _run_repair(project_root, timeout_seconds=timeout_seconds)

    catalog = _load_json(catalog_path)
    entries = catalog.get("entries") if isinstance(catalog.get("entries"), list) else []
    checks: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    degraded: list[dict[str, Any]] = []

    if not entries:
        blockers.append({"name": "report_catalog_missing_or_empty", "detail": str(catalog_path)})

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        slug = str(entry.get("slug") or "")
        pdf_path = Path(str(entry.get("pdf_path") or ""))
        if not pdf_path.is_absolute():
            pdf_path = project_root / pdf_path
        pdf = _pdf_check(pdf_path)
        detail = str(entry.get("detail") or "")
        check = {
            "slug": slug,
            "title": str(entry.get("title") or ""),
            "detail": detail,
            "pdf": pdf,
            "ok": bool(pdf.get("ok")),
            "report_ready_required": slug in REPORT_READY_RENDERERS,
            "report_ready_ok": True,
        }
        required_detail = REPORT_READY_RENDERERS.get(slug)
        if required_detail and detail != required_detail:
            check["report_ready_ok"] = False
            check["ok"] = False
            degraded.append(
                {
                    "name": "report_ready_renderer_missing",
                    "slug": slug,
                    "expected_detail": required_detail,
                    "actual_detail": detail,
                }
            )
        if not pdf.get("ok"):
            blockers.append({"name": "pdf_integrity_failed", "slug": slug, "pdf": pdf})
        checks.append(check)

    blocked_count = len(blockers)
    degraded_count = len(degraded)
    ok = blocked_count == 0 and degraded_count == 0
    payload = {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "ok": bool(ok),
        "overall_status": "ready" if ok else "blocked" if blocked_count else "degraded",
        "repair": repair_result,
        "catalog_path": str(catalog_path),
        "catalog_status": catalog.get("overall_status", "missing"),
        "metrics": {
            "entry_count": int(len(checks)),
            "blocked_count": int(blocked_count),
            "degraded_count": int(degraded_count),
            "missing_pdf_count": int(sum(1 for row in checks if not row["pdf"].get("exists"))),
            "small_pdf_count": int(sum(1 for row in checks if not row["pdf"].get("large_enough"))),
            "report_ready_renderer_count": int(sum(1 for row in checks if row.get("report_ready_required"))),
        },
        "blockers": blockers,
        "degraded_checks": degraded,
        "checks": checks,
        "recommended_actions": [
            "run ./scripts/ops/opsctl.sh report-quality-guard --repair --json after changing report renderers or COMMANDS.md report entries",
            "keep paper_performance and post_trade_analysis on report-ready renderers before sending documents externally",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate report-ready PDF artifacts and repair the sendout bundle when requested.")
    parser.add_argument("--repair", action="store_true", help="Regenerate the PDF bundle before validating.")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build_payload(PROJECT_ROOT, repair=bool(args.repair), timeout_seconds=max(int(args.timeout_seconds), 5))
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "report_quality_guard "
            f"status={payload['overall_status']} "
            f"entries={payload['metrics']['entry_count']} "
            f"blocked={payload['metrics']['blocked_count']} "
            f"degraded={payload['metrics']['degraded_count']}"
        )
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
