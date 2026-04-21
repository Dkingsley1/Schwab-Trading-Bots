#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.collect_options_flow_context import (
        DEFAULT_UNUSUAL_WHALES_EXPORT_MAX_AGE_SECONDS,
        DEFAULT_UNUSUAL_WHALES_EXPORT_MIN_STABLE_SECONDS,
        inspect_unusual_whales_export,
    )
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_timestamp, utc_now, write_payload
    from scripts.ops.options_flow_export_hygiene_bot import build_payload as build_export_hygiene_payload
else:
    from ..collect_options_flow_context import (
        DEFAULT_UNUSUAL_WHALES_EXPORT_MAX_AGE_SECONDS,
        DEFAULT_UNUSUAL_WHALES_EXPORT_MIN_STABLE_SECONDS,
        inspect_unusual_whales_export,
    )
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_timestamp, utc_now, write_payload
    from .options_flow_export_hygiene_bot import build_payload as build_export_hygiene_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "options_flow_efficiency_latest.json"
PYTHON_BIN = Path(sys.executable)
COLLECTOR_SCRIPT = PROJECT_ROOT / "scripts" / "collect_options_flow_context.py"


def _run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = int(proc.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        rc = 124
        timed_out = True
    payload = {}
    for raw in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break
    return {
        "cmd": list(cmd),
        "rc": rc,
        "timed_out": timed_out,
        "stdout_tail": "\n".join(stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-10:]),
        "payload": payload,
    }


def _status_path(project_root: Path) -> Path:
    primary = project_root / "governance" / "health" / "options_flow_context_sync_latest.json"
    return primary if primary.exists() else project_root / "governance" / "health" / "tastytrade_context_sync_latest.json"


def _status_age_seconds(payload: dict[str, Any], path: Path) -> float | None:
    ts = payload_timestamp(payload, path)
    if ts is None:
        return None
    return max((utc_now() - ts).total_seconds(), 0.0)


def _refresh_reasons(
    *,
    status_payload: dict[str, Any],
    status_path: Path,
    status_max_age_seconds: int,
    export_inspection: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    status_age = _status_age_seconds(status_payload, status_path)
    if not status_path.exists():
        reasons.append("status_missing")
    if status_age is None:
        reasons.append("status_timestamp_missing")
    elif status_age > max(int(status_max_age_seconds), 1):
        reasons.append("status_stale")
    overall_status = str(status_payload.get("overall_status") or ("ready" if status_payload.get("ok", False) else "blocked")).strip()
    if overall_status in {"blocked", "degraded"}:
        reasons.append(f"context_{overall_status}")

    current_sources = status_payload.get("sources") if isinstance(status_payload.get("sources"), dict) else {}
    current_export = current_sources.get("unusual_whales_export") if isinstance(current_sources.get("unusual_whales_export"), dict) else {}
    current_candidate = str(current_export.get("selected_candidate") or "")
    inspected_candidate = str(export_inspection.get("selected_candidate") or "")
    if inspected_candidate and inspected_candidate != current_candidate:
        reasons.append("export_candidate_changed")
    if bool(export_inspection.get("configured", False)) and bool(export_inspection.get("usable", False)) != bool(current_export.get("ok", False)):
        reasons.append("export_hygiene_state_changed")
    export_ts = payload_timestamp({"timestamp_utc": export_inspection.get("timestamp_utc")})
    status_ts = payload_timestamp(status_payload, status_path)
    if export_ts is not None and status_ts is not None and export_ts > status_ts:
        reasons.append("export_newer_than_status")
    if bool(export_inspection.get("configured", False)) and not bool(export_inspection.get("usable", False)):
        reasons.append("export_unusable")
    return ordered_unique(reasons)


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    export_path: str | None = None,
    status_max_age_seconds: int = 14400,
    export_max_age_seconds: int = DEFAULT_UNUSUAL_WHALES_EXPORT_MAX_AGE_SECONDS,
    export_min_stable_seconds: int = DEFAULT_UNUSUAL_WHALES_EXPORT_MIN_STABLE_SECONDS,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    configured_export_path = str(export_path if export_path is not None else os.getenv("UNUSUAL_WHALES_EXPORT_PATH", "")).strip()
    status_path = _status_path(project_root)
    status_payload = load_json(status_path)

    export_hygiene = build_export_hygiene_payload(
        project_root,
        export_path=configured_export_path,
        max_age_seconds=max(int(export_max_age_seconds), 1),
        min_stable_seconds=max(int(export_min_stable_seconds), 0),
        apply=bool(apply),
    )
    _, export_inspection = inspect_unusual_whales_export(
        configured_export_path,
        max_age_seconds=max(int(export_max_age_seconds), 1),
        min_stable_seconds=max(int(export_min_stable_seconds), 0),
    )
    refresh_reasons = _refresh_reasons(
        status_payload=status_payload,
        status_path=status_path,
        status_max_age_seconds=max(int(status_max_age_seconds), 1),
        export_inspection=export_inspection,
    )
    repair_plan = []
    if bool(export_inspection.get("configured", False)):
        repair_plan.append(
            {
                "name": "options_flow_export_hygiene",
                "reason": "inspect_and_promote_latest_unusual_whales_export",
            }
        )
    if refresh_reasons:
        repair_plan.append(
            {
                "name": "options_flow_context_refresh",
                "reason": ",".join(refresh_reasons),
            }
        )

    collector = {
        "executed": False,
        "rc": None,
        "timed_out": False,
        "payload": {},
        "stdout_tail": "",
        "stderr_tail": "",
    }
    actions_taken: list[str] = []
    if bool((export_hygiene.get("promotion") or {}).get("promoted", False)):
        actions_taken.append("promoted_export")

    if apply and refresh_reasons:
        cmd = [str(PYTHON_BIN), str(COLLECTOR_SCRIPT), "--json"]
        if configured_export_path:
            cmd.extend(["--unusual-whales-export-path", configured_export_path])
        cmd.extend(
            [
                "--unusual-whales-export-max-age-seconds",
                str(max(int(export_max_age_seconds), 1)),
                "--unusual-whales-export-min-stable-seconds",
                str(max(int(export_min_stable_seconds), 0)),
            ]
        )
        collector = _run_json(cmd, cwd=project_root, timeout_sec=max(int(timeout_sec), 1))
        collector["executed"] = True
        actions_taken.append("refreshed_context")
        status_payload = load_json(status_path)

    current_overall_status = str(status_payload.get("overall_status") or ("ready" if status_payload.get("ok", False) else "blocked")).strip()
    operator_followups: list[str] = []
    if "export_unusable" in refresh_reasons or not bool((export_hygiene.get("inspection") or {}).get("usable", False)):
        operator_followups.append("repair the Unusual Whales export handoff so the collector stops ingesting stale or malformed overlays")
    if current_overall_status == "degraded":
        operator_followups.append("restore Polygon backbone coverage if you want options-flow completeness to return to ready")
    if current_overall_status == "blocked":
        operator_followups.append("restore at least one working options-flow source because the context is currently blocked")

    overall_status = current_overall_status or "blocked"
    if not status_payload and refresh_reasons:
        overall_status = "blocked"

    payload = {
        "timestamp_utc": iso_now(),
        "overall_status": overall_status,
        "apply_requested": bool(apply),
        "status_path": str(status_path),
        "status_age_seconds": _status_age_seconds(status_payload, status_path),
        "refresh_needed": bool(refresh_reasons),
        "refresh_reasons": refresh_reasons,
        "repair_plan": repair_plan,
        "actions_taken": actions_taken,
        "export_hygiene": export_hygiene,
        "collector": collector,
        "latest_context": status_payload,
        "operator_followups": ordered_unique(operator_followups),
        "metrics": {
            "refresh_reason_count": len(refresh_reasons),
            "collector_executed": bool(collector.get("executed", False)),
            "collector_rc": collector.get("rc"),
            "context_ok": bool(status_payload.get("ok", False)),
            "context_profile": str(status_payload.get("context_profile") or ""),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep the options-flow stack efficient by hardening export handoff and refreshing only when needed.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--export-path", default=os.getenv("UNUSUAL_WHALES_EXPORT_PATH", ""))
    parser.add_argument(
        "--status-max-age-seconds",
        type=int,
        default=int(os.getenv("OPTIONS_FLOW_MAX_STALE_SECONDS", "14400")),
    )
    parser.add_argument(
        "--export-max-age-seconds",
        type=int,
        default=int(os.getenv("UNUSUAL_WHALES_EXPORT_MAX_AGE_SECONDS", str(DEFAULT_UNUSUAL_WHALES_EXPORT_MAX_AGE_SECONDS))),
    )
    parser.add_argument(
        "--export-min-stable-seconds",
        type=int,
        default=int(os.getenv("UNUSUAL_WHALES_EXPORT_MIN_STABLE_SECONDS", str(DEFAULT_UNUSUAL_WHALES_EXPORT_MIN_STABLE_SECONDS))),
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=int(os.getenv("OPTIONS_FLOW_EFFICIENCY_TIMEOUT_SECONDS", "900")),
    )
    parser.add_argument("--out-path", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser()
    payload = build_payload(
        project_root,
        apply=bool(args.apply),
        export_path=str(args.export_path or ""),
        status_max_age_seconds=int(args.status_max_age_seconds),
        export_max_age_seconds=int(args.export_max_age_seconds),
        export_min_stable_seconds=int(args.export_min_stable_seconds),
        timeout_sec=int(args.timeout_sec),
    )
    out_path = Path(args.out_path).expanduser()
    write_payload(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "options_flow_efficiency overall_status={status} refresh_needed={refresh_needed} collector_executed={collector}".format(
                status=str(payload.get("overall_status") or ""),
                refresh_needed=str(bool(payload.get("refresh_needed", False))).lower(),
                collector=str(bool((payload.get("metrics") or {}).get("collector_executed", False))).lower(),
            )
        )
    return 0 if str(payload.get("overall_status") or "") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
