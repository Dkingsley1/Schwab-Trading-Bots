#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from strategy_attribution_report import build_strategy_attribution_report, render_strategy_attribution_markdown


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_PATH = PROJECT_ROOT / "governance" / "health" / "post_trade_analysis_latest.json"
DEFAULT_MD_PATH = PROJECT_ROOT / "exports" / "reports" / "post_trade_analysis_latest.md"


def _python_bin(project_root: Path) -> str:
    candidate = project_root / ".venv312" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return str(Path(sys.executable))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_json_command(cmd: list[str], *, cwd: Path) -> tuple[int, dict[str, Any], str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    payload: dict[str, Any] = {}
    if stdout.startswith("{") and stdout.endswith("}"):
        try:
            obj = json.loads(stdout)
            if isinstance(obj, dict):
                payload = obj
        except Exception:
            payload = {}
    return int(proc.returncode), payload, stderr


def _softguard_summary(project_root: Path, *, day: str) -> dict[str, Any]:
    path = project_root / "governance" / "events" / f"live_softguard_{day}.jsonl"
    reason_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    rows = 0
    latest_ts = ""

    if path.exists():
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                rows += 1
                ts = str(row.get("timestamp_utc") or "").strip()
                if ts and ts > latest_ts:
                    latest_ts = ts
                reason_counts[str(row.get("reason") or "unknown")] += 1
                mode_counts[str(row.get("mode_label") or "unknown")] += 1

    return {
        "path": str(path),
        "rows": int(rows),
        "latest_timestamp_utc": latest_ts,
        "reason_counts": dict(sorted(reason_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def _assessment_lines(
    *,
    strategy_payload: dict[str, Any],
    calibration_payload: dict[str, Any],
    runtime_payload: dict[str, Any],
    softguard_payload: dict[str, Any],
) -> list[str]:
    lines: list[str] = []

    total_pnl_proxy = float(strategy_payload.get("total_pnl_proxy", 0.0) or 0.0)
    top_lane = str(strategy_payload.get("top_lane") or "none")
    if strategy_payload.get("row_count", 0):
        direction = "positive" if total_pnl_proxy >= 0 else "negative"
        lines.append(
            f"Attribution captured {int(strategy_payload.get('row_count', 0))} rows with {direction} total pnl_proxy "
            f"({total_pnl_proxy:.8f}); top lane was {top_lane}."
        )
    else:
        lines.append("No strategy attribution rows were available for the selected day.")

    if calibration_payload:
        mae_bps = float(((calibration_payload.get("metrics") or {}).get("mae_bps", 0.0) or 0.0))
        max_mae = float(((calibration_payload.get("thresholds") or {}).get("max_mae_bps", 0.0) or 0.0))
        if calibration_payload.get("ok", False):
            lines.append(f"Paper execution calibration stayed inside guardrails with mae_bps={mae_bps:.4f}/{max_mae:.4f}.")
        else:
            lines.append(f"Paper execution calibration breached guardrails with mae_bps={mae_bps:.4f}/{max_mae:.4f}.")
    else:
        lines.append("Paper execution calibration data was not available.")

    decision = runtime_payload.get("decision") if isinstance(runtime_payload.get("decision"), dict) else {}
    watchdog = runtime_payload.get("watchdog") if isinstance(runtime_payload.get("watchdog"), dict) else {}
    decision_rows = int(decision.get("rows", 0) or 0)
    stale_windows = int(decision.get("stale_windows", 0) or 0)
    restarts = int(watchdog.get("restarts", 0) or 0)
    if decision_rows > 0:
        lines.append(
            f"Runtime summary saw decision_rows={decision_rows}, decision_stale_windows={stale_windows}, "
            f"watchdog_restarts={restarts}."
        )
    else:
        lines.append("Runtime summary did not find any decision rows for the selected day.")

    softguard_rows = int(softguard_payload.get("rows", 0) or 0)
    if softguard_rows > 0:
        top_reason = next(iter((softguard_payload.get("reason_counts") or {}).keys()), "unknown")
        lines.append(f"Softguard logged {softguard_rows} halt events; leading reason was {top_reason}.")
    else:
        lines.append("No live softguard halt events were recorded for the selected day.")

    return lines


def render_post_trade_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    calibration = payload.get("paper_execution_calibration") if isinstance(payload.get("paper_execution_calibration"), dict) else {}
    runtime = payload.get("daily_runtime_summary") if isinstance(payload.get("daily_runtime_summary"), dict) else {}
    softguard = payload.get("softguard") if isinstance(payload.get("softguard"), dict) else {}

    lines = [
        "# Post-Trade Analysis",
        "",
        f"- generated_utc: {payload.get('timestamp_utc', '')}",
        f"- day: {payload.get('day', '')}",
        f"- lookback_hours: {payload.get('lookback_hours', 0)}",
        f"- total_pnl_proxy: {float(summary.get('total_pnl_proxy', 0.0) or 0.0):.8f}",
        f"- top_lane: {summary.get('top_lane', '')}",
        f"- paper_mae_bps: {float(summary.get('paper_mae_bps', 0.0) or 0.0):.4f}",
        f"- global_halt_events: {int(summary.get('global_halt_events', 0) or 0)}",
        "",
        "## Assessment",
        "",
    ]
    for line in payload.get("assessment", []) or []:
        lines.append(f"- {line}")

    lines.extend(
        [
            "",
            "## Runtime Snapshot",
            "",
            f"- decision_rows: {int(((runtime.get('decision') or {}).get('rows', 0) or 0))}",
            f"- decision_stale_windows: {int(((runtime.get('decision') or {}).get('stale_windows', 0) or 0))}",
            f"- watchdog_restarts: {int(((runtime.get('watchdog') or {}).get('restarts', 0) or 0))}",
            "",
            "## Calibration Snapshot",
            "",
            f"- ok: {bool(calibration.get('ok', False))}",
            f"- samples: {int(calibration.get('samples', 0) or 0)}",
            f"- mae_bps: {float(((calibration.get('metrics') or {}).get('mae_bps', 0.0) or 0.0)):.4f}",
            f"- p95_bps: {float(((calibration.get('metrics') or {}).get('p95_bps', 0.0) or 0.0)):.4f}",
            "",
            "## Softguard Snapshot",
            "",
            f"- rows: {int(softguard.get('rows', 0) or 0)}",
            f"- latest_timestamp_utc: {softguard.get('latest_timestamp_utc', '')}",
            "",
            "## Strategy Attribution",
            "",
        ]
    )
    lines.extend(render_strategy_attribution_markdown(payload.get("strategy_attribution", {})).splitlines()[2:])
    return "\n".join(lines).strip() + "\n"


def build_post_trade_analysis(
    project_root: Path,
    *,
    day: str,
    hours: int,
    runner: Callable[[list[str], Path], tuple[int, dict[str, Any], str]] | None = None,
) -> dict[str, Any]:
    py = _python_bin(project_root)
    exec_runner = runner or (lambda cmd, cwd: _run_json_command(cmd, cwd=cwd))

    strategy_payload = build_strategy_attribution_report(project_root, day=day)
    calibration_cmd = [py, str(project_root / "scripts" / "paper_execution_calibration_report.py"), "--hours", str(max(int(hours), 1)), "--json"]
    runtime_cmd = [py, str(project_root / "scripts" / "daily_runtime_summary.py"), "--day", str(day), "--json"]

    calibration_rc, calibration_payload, calibration_err = exec_runner(calibration_cmd, project_root)
    runtime_rc, runtime_payload, runtime_err = exec_runner(runtime_cmd, project_root)
    softguard_payload = _softguard_summary(project_root, day=day)

    assessment = _assessment_lines(
        strategy_payload=strategy_payload,
        calibration_payload=calibration_payload,
        runtime_payload=runtime_payload,
        softguard_payload=softguard_payload,
    )

    summary = {
        "total_pnl_proxy": float(strategy_payload.get("total_pnl_proxy", 0.0) or 0.0),
        "top_lane": str(strategy_payload.get("top_lane") or ""),
        "paper_mae_bps": float(((calibration_payload.get("metrics") or {}).get("mae_bps", 0.0) or 0.0)),
        "paper_ok": bool(calibration_payload.get("ok", False)),
        "decision_rows": int(((runtime_payload.get("decision") or {}).get("rows", 0) or 0)),
        "decision_stale_windows": int(((runtime_payload.get("decision") or {}).get("stale_windows", 0) or 0)),
        "watchdog_restarts": int(((runtime_payload.get("watchdog") or {}).get("restarts", 0) or 0)),
        "global_halt_events": int(softguard_payload.get("rows", 0) or 0),
        "top_halt_reason": next(iter((softguard_payload.get("reason_counts") or {}).keys()), ""),
    }

    ok = bool(strategy_payload.get("ok", False)) and (runtime_rc == 0) and (calibration_rc in {0, 2})
    return {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "ok": ok,
        "day": day,
        "lookback_hours": int(hours),
        "summary": summary,
        "assessment": assessment,
        "strategy_attribution": strategy_payload,
        "paper_execution_calibration": calibration_payload,
        "daily_runtime_summary": runtime_payload,
        "softguard": softguard_payload,
        "sources": {
            "strategy_attribution": str(project_root / "governance" / "health" / "strategy_attribution_latest.json"),
            "paper_execution_calibration_rc": int(calibration_rc),
            "paper_execution_calibration_error": calibration_err,
            "daily_runtime_summary_rc": int(runtime_rc),
            "daily_runtime_summary_error": runtime_err,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a post-trade analysis artifact from attribution, calibration, runtime, and halt data.")
    parser.add_argument("--day", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--out-file", default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_post_trade_analysis(PROJECT_ROOT, day=str(args.day), hours=max(int(args.hours), 1))
    out_path = Path(args.out_file)
    md_path = Path(args.md_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    md_path.write_text(render_post_trade_markdown(payload), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "post_trade_analysis "
            f"ok={int(bool(payload.get('ok', False)))} "
            f"day={payload.get('day', '')} "
            f"top_lane={((payload.get('summary') or {}).get('top_lane', '') or 'none')} "
            f"global_halt_events={int(((payload.get('summary') or {}).get('global_halt_events', 0) or 0))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
