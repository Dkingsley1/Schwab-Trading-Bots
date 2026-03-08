import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODE_FILE = PROJECT_ROOT / "governance" / "health" / "shadow_watchdog_halt_recovery_latest.json"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except Exception:
        return []
    return rows


def _parse_iso_utc(value: str) -> datetime | None:
    if not value:
        return None
    raw = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _resolve_execution_mode(mode_payload: Dict[str, Any], reconcile_events: int) -> Dict[str, Any]:
    env_market_data_only = os.getenv("MARKET_DATA_ONLY")
    env_allow_order_execution = os.getenv("ALLOW_ORDER_EXECUTION")

    market_data_only = _as_bool(
        env_market_data_only if env_market_data_only is not None else mode_payload.get("market_data_only"),
        default=True,
    )
    allow_order_execution = _as_bool(
        env_allow_order_execution if env_allow_order_execution is not None else mode_payload.get("allow_order_execution"),
        default=False,
    )

    execution_expected = bool(allow_order_execution and (not market_data_only))
    inferred_from_events = int(reconcile_events) > 0
    if (not execution_expected) and inferred_from_events:
        execution_expected = True

    return {
        "market_data_only": bool(market_data_only),
        "allow_order_execution": bool(allow_order_execution),
        "execution_expected": bool(execution_expected),
        "inferred_execution_from_events": bool(inferred_from_events),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check reconciliation SLOs for live execution telemetry.")
    parser.add_argument("--lookback-minutes", type=int, default=30)
    parser.add_argument("--max-mismatch-rate", type=float, default=0.02)
    parser.add_argument("--max-error-rate", type=float, default=0.03)
    parser.add_argument("--max-staleness-minutes", type=float, default=5.0)
    parser.add_argument("--in-file", default="")
    parser.add_argument("--mode-file", default=str(DEFAULT_MODE_FILE))
    parser.add_argument(
        "--enforce-only-when-execution-enabled",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("LIVE_RECON_ENFORCE_ONLY_WHEN_EXECUTION_ENABLED", "1").strip() == "1",
    )
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "live_reconciliation_slo_latest.json"))
    parser.add_argument("--alert-file", default=str(PROJECT_ROOT / "governance" / "alerts" / "live_reconciliation_slo_events.jsonl"))
    parser.add_argument("--latest-alert-file", default=str(PROJECT_ROOT / "governance" / "alerts" / "live_reconciliation_slo_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    lookback = max(int(args.lookback_minutes), 1)
    window_start = now - timedelta(minutes=lookback)

    if args.in_file:
        src = Path(args.in_file)
    else:
        day = now.strftime("%Y%m%d")
        src = PROJECT_ROOT / "governance" / "events" / f"live_execution_guard_{day}.jsonl"

    rows = _read_jsonl(src)
    window_rows: List[Dict[str, Any]] = []
    for r in rows:
        ts = _parse_iso_utc(str(r.get("timestamp_utc", "")))
        if ts is None or ts < window_start:
            continue
        window_rows.append(r)

    reconcile_rows = [r for r in window_rows if str(r.get("event", "")).strip() == "position_reconcile"]

    mismatch_count = sum(1 for r in reconcile_rows if str(r.get("status", "")).strip() == "mismatch")
    error_count = sum(1 for r in reconcile_rows if str(r.get("status", "")).strip() == "error")

    total = len(reconcile_rows)
    mismatch_rate = (mismatch_count / total) if total > 0 else 0.0
    error_rate = (error_count / total) if total > 0 else 0.0

    last_ts = None
    if reconcile_rows:
        last_raw = str(reconcile_rows[-1].get("timestamp_utc", ""))
        last_ts = _parse_iso_utc(last_raw)

    staleness_minutes = ((now - last_ts).total_seconds() / 60.0) if last_ts else 1e9
    stale = staleness_minutes > float(args.max_staleness_minutes)

    mode_payload = _read_json(Path(args.mode_file)) if args.mode_file else {}
    mode = _resolve_execution_mode(mode_payload, reconcile_events=total)

    raw_failed_checks: List[str] = []
    if mismatch_rate > float(args.max_mismatch_rate):
        raw_failed_checks.append("mismatch_rate")
    if error_rate > float(args.max_error_rate):
        raw_failed_checks.append("error_rate")
    if stale:
        raw_failed_checks.append("staleness")

    enforcement_suppressed = bool(args.enforce_only_when_execution_enabled) and (not mode["execution_expected"])
    failed_checks = [] if enforcement_suppressed else list(raw_failed_checks)

    out = {
        "timestamp_utc": now.isoformat(),
        "ok": len(failed_checks) == 0,
        "source_file": str(src),
        "lookback_minutes": lookback,
        "window_start_utc": window_start.isoformat(),
        "mode": mode,
        "enforcement": {
            "enforce_only_when_execution_enabled": bool(args.enforce_only_when_execution_enabled),
            "suppressed": bool(enforcement_suppressed),
            "suppressed_checks": list(raw_failed_checks) if enforcement_suppressed else [],
        },
        "metrics": {
            "reconcile_events": total,
            "mismatch_count": mismatch_count,
            "error_count": error_count,
            "mismatch_rate": round(float(mismatch_rate), 6),
            "error_rate": round(float(error_rate), 6),
            "last_reconcile_timestamp_utc": (last_ts.isoformat() if last_ts else ""),
            "staleness_minutes": round(float(staleness_minutes), 4),
        },
        "thresholds": {
            "max_mismatch_rate": float(args.max_mismatch_rate),
            "max_error_rate": float(args.max_error_rate),
            "max_staleness_minutes": float(args.max_staleness_minutes),
        },
        "failed_checks": failed_checks,
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    if failed_checks:
        alert = {
            "timestamp_utc": now.isoformat(),
            "event": "live_reconciliation_slo_breach",
            "severity": "critical",
            "failed_checks": failed_checks,
            "metrics": out["metrics"],
            "thresholds": out["thresholds"],
            "mode": mode,
        }
        _append_jsonl(Path(args.alert_file), alert)
        latest_path = Path(args.latest_alert_file)
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(json.dumps(alert, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(
            "live_reconciliation_slo "
            f"ok={int(bool(out['ok']))} events={total} "
            f"mismatch_rate={mismatch_rate:.6f}/{float(args.max_mismatch_rate):.6f} "
            f"error_rate={error_rate:.6f}/{float(args.max_error_rate):.6f} "
            f"staleness_min={staleness_minutes:.3f}/{float(args.max_staleness_minutes):.3f} "
            f"execution_expected={int(mode['execution_expected'])} suppressed={int(enforcement_suppressed)}"
        )

    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
