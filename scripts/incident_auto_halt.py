import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HALT_FLAG = PROJECT_ROOT / "governance" / "health" / "GLOBAL_TRADING_HALT.flag"
STATE_FILE = PROJECT_ROOT / "governance" / "health" / "incident_auto_halt_state.json"
EVENT_LOG = PROJECT_ROOT / "governance" / "watchdog" / "incident_auto_halt_events.jsonl"
LATEST_ALERT = PROJECT_ROOT / "governance" / "alerts" / "incident_auto_halt_latest.json"
DEFAULT_MODE_FILE = PROJECT_ROOT / "governance" / "health" / "shadow_watchdog_halt_recovery_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
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


def _resolve_execution_expected(mode_payload: dict[str, Any], reconciliation_slo: dict[str, Any]) -> dict[str, Any]:
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
    reconcile_events = int((reconciliation_slo.get("metrics") or {}).get("reconcile_events", 0) or 0)
    inferred_from_events = reconcile_events > 0
    if (not execution_expected) and inferred_from_events:
        execution_expected = True

    return {
        "market_data_only": bool(market_data_only),
        "allow_order_execution": bool(allow_order_execution),
        "execution_expected": bool(execution_expected),
        "inferred_execution_from_events": bool(inferred_from_events),
        "reconcile_events": int(reconcile_events),
    }


def evaluate_incident(
    daily_verify: dict[str, Any],
    quality_gate: dict[str, Any],
    reconciliation_slo: dict[str, Any],
    *,
    require_daily_verify: bool,
    require_quality_gate: bool,
    require_reconciliation_slo: bool,
) -> tuple[bool, list[str], dict[str, Any]]:
    failed: list[str] = []

    if require_daily_verify and not bool(daily_verify.get("ok", False)):
        failed.append("daily_verify_not_ok")

    if require_quality_gate and not bool(quality_gate.get("ok", False)):
        failed.append("promotion_quality_gate_not_ok")

    if require_reconciliation_slo and not bool(reconciliation_slo.get("ok", False)):
        failed.append("reconciliation_slo_not_ok")

    detail = {
        "daily_verify_ok": bool(daily_verify.get("ok", False)),
        "promotion_quality_gate_ok": bool(quality_gate.get("ok", False)),
        "reconciliation_slo_ok": bool(reconciliation_slo.get("ok", False)),
        "require_daily_verify": bool(require_daily_verify),
        "require_quality_gate": bool(require_quality_gate),
        "require_reconciliation_slo": bool(require_reconciliation_slo),
    }
    return len(failed) == 0, failed, detail


def main() -> int:
    parser = argparse.ArgumentParser(description="Trip GLOBAL_TRADING_HALT on persistent critical health failures.")
    parser.add_argument("--daily-verify-file", default=str(PROJECT_ROOT / "governance" / "health" / "daily_auto_verify_latest.json"))
    parser.add_argument("--quality-gate-file", default=str(PROJECT_ROOT / "governance" / "health" / "promotion_quality_gate_latest.json"))
    parser.add_argument("--reconciliation-file", default=str(PROJECT_ROOT / "governance" / "health" / "live_reconciliation_slo_latest.json"))
    parser.add_argument("--mode-file", default=str(DEFAULT_MODE_FILE))
    parser.add_argument("--state-file", default=str(STATE_FILE))
    parser.add_argument("--event-log", default=str(EVENT_LOG))
    parser.add_argument("--latest-alert-file", default=str(LATEST_ALERT))
    parser.add_argument("--halt-flag", default=str(HALT_FLAG))
    parser.add_argument("--trip-streak", type=int, default=3)
    parser.add_argument("--clear-streak", type=int, default=2)
    parser.add_argument("--auto-clear", action="store_true", default=False)
    parser.add_argument("--force-clear", action="store_true")
    parser.add_argument(
        "--enforce-only-when-execution-enabled",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("INCIDENT_AUTO_HALT_ENFORCE_ONLY_WHEN_EXECUTION_ENABLED", "1").strip() == "1",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc).isoformat()
    halt_flag = Path(args.halt_flag)
    state_path = Path(args.state_file)

    state = _load_json(state_path)
    state.setdefault("fail_streak", 0)
    state.setdefault("clear_streak", 0)

    if args.force_clear:
        if halt_flag.exists():
            halt_flag.unlink()
        state["fail_streak"] = 0
        state["clear_streak"] = 0
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
        event = {"timestamp_utc": now, "event": "halt_force_cleared", "halt": False}
        _append_jsonl(Path(args.event_log), event)
        if args.json:
            print(json.dumps(event, ensure_ascii=True))
        else:
            print("incident_auto_halt_event=halt_force_cleared")
        return 0

    daily_verify = _load_json(Path(args.daily_verify_file))
    quality_gate = _load_json(Path(args.quality_gate_file))
    reconciliation = _load_json(Path(args.reconciliation_file))
    mode_payload = _load_json(Path(args.mode_file)) if args.mode_file else {}
    mode = _resolve_execution_expected(mode_payload, reconciliation)

    effective_reconciliation_required = True
    enforcement_suppressed = bool(args.enforce_only_when_execution_enabled) and (not mode["execution_expected"])
    if enforcement_suppressed:
        effective_reconciliation_required = False

    ok, failed_checks, detail = evaluate_incident(
        daily_verify,
        quality_gate,
        reconciliation,
        require_daily_verify=True,
        require_quality_gate=True,
        require_reconciliation_slo=bool(effective_reconciliation_required),
    )

    detail["mode"] = mode
    detail["enforcement_suppressed"] = bool(enforcement_suppressed)

    if ok:
        state["clear_streak"] = int(state.get("clear_streak", 0)) + 1
        state["fail_streak"] = 0
    else:
        state["fail_streak"] = int(state.get("fail_streak", 0)) + 1
        state["clear_streak"] = 0

    if enforcement_suppressed:
        state["fail_streak"] = 0
        state["clear_streak"] = max(int(state.get("clear_streak", 0)), max(int(args.clear_streak), 1))

    event: dict[str, Any] = {
        "timestamp_utc": now,
        "ok": ok,
        "failed_checks": failed_checks,
        "fail_streak": state["fail_streak"],
        "clear_streak": state["clear_streak"],
        "detail": detail,
    }

    effective_auto_clear = bool(args.auto_clear) or bool(enforcement_suppressed)

    if state["fail_streak"] >= max(int(args.trip_streak), 1):
        halt_flag.parent.mkdir(parents=True, exist_ok=True)
        halt_payload = {
            "timestamp_utc": now,
            "reason": "incident_auto_halt",
            "failed_checks": failed_checks,
            "fail_streak": state["fail_streak"],
            "mode": mode,
        }
        halt_flag.write_text(json.dumps(halt_payload, ensure_ascii=True), encoding="utf-8")
        event["event"] = "halt_set"
    elif halt_flag.exists() and effective_auto_clear and state["clear_streak"] >= max(int(args.clear_streak), 1):
        halt_flag.unlink()
        event["event"] = "halt_cleared" if not enforcement_suppressed else "halt_cleared_execution_not_expected"
    else:
        event["event"] = "state_update"

    event["halt"] = halt_flag.exists()

    latest_alert_path = Path(args.latest_alert_file)
    latest_alert_path.parent.mkdir(parents=True, exist_ok=True)
    latest_alert_path.write_text(json.dumps(event, ensure_ascii=True, indent=2), encoding="utf-8")

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
    _append_jsonl(Path(args.event_log), event)

    if args.json:
        print(json.dumps(event, ensure_ascii=True))
    else:
        checks = ",".join(failed_checks) if failed_checks else "none"
        print(
            "incident_auto_halt "
            f"event={event['event']} fail_streak={state['fail_streak']} "
            f"clear_streak={state['clear_streak']} halt={int(halt_flag.exists())} "
            f"checks={checks} execution_expected={int(mode['execution_expected'])}"
        )

    if not ok and halt_flag.exists() and (not enforcement_suppressed):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
