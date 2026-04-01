#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEALTH_DIR = PROJECT_ROOT / "governance" / "health"
REPORTS_DIR = PROJECT_ROOT / "exports" / "reports"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _age_hours(ts: datetime | None, now: datetime) -> float | None:
    if ts is None:
        return None
    return max((now - ts).total_seconds() / 3600.0, 0.0)


def _is_fresh(ts: datetime | None, now: datetime, max_age_hours: float) -> bool:
    age = _age_hours(ts, now)
    if age is None:
        return False
    return age <= max(max_age_hours, 0.25)


def _round_age(age: float | None) -> float | None:
    if age is None:
        return None
    return round(age, 3)


def build_macro_crosscheck_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    health_dir = project_root / "governance" / "health"
    latest_status_path = project_root / "exports" / "external_feeds" / "latest_status.json"
    official_macro_path = health_dir / "official_macro_context_sync_latest.json"
    market_micro_path = health_dir / "market_micro_sync_latest.json"

    latest_status = _read_json(latest_status_path)
    official_macro = _read_json(official_macro_path)
    market_micro = _read_json(market_micro_path)

    latest_status_ts = _parse_ts(latest_status.get("timestamp_utc"))
    official_macro_ts = _parse_ts(official_macro.get("timestamp_utc"))
    market_micro_ts = _parse_ts(market_micro.get("timestamp_utc"))

    latest_status_fresh = _is_fresh(latest_status_ts, now, 24.0)
    official_macro_fresh = _is_fresh(official_macro_ts, now, 24.0)
    market_micro_fresh = _is_fresh(market_micro_ts, now, 24.0)

    latest_status_bls_ok = bool(((latest_status.get("bls") or {}).get("ok", False)))
    latest_status_bea_ok = bool(((latest_status.get("bea") or {}).get("ok", False)))

    official_sources = official_macro.get("sources") if isinstance(official_macro.get("sources"), dict) else {}
    official_bls_ok = bool(((official_sources.get("bls") or {}).get("ok", False)))
    official_bls_calendar_ok = bool(((official_sources.get("bls_calendar") or {}).get("ok", False)))
    official_bea_ok = bool(((official_sources.get("bea") or {}).get("ok", False)))
    official_treasury_ok = bool(((official_sources.get("treasury") or {}).get("ok", False)))

    market_micro_sources = market_micro.get("sources") if isinstance(market_micro.get("sources"), dict) else {}
    market_micro_treasury_ok = bool(((market_micro_sources.get("treasury_auctions") or {}).get("ok", False)))

    checks = {
        "artifacts_fresh": {
            "ok": latest_status_fresh and official_macro_fresh and market_micro_fresh,
            "artifacts": {
                "latest_status_age_hours": _round_age(_age_hours(latest_status_ts, now)),
                "official_macro_age_hours": _round_age(_age_hours(official_macro_ts, now)),
                "market_micro_age_hours": _round_age(_age_hours(market_micro_ts, now)),
            },
        },
        "bls_dual_source": {
            "ok": latest_status_bls_ok and (official_bls_ok or official_bls_calendar_ok),
            "latest_status_bls_ok": latest_status_bls_ok,
            "official_bls_ok": official_bls_ok,
            "official_bls_calendar_ok": official_bls_calendar_ok,
        },
        "bea_dual_source": {
            "ok": latest_status_bea_ok and official_bea_ok,
            "latest_status_bea_ok": latest_status_bea_ok,
            "official_bea_ok": official_bea_ok,
        },
        "treasury_dual_source": {
            "ok": official_treasury_ok and market_micro_treasury_ok,
            "official_macro_treasury_ok": official_treasury_ok,
            "market_micro_treasury_ok": market_micro_treasury_ok,
            "official_macro_rows": int(((official_sources.get("treasury") or {}).get("rows", 0)) or 0),
            "market_micro_rows": int(((market_micro_sources.get("treasury_auctions") or {}).get("rows", 0)) or 0),
        },
    }

    passed_checks = sum(1 for check in checks.values() if bool(check.get("ok", False)))
    total_checks = len(checks)
    notes: list[str] = []
    if bool(((official_sources.get("treasury") or {}).get("fallback"))) and not bool(((official_sources.get("treasury") or {}).get("ok"))):
        notes.append(f"official_treasury_fallback={((official_sources.get('treasury') or {}).get('fallback'))}")
    if bool(((official_sources.get("bls") or {}).get("fallback"))) and not bool(((official_sources.get("bls") or {}).get("ok"))):
        notes.append(f"official_bls_fallback={((official_sources.get('bls') or {}).get('fallback'))}")
    fed_calendar_err = (official_sources.get("federal_reserve_calendar") or {}).get("error")
    if fed_calendar_err:
        notes.append("fed_calendar_partial_error")

    return {
        "timestamp_utc": now.isoformat(),
        "ok": passed_checks == total_checks and total_checks > 0,
        "passed_checks": passed_checks,
        "total_checks": total_checks,
        "notes": notes,
        "checks": checks,
        "artifacts": {
            "latest_status": str(latest_status_path),
            "official_macro_context_sync": str(official_macro_path),
            "market_micro_sync": str(market_micro_path),
        },
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Macro Crosscheck ({payload.get('timestamp_utc', '')})",
        f"- ok: {bool(payload.get('ok', False))}",
        f"- passed_checks: {int(payload.get('passed_checks', 0) or 0)}",
        f"- total_checks: {int(payload.get('total_checks', 0) or 0)}",
        f"- notes: {','.join(payload.get('notes', [])) if isinstance(payload.get('notes'), list) and payload.get('notes') else 'none'}",
        "",
    ]
    checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
    for name, check in checks.items():
        if not isinstance(check, dict):
            continue
        lines.append(f"## {name}")
        lines.append(f"- ok: {bool(check.get('ok', False))}")
        for key, value in check.items():
            if key == "ok":
                continue
            lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-check overlapping macro sources across artifacts.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_macro_crosscheck_payload(PROJECT_ROOT)

    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_json = REPORTS_DIR / f"macro_crosscheck_{day}.json"
    out_md = REPORTS_DIR / f"macro_crosscheck_{day}.md"
    latest_json = HEALTH_DIR / "macro_crosscheck_latest.json"
    latest_md = REPORTS_DIR / "macro_crosscheck_latest.md"

    json_text = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
    md_text = _render_markdown(payload)
    out_json.write_text(json_text, encoding="utf-8")
    out_md.write_text(md_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "macro_crosscheck ok={ok} passed_checks={passed} total_checks={total}".format(
                ok=str(bool(payload.get("ok", False))).lower(),
                passed=int(payload.get("passed_checks", 0) or 0),
                total=int(payload.get("total_checks", 0) or 0),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
