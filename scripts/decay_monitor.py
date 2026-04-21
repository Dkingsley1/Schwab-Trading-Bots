#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "research" / "decay_monitor_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    health_root = project_root / "governance" / "health"

    paper = _load_json(health_root / "paper_performance_latest.json")
    promotion = _load_json(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json")

    sleeve_latest = paper.get("sleeve_latest") if isinstance(paper.get("sleeve_latest"), list) else []
    history_daily = paper.get("history_daily_series") if isinstance(paper.get("history_daily_series"), list) else []
    period_change = paper.get("period_change_series") if isinstance(paper.get("period_change_series"), list) else []
    sleeve_daily = paper.get("sleeve_daily_series") if isinstance(paper.get("sleeve_daily_series"), dict) else {}

    weak_sleeves: list[dict[str, Any]] = []
    active_sleeves = 0
    for row in sleeve_latest:
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile") or "").strip().lower()
        data_status = str(row.get("data_status") or "").strip().lower()
        if data_status not in {"current", "stale", "partial"}:
            continue
        active_sleeves += 1
        pnl = _safe_float(row.get("ending_net_pnl_total"), 0.0)
        win_rate_raw = row.get("win_rate")
        win_rate = _safe_float(win_rate_raw, -1.0) if win_rate_raw is not None else None
        if pnl < 0.0 or (win_rate is not None and win_rate < 0.45):
            weak_sleeves.append(
                {
                    "profile": profile,
                    "ending_net_pnl_total": round(pnl, 6),
                    "win_rate": round(win_rate, 6) if win_rate is not None else None,
                    "top_loss_causes": row.get("top_loss_causes") if isinstance(row.get("top_loss_causes"), list) else [],
                }
            )

    latest_change = _safe_float((history_daily[-1] if history_daily else {}).get("change_vs_previous_day"), 0.0)
    latest_net_pnl = _safe_float((history_daily[-1] if history_daily else {}).get("ending_net_pnl_total"), 0.0)
    previous_net_pnl = _safe_float((history_daily[-2] if len(history_daily) >= 2 else {}).get("ending_net_pnl_total"), 0.0)
    pnl_slope = round(latest_net_pnl - previous_net_pnl, 6) if len(history_daily) >= 2 else None

    sleeve_regimes = []
    for profile, rows in sleeve_daily.items():
        if not isinstance(rows, list) or not rows:
            continue
        latest = rows[-1] if isinstance(rows[-1], dict) else {}
        sleeve_regimes.append(
            {
                "profile": str(profile),
                "latest_net_pnl_total": round(_safe_float(latest.get("ending_net_pnl_total"), 0.0), 6),
                "change_vs_previous_day": round(_safe_float(latest.get("change_vs_previous_day"), 0.0), 6),
            }
        )
    sleeve_regimes.sort(key=lambda row: (float(row.get("latest_net_pnl_total", 0.0) or 0.0), str(row.get("profile") or "")))

    trailing_periods = []
    for row in period_change[:6]:
        if not isinstance(row, dict):
            continue
        trailing_periods.append(
            {
                "label": str(row.get("label") or ""),
                "window_days": _safe_int(row.get("window_days"), 0),
                "change": round(_safe_float(row.get("change"), 0.0), 6),
                "available_days": _safe_int(row.get("available_days"), 0),
            }
        )

    ok = bool(paper.get("ok", False) and len(history_daily) >= 1)
    overall_status = "ready"
    if not ok:
        overall_status = "blocked"
    elif weak_sleeves or (pnl_slope is not None and pnl_slope < 0.0) or latest_change < 0.0 or bool(promotion.get("promote_ok") is False):
        overall_status = "needs_work"

    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": overall_status,
        "history_days_available": len(history_daily),
        "active_sleeves": active_sleeves,
        "weak_sleeve_count": len(weak_sleeves),
        "weak_sleeves": weak_sleeves[:10],
        "latest_change_vs_previous_day": round(latest_change, 6),
        "latest_net_pnl_total": round(latest_net_pnl, 6),
        "pnl_slope": pnl_slope,
        "trailing_periods": trailing_periods,
        "regime_segments": sleeve_regimes[:12],
        "promotion_ready": bool(promotion.get("promote_ok", False)),
        "recommendations": [
            "Refresh or demote sleeves that stay loss-making across consecutive periods instead of letting them quietly dilute training.",
            "Segment decay review by sleeve and regime before promoting threshold or label changes across the full registry.",
        ],
        "source_files": {
            "paper_performance": str(health_root / "paper_performance_latest.json"),
            "promotion_readiness": str(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json"),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a replay/paper decay monitor artifact.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "decay_monitor "
            f"status={payload['overall_status']} "
            f"weak_sleeves={int(payload.get('weak_sleeve_count', 0) or 0)} "
            f"history_days={int(payload.get('history_days_available', 0) or 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
