#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, parse_iso_utc, utc_now, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, parse_iso_utc, utc_now, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "release_freeze_guard_latest.json"
DEFAULT_WINDOW_PATH = PROJECT_ROOT / "governance" / "runtime" / "release_freeze_window.json"


def _load_window(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload:
        return payload
    return {"active": False}


def _save_window(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def build_payload(project_root: Path = PROJECT_ROOT, *, window_path: Path = DEFAULT_WINDOW_PATH) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    promotion_readiness = load_json(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json")
    new_bot_graduation = load_json(project_root / "governance" / "walk_forward" / "new_bot_graduation_latest.json")
    supportability_control = load_json(health_root / "supportability_control_latest.json")
    window = _load_window(window_path)
    ends_at = parse_iso_utc(window.get("ends_at_utc"))
    active = bool(window.get("active", False)) and (ends_at is None or ends_at > utc_now())

    overall_status = "ready" if active else "degraded"
    if active and str(supportability_control.get("overall_status") or "") == "blocked":
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "./scripts/ops/opsctl.sh release-freeze --activate-days 21 --reason multi_week_runtime_window --json" if not active else "",
            "hold promotions, schema churn, and experimental bot activations while the long-run window is active" if active else "",
            "only thaw the window after promotion readiness, supportability, and freshness lanes are back inside budget" if active else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "window": {
            "active": active,
            "started_at_utc": str(window.get("started_at_utc") or ""),
            "ends_at_utc": str(window.get("ends_at_utc") or ""),
            "reason": str(window.get("reason") or ""),
        },
        "frozen_surfaces": {
            "allow_promotions": not active,
            "allow_schema_changes": not active,
            "allow_experimental_bots": not active,
        },
        "gating_context": {
            "promotion_ready": bool(promotion_readiness.get("promote_ok", False)),
            "new_bot_graduation_ok": bool(new_bot_graduation.get("ok", False)),
            "supportability_status": str(supportability_control.get("overall_status") or ""),
        },
        "infra_bots": ["release_freeze_guard", "promotion_readiness_summary", "supportability_control"],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage release freeze windows for long runtime runs.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--window-path", default=str(DEFAULT_WINDOW_PATH))
    parser.add_argument("--activate-days", type=int, default=0)
    parser.add_argument("--clear-window", action="store_true")
    parser.add_argument("--reason", default="")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    window_path = Path(args.window_path).expanduser()
    if args.clear_window:
        _save_window(window_path, {"active": False, "cleared_at_utc": iso_now(), "reason": str(args.reason or "")})
    elif int(args.activate_days) > 0:
        started = utc_now()
        payload = {
            "active": True,
            "started_at_utc": started.isoformat(),
            "ends_at_utc": (started + timedelta(days=int(args.activate_days))).isoformat(),
            "reason": str(args.reason or "runtime_freeze_window"),
        }
        _save_window(window_path, payload)

    payload = build_payload(Path(args.project_root).resolve(), window_path=window_path)
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "release_freeze_guard "
            f"overall_status={payload.get('overall_status', '')} "
            f"active={int(bool(((payload.get('window') or {}).get('active', False))))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
