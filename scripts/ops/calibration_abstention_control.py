#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "calibration_abstention_control_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "governance" / "health" / "calibration_abstention_overrides_latest.json"


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


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _infer_family(bot_id: str) -> str:
    lowered = str(bot_id or "").strip().lower()
    for token, family in (
        ("intraday", "intraday"),
        ("swing", "swing"),
        ("crypto", "crypto"),
        ("bond", "bond"),
        ("fx", "fx"),
        ("dividend", "dividend"),
        ("futures", "futures"),
        ("risk_budget", "infrastructure"),
        ("allocator", "infrastructure"),
    ):
        if token in lowered:
            return family
    return "general"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    label_audit = _load_json(project_root / "governance" / "health" / "training_label_audit_latest.json")
    training_quality = _load_json(project_root / "governance" / "health" / "training_quality_control_latest.json")
    existing_overrides = _load_json(DEFAULT_OVERRIDE_PATH if project_root == PROJECT_ROOT else project_root / "governance" / "health" / "calibration_abstention_overrides_latest.json")
    overacting = label_audit.get("active_overacting") if isinstance(label_audit.get("active_overacting"), list) else []
    underacting = label_audit.get("active_underacting") if isinstance(label_audit.get("active_underacting"), list) else []
    recommendations: list[dict[str, Any]] = []

    for row in overacting[:10]:
        if not isinstance(row, dict):
            continue
        accuracy_lift = _safe_float(row.get("accuracy_lift_over_majority"), 0.0)
        acceptance_rate = max(_safe_float(row.get("acceptance_rate"), 0.0), 0.0)
        acted_accuracy = max(_safe_float(row.get("acted_accuracy"), 0.0), 0.0)
        confidence_uplift = round(min(0.25, 0.04 + max(-accuracy_lift, 0.0) * 1.8), 6)
        target_acceptance_rate = round(max(0.05, min(0.18, acceptance_rate * (0.78 if accuracy_lift < 0 else 0.90))), 6)
        recommendations.append(
            {
                "bot_id": str(row.get("bot_id") or ""),
                "family": _infer_family(str(row.get("bot_id") or "")),
                "mode": "tighten",
                "acted_accuracy": round(acted_accuracy, 6),
                "accuracy_lift_over_majority": round(accuracy_lift, 6),
                "current_acceptance_rate": round(acceptance_rate, 6),
                "target_acceptance_rate": target_acceptance_rate,
                "confidence_threshold_uplift": confidence_uplift,
                "recommended_abstention_budget": round(max(0.0, 1.0 - target_acceptance_rate), 6),
            }
        )

    for row in underacting[:10]:
        if not isinstance(row, dict):
            continue
        acceptance_rate = max(_safe_float(row.get("acceptance_rate"), 0.0), 0.0)
        recommendations.append(
            {
                "bot_id": str(row.get("bot_id") or ""),
                "family": _infer_family(str(row.get("bot_id") or "")),
                "mode": "loosen",
                "current_acceptance_rate": round(acceptance_rate, 6),
                "target_acceptance_rate": round(min(0.22, max(acceptance_rate * 1.15, 0.08)), 6),
                "confidence_threshold_uplift": 0.0,
                "recommended_abstention_budget": round(max(0.0, 1.0 - min(0.22, max(acceptance_rate * 1.15, 0.08))), 6),
            }
        )

    targeted_actions = training_quality.get("targeted_actions") if isinstance(training_quality.get("targeted_actions"), dict) else {}
    weak_sleeves = targeted_actions.get("weak_sleeves") if isinstance(targeted_actions.get("weak_sleeves"), list) else []
    family_overrides: list[dict[str, Any]] = []
    for row in weak_sleeves:
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile") or "").strip().lower()
        if not profile:
            continue
        family = "dividend" if "dividend" in profile else ("bond" if "bond" in profile else "")
        if not family:
            continue
        ending_net = _safe_float(row.get("ending_net_pnl_total"), 0.0)
        win_rate = _safe_float(row.get("win_rate"), 0.0)
        uplift = min(0.08, 0.02 + max(abs(min(ending_net, 0.0)) / 500.0, 0.0) + max(0.45 - win_rate, 0.0) * 0.08)
        family_overrides.append(
            {
                "family": family,
                "mode": "tighten",
                "source_profile": profile,
                "ending_net_pnl_total": round(ending_net, 6),
                "win_rate": round(win_rate, 6),
                "confidence_threshold_uplift": round(uplift, 6),
                "recommended_abstention_budget": round(min(0.96, 0.80 + uplift), 6),
            }
        )

    bot_override_count = len((existing_overrides.get("bot_overrides") or {})) if isinstance(existing_overrides, dict) else 0
    family_override_count = len((existing_overrides.get("family_overrides") or {})) if isinstance(existing_overrides, dict) else 0
    total_recommendations = len(recommendations) + len(family_overrides)
    calibration_confidence_score = 100.0 - min(total_recommendations * 6.0, 42.0)
    if total_recommendations > 0 and (bot_override_count > 0 or family_override_count > 0):
        calibration_confidence_score += 6.0
    calibration_confidence_score = min(max(round(calibration_confidence_score, 2), 0.0), 100.0)
    overall_status = "ready" if not recommendations and not family_overrides else "needs_tuning"
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "overacting_count": len(overacting),
        "underacting_count": len(underacting),
        "calibration_confidence_score": calibration_confidence_score,
        "recommendations": recommendations,
        "family_recommendations": family_overrides,
        "override_state": {
            "bot_override_count": bot_override_count,
            "family_override_count": family_override_count,
        },
        "a_plus_contract": {
            "calibration_confidence_target": 90.0,
            "calibration_confidence_score": calibration_confidence_score,
            "recommendation_count": total_recommendations,
            "override_ready": bool(bot_override_count > 0 or family_override_count > 0 or total_recommendations <= 0),
        },
        "top_actions": [
            "treat abstention thresholds as learned controls rather than fixed constants",
            "tighten confidence gates for overacting bots before widening feature scope",
            "calibrate acceptance rate by lane and regime rather than globally",
        ],
    }
    return payload


def build_override_payload(control_payload: dict[str, Any]) -> dict[str, Any]:
    recommendations = control_payload.get("recommendations") if isinstance(control_payload.get("recommendations"), list) else []
    family_recommendations = control_payload.get("family_recommendations") if isinstance(control_payload.get("family_recommendations"), list) else []
    bot_overrides: dict[str, dict[str, Any]] = {}
    family_overrides: dict[str, dict[str, Any]] = {}

    for row in recommendations:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if not bot_id:
            continue
        mode = str(row.get("mode") or "tighten").strip().lower()
        uplift = _safe_float(row.get("confidence_threshold_uplift"), 0.0)
        bot_overrides[bot_id] = {
            "mode": mode,
            "family": str(row.get("family") or _infer_family(bot_id)),
            "acted_prob_threshold_uplift": round(uplift if mode == "tighten" else -uplift, 6),
            "target_acceptance_rate": round(_safe_float(row.get("target_acceptance_rate"), 0.0), 6),
            "recommended_abstention_budget": round(_safe_float(row.get("recommended_abstention_budget"), 0.0), 6),
        }

    for row in family_recommendations:
        if not isinstance(row, dict):
            continue
        family = str(row.get("family") or "").strip().lower()
        if not family:
            continue
        mode = str(row.get("mode") or "tighten").strip().lower()
        uplift = _safe_float(row.get("confidence_threshold_uplift"), 0.0)
        family_overrides[family] = {
            "mode": mode,
            "acted_prob_threshold_uplift": round(uplift if mode == "tighten" else -uplift, 6),
            "recommended_abstention_budget": round(_safe_float(row.get("recommended_abstention_budget"), 0.0), 6),
            "source_profile": str(row.get("source_profile") or ""),
        }

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "bot_overrides": bot_overrides,
        "family_overrides": family_overrides,
        "recommended_actions": _ordered_unique(
            [
                "apply per-bot threshold uplifts first for active overacting bots",
                "use family-level tightening for sleeves with persistent negative paper behavior",
                "revisit overrides after the next targeted retrain refreshes acted accuracy and coverage",
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish calibration and abstention recommendations from label-audit behavior.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override-out", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    if args.apply:
        override_payload = build_override_payload(payload)
        _write_json(Path(args.override_out).expanduser(), override_payload)
        payload["applied_override_file"] = str(Path(args.override_out).expanduser())
        payload["applied_override_summary"] = {
            "bot_override_count": len(override_payload.get("bot_overrides") or {}),
            "family_override_count": len(override_payload.get("family_overrides") or {}),
        }
    out_path = Path(args.out_file).expanduser()
    _write_json(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "calibration_abstention_control "
            f"overall_status={payload.get('overall_status', '')} "
            f"recommendations={len(payload.get('recommendations') or [])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
