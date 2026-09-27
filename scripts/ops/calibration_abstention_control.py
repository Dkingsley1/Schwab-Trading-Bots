#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "calibration_abstention_control_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "governance" / "health" / "calibration_abstention_overrides_latest.json"
BOT_NEEDS_CALIBRATION_ACTIONS = {
    "apply_abstention_calibration",
    "use_side_specific_thresholds",
    "repair_long_precision",
    "repair_short_precision",
    "repair_precision_balance",
    "repair_options_structure_precision",
    "repair_guard_false_positive_control",
}


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


def _source_receipt(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        digest = ""
    return {
        "path": str(path),
        "present": bool(payload),
        "timestamp_utc": str(payload.get("timestamp_utc") or ""),
        "sha256": digest,
    }


def _declared_candidate_id(payload: dict[str, Any]) -> str:
    binding = payload.get("candidate_binding") if isinstance(payload.get("candidate_binding"), dict) else {}
    return str(
        payload.get("candidate_id")
        or binding.get("candidate_id")
        or binding.get("valid_candidate_id")
        or ""
    ).strip()


def _candidate_binding(
    project_root: Path,
    *,
    source_payloads: dict[str, tuple[dict[str, Any], Path]],
) -> dict[str, Any]:
    state_path = project_root / "governance" / "runtime" / "production_candidate_state.json"
    state = _load_json(state_path)
    candidate_id = str(state.get("candidate_id") or "").strip()
    declared = {
        name: _declared_candidate_id(payload)
        for name, (payload, _path) in source_payloads.items()
        if _declared_candidate_id(payload)
    }
    mismatches = sorted(
        name for name, declared_id in declared.items() if candidate_id and declared_id != candidate_id
    )
    source_receipts = {
        name: _source_receipt(path, payload)
        for name, (payload, path) in source_payloads.items()
    }
    source_receipts["production_candidate"] = _source_receipt(state_path, state)
    evidence_scope = (
        "candidate_bound"
        if candidate_id and declared and not mismatches and all(value == candidate_id for value in declared.values())
        else "historical_diagnostic"
    )
    return {
        "candidate_id": candidate_id,
        "generation": int(state.get("generation", 0) or 0),
        "accepted_at_utc": str(state.get("accepted_at_utc") or ""),
        "candidate_state_receipt_sha256": str(state.get("overall_sha256") or ""),
        "identity_consistent": bool(candidate_id and not mismatches),
        "declared_source_candidate_ids": declared,
        "mismatch_sources": mismatches,
        "evidence_scope": evidence_scope,
        "safe_application_scope": "paper_only_tightening",
        "valid_until_candidate_changes": True,
        "source_receipts": source_receipts,
        "policy": "historical diagnostics may tighten the current paper candidate but may never loosen, promote, or cross a candidate boundary",
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    label_path = health / "training_label_audit_latest.json"
    quality_path = health / "training_quality_control_latest.json"
    bot_needs_path = health / "bot_needs_intelligence_latest.json"
    override_path = health / "calibration_abstention_overrides_latest.json"
    label_audit = _load_json(label_path)
    training_quality = _load_json(quality_path)
    bot_needs = _load_json(bot_needs_path)
    existing_overrides = _load_json(override_path)
    candidate_binding = _candidate_binding(
        project_root,
        source_payloads={
            "training_label_audit": (label_audit, label_path),
            "training_quality_control": (training_quality, quality_path),
            "bot_needs_intelligence": (bot_needs, bot_needs_path),
        },
    )
    overacting = label_audit.get("active_overacting") if isinstance(label_audit.get("active_overacting"), list) else []
    underacting = label_audit.get("active_underacting") if isinstance(label_audit.get("active_underacting"), list) else []
    recommendations: list[dict[str, Any]] = []

    for row in overacting:
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

    existing_recommendation_ids = {str(row.get("bot_id") or "").strip().lower() for row in recommendations if isinstance(row, dict)}
    needs_rows = bot_needs.get("bot_needs") if isinstance(bot_needs.get("bot_needs"), list) else []
    for row in needs_rows:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip()
        if not bot_id or bot_id.lower() in existing_recommendation_ids:
            continue
        primary_need = str(row.get("primary_need") or "").strip()
        if primary_need not in BOT_NEEDS_CALIBRATION_ACTIONS:
            continue
        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        accuracy_lift = _safe_float(evidence.get("accuracy_lift_over_majority"), 0.0)
        acted_coverage = max(_safe_float(evidence.get("acted_coverage"), 0.0), 0.0)
        acted_accuracy = max(_safe_float(evidence.get("acted_accuracy"), 0.0), 0.0)
        precision_gap_bonus = 0.04 if primary_need in BOT_NEEDS_CALIBRATION_ACTIONS - {"apply_abstention_calibration"} else 0.0
        guard_bonus = 0.02 if primary_need == "repair_guard_false_positive_control" else 0.0
        confidence_uplift = round(min(0.25, 0.05 + max(-accuracy_lift, 0.0) * 1.5 + precision_gap_bonus + guard_bonus), 6)
        target_acceptance_rate = round(max(0.05, min(0.18, acted_coverage * 0.72 if acted_coverage > 0 else 0.12)), 6)
        recommendations.append(
            {
                "bot_id": bot_id,
                "family": _infer_family(bot_id),
                "mode": "tighten",
                "acted_accuracy": round(acted_accuracy, 6),
                "accuracy_lift_over_majority": round(accuracy_lift, 6),
                "current_acceptance_rate": round(acted_coverage, 6),
                "target_acceptance_rate": target_acceptance_rate,
                "confidence_threshold_uplift": confidence_uplift,
                "recommended_abstention_budget": round(max(0.0, 1.0 - target_acceptance_rate), 6),
                "source": "bot_needs_intelligence",
                "source_need": primary_need,
            }
        )
        existing_recommendation_ids.add(bot_id.lower())

    for row in underacting:
        if not isinstance(row, dict):
            continue
        acceptance_rate = max(_safe_float(row.get("acted_coverage", row.get("acceptance_rate")), 0.0), 0.0)
        evidence_sufficient = bool(row.get("abstention_evidence_sufficient", False))
        recommendations.append(
            {
                "bot_id": str(row.get("bot_id") or ""),
                "family": _infer_family(str(row.get("bot_id") or "")),
                "mode": "counterfactual_replay" if evidence_sufficient else "collect_evidence",
                "current_acceptance_rate": round(acceptance_rate, 6),
                "target_acceptance_rate": round(acceptance_rate, 6),
                "confidence_threshold_uplift": 0.0,
                "recommended_abstention_budget": round(max(0.0, 1.0 - acceptance_rate), 6),
                "abstention_evidence_sufficient": evidence_sufficient,
                "direct_loosen_allowed": False,
                "next_action": (
                    "run_counterfactual_threshold_replay_and_require_positive_post_cost_expectancy"
                    if evidence_sufficient
                    else "collect_more_acted_outcomes_before_threshold_replay"
                ),
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
    existing_binding = (
        existing_overrides.get("candidate_binding")
        if isinstance(existing_overrides.get("candidate_binding"), dict)
        else {}
    )
    existing_candidate_bound = bool(
        int(existing_overrides.get("schema_version", 0) or 0) >= 2
        and str(existing_binding.get("valid_candidate_id") or "")
        == str(candidate_binding.get("candidate_id") or "")
        and bool(existing_binding.get("valid_until_candidate_changes", False))
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 2,
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
            "candidate_bound": existing_candidate_bound,
        },
        "candidate_binding": candidate_binding,
        "application_contract": {
            "paper_only": True,
            "tightening_only": True,
            "direct_loosen_allowed": False,
            "candidate_change_invalidates_overrides": True,
            "live_execution_allowed": False,
            "grant_promotion": False,
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


def _candidate_id(control_payload: dict[str, Any]) -> str:
    binding = control_payload.get("candidate_binding") if isinstance(control_payload.get("candidate_binding"), dict) else {}
    return str(binding.get("candidate_id") or "")


def _is_override_leaf(value: dict[str, Any]) -> bool:
    return any(
        key in value
        for key in (
            "mode",
            "acted_prob_threshold_uplift",
            "target_acceptance_rate",
            "recommended_abstention_budget",
            "valid_candidate_id",
        )
    )


def _is_safe_tightening(value: dict[str, Any]) -> bool:
    return bool(
        str(value.get("mode") or "tighten").strip().lower() == "tighten"
        and _safe_float(value.get("acted_prob_threshold_uplift"), 0.0) >= 0.0
    )


def _bind_tightening_rows(values: dict[str, Any], candidate_id: str) -> None:
    for row in values.values():
        if not isinstance(row, dict):
            continue
        if not _is_override_leaf(row):
            _bind_tightening_rows(row, candidate_id)
            continue
        row["mode"] = "tighten"
        row["acted_prob_threshold_uplift"] = round(
            max(_safe_float(row.get("acted_prob_threshold_uplift"), 0.0), 0.0),
            6,
        )
        row["valid_candidate_id"] = candidate_id


def build_override_payload(control_payload: dict[str, Any], existing_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    recommendations = control_payload.get("recommendations") if isinstance(control_payload.get("recommendations"), list) else []
    family_recommendations = control_payload.get("family_recommendations") if isinstance(control_payload.get("family_recommendations"), list) else []
    existing = existing_overrides if isinstance(existing_overrides, dict) else {}
    retired_overrides: list[dict[str, str]] = []
    bot_overrides: dict[str, dict[str, Any]] = {
        str(bot_id): dict(value)
        for bot_id, value in (existing.get("bot_overrides") or {}).items()
        if isinstance(value, dict)
        and str(value.get("mode") or "tighten").strip().lower() == "tighten"
        and _safe_float(value.get("acted_prob_threshold_uplift"), 0.0) >= 0.0
    }
    family_overrides: dict[str, dict[str, Any]] = {
        str(family): dict(value)
        for family, value in (existing.get("family_overrides") or {}).items()
        if isinstance(value, dict)
        and str(value.get("mode") or "tighten").strip().lower() == "tighten"
        and _safe_float(value.get("acted_prob_threshold_uplift"), 0.0) >= 0.0
    }
    regime_overrides: dict[str, Any] = {}
    raw_regime_overrides = existing.get("regime_overrides") or {}
    if isinstance(raw_regime_overrides, dict):
        for family_or_key, value in raw_regime_overrides.items():
            if not isinstance(value, dict):
                continue
            if _is_override_leaf(value):
                if _is_safe_tightening(value):
                    regime_overrides[str(family_or_key)] = dict(value)
                else:
                    retired_overrides.append(
                        {
                            "scope": "regime",
                            "key": str(family_or_key),
                            "reason": "unsafe_direct_loosen_retired",
                        }
                    )
                continue
            nested: dict[str, Any] = {}
            for regime, row in value.items():
                if not isinstance(row, dict):
                    continue
                if _is_safe_tightening(row):
                    nested[str(regime)] = dict(row)
                else:
                    retired_overrides.append(
                        {
                            "scope": "regime",
                            "key": f"{family_or_key}:{regime}",
                            "reason": "unsafe_direct_loosen_retired",
                        }
                    )
            if nested:
                regime_overrides[str(family_or_key)] = nested
    for scope, values in (("bot", existing.get("bot_overrides") or {}), ("family", existing.get("family_overrides") or {})):
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if not isinstance(value, dict):
                continue
            if (
                str(value.get("mode") or "").strip().lower() == "loosen"
                or _safe_float(value.get("acted_prob_threshold_uplift"), 0.0) < 0.0
            ):
                retired_overrides.append({"scope": scope, "key": str(key), "reason": "unsafe_direct_loosen_retired"})

    for row in recommendations:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if not bot_id:
            continue
        mode = str(row.get("mode") or "tighten").strip().lower()
        if mode != "tighten":
            continue
        uplift = _safe_float(row.get("confidence_threshold_uplift"), 0.0)
        bot_overrides[bot_id] = {
            "mode": mode,
            "family": str(row.get("family") or _infer_family(bot_id)),
            "acted_prob_threshold_uplift": round(uplift if mode == "tighten" else -uplift, 6),
            "target_acceptance_rate": round(_safe_float(row.get("target_acceptance_rate"), 0.0), 6),
            "recommended_abstention_budget": round(_safe_float(row.get("recommended_abstention_budget"), 0.0), 6),
            "valid_candidate_id": _candidate_id(control_payload),
        }

    for row in family_recommendations:
        if not isinstance(row, dict):
            continue
        family = str(row.get("family") or "").strip().lower()
        if not family:
            continue
        mode = str(row.get("mode") or "tighten").strip().lower()
        if mode != "tighten":
            continue
        uplift = _safe_float(row.get("confidence_threshold_uplift"), 0.0)
        family_overrides[family] = {
            "mode": mode,
            "acted_prob_threshold_uplift": round(uplift if mode == "tighten" else -uplift, 6),
            "recommended_abstention_budget": round(_safe_float(row.get("recommended_abstention_budget"), 0.0), 6),
            "source_profile": str(row.get("source_profile") or ""),
            "valid_candidate_id": _candidate_id(control_payload),
        }

    candidate_binding = (
        dict(control_payload.get("candidate_binding"))
        if isinstance(control_payload.get("candidate_binding"), dict)
        else {}
    )
    candidate_binding["valid_candidate_id"] = str(candidate_binding.get("candidate_id") or "")
    candidate_binding["valid_candidate_generation"] = int(candidate_binding.get("generation", 0) or 0)
    candidate_binding["valid_until_candidate_changes"] = True
    valid_candidate_id = str(candidate_binding.get("valid_candidate_id") or "")
    for values in (bot_overrides, family_overrides, regime_overrides):
        _bind_tightening_rows(values, valid_candidate_id)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 2,
        "bot_overrides": bot_overrides,
        "family_overrides": family_overrides,
        "regime_overrides": regime_overrides,
        "candidate_binding": candidate_binding,
        "retired_overrides": retired_overrides,
        "direct_loosen_policy": "never_apply_without_counterfactual_replay_and_positive_post_cost_expectancy",
        "application_contract": {
            "paper_only": True,
            "tightening_only": True,
            "candidate_change_invalidates_overrides": True,
            "live_execution_allowed": False,
            "grant_promotion": False,
        },
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
        existing_override_payload = _load_json(Path(args.override_out).expanduser())
        override_payload = build_override_payload(payload, existing_override_payload)
        _write_json(Path(args.override_out).expanduser(), override_payload)
        payload["applied_override_file"] = str(Path(args.override_out).expanduser())
        payload["applied_override_summary"] = {
            "bot_override_count": len(override_payload.get("bot_overrides") or {}),
            "family_override_count": len(override_payload.get("family_overrides") or {}),
            "regime_override_count": len(override_payload.get("regime_overrides") or {}),
            "valid_candidate_id": str(
                (override_payload.get("candidate_binding") or {}).get("valid_candidate_id")
                if isinstance(override_payload.get("candidate_binding"), dict)
                else ""
            ),
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
