#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "bot_needs_intelligence_latest.json"
OVERFIT_BLOCKING_STATUSES = {"leak_like", "severe_overfit", "overfit_watch", "high_accuracy_guarded"}
PRECISION_REPAIR_NEEDS = {
    "repair_long_precision",
    "repair_short_precision",
    "repair_precision_balance",
    "repair_options_structure_precision",
    "repair_guard_false_positive_control",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _parse_dt(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _age_hours(raw: Any, now: datetime) -> float | None:
    parsed = _parse_dt(raw)
    if parsed is None:
        return None
    return round(max((now - parsed).total_seconds(), 0.0) / 3600.0, 3)


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _index_label_rows(label_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for bucket in (
        "active_zero_sample",
        "active_label_contract_upgrades",
        "active_unbalanced_labels",
        "active_overacting",
        "active_underacting",
    ):
        for row in _as_list(label_audit.get(bucket)):
            if not isinstance(row, dict):
                continue
            bot_id = str(row.get("bot_id") or "").strip()
            if not bot_id:
                continue
            existing = indexed.setdefault(bot_id, {})
            existing.update(row)
            buckets = _as_list(existing.get("_audit_buckets"))
            buckets.append(bucket)
            existing["_audit_buckets"] = _unique([str(item) for item in buckets])
    return indexed


def _index_quality_queue(bot_quality: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for queue_name in ("quality_upgrade_queue", "infrastructure_helper_queue"):
        for row in _as_list(bot_quality.get(queue_name)):
            if not isinstance(row, dict):
                continue
            bot_id = str(row.get("bot_id") or "").strip()
            if not bot_id:
                continue
            item = indexed.setdefault(bot_id, {})
            item.update(row)
            queues = _as_list(item.get("_queues"))
            queues.append(queue_name)
            item["_queues"] = _unique([str(value) for value in queues])
    return indexed


def _index_overfit_rows(overfit_awareness: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in _as_list(overfit_awareness.get("bot_risk")):
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip()
        if not bot_id:
            continue
        indexed[bot_id] = row
        indexed[bot_id.lower()] = row
    return indexed


def _membership_sets(training_quality: dict[str, Any], bot_quality: dict[str, Any]) -> dict[str, set[str]]:
    targeted = _as_dict(training_quality.get("targeted_actions"))
    blockers = _as_dict(bot_quality.get("quality_blockers"))
    keys = [
        "refresh_diagnostics_bot_ids",
        "unsupported_stale_bot_ids",
        "provisional_registry_backed_bot_ids",
        "repair_runtime_input_bot_ids",
        "quality_probation_bot_ids",
        "targeted_retrain_bot_ids",
    ]
    out: dict[str, set[str]] = {key: set() for key in keys}
    for source in (targeted, blockers):
        for key in keys:
            out[key].update(str(item) for item in _as_list(source.get(key)) if str(item or "").strip())
    return out


def _minimum_observations(bot: dict[str, Any]) -> int:
    standard = _as_dict(bot.get("paper_promotion_standard"))
    if _safe_int(standard.get("minimum_observations"), 0) > 0:
        return _safe_int(standard.get("minimum_observations"), 0)
    threshold = _as_dict(bot.get("data_collection_threshold"))
    if _safe_int(threshold.get("minimum_training_observations"), 0) > 0:
        return _safe_int(threshold.get("minimum_training_observations"), 0)
    return _safe_int(bot.get("minimum_training_observations"), 0)


def _bot_search_text(bot: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "bot_id",
        "bot_role",
        "slot_kind",
        "lifecycle_state",
        "reason",
        "promotion_reason",
        "execution_policy_label",
        "core_module_path",
    ):
        parts.append(str(bot.get(key) or ""))
    return " ".join(parts).lower()


def _precision_contract(bot: dict[str, Any]) -> dict[str, Any]:
    role = str(bot.get("bot_role") or "").strip()
    text = _bot_search_text(bot)
    infra_terms = (
        "infrastructure",
        "guard",
        "sentinel",
        "allocator",
        "throttle",
        "governor",
        "backlog",
        "writer",
        "runtime",
        "memory",
        "storage",
        "verification",
        "integrity",
        "drift",
        "execution_feasibility",
        "correlation_penalty",
        "controller",
    )
    short_terms = (
        "defensive",
        "hedge",
        "short",
        "risk",
        "drawdown",
        "crisis",
        "tail",
        "downside",
        "bear",
        "rotation",
    )
    long_terms = (
        "long",
        "income",
        "dividend",
        "buy",
        "growth",
        "trend",
        "momentum",
        "breakout",
        "accumulation",
        "bull",
    )
    option_terms = ("option", "options", "volatility", "skew", "iv_", "greeks", "straddle", "strangle", "spread")

    if role == "infrastructure_sub_bot" or any(term in text for term in infra_terms):
        return {
            "type": "guard_control",
            "required_sides": [],
            "required_metrics": ["acted_accuracy", "acted_coverage", "false_positive_guard"],
            "min_acted_accuracy": 0.54,
            "max_acted_coverage": 0.72,
            "why": "Infrastructure/risk-control bots are judged on correct guard firing and false-positive control, not directional long precision.",
        }
    if any(term in text for term in short_terms):
        return {
            "type": "defensive_or_short_bias",
            "required_sides": ["short"],
            "required_metrics": ["short_precision", "acted_accuracy", "false_positive_guard"],
            "min_short_precision": 0.52,
            "min_acted_accuracy": 0.52,
            "max_acted_coverage": 0.70,
            "why": "Defensive/risk bots need reliable downside or risk-off precision before long-side symmetry matters.",
        }
    if role == "options_sub_bot" or any(term in text for term in option_terms):
        return {
            "type": "options_structure",
            "required_sides": ["long", "short"],
            "required_metrics": ["long_precision", "short_precision", "acted_accuracy", "options_structure_precision"],
            "min_long_precision": 0.50,
            "min_short_precision": 0.50,
            "min_acted_accuracy": 0.51,
            "max_acted_coverage": 0.68,
            "why": "Options bots need both structure-side precision and abstention discipline because payoff shape matters as much as direction.",
        }
    if any(term in text for term in long_terms):
        return {
            "type": "long_or_growth_bias",
            "required_sides": ["long"],
            "required_metrics": ["long_precision", "acted_accuracy"],
            "min_long_precision": 0.52,
            "min_acted_accuracy": 0.52,
            "max_acted_coverage": 0.72,
            "why": "Long/growth bots must prove long-side precision first; short-side weakness is secondary unless it overacts.",
        }
    return {
        "type": "balanced_directional",
        "required_sides": ["long", "short"],
        "required_metrics": ["long_precision", "short_precision", "precision_balance_score", "acted_accuracy"],
        "min_long_precision": 0.52,
        "min_short_precision": 0.52,
        "min_precision_balance_score": 0.72,
        "min_acted_accuracy": 0.52,
        "max_acted_coverage": 0.70,
        "why": "General signal bots should not collapse into one-sided behavior unless they declare a directional specialty.",
    }


def _precision_repair_needs(
    *,
    contract: dict[str, Any],
    long_precision: float,
    short_precision: float,
    acted_accuracy: float,
    acted_coverage: float,
    sample_count: int,
    calibration_override: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract_type = str(contract.get("type") or "balanced_directional")
    required_sides = {str(item) for item in _as_list(contract.get("required_sides"))}
    long_floor = _safe_float(contract.get("min_long_precision"), 0.0)
    short_floor = _safe_float(contract.get("min_short_precision"), 0.0)
    acted_floor = _safe_float(contract.get("min_acted_accuracy"), 0.0)
    max_coverage = _safe_float(contract.get("max_acted_coverage"), 1.0)
    balance_score = 1.0 - min(abs(float(long_precision) - float(short_precision)), 1.0)
    needs: list[dict[str, Any]] = []
    gaps: dict[str, Any] = {
        "long_precision_gap": round(max(long_floor - long_precision, 0.0), 6) if "long" in required_sides else 0.0,
        "short_precision_gap": round(max(short_floor - short_precision, 0.0), 6) if "short" in required_sides else 0.0,
        "acted_accuracy_gap": round(max(acted_floor - acted_accuracy, 0.0), 6) if acted_floor > 0 else 0.0,
        "acted_coverage_excess": round(max(acted_coverage - max_coverage, 0.0), 6) if acted_coverage >= 0 else 0.0,
        "precision_balance_score": round(balance_score, 6),
        "calibration_override_applied": bool(calibration_override),
    }

    if not required_sides and contract_type == "guard_control":
        if acted_coverage > max_coverage or (acted_accuracy >= 0.0 and acted_accuracy < acted_floor):
            if calibration_override:
                needs.append(
                    _need_record(
                        "targeted_quality_retrain",
                        "Guard calibration is applied; run a small canary to verify false-positive control.",
                        84,
                    )
                )
            else:
                needs.append(
                    _need_record(
                        "repair_guard_false_positive_control",
                        "Guard/infrastructure bot needs false-positive control, acted-accuracy repair, or tighter firing thresholds.",
                        88,
                        command_key="apply_abstention_calibration",
                    )
                )
        return needs, gaps

    if required_sides and sample_count > 0 and long_precision <= 0.0 and short_precision <= 0.0:
        needs.append(
            _need_record(
                "collect_side_specific_outcomes",
                "Precision sides are missing; enrich labels with side-specific outcome joins before more blind training.",
                86,
                command_key="upgrade_label_contract",
            )
        )
        return needs, gaps

    if contract_type == "options_structure":
        side_gap = ("long" in required_sides and long_precision < long_floor) or ("short" in required_sides and short_precision < short_floor)
        if side_gap:
            key = "targeted_quality_retrain" if calibration_override else "repair_options_structure_precision"
            summary = (
                "Options structure calibration is applied; run a small canary to verify payoff-side precision."
                if calibration_override
                else "Options bot needs payoff-side precision repair before more promotion pressure."
            )
            needs.append(_need_record(key, summary, 84 if calibration_override else 88, command_key=None if calibration_override else "apply_abstention_calibration"))
        if acted_coverage > max_coverage and not calibration_override:
            needs.append(
                _need_record(
                    "apply_abstention_calibration",
                    "Options bot is acting too broadly for its precision contract; tighten abstention before widening.",
                    87,
                )
            )
        return needs, gaps

    if "long" in required_sides and long_precision < long_floor:
        if calibration_override:
            needs.append(_need_record("targeted_quality_retrain", "Long-side calibration is applied; run a small canary to validate long precision.", 84))
        else:
            needs.append(
                _need_record(
                    "repair_long_precision",
                    f"Long-side precision is {long_precision:.3f} vs required {long_floor:.3f}; add long-side calibration or rebalance long examples.",
                    88 if long_precision <= 0.0 else 84,
                    command_key="apply_abstention_calibration",
                )
            )
    if "short" in required_sides and short_precision < short_floor:
        if calibration_override:
            needs.append(_need_record("targeted_quality_retrain", "Short-side calibration is applied; run a small canary to validate short precision.", 84))
        else:
            needs.append(
                _need_record(
                    "repair_short_precision",
                    f"Short-side precision is {short_precision:.3f} vs required {short_floor:.3f}; add short-side calibration or rebalance downside examples.",
                    88 if short_precision <= 0.0 else 84,
                    command_key="apply_abstention_calibration",
                )
            )

    min_balance = _safe_float(contract.get("min_precision_balance_score"), 0.0)
    if min_balance > 0 and balance_score < min_balance and "long" in required_sides and "short" in required_sides:
        if calibration_override:
            needs.append(_need_record("targeted_quality_retrain", "Side-balance calibration is applied; run a small canary to validate both sides.", 83))
        else:
            needs.append(
                _need_record(
                    "repair_precision_balance",
                    f"Long/short precision balance is {balance_score:.3f} vs required {min_balance:.3f}; repair the weaker side before promotion pressure.",
                    87,
                    command_key="apply_abstention_calibration",
                )
            )
    if acted_coverage > max_coverage and not calibration_override:
        needs.append(
            _need_record(
                "apply_abstention_calibration",
                f"Acted coverage is {acted_coverage:.3f} vs precision-contract cap {max_coverage:.3f}; tighten thresholds.",
                86,
            )
        )
    return needs, gaps


def _bot_command(bot_id: str, need_key: str) -> list[str]:
    if need_key == "repair_runtime_inputs":
        return [
            "./scripts/ops/opsctl.sh",
            "training-requalification",
            "--include-bot-ids",
            bot_id,
            "--apply-repair",
            "--json",
        ]
    if need_key in {"top_off_walk_forward_runs", "targeted_quality_retrain", "targeted_retrain", "seed_walk_forward_coverage", "generate_walk_forward_runs"}:
        return [
            "./scripts/ops/opsctl.sh",
            "retrain-force-targeted",
            "--include-bot-ids",
            bot_id,
            "--retrain-profile",
            "coverage_canary",
            "--skip-master-update",
        ]
    if need_key == "apply_abstention_calibration":
        return ["./scripts/ops/opsctl.sh", "calibration-control", "--apply", "--json"]
    if need_key == "upgrade_label_contract":
        return ["./scripts/ops/opsctl.sh", "training-labeling-intelligence", "--apply", "--json"]
    if need_key == "relax_sample_filter":
        return ["./scripts/ops/opsctl.sh", "training-label-audit", "--json"]
    if need_key in {"refresh_training_diagnostics", "create_collect_only_diagnostics"}:
        return ["./scripts/ops/opsctl.sh", "bot-quality-autopilot", "--apply", "--json"]
    if need_key == "collect_more_data":
        return ["./scripts/ops/opsctl.sh", "training-label-audit", "--json"]
    if need_key in {"reduce_overfitting", "overfitting_awareness"}:
        return ["./scripts/ops/opsctl.sh", "overfitting-awareness", "--json"]
    return ["./scripts/ops/opsctl.sh", "bot-needs", "--include-bot-ids", bot_id, "--json"]


def _need_record(key: str, summary: str, priority: float, *, command_key: str | None = None) -> dict[str, Any]:
    return {
        "key": key,
        "summary": summary,
        "priority": round(float(priority), 3),
        "command_key": command_key or key,
    }


def _effectiveness_prescription(
    *,
    bot_id: str,
    primary_need: str,
    summary: str,
    evidence: dict[str, Any],
    next_command: list[str],
) -> dict[str, Any]:
    sample_count = _safe_int(evidence.get("sample_count"), 0)
    observations = _safe_int(evidence.get("observation_count"), 0)
    min_observations = _safe_int(evidence.get("minimum_observations"), 0)
    runs_remaining = _safe_int(evidence.get("walk_forward_runs_remaining"), 0)
    positive_rate = _safe_float(evidence.get("positive_rate"), 0.0)
    stage = str(primary_need or "monitor")
    risk = "low"
    expected = "Improves bot readiness by resolving the current highest-priority bottleneck."
    stop_when = "the bot primary_need changes to monitor or promotion review"
    can_train_now = False
    if stage == "collect_more_data":
        gap = max(min_observations - observations, 200 - sample_count, 0)
        expected = "Raises usable training sample depth so canary runs are not sample-starved."
        stop_when = f"sample_count is at least 200 and observation_count reaches {min_observations}" if min_observations else "sample_count is at least 200"
        risk = "none"
    elif stage in {"rebalance_labels", "relax_sample_filter", "upgrade_label_contract"}:
        expected = "Improves label quality and side balance before spending training cycles."
        stop_when = "positive_rate is between 0.25 and 0.75 and eligible sequences produce usable samples"
        risk = "low"
        gap = 0
    elif stage in {"repair_runtime_inputs", "refresh_training_diagnostics", "create_collect_only_diagnostics"}:
        expected = "Repairs or refreshes the runtime evidence needed to judge the bot correctly."
        stop_when = "fresh diagnostics exist and runtime_input repair flags clear"
        risk = "low"
        gap = 0
    elif stage in {"top_off_walk_forward_runs", "targeted_quality_retrain"}:
        expected = "Adds confirmation runs without promoting or widening live exposure."
        stop_when = f"walk_forward_runs_remaining reaches 0 and quality gate is not failing"
        risk = "medium" if stage == "targeted_quality_retrain" else "low"
        can_train_now = runs_remaining > 0 or stage == "targeted_quality_retrain"
        gap = runs_remaining
    elif stage in {"apply_abstention_calibration", "use_side_specific_thresholds"} | PRECISION_REPAIR_NEEDS:
        expected = "Reduces overacting, one-sided precision collapse, or noisy guard firing using the bot's role-specific precision contract."
        stop_when = "the precision_contract gaps clear and acted coverage is inside guardrails"
        risk = "low"
        gap = 0
    elif stage == "collect_side_specific_outcomes":
        expected = "Adds long/short outcome evidence so the next calibration or training pass can target the weak side."
        stop_when = "long_precision and short_precision are present for every required side in the precision contract"
        risk = "low"
        gap = 0
    elif stage == "reduce_overfitting":
        expected = "Reduces memorized or fragile behavior before the bot can teach, promote, or widen exposure."
        stop_when = "overfit status is generalization_clean and train-forward gap is at or below the configured threshold"
        risk = "medium"
        gap = 0
    elif stage == "monitor_passing_candidate":
        expected = "Keeps a passing candidate under observation instead of wasting retrain cycles."
        stop_when = "promotion review either accepts, rejects, or asks for targeted confirmation"
        risk = "none"
        gap = 0
    elif stage == "leave_inactive_or_retired":
        expected = "Avoids spending system resources on inactive or retired bots."
        stop_when = "operator explicitly reactivates the bot"
        risk = "none"
        gap = 0
    else:
        gap = 0
    return {
        "stage": stage,
        "bot_id": bot_id,
        "next_step": str(summary or ""),
        "next_command": next_command,
        "expected_impact": expected,
        "risk_level": risk,
        "stop_when": stop_when,
        "can_train_now": bool(can_train_now),
        "data_gap": int(max(gap, 0)),
        "positive_rate": round(float(positive_rate), 6),
        "policy": "one_actionable_prescription_per_bot_with_stop_condition_and_expected_impact",
    }


def _classify_bot(
    bot: dict[str, Any],
    *,
    label_row: dict[str, Any],
    quality_row: dict[str, Any],
    memberships: dict[str, set[str]],
    walk_forward: dict[str, Any],
    diagnostic: dict[str, Any],
    diagnostic_path: Path,
    calibration_override: dict[str, Any],
    min_runs: int,
    now: datetime,
    overfit_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bot_id = str(bot.get("bot_id") or "").strip()
    active = bool(bot.get("active", False))
    deleted = bool(bot.get("deleted_from_rotation", False))
    role = str(bot.get("bot_role") or "")
    lifecycle = str(bot.get("lifecycle_state") or "")
    training_excluded = bool(bot.get("training_excluded") or bot.get("exclude_from_training"))
    collection_active = bool(bot.get("data_collection_active"))
    diagnostic_present = diagnostic_path.exists() or bool(label_row.get("diagnostic_present"))
    diagnostic_age = _age_hours(diagnostic.get("timestamp_utc") or label_row.get("diagnostic_age_hours"), now)
    if diagnostic_age is None and label_row.get("diagnostic_age_hours") is not None:
        diagnostic_age = _safe_float(label_row.get("diagnostic_age_hours"), 0.0)

    metrics = _as_dict(diagnostic.get("metrics"))
    sample_count = max(
        _safe_int(diagnostic.get("sample_count"), 0),
        _safe_int(_as_dict(diagnostic.get("runtime_meta")).get("sample_count"), 0),
        _safe_int(label_row.get("sample_count"), 0),
    )
    observation_count = max(
        _safe_int(diagnostic.get("observation_count"), 0),
        _safe_int(_as_dict(diagnostic.get("runtime_meta")).get("observation_count"), 0),
        _safe_int(label_row.get("observation_count"), 0),
        _safe_int(bot.get("data_collection_observations"), 0),
        _safe_int(bot.get("collected_observation_count"), 0),
        _safe_int(bot.get("observations"), 0),
    )
    eligible_sequences = max(
        _safe_int(diagnostic.get("eligible_sequences"), 0),
        _safe_int(_as_dict(diagnostic.get("runtime_meta")).get("eligible_sequences"), 0),
        _safe_int(label_row.get("eligible_sequences"), 0),
    )
    positive_rate = max(
        _safe_float(diagnostic.get("positive_rate"), 0.0),
        _safe_float(metrics.get("positive_rate"), 0.0),
        _safe_float(label_row.get("positive_rate"), 0.0),
    )
    acted_coverage = max(_safe_float(metrics.get("acted_coverage"), -1.0), _safe_float(label_row.get("acted_coverage"), -1.0))
    acted_accuracy = max(_safe_float(metrics.get("acted_accuracy"), -1.0), _safe_float(label_row.get("acted_accuracy"), -1.0))
    accuracy_lift = max(
        _safe_float(metrics.get("accuracy_lift_over_majority"), -999.0),
        _safe_float(label_row.get("accuracy_lift_over_majority"), -999.0),
    )
    long_precision = max(_safe_float(metrics.get("long_precision"), 0.0), _safe_float(label_row.get("long_precision"), 0.0))
    short_precision = max(_safe_float(metrics.get("short_precision"), 0.0), _safe_float(label_row.get("short_precision"), 0.0))
    precision_contract = _precision_contract(bot)
    precision_needs, precision_gaps = _precision_repair_needs(
        contract=precision_contract,
        long_precision=long_precision,
        short_precision=short_precision,
        acted_accuracy=acted_accuracy,
        acted_coverage=acted_coverage,
        sample_count=sample_count,
        calibration_override=calibration_override,
    )
    test_accuracy = max(
        _safe_float(metrics.get("test_accuracy"), 0.0),
        _safe_float(bot.get("test_accuracy"), 0.0),
        _safe_float(quality_row.get("test_accuracy"), 0.0),
    )
    quality_score = max(_safe_float(bot.get("quality_score"), 0.0), _safe_float(quality_row.get("quality_score"), 0.0))
    min_observations = _minimum_observations(bot)
    wf_runs = _safe_int(walk_forward.get("runs"), 0)
    wf_status = str(walk_forward.get("status") or "")
    runs_remaining = max(int(min_runs) - wf_runs, 0) if active and not deleted else 0
    overfit = _as_dict(overfit_row)
    overfit_status = str(overfit.get("status") or "").strip().lower()
    overfit_policy = _as_dict(overfit.get("policy"))
    overfit_risk_score = _safe_float(overfit.get("risk_score"), 0.0)
    overfit_gap = _safe_float(overfit.get("train_forward_gap"), 0.0)

    needs: list[dict[str, Any]] = []
    if deleted or not active:
        needs.append(_need_record("leave_inactive_or_retired", "Inactive/deleted; do not spend training cycles unless explicitly reactivated.", 5))
    label_recommendation = str(label_row.get("recommendation") or "")
    if active and not diagnostic_present and (training_excluded or label_recommendation == "create_collect_only_diagnostics"):
        needs.append(_need_record("create_collect_only_diagnostics", "Collection-only bot needs a diagnostic snapshot before training eligibility can be judged.", 100))
    elif active and not diagnostic_present:
        needs.append(_need_record("refresh_training_diagnostics", "No fresh diagnostic artifact; create or refresh diagnostics before judging it.", 100))
    elif active and diagnostic_age is not None and diagnostic_age > 48:
        needs.append(_need_record("refresh_training_diagnostics", f"Diagnostic is stale at {diagnostic_age:.1f}h; refresh before retraining.", 90))
    if bot_id in memberships.get("repair_runtime_input_bot_ids", set()):
        repair_priority = 80.0 if training_excluded and min_observations > 0 and observation_count < min_observations else 102.0
        needs.append(_need_record("repair_runtime_inputs", "Runtime inputs are flagged for repair before another retrain.", repair_priority))
    if active and not deleted and overfit_status in OVERFIT_BLOCKING_STATUSES and not bool(overfit_policy.get("may_promote", False)):
        priority = 106.0 if overfit_status == "leak_like" else 104.0 if overfit_status == "severe_overfit" else 93.0
        summary = (
            f"Overfitting awareness is {overfit_status}; hold teacher/promotion duty and run generalization repair before more blind training."
        )
        needs.append(_need_record("reduce_overfitting", summary, priority, command_key="overfitting_awareness"))
    if label_recommendation == "upgrade_label_contract":
        needs.append(_need_record("upgrade_label_contract", "Observed diagnostics are missing the expected label contract.", 98))
    if training_excluded and min_observations > 0 and observation_count < min_observations:
        gap = min_observations - observation_count
        needs.append(_need_record("collect_more_data", f"Collect {gap} more observations to reach the {min_observations} training floor.", 94))
    elif active and sample_count > 0 and sample_count < 200:
        needs.append(_need_record("collect_more_data", f"Only {sample_count} usable samples; collect more before canary training.", 92))
    sample_filter_blocked = bool(
        active
        and not deleted
        and not training_excluded
        and sample_count <= 0
        and (eligible_sequences > 0 or label_recommendation == "relax_sample_filter")
    )
    if sample_filter_blocked:
        needs.append(
            _need_record(
                "relax_sample_filter",
                f"Runtime snapshot has {eligible_sequences} eligible sequences but 0 usable samples; relax the sample filter or repair label eligibility before retraining.",
                96,
            )
        )
    if positive_rate > 0 and (positive_rate < 0.25 or positive_rate > 0.75):
        needs.append(_need_record("rebalance_labels", f"Positive rate is {positive_rate:.3f}; rebalance labels or widen counterexamples.", 86, command_key="upgrade_label_contract"))
    if active and not deleted and runs_remaining > 0 and not training_excluded and not sample_filter_blocked:
        needs.append(_need_record("top_off_walk_forward_runs", f"Needs {runs_remaining} more walk-forward runs to reach {min_runs}.", 82))
    if bot_id in memberships.get("quality_probation_bot_ids", set()) or wf_status == "fail":
        if wf_status == "pass" and runs_remaining <= 0:
            needs.append(_need_record("monitor_passing_candidate", "Walk-forward gate is passing; monitor or route to promotion review instead of retraining.", 42))
        elif acted_coverage >= 0.75 or label_recommendation == "tighten_abstention_thresholds":
            if calibration_override:
                needs.append(_need_record("targeted_quality_retrain", "Calibration override is applied; run a small canary to validate the tighter abstention.", 84))
            else:
                needs.append(_need_record("apply_abstention_calibration", "Quality probation with high acted coverage; tighten abstention before widening.", 89))
        else:
            needs.append(_need_record("targeted_quality_retrain", "Quality probation/failing gate; run a small targeted canary after repairs.", 76))
    needs.extend(precision_needs)
    quality_next_step = str(quality_row.get("next_step") or "")
    suppress_redundant_autopilot_retrain = bool(
        quality_next_step == "targeted_retrain" and wf_status == "pass" and runs_remaining <= 0
    )
    if quality_next_step and not suppress_redundant_autopilot_retrain:
        needs.append(
            _need_record(
                f"autopilot_{quality_next_step}",
                f"Bot-quality autopilot next step: {quality_next_step}.",
                min(_safe_float(quality_row.get("priority"), 50.0), 74.0),
                command_key=quality_next_step,
            )
        )
    if not needs:
        needs.append(_need_record("monitor", "No immediate blocker; keep collecting evidence and watch quality drift.", 10))

    needs.sort(key=lambda item: _safe_float(item.get("priority"), 0.0), reverse=True)
    primary = needs[0]
    command_key = str(primary.get("command_key") or primary.get("key") or "bot-needs")
    exact_files = _unique(
        [
            str(diagnostic_path) if diagnostic_path.exists() else "",
            "governance/health/training_label_audit_latest.json" if label_row else "",
            "governance/walk_forward/walk_forward_latest.json" if walk_forward else "",
            "governance/health/training_quality_control_latest.json"
            if bot_id in memberships.get("targeted_retrain_bot_ids", set()) or bot_id in memberships.get("quality_probation_bot_ids", set())
            else "",
            "governance/health/overfitting_awareness_latest.json" if overfit else "",
        ]
    )
    evidence = {
        "sample_count": sample_count,
        "observation_count": observation_count,
        "minimum_observations": min_observations,
        "eligible_sequences": eligible_sequences,
        "positive_rate": round(positive_rate, 6),
        "acted_coverage": round(acted_coverage, 6),
        "acted_accuracy": round(acted_accuracy, 6),
        "accuracy_lift_over_majority": round(accuracy_lift, 6) if accuracy_lift > -100 else None,
        "long_precision": round(long_precision, 6),
        "short_precision": round(short_precision, 6),
        "precision_contract": precision_contract,
        "precision_gaps": precision_gaps,
        "test_accuracy": round(test_accuracy, 6),
        "quality_score": round(quality_score, 6),
        "walk_forward_runs": wf_runs,
        "walk_forward_status": wf_status,
        "walk_forward_runs_remaining": runs_remaining,
        "diagnostic_present": diagnostic_present,
        "diagnostic_age_hours": diagnostic_age,
        "label_recommendation": str(label_row.get("recommendation") or ""),
        "audit_buckets": _as_list(label_row.get("_audit_buckets")),
        "quality_queue_reasons": _as_list(quality_row.get("reasons")),
        "quality_queue_next_step": str(quality_row.get("next_step") or ""),
        "calibration_override_applied": bool(calibration_override),
        "overfit_status": overfit_status or None,
        "overfit_risk_score": round(overfit_risk_score, 6),
        "train_forward_gap": round(overfit_gap, 6),
        "overfit_may_teach": bool(overfit_policy.get("may_teach", True)),
        "overfit_may_promote": bool(overfit_policy.get("may_promote", True)),
    }
    next_command = _bot_command(bot_id, command_key)
    return {
        "bot_id": bot_id,
        "bot_role": role,
        "lifecycle_state": lifecycle,
        "active": active,
        "training_excluded": training_excluded,
        "data_collection_active": collection_active,
        "primary_need": primary.get("key"),
        "primary_need_summary": primary.get("summary"),
        "priority": primary.get("priority"),
        "next_command": next_command,
        "all_needs": [{k: v for k, v in need.items() if k != "command_key"} for need in needs],
        "evidence": evidence,
        "effectiveness_prescription": _effectiveness_prescription(
            bot_id=bot_id,
            primary_need=str(primary.get("key") or ""),
            summary=str(primary.get("summary") or ""),
            evidence=evidence,
            next_command=next_command,
        ),
        "exact_files": exact_files,
    }


def _summary_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        key = str(row.get("primary_need") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _prescription_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        prescription = _as_dict(row.get("effectiveness_prescription"))
        key = str(prescription.get("stage") or row.get("primary_need") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _training_readiness_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    out = {"can_train_now": 0, "repair_or_calibrate_first": 0, "collect_more_data_first": 0, "monitor_or_inactive": 0}
    for row in records:
        prescription = _as_dict(row.get("effectiveness_prescription"))
        stage = str(prescription.get("stage") or row.get("primary_need") or "")
        if bool(prescription.get("can_train_now", False)):
            out["can_train_now"] += 1
        elif stage in {"collect_more_data", "rebalance_labels", "relax_sample_filter", "collect_side_specific_outcomes"}:
            out["collect_more_data_first"] += 1
        elif stage in {"monitor", "monitor_passing_candidate", "leave_inactive_or_retired"}:
            out["monitor_or_inactive"] += 1
        else:
            out["repair_or_calibrate_first"] += 1
    return out


def _next_batches(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    buckets = {
        "diagnostics": {"create_collect_only_diagnostics", "refresh_training_diagnostics"},
        "repair_first": {"repair_runtime_inputs", "upgrade_label_contract"},
        "collect_more_data": {"collect_more_data", "rebalance_labels", "relax_sample_filter", "collect_side_specific_outcomes"},
        "training_topoff": {"top_off_walk_forward_runs", "targeted_quality_retrain"},
        "calibration": {"apply_abstention_calibration", "use_side_specific_thresholds"} | PRECISION_REPAIR_NEEDS,
        "overfitting": {"reduce_overfitting"},
    }
    out: dict[str, list[str]] = {key: [] for key in buckets}
    for row in sorted(records, key=lambda item: _safe_float(item.get("priority"), 0.0), reverse=True):
        need = str(row.get("primary_need") or "")
        for bucket, allowed in buckets.items():
            if need in allowed and len(out[bucket]) < 20:
                out[bucket].append(str(row.get("bot_id") or ""))
    return out


def _training_candidate_selector(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    near_ready: list[dict[str, Any]] = []
    blocked_reasons = {
        "repair_first": {"repair_runtime_inputs", "upgrade_label_contract", "refresh_training_diagnostics", "create_collect_only_diagnostics"},
        "data_first": {"collect_more_data", "rebalance_labels", "relax_sample_filter", "collect_side_specific_outcomes"},
        "calibration_first": {"apply_abstention_calibration", "use_side_specific_thresholds"} | PRECISION_REPAIR_NEEDS,
        "overfit_first": {"reduce_overfitting"},
    }
    blocked_counts = {key: 0 for key in blocked_reasons}
    for row in records:
        prescription = _as_dict(row.get("effectiveness_prescription"))
        evidence = _as_dict(row.get("evidence"))
        need = str(row.get("primary_need") or "")
        can_train = bool(prescription.get("can_train_now", False))
        overfit_status = str(evidence.get("overfit_status") or "")
        sample_count = _safe_int(evidence.get("sample_count"), 0)
        observation_count = _safe_int(evidence.get("observation_count"), 0)
        positive_rate = _safe_float(evidence.get("positive_rate"), 0.5)
        runs_remaining = _safe_int(evidence.get("walk_forward_runs_remaining"), 0)
        quality_score = _safe_float(evidence.get("quality_score"), 0.0)
        test_accuracy = _safe_float(evidence.get("test_accuracy"), 0.0)
        if need == "monitor_passing_candidate":
            near_ready.append(
                {
                    "bot_id": row.get("bot_id"),
                    "state": "promotion_review_not_blind_retrain",
                    "quality_score": round(quality_score, 6),
                    "test_accuracy": round(test_accuracy, 6),
                    "walk_forward_runs_remaining": runs_remaining,
                }
            )
            continue
        for bucket, needs in blocked_reasons.items():
            if need in needs:
                blocked_counts[bucket] += 1
        if not can_train:
            continue
        if overfit_status in OVERFIT_BLOCKING_STATUSES:
            blocked_counts["overfit_first"] += 1
            continue
        if sample_count < 200 or observation_count < 200:
            blocked_counts["data_first"] += 1
            continue
        if not (0.20 <= positive_rate <= 0.80):
            blocked_counts["data_first"] += 1
            continue
        candidates.append(
            {
                "bot_id": row.get("bot_id"),
                "primary_need": need,
                "priority": round(_safe_float(row.get("priority"), 0.0), 6),
                "quality_score": round(quality_score, 6),
                "test_accuracy": round(test_accuracy, 6),
                "walk_forward_runs_remaining": runs_remaining,
                "sample_count": sample_count,
                "observation_count": observation_count,
                "positive_rate": round(positive_rate, 6),
                "recommended_command": [
                    "./scripts/ops/opsctl.sh",
                    "retrain-force-targeted",
                    "--include-bot-ids",
                    str(row.get("bot_id") or ""),
                    "--skip-master-update",
                ],
            }
        )
    candidates.sort(
        key=lambda row: (
            _safe_float(row.get("quality_score"), 0.0),
            _safe_float(row.get("test_accuracy"), 0.0),
            -_safe_int(row.get("walk_forward_runs_remaining"), 0),
            _safe_float(row.get("priority"), 0.0),
        ),
        reverse=True,
    )
    selected = candidates[:20]
    selected_ids = [str(row.get("bot_id") or "") for row in selected if str(row.get("bot_id") or "")]
    return {
        "active": True,
        "mode": "training_candidate_selector_v2",
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "near_ready_promotion_review_count": len(near_ready),
        "selected_candidates": selected,
        "near_ready_promotion_review": near_ready[:20],
        "blocked_counts": blocked_counts,
        "batch_policy": {
            "micro_canary_first": True,
            "max_recommended_batch_size_now": min(len(selected), 5),
            "require_training_runtime_clear": True,
            "require_writer_idle": True,
            "require_fresh_diagnostics": True,
            "require_overfit_clear": True,
        },
        "recommended_batch_command": [
            "./scripts/ops/opsctl.sh",
            "retrain-force-targeted",
            "--include-bot-ids",
            ",".join(selected_ids[:5]),
            "--skip-master-update",
        ]
        if selected_ids
        else [],
        "policy": "train only fresh, balanced, overfit-clear candidates; route passing bots to promotion review instead of blind retrain",
    }


def _zero_observation_repair_contract(records: list[dict[str, Any]]) -> dict[str, Any]:
    zero_rows: list[dict[str, Any]] = []
    near_zero_rows: list[dict[str, Any]] = []
    for row in records:
        evidence = _as_dict(row.get("evidence"))
        active = bool(row.get("active", False))
        collecting = bool(row.get("data_collection_active", False))
        if not active or not collecting:
            continue
        observation_count = _safe_int(evidence.get("observation_count"), 0)
        sample_count = _safe_int(evidence.get("sample_count"), 0)
        item = {
            "bot_id": row.get("bot_id"),
            "sample_count": sample_count,
            "observation_count": observation_count,
            "primary_need": row.get("primary_need"),
            "next_command": row.get("next_command"),
        }
        if observation_count <= 0 and sample_count <= 0:
            zero_rows.append(item)
        elif observation_count < 25 or sample_count <= 0:
            near_zero_rows.append(item)
    zero_ids = [str(row.get("bot_id") or "") for row in zero_rows if str(row.get("bot_id") or "")]
    near_zero_ids = [str(row.get("bot_id") or "") for row in near_zero_rows if str(row.get("bot_id") or "")]
    return {
        "active": bool(zero_rows or near_zero_rows),
        "mode": "zero_observation_collector_repair_v2",
        "zero_observation_count": len(zero_rows),
        "near_zero_observation_count": len(near_zero_rows),
        "zero_observation_bots": zero_rows[:40],
        "near_zero_observation_bots": near_zero_rows[:40],
        "repair_commands": [
            [
                "./scripts/ops/opsctl.sh",
                "training-data-intake",
                "--apply",
                "--include-bot-ids",
                ",".join(zero_ids[:25]),
                "--json",
            ]
            if zero_ids
            else [],
            [
                "./scripts/ops/opsctl.sh",
                "training-labeling-intelligence",
                "--apply",
                "--materialize-collect-only-diagnostics",
                "--json",
            ],
        ],
        "expected_impact": "restores missing collector observations before spending canary training cycles",
        "stop_condition": "zero_observation_count == 0 and near_zero_observation_count trends down after the next observation rollup",
        "protected_volumes": ["/Volumes/VIDEO"],
        "policy": "repair collection and diagnostics first; do not train zero-observation bots",
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, include_bot_ids: set[str] | None = None, limit: int = 0) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    health = project_root / "governance" / "health"
    walk_dir = project_root / "governance" / "walk_forward"
    diagnostics_dir = project_root / "governance" / "training_diagnostics"
    registry = load_json(project_root / "master_bot_registry.json")
    bots = [row for row in _as_list(registry.get("sub_bots")) if isinstance(row, dict)]
    label_audit = load_json(health / "training_label_audit_latest.json")
    training_quality = load_json(health / "training_quality_control_latest.json")
    bot_quality = load_json(health / "bot_quality_autopilot_latest.json")
    overfit_awareness = load_json(health / "overfitting_awareness_latest.json")
    calibration_overrides = _as_dict(load_json(health / "calibration_abstention_overrides_latest.json").get("bot_overrides"))
    walk_forward = _as_dict(load_json(walk_dir / "walk_forward_latest.json").get("bots"))
    min_runs = _safe_int(load_json(walk_dir / "walk_forward_latest.json").get("min_runs"), 12) or 12
    label_index = _index_label_rows(label_audit)
    quality_index = _index_quality_queue(bot_quality)
    overfit_index = _index_overfit_rows(overfit_awareness)
    memberships = _membership_sets(training_quality, bot_quality)
    selected = bots
    if include_bot_ids:
        selected = [bot for bot in bots if str(bot.get("bot_id") or "") in include_bot_ids]

    records: list[dict[str, Any]] = []
    for bot in selected:
        bot_id = str(bot.get("bot_id") or "").strip()
        if not bot_id:
            continue
        diagnostic_path = diagnostics_dir / f"{bot_id}_latest.json"
        diagnostic = load_json(diagnostic_path)
        records.append(
            _classify_bot(
                bot,
                label_row=label_index.get(bot_id, {}),
                quality_row=quality_index.get(bot_id, {}),
                memberships=memberships,
                walk_forward=_as_dict(walk_forward.get(bot_id)),
                diagnostic=diagnostic,
                diagnostic_path=diagnostic_path,
                calibration_override=_as_dict(calibration_overrides.get(bot_id.lower()) or calibration_overrides.get(bot_id)),
                overfit_row=_as_dict(overfit_index.get(bot_id) or overfit_index.get(bot_id.lower())),
                min_runs=min_runs,
                now=now,
            )
        )
    records.sort(key=lambda item: _safe_float(item.get("priority"), 0.0), reverse=True)
    limited_records = records[:limit] if limit and limit > 0 else records
    counts = _summary_counts(records)
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "needs_action" if any(str(row.get("primary_need")) != "monitor" for row in records) else "ready",
        "registry_bot_count": len(bots),
        "included_bot_count": len(records),
        "returned_bot_count": len(limited_records),
        "limit": limit,
        "need_counts": counts,
        "prescription_counts": _prescription_counts(records),
        "training_readiness_counts": _training_readiness_counts(records),
        "next_batches": _next_batches(records),
        "training_candidate_selector": _training_candidate_selector(records),
        "zero_observation_repair_contract": _zero_observation_repair_contract(records),
        "bot_needs": limited_records,
        "artifacts": {
            "registry": str(project_root / "master_bot_registry.json"),
            "training_label_audit": str(health / "training_label_audit_latest.json"),
            "training_quality": str(health / "training_quality_control_latest.json"),
            "bot_quality_autopilot": str(health / "bot_quality_autopilot_latest.json"),
            "overfitting_awareness": str(health / "overfitting_awareness_latest.json"),
            "walk_forward": str(walk_dir / "walk_forward_latest.json"),
            "diagnostics_dir": str(diagnostics_dir),
        },
        "contract": {
            "one_primary_need_per_bot": True,
            "includes_exact_files": True,
            "includes_next_command": True,
            "safe_by_default": "training commands use skip-master-update; collection and calibration commands are advisory/control-plane first",
            "overfit_aware": True,
            "training_candidate_selector_v2": True,
            "zero_observation_collector_repair_v2": True,
            "protected_volumes": ["/Volumes/VIDEO"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tell the operator what each bot needs to become more effective.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--include-bot-ids", default="")
    parser.add_argument("--limit", type=int, default=0, help="Limit returned bot rows; 0 returns every selected bot.")
    args = parser.parse_args()
    include_bot_ids = {item.strip() for item in args.include_bot_ids.split(",") if item.strip()} or None
    payload = build_payload(PROJECT_ROOT, include_bot_ids=include_bot_ids, limit=max(int(args.limit), 0))
    write_payload(Path(args.out), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "bot_needs_intelligence "
            f"status={payload['overall_status']} "
            f"included={payload['included_bot_count']} "
            f"returned={payload['returned_bot_count']} "
            f"top_need={next(iter(payload['need_counts']), 'none')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
