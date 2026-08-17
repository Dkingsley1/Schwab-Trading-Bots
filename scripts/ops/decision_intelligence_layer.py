#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "decision_intelligence_layer_latest.json"
DEFAULT_MARKET_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "market_move_explainer_latest.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _symbol_aliases(symbol: str) -> set[str]:
    raw = str(symbol or "BTC").strip().upper() or "BTC"
    compact = raw.replace("-", "").replace("/", "")
    aliases = {raw, compact}
    if raw in {"BTC", "BTC-USD", "BTCUSD", "XBT", "XBTUSD"} or compact in {"BTCUSD", "XBTUSD"}:
        aliases.update({"BTC", "BTC-USD", "BTCUSD", "XBT", "XBTUSD", "BTC-PERP", "BTCUSDT"})
    return aliases


def _row_symbol(row: dict[str, Any]) -> str:
    for key in ("symbol", "underlying", "asset", "ticker"):
        text = str(row.get(key) or "").strip().upper()
        if text:
            return text
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    return str(features.get("symbol") or "").strip().upper()


def _row_features(row: dict[str, Any]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    raw = row.get("features")
    if isinstance(raw, dict):
        features.update(raw)
    raw_meta = row.get("grand_master_meta")
    if isinstance(raw_meta, dict):
        for key, value in raw_meta.items():
            features[f"grand_master_{key}"] = value
            if str(key).startswith("quant_strategy_"):
                features[str(key)] = value
    for key in (
        "action",
        "score",
        "threshold",
        "pnl_proxy",
        "return_1m",
        "grand_master_vote",
        "master_vote",
    ):
        if key in row:
            features[key] = row.get(key)
    return features


def _iter_recent_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    rows: deque[dict[str, Any]] = deque(maxlen=max(int(limit), 1))
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    return list(rows)


def _recent_shadow_rows(project_root: Path, *, symbol: str, max_files: int, max_rows: int) -> list[dict[str, Any]]:
    aliases = _symbol_aliases(symbol)
    candidates = []
    for root in (project_root / "governance", project_root / "local_fallback_storage" / "governance"):
        candidates.extend(root.glob("shadow*/shadow_pnl_attribution_*.jsonl"))
    candidates = sorted(
        [path for path in candidates if path.is_file()],
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )[: max(int(max_files), 1)]
    rows: list[dict[str, Any]] = []
    per_file_limit = max(max(int(max_rows) // max(len(candidates), 1), 25), 1)
    for path in candidates:
        for row in _iter_recent_jsonl(path, limit=per_file_limit):
            row_symbol = _row_symbol(row)
            if not row_symbol or row_symbol in aliases or row_symbol.replace("-", "") in aliases:
                rows.append({**row, "_source_file": str(path)})
    return rows[-max(int(max_rows), 1) :]


def _avg(values: Iterable[float]) -> float:
    clean = [float(value) for value in values]
    return sum(clean) / len(clean) if clean else 0.0


def _feature_average(rows: list[dict[str, Any]], keys: list[str], default: float = 0.0) -> float:
    values = []
    for row in rows:
        features = _row_features(row)
        for key in keys:
            if key in features:
                values.append(_safe_float(features.get(key), default))
                break
    return _avg(values) if values else float(default)


def _latest_feature(rows: list[dict[str, Any]], keys: list[str], default: float = 0.0) -> float:
    for row in reversed(rows):
        features = _row_features(row)
        for key in keys:
            if key in features:
                return _safe_float(features.get(key), default)
    return float(default)


def _factor_strength(name: str, value: float, *, neutral: float = 0.5, pressure: bool = False, signed: bool = False) -> dict[str, Any]:
    if signed:
        strength = min(abs(float(value)), 1.0)
    elif pressure:
        strength = _clamp01(float(value))
    else:
        strength = min(abs(float(value) - neutral) * 2.0, 1.0)
    return {"name": name, "value": round(float(value), 6), "strength": round(strength, 6)}


def build_quant_attribution_ledger(rows: list[dict[str, Any]], attribution: dict[str, Any]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    latest_rows = rows[-25:]
    for row in latest_rows:
        action = str(row.get("action") or "").strip().upper()
        if action:
            action_counts[action] += 1
        for reason in row.get("reasons") or []:
            if isinstance(reason, str):
                reason_counts[reason.split()[0].strip()] += 1

    factors = [
        _factor_strength("kelly_signal", _feature_average(rows, ["quant_kelly_fraction_norm"], 0.5)),
        _factor_strength("strategy_conviction", _feature_average(rows, ["quant_strategy_risk_adjusted_conviction_norm", "quant_strategy_conviction"], 0.0), neutral=0.0, pressure=True),
        _factor_strength("portfolio_fit", _feature_average(rows, ["quant_strategy_portfolio_fit_norm", "quant_strategy_fit"], 0.0), neutral=0.0, pressure=True),
        _factor_strength("execution_alignment", _feature_average(rows, ["quant_strategy_execution_alignment_norm", "quant_strategy_execution_alignment"], 0.0), neutral=0.0, pressure=True),
        _factor_strength("tail_pressure", _feature_average(rows, ["quant_cvar_tail_risk_norm", "quant_tail_pressure"], 0.0), pressure=True),
        _factor_strength("micro_tradeability", _feature_average(rows, ["market_micro_tradeability_score_norm", "execution_fitness_norm"], 0.5)),
        _factor_strength("allocation_confidence", _feature_average(rows, ["allocation_confidence_norm"], 0.0), neutral=0.0, pressure=True),
        _factor_strength("flow_direction", _feature_average(rows, ["flow_direction_signed"], 0.0), signed=True),
        _factor_strength("lead_lag_signal", _feature_average(rows, ["lead_lag_signal_signed"], 0.0), signed=True),
        _factor_strength("duplicate_alpha_pressure", _feature_average(rows, ["duplicate_alpha_pressure_norm", "strategy_overlap_pressure_norm"], 0.0), pressure=True),
    ]
    strongest = sorted(factors, key=lambda item: (-float(item["strength"]), str(item["name"])))[:6]

    status = "ready" if rows else "degraded"
    if attribution.get("ok") and int(attribution.get("row_count", 0) or 0) > 0 and rows:
        status = "ready"
    elif attribution.get("ok") or rows:
        status = "thin"
    if rows and max((float(item.get("strength", 0.0) or 0.0) for item in factors), default=0.0) <= 0.0:
        status = "thin"

    return {
        "status": status,
        "recent_decision_rows": len(rows),
        "latest_event_timestamp_utc": str((rows[-1] if rows else {}).get("timestamp_utc") or attribution.get("latest_event_timestamp_utc") or ""),
        "action_counts": dict(sorted(action_counts.items())),
        "top_reasons": [{"reason": key, "count": count} for key, count in reason_counts.most_common(8)],
        "factor_scores": factors,
        "strongest_factors": strongest,
        "strategy_attribution": {
            "ok": bool(attribution.get("ok", False)),
            "day": str(attribution.get("day") or ""),
            "row_count": int(attribution.get("row_count", 0) or 0),
            "total_pnl_proxy": round(_safe_float(attribution.get("total_pnl_proxy"), 0.0), 8),
            "top_lane": str(attribution.get("top_lane") or ""),
            "top_layer": str(attribution.get("top_layer") or ""),
        },
    }


def build_promotion_gate_intelligence(project_root: Path) -> dict[str, Any]:
    readiness_path = project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json"
    gate_path = project_root / "governance" / "walk_forward" / "promotion_gate_latest.json"
    bottleneck_path = project_root / "governance" / "walk_forward" / "promotion_bottleneck_latest.json"
    readiness = load_json(readiness_path)
    gate = load_json(gate_path)
    bottleneck = load_json(bottleneck_path)

    near_pass = readiness.get("near_pass_examples") if isinstance(readiness.get("near_pass_examples"), list) else []
    fail_examples = readiness.get("top_fail_examples") if isinstance(readiness.get("top_fail_examples"), list) else []
    candidates = []
    for row in list(near_pass)[:8] + list(fail_examples)[:8]:
        if not isinstance(row, dict):
            continue
        failed = row.get("failed_gates") if isinstance(row.get("failed_gates"), dict) else {}
        missing = [key for key, value in failed.items() if bool(value)]
        bot_id = str(row.get("bot_id") or "").strip()
        if not bot_id:
            continue
        forward_mean = _safe_float(row.get("forward_mean"), 0.0)
        delta = _safe_float(row.get("delta"), 0.0)
        runs = _safe_int(row.get("runs"), 0)
        readiness_score = _clamp01(0.45 + 0.35 * forward_mean + 0.20 * max(delta, 0.0) - 0.08 * len(missing))
        candidates.append(
            {
                "bot_id": bot_id,
                "readiness_score": round(readiness_score, 6),
                "runs": runs,
                "forward_mean": round(forward_mean, 6),
                "delta": round(delta, 6),
                "missing": missing,
            }
        )
    candidates.sort(key=lambda row: (-float(row["readiness_score"]), -int(row["runs"]), str(row["bot_id"])))

    blockers = []
    blockers.extend(str(item) for item in readiness.get("blocking_reasons", []) if str(item))
    if not bool(readiness.get("promote_ok", gate.get("promote_ok", False))):
        blockers.append("promotion_gate_not_cleared")
    if _safe_int(readiness.get("coverage_shortfall_bots"), 0) > 0:
        blockers.append("coverage_shortfall")
    blockers = ordered_unique(blockers)

    return {
        "status": "ready" if bool(readiness.get("promote_ok", False)) else ("needs_work" if readiness else "degraded"),
        "promote_ok": bool(readiness.get("promote_ok", gate.get("promote_ok", False))),
        "coverage_ok": bool(readiness.get("coverage_ok", gate.get("coverage_ok", False))),
        "considered_bots": _safe_int(readiness.get("considered_bots", gate.get("considered_bots")), 0),
        "fail_share": round(_safe_float(readiness.get("fail_share", gate.get("fail_share")), 1.0), 6),
        "readiness_margin": round(_safe_float(readiness.get("readiness_margin"), 0.0), 6),
        "blockers": blockers,
        "closest_candidates": candidates[:8],
        "recommended_retrain": readiness.get("recommended_retrain") if isinstance(readiness.get("recommended_retrain"), dict) else {},
        "bottleneck_status": str(bottleneck.get("overall_status") or ""),
        "source_files": [str(readiness_path), str(gate_path), str(bottleneck_path)],
    }


def build_shadow_paper_feedback(project_root: Path, attribution: dict[str, Any]) -> dict[str, Any]:
    paper_path = project_root / "governance" / "health" / "paper_performance_latest.json"
    paper = load_json(paper_path)
    history = paper.get("history_daily_series") if isinstance(paper.get("history_daily_series"), list) else []
    latest = history[-1] if history and isinstance(history[-1], dict) else {}
    paper_net = _safe_float(paper.get("latest_net_pnl_total", latest.get("ending_net_pnl_total")), 0.0)
    paper_change = _safe_float(paper.get("latest_change_vs_previous_day", latest.get("change_vs_previous_day")), 0.0)
    shadow_proxy = _safe_float(attribution.get("total_pnl_proxy"), 0.0)
    shadow_ok = bool(attribution.get("ok", False))
    paper_ok = bool(paper.get("ok", bool(history)))
    agreement = "unknown"
    if shadow_ok and paper_ok:
        if shadow_proxy == 0.0 or paper_change == 0.0:
            agreement = "flat"
        elif (shadow_proxy > 0) == (paper_change > 0):
            agreement = "aligned"
        else:
            agreement = "divergent"
    recommendations = []
    if agreement == "divergent":
        recommendations.append("compare shadow conviction against paper fills before raising sizing or promotion thresholds")
    if not paper_ok:
        recommendations.append("refresh paper performance before using feedback to tune timidness")
    if not shadow_ok:
        recommendations.append("refresh strategy attribution so shadow-paper feedback has current decision evidence")
    if agreement == "aligned" and shadow_proxy > 0 and paper_change > 0:
        recommendations.append("allow cautious conviction lift for matching shadow and paper winners")

    return {
        "status": "ready" if shadow_ok and paper_ok else "degraded",
        "agreement": agreement,
        "shadow_total_pnl_proxy": round(shadow_proxy, 8),
        "paper_latest_net_pnl_total": round(paper_net, 6),
        "paper_latest_change": round(paper_change, 6),
        "paper_history_days": len(history),
        "recommendations": recommendations,
        "source_files": [str(paper_path), str(project_root / "governance" / "health" / "strategy_attribution_latest.json")],
    }


def build_strategy_decay_intelligence(project_root: Path) -> dict[str, Any]:
    path = project_root / "governance" / "research" / "decay_monitor_latest.json"
    decay = load_json(path)
    weak = decay.get("weak_sleeves") if isinstance(decay.get("weak_sleeves"), list) else []
    trend = str((decay.get("trend") if isinstance(decay.get("trend"), str) else "") or "")
    status = str(decay.get("overall_status") or ("ready" if decay.get("ok") else "degraded"))
    return {
        "status": status,
        "weak_sleeve_count": _safe_int(decay.get("weak_sleeve_count"), len(weak)),
        "weak_sleeves": weak[:8],
        "latest_change_vs_previous_day": round(_safe_float(decay.get("latest_change_vs_previous_day"), 0.0), 6),
        "pnl_slope": decay.get("pnl_slope"),
        "promotion_ready": bool(decay.get("promotion_ready", False)),
        "trend": trend,
        "recommendations": decay.get("recommendations") if isinstance(decay.get("recommendations"), list) else [],
        "source_files": [str(path)],
    }


def build_duplicate_alpha_governor(project_root: Path) -> dict[str, Any]:
    overlap_path = project_root / "governance" / "platform_intelligence" / "duplicate_alpha_overlap_latest.json"
    compression_path = project_root / "governance" / "platform_stabilization_quality" / "duplicate_alpha_compression_latest.json"
    overlap = load_json(overlap_path)
    compression = load_json(compression_path)
    cluster_count = _safe_int(overlap.get("overlap_cluster_count", compression.get("overlap_cluster_count")), 0)
    high_count = _safe_int(overlap.get("high_overlap_cluster_count"), 0)
    status = "ready"
    if high_count > 0:
        status = "needs_work"
    if not overlap and not compression:
        status = "degraded"
    clusters = overlap.get("overlap_clusters") if isinstance(overlap.get("overlap_clusters"), list) else []
    actions = []
    if high_count > 0:
        actions.append("downweight high-overlap clusters before promotion or training expansion")
    if cluster_count > 0:
        actions.append("prefer novel candidates over overlapping alpha families in the next retrain batch")
    for item in compression.get("recommended_commands", []) if isinstance(compression.get("recommended_commands"), list) else []:
        if isinstance(item, list):
            actions.append(" ".join(str(part) for part in item if str(part)))
        else:
            actions.append(str(item))
    return {
        "status": status,
        "overlap_cluster_count": cluster_count,
        "high_overlap_cluster_count": high_count,
        "top_overlap_clusters": clusters[:8],
        "recommended_actions": ordered_unique(actions),
        "source_files": [str(overlap_path), str(compression_path)],
    }


def build_training_launch_preflight(project_root: Path) -> dict[str, Any]:
    path = project_root / "governance" / "health" / "training_runtime_control_latest.json"
    control = load_json(path)
    if not control:
        return {
            "status": "degraded",
            "launch_allowed": False,
            "prep_allowed": False,
            "mode": "refresh_required",
            "blockers": ["training_runtime_control_missing"],
            "safe_prep_targets": [],
            "recommended_commands": [["./scripts/ops/opsctl.sh", "training-runtime-control", "--json"]],
            "recommended_actions": ["refresh training-runtime-control before launching or staging retrains"],
            "source_files": [str(path)],
        }

    runtime_status = str(control.get("overall_status") or "").strip().lower() or "unknown"
    training_quality = control.get("training_quality") if isinstance(control.get("training_quality"), dict) else {}
    quality_status = str(training_quality.get("overall_status") or "").strip().lower()
    resource_guard = control.get("resource_guard") if isinstance(control.get("resource_guard"), dict) else {}
    backend = control.get("runtime_backend_parity") if isinstance(control.get("runtime_backend_parity"), dict) else {}
    parity_state = str(backend.get("parity_state") or "").strip().lower()
    snapshot_ready = bool(control.get("snapshot_ready", False))
    resource_ok = bool(resource_guard.get("ok", True))
    coverage_repair_ready = bool(control.get("coverage_repair_ready", False))
    precompute_targets = control.get("precompute_targets") if isinstance(control.get("precompute_targets"), list) else []
    launch_contract = control.get("training_launch_contract") if isinstance(control.get("training_launch_contract"), dict) else {}

    blockers: list[str] = []
    if runtime_status == "blocked":
        blockers.append("training_runtime_blocked")
    if quality_status == "blocked":
        blockers.append("training_quality_blocked")
    if not snapshot_ready:
        blockers.append("runtime_snapshot_not_ready")
    if not resource_ok:
        blockers.append("resource_guard_not_green")
    if parity_state and parity_state not in {"ready", "portable_only"}:
        blockers.append(f"backend_parity_{parity_state}")
    if not coverage_repair_ready:
        blockers.append("coverage_repair_not_ready")
    for blocker in launch_contract.get("launch_blockers", []) if isinstance(launch_contract.get("launch_blockers"), list) else []:
        blockers.append(str(blocker))
    blockers = ordered_unique(blockers)

    contract_prep_targets = launch_contract.get("prep_targets") if isinstance(launch_contract.get("prep_targets"), list) else []
    contract_canary_ids = {
        str(row.get("bot_id") or "").strip()
        for row in (launch_contract.get("canary_batch") if isinstance(launch_contract.get("canary_batch"), list) else [])
        if isinstance(row, dict)
    }
    contract_repair_ids = {
        str(row.get("bot_id") or "").strip()
        for row in (launch_contract.get("repair_first_targets") if isinstance(launch_contract.get("repair_first_targets"), list) else [])
        if isinstance(row, dict)
    }
    target_source = contract_prep_targets if contract_prep_targets else precompute_targets
    safe_targets = []
    for row in target_source[:12]:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip()
        if not bot_id:
            continue
        actions = row.get("actions") if isinstance(row.get("actions"), list) else []
        reasons = row.get("reasons") if isinstance(row.get("reasons"), list) else []
        safe_targets.append(
            {
                "bot_id": bot_id,
                "family": str(row.get("family") or ""),
                "priority": round(_safe_float(row.get("priority"), 0.0), 6),
                "training_stage": str(row.get("training_stage") or ""),
                "current_runs": _safe_int(row.get("current_runs"), 0),
                "runs_remaining": _safe_int(row.get("runs_remaining"), 0),
                "needs_runtime_input_repair": bool(row.get("needs_runtime_input_repair", False)),
                "reasons": [str(item) for item in reasons if str(item)],
                "actions": [str(item) for item in actions if str(item)],
                "launch_recommendation": (
                    "repair_first"
                    if bot_id in contract_repair_ids
                    else (
                        "canary_ok"
                        if (bot_id in contract_canary_ids and bool(launch_contract.get("launch_allowed", False))) or (not launch_contract and not blockers)
                        else "prep_only"
                    )
                ),
            }
        )

    launch_allowed = bool(launch_contract.get("launch_allowed", False)) if launch_contract else bool(not blockers and runtime_status in {"ready", "clear", "ok"} and snapshot_ready and resource_ok)
    prep_allowed = bool(launch_contract.get("prep_allowed", False)) if launch_contract else bool(snapshot_ready and bool(safe_targets) and resource_ok and parity_state in {"ready", "portable_only", ""})
    if launch_allowed:
        mode = str(launch_contract.get("mode") or "canary_training_allowed")
        status = "ready"
    elif prep_allowed:
        mode = str(launch_contract.get("mode") or "prep_only")
        status = "needs_work"
    else:
        mode = str(launch_contract.get("mode") or "refresh_required")
        status = "blocked" if blockers else "degraded"

    commands: list[list[str]] = []
    for command in launch_contract.get("recommended_prep_commands", []) if isinstance(launch_contract.get("recommended_prep_commands"), list) else []:
        if isinstance(command, list):
            commands.append([str(part) for part in command])
    if not commands and not snapshot_ready:
        commands.append(["./scripts/ops/opsctl.sh", "runtime-training-snapshot", "--json"])
    if not any(command[:2] == ["./scripts/ops/opsctl.sh", "training-runtime-control"] or (len(command) > 1 and command[1] == "training-runtime-control") for command in commands):
        commands.append(["./scripts/ops/opsctl.sh", "training-runtime-control", "--json"])
    retrain_command = launch_contract.get("recommended_retrain_command") if isinstance(launch_contract.get("recommended_retrain_command"), list) else []
    if launch_allowed and retrain_command:
        commands.append([str(part) for part in retrain_command])
    elif launch_allowed and safe_targets:
        commands.append(["./scripts/ops/opsctl.sh", "coverage-gap-closer", "--apply-stage", "--launch", "--retrain-profile", "coverage_canary", "--json"])

    actions = []
    if mode == "prep_only":
        actions.append("reuse the shared runtime snapshot and keep retrain workers parked until training quality clears")
    if "training_runtime_blocked" in blockers or "training_quality_blocked" in blockers:
        actions.append("treat requested trainings as prep-only while ingestion/backpressure quality remains blocked")
    if launch_allowed:
        actions.append("run a small coverage_canary batch before any wider training expansion")
    for item in control.get("recommended_actions", []) if isinstance(control.get("recommended_actions"), list) else []:
        actions.append(str(item))

    return {
        "status": status,
        "runtime_status": runtime_status,
        "quality_status": quality_status,
        "launch_allowed": launch_allowed,
        "prep_allowed": prep_allowed,
        "mode": mode,
        "snapshot_ready": snapshot_ready,
        "snapshot_age_minutes": round(_safe_float(control.get("snapshot_age_minutes"), 0.0), 3),
        "resource_guard": {
            "ok": resource_ok,
            "memory_pressure_state": str(resource_guard.get("memory_pressure_state") or ""),
            "swap_used_gb": round(_safe_float(resource_guard.get("swap_used_gb"), 0.0), 3),
        },
        "backend_parity_state": parity_state,
        "coverage_repair_ready": coverage_repair_ready,
        "blockers": blockers,
        "backpressure_gate": launch_contract.get("backpressure_gate") if isinstance(launch_contract.get("backpressure_gate"), dict) else control.get("backpressure_training_gate", {}),
        "canary_batch_bot_ids": sorted(contract_canary_ids),
        "repair_first_bot_ids": sorted(contract_repair_ids),
        "safe_prep_targets": safe_targets,
        "recommended_commands": commands,
        "recommended_retrain_command": [str(part) for part in retrain_command],
        "recommended_actions": ordered_unique(actions),
        "source_files": [str(path)],
    }


def _market_context_sources(project_root: Path) -> dict[str, dict[str, Any]]:
    health = project_root / "governance" / "health"
    return {
        "market_micro": load_json(health / "market_micro_sync_latest.json"),
        "crypto_market": load_json(health / "crypto_market_context_sync_latest.json"),
        "crypto_correlation": load_json(health / "market_crypto_correlation_sync_latest.json"),
        "fx_market": load_json(health / "fx_market_context_sync_latest.json"),
        "source_verification": load_json(health / "source_verification_latest.json"),
    }


def _context_evidence_events(project_root: Path, *, symbol: str, contexts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    aliases = _symbol_aliases(symbol)
    events: list[dict[str, Any]] = []

    def add(source: str, event_type: str, strength: float, summary: str, evidence: dict[str, Any] | None = None) -> None:
        events.append(
            {
                "source": source,
                "event_type": event_type,
                "symbol": symbol.upper(),
                "strength": round(_clamp01(strength), 6),
                "summary": summary,
                "evidence": evidence or {},
            }
        )

    crypto = contexts.get("crypto_market") if isinstance(contexts.get("crypto_market"), dict) else {}
    if crypto:
        sources = crypto.get("sources") if isinstance(crypto.get("sources"), dict) else {}
        ok_sources = _safe_int(crypto.get("ok_source_count"), sum(1 for row in sources.values() if isinstance(row, dict) and row.get("ok")))
        total_sources = _safe_int(crypto.get("source_count"), len(sources))
        compared_assets = _safe_int(crypto.get("compared_assets"), 0)
        tracked = _safe_int(crypto.get("tracked_assets"), _safe_int(crypto.get("tracked_symbols"), 0))
        warning_count = _safe_int(crypto.get("warning_count"), 0)
        if bool(crypto.get("ok", False)) or ok_sources > 0 or compared_assets > 0:
            add(
                "crypto_market_context",
                "multi_provider_crypto_context",
                min((ok_sources / max(total_sources, 1)) * 0.7 + min(compared_assets, 5) * 0.06, 1.0),
                f"crypto context has {ok_sources}/{total_sources} sources ok and {compared_assets} cross-provider overlaps",
                {
                    "tracked_assets": tracked,
                    "ok_sources": ok_sources,
                    "total_sources": total_sources,
                    "compared_assets": compared_assets,
                    "warning_count": warning_count,
                    "aliases": sorted(aliases),
                },
            )
        if warning_count > 0:
            add(
                "crypto_market_context",
                "crypto_context_warning",
                min(warning_count / 10.0, 1.0),
                f"crypto context has {warning_count} provider warnings",
                {"warning_count": warning_count},
            )

    corr = contexts.get("crypto_correlation") if isinstance(contexts.get("crypto_correlation"), dict) else {}
    if corr:
        status = str(corr.get("overall_status") or corr.get("status") or "").strip().lower()
        row_count = max(_safe_int(corr.get("row_count"), 0), _safe_int(corr.get("correlation_count"), 0), _safe_int(corr.get("pair_count"), 0))
        if bool(corr.get("ok", False)) or status in {"ready", "advisory", "ok"} or row_count > 0:
            add(
                "market_crypto_correlation",
                "cross_asset_crypto_correlation",
                min(0.35 + row_count / 200.0, 1.0),
                "crypto correlation context is available for cross-asset move explanation",
                {"status": status, "row_count": row_count},
            )
        elif status or corr:
            add(
                "market_crypto_correlation",
                "correlation_context_degraded",
                0.25,
                "crypto correlation context is present but not clean",
                {"status": status, "ok": bool(corr.get("ok", False))},
            )

    micro = contexts.get("market_micro") if isinstance(contexts.get("market_micro"), dict) else {}
    if micro:
        sources = micro.get("sources") if isinstance(micro.get("sources"), dict) else {}
        critical = {
            "local_micro": bool((sources.get("local_micro") or {}).get("ok", False)),
            "finra_short_volume": bool((sources.get("finra_short_volume") or {}).get("ok", False)),
            "nasdaq_trade_halts": bool((sources.get("nasdaq_trade_halts") or {}).get("ok", False)),
            "treasury_auctions": bool((sources.get("treasury_auctions") or {}).get("ok", False)),
        }
        ready_count = sum(1 for value in critical.values() if value)
        if ready_count:
            add(
                "market_micro_context",
                "microstructure_context",
                ready_count / max(len(critical), 1),
                f"market micro context has {ready_count}/{len(critical)} critical lanes ready",
                {"critical_sources": critical},
            )

    verification = contexts.get("source_verification") if isinstance(contexts.get("source_verification"), dict) else {}
    if verification:
        degraded = verification.get("degraded_artifacts") if isinstance(verification.get("degraded_artifacts"), list) else []
        overall = verification.get("overall") if isinstance(verification.get("overall"), dict) else {}
        if degraded:
            add(
                "source_verification",
                "source_verification_degraded_context",
                min(len(degraded) / 8.0, 1.0),
                f"source verification still has {len(degraded)} degraded artifacts",
                {"degraded_artifacts": degraded[:8]},
            )
        elif bool(verification.get("ok", False)) or bool(overall.get("all_verified", False)):
            add("source_verification", "source_verification_clean", 0.75, "source verification is clean", {})

    return sorted(events, key=lambda row: (-float(row.get("strength", 0.0)), str(row.get("source") or ""), str(row.get("event_type") or "")))[:12]


def build_market_move_explainer(project_root: Path, *, symbol: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    contexts = _market_context_sources(project_root)
    context_evidence = _context_evidence_events(project_root, symbol=symbol, contexts=contexts)
    latest_features = _row_features(rows[-1]) if rows else {}
    evidence_count = len(rows)
    context_evidence_count = len(context_evidence)
    mom_1m = _latest_feature(rows, ["mom_1m", "return_1m"], 0.0)
    mom_5m = _latest_feature(rows, ["mom_5m", "pct_from_close"], 0.0)
    flow = _latest_feature(rows, ["flow_direction_signed"], 0.0)
    micro_imbalance = _latest_feature(rows, ["market_micro_order_flow_imbalance_norm"], 0.5)
    tradeability = _latest_feature(rows, ["market_micro_tradeability_score_norm", "execution_fitness_norm"], 0.5)
    tail = max(
        _latest_feature(rows, ["quant_cvar_tail_risk_norm"], 0.0),
        _latest_feature(rows, ["quant_copula_dependency_norm"], 0.0),
        _latest_feature(rows, ["quant_tail_pressure"], 0.0),
    )
    sell_pressure = _latest_feature(rows, ["decision_driver_sell_pressure_norm"], 0.0)
    buy_pressure = _latest_feature(rows, ["decision_driver_buy_pressure_norm"], 0.0)
    crypto_basis = max(
        _latest_feature(rows, ["quant_strategy_crypto_basis_edge_norm"], 0.0),
        _latest_feature(rows, ["crypto_basis_norm", "crypto_perp_basis_norm"], 0.0),
        _latest_feature(rows, ["crypto_hyperliquid_funding_norm", "crypto_funding_norm"], 0.0),
    )
    strategy_conviction = _latest_feature(rows, ["quant_strategy_risk_adjusted_conviction_norm", "quant_strategy_conviction"], 0.0)

    drivers: list[dict[str, Any]] = []
    if mom_1m < -0.0001 or mom_5m < -0.0001:
        drivers.append({"driver": "negative_short_term_momentum", "direction": "selling", "strength": round(min(abs(mom_1m) * 80.0 + abs(mom_5m) * 35.0, 1.0), 6)})
    if flow < -0.05:
        drivers.append({"driver": "negative_system_flow", "direction": "selling", "strength": round(min(abs(flow), 1.0), 6)})
    if micro_imbalance < 0.45:
        drivers.append({"driver": "order_flow_imbalance", "direction": "selling", "strength": round(min((0.45 - micro_imbalance) * 2.5, 1.0), 6)})
    if sell_pressure > max(buy_pressure, 0.25):
        drivers.append({"driver": "system_sell_pressure_stack", "direction": "selling", "strength": round(sell_pressure, 6)})
    elif buy_pressure > max(sell_pressure, 0.25):
        drivers.append({"driver": "system_buy_pressure_stack", "direction": "buying", "strength": round(buy_pressure, 6)})
    if tail > 0.45:
        drivers.append({"driver": "tail_or_correlation_risk", "direction": "risk_off", "strength": round(tail, 6)})
    if crypto_basis > 0.45:
        drivers.append({"driver": "crypto_basis_or_funding_context", "direction": "context", "strength": round(crypto_basis, 6)})
    if strategy_conviction > 0.45:
        drivers.append({"driver": "system_strategy_conviction", "direction": "decision_support", "strength": round(strategy_conviction, 6)})
    if tradeability < 0.35:
        drivers.append({"driver": "thin_tradeability", "direction": "caution", "strength": round(1.0 - tradeability, 6)})
    direction_priority = {
        "selling": 0,
        "buying": 0,
        "risk_off": 1,
        "risk_on": 1,
        "caution": 2,
        "context": 3,
        "decision_support": 4,
    }
    drivers.sort(
        key=lambda row: (
            direction_priority.get(str(row.get("direction") or ""), 5),
            -float(row.get("strength", 0.0)),
            str(row.get("driver") or ""),
        )
    )

    source_coverage = {
        name: bool(payload)
        for name, payload in contexts.items()
    }
    unknowns = []
    if evidence_count == 0 and context_evidence_count == 0:
        unknowns.append(f"{symbol.upper()} has no recent symbol-specific shadow attribution rows")
    elif evidence_count == 0:
        unknowns.append(f"{symbol.upper()} has no recent shadow attribution rows; explanation is context-backed instead of decision-row backed")
    if not source_coverage.get("crypto_market"):
        unknowns.append("crypto market context artifact is missing")
    if not source_coverage.get("market_micro"):
        unknowns.append("market micro context artifact is missing")

    primary_readout = "insufficient symbol-specific evidence"
    if drivers:
        top = drivers[0]
        primary_readout = f"{symbol.upper()} move is most explained by {top['driver']} ({top['direction']})"
    elif evidence_count > 0:
        primary_readout = f"{symbol.upper()} is mostly sideways or evidence is mixed"
    confidence = _clamp01(
        0.18
        + 0.10 * min(evidence_count, 5)
        + 0.06 * min(context_evidence_count, 5)
        + 0.10 * sum(1 for ok in source_coverage.values() if ok)
        + (0.20 if drivers else 0.0)
    )
    if not drivers:
        confidence = min(confidence, 0.64 if context_evidence_count else 0.55)
    overall_status = "ready" if evidence_count > 0 and drivers else ("thin" if evidence_count > 0 or context_evidence_count > 0 else "degraded")

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 2,
        "ok": bool(evidence_count > 0 or drivers or context_evidence_count > 0),
        "overall_status": overall_status,
        "symbol": symbol.upper(),
        "symbol_evidence_count": evidence_count,
        "context_evidence_count": context_evidence_count,
        "symbol_evidence_events": context_evidence,
        "primary_confidence": round(confidence, 6),
        "primary_readout": primary_readout,
        "ranked_drivers": drivers[:8],
        "latest_system_action": str((rows[-1] if rows else {}).get("action") or ""),
        "latest_system_score": round(_safe_float((rows[-1] if rows else {}).get("score"), 0.0), 6),
        "latest_features": {
            "mom_1m": round(mom_1m, 8),
            "mom_5m": round(mom_5m, 8),
            "flow_direction_signed": round(flow, 6),
            "market_micro_order_flow_imbalance_norm": round(micro_imbalance, 6),
            "tradeability": round(tradeability, 6),
            "tail_pressure": round(tail, 6),
            "sell_pressure": round(sell_pressure, 6),
            "buy_pressure": round(buy_pressure, 6),
            "crypto_basis_context": round(crypto_basis, 6),
            "strategy_conviction": round(strategy_conviction, 6),
            "grand_master_vote": round(_safe_float(latest_features.get("grand_master_vote"), 0.0), 6),
        },
        "source_coverage": source_coverage,
        "unknowns": unknowns,
        "operator_actions": ordered_unique(
            [
                "refresh crypto-market-sync and market-correlation-sync for stronger BTC explanations" if not source_coverage.get("crypto_market") else "",
                "collect more symbol-specific BTC evidence before making high-confidence narrative claims" if evidence_count < 3 and context_evidence_count < 3 else "",
                "add symbol-level BTC driver features to shadow attribution rows" if evidence_count >= 3 and not drivers else "",
            ]
        ),
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, symbol: str = "BTC", max_files: int = 12, max_rows: int = 500) -> dict[str, Any]:
    project_root = project_root.resolve()
    rows = _recent_shadow_rows(project_root, symbol=symbol, max_files=max_files, max_rows=max_rows)
    health = project_root / "governance" / "health"
    attribution = load_json(health / "strategy_attribution_latest.json")

    quant_ledger = build_quant_attribution_ledger(rows, attribution)
    promotion = build_promotion_gate_intelligence(project_root)
    feedback = build_shadow_paper_feedback(project_root, attribution)
    decay = build_strategy_decay_intelligence(project_root)
    duplicate = build_duplicate_alpha_governor(project_root)
    training_preflight = build_training_launch_preflight(project_root)
    market_move = build_market_move_explainer(project_root, symbol=symbol, rows=rows)

    sections = {
        "quant_attribution_ledger": quant_ledger,
        "promotion_gate_intelligence": promotion,
        "shadow_paper_feedback_loop": feedback,
        "strategy_decay_monitor": decay,
        "duplicate_alpha_governor": duplicate,
        "training_launch_preflight": training_preflight,
        "market_move_explainer": market_move,
    }
    statuses = [str(section.get("status") or section.get("overall_status") or "ready") for section in sections.values()]
    degraded = [name for name, section in sections.items() if str(section.get("status") or section.get("overall_status") or "ready") in {"blocked", "critical", "degraded"}]
    needs_work = [name for name, section in sections.items() if str(section.get("status") or section.get("overall_status") or "ready") in {"needs_work", "thin"}]

    integrated_actions = ordered_unique(
        [
            "refresh strategy-attribution before tuning conviction" if quant_ledger["status"] == "degraded" else "",
            "run targeted retraining for closest promotion candidates" if promotion["blockers"] else "",
            "hold promotion or sizing increases until shadow-paper feedback is aligned" if feedback["agreement"] == "divergent" else "",
            "quarantine or downweight high-overlap alpha clusters" if duplicate["high_overlap_cluster_count"] > 0 else "",
            "keep training prep-only until training-runtime-control clears" if not training_preflight["launch_allowed"] and training_preflight["prep_allowed"] else "",
            "refresh BTC market context collectors for stronger move explanations" if market_move["overall_status"] != "ready" else "",
        ]
    )
    overall_status = "ready"
    if degraded:
        overall_status = "degraded"
    elif needs_work:
        overall_status = "needs_work"

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": overall_status in {"ready", "needs_work"},
        "overall_status": overall_status,
        "symbol": symbol.upper(),
        "degraded_sections": degraded,
        "needs_work_sections": needs_work,
        "sections": sections,
        "integrated_actions": integrated_actions,
        "source_files": {
            "strategy_attribution": str(health / "strategy_attribution_latest.json"),
            "paper_performance": str(health / "paper_performance_latest.json"),
            "promotion_readiness": str(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json"),
            "decay_monitor": str(project_root / "governance" / "research" / "decay_monitor_latest.json"),
            "duplicate_alpha_overlap": str(project_root / "governance" / "platform_intelligence" / "duplicate_alpha_overlap_latest.json"),
            "training_runtime_control": str(health / "training_runtime_control_latest.json"),
            "market_move_explainer": str(health / "market_move_explainer_latest.json"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build integrated decision intelligence for attribution, promotion, paper feedback, decay, duplicate alpha, and market moves.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--max-files", type=int, default=12)
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--market-out-file", default=str(DEFAULT_MARKET_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        symbol=str(args.symbol),
        max_files=int(args.max_files),
        max_rows=int(args.max_rows),
    )
    out_path = Path(args.out_file).expanduser()
    market_out_path = Path(args.market_out_file).expanduser()
    write_payload(out_path, payload)
    write_payload(market_out_path, payload["sections"]["market_move_explainer"])

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "decision_intelligence "
            f"overall_status={payload.get('overall_status', '')} "
            f"symbol={payload.get('symbol', '')} "
            f"degraded_sections={len(payload.get('degraded_sections', []))} "
            f"needs_work_sections={len(payload.get('needs_work_sections', []))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "needs_work", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
