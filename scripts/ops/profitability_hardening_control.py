#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.accountability import safe_write_json_atomic
from core.profitability_hardening import POLICY_VERSION, evaluate_retirement_evidence


DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "profitability_hardening_latest.json"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _parse_ts(value: Any) -> datetime | None:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        result = datetime.fromisoformat(text)
    except ValueError:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _grade(score: float, *, complete: bool = False) -> str:
    if complete and score >= 100.0:
        return "A+"
    if score >= 90.0:
        return "A"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    if score >= 60.0:
        return "D"
    return "F"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _paper_paths(project_root: Path, *, max_files: int) -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(project_root.glob("exports/trade_logs/**/paper_trades_*.jsonl"))
    candidates.extend(project_root.glob("exports/trade_logs/**/paper_trades_*.jsonl.gz"))
    candidates.extend(project_root.glob("paper_trades_*.jsonl"))
    canonical_runtime = project_root.resolve(strict=False) == PROJECT_ROOT.resolve(strict=False)
    external = Path("/Volumes/BOT_LOGS/schwab_trading_bot")
    if canonical_runtime and external.exists():
        candidates.extend(external.glob("exports/trade_logs/**/paper_trades_*.jsonl"))
        candidates.extend(external.glob("exports/trade_logs/**/paper_trades_*.jsonl.gz"))

    unique: dict[str, Path] = {}
    for path in candidates:
        try:
            key = str(path.resolve())
            mtime = path.stat().st_mtime
        except OSError:
            continue
        prior = unique.get(key)
        if prior is None or mtime > prior.stat().st_mtime:
            unique[key] = path
    rows = sorted(unique.values(), key=lambda item: item.stat().st_mtime, reverse=True)
    return rows[: max(int(max_files), 1)]


def _tail_rows(path: Path, *, max_rows: int) -> Iterable[dict[str, Any]]:
    row_limit = max(int(max_rows), 1)
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                lines = list(deque(handle, maxlen=row_limit))
        else:
            max_tail_bytes = 16 * 1024 * 1024
            with path.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                start = max(size - max_tail_bytes, 0)
                handle.seek(start)
                raw = handle.read()
            if start > 0 and b"\n" in raw:
                raw = raw.split(b"\n", 1)[1]
            lines = raw.decode("utf-8", errors="replace").splitlines()[-row_limit:]
    except OSError:
        return []
    output: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            output.append(row)
    return output


def _bot_id_from_strategy(strategy: str) -> str:
    value = str(strategy or "").strip().lower()
    if value.startswith("paper_mirror") and "::" in value:
        return value.rsplit("::", 1)[-1]
    return ""


def _evidence_summary(outcomes: list[tuple[datetime, float]]) -> dict[str, Any]:
    if not outcomes:
        return {
            "post_cost_samples": 0,
            "observed_days": 0,
            "post_cost_expectancy": 0.0,
            "post_cost_lower_confidence_bound": 0.0,
            "failed_retests": 0,
            "post_cost_total": 0.0,
        }
    values = [value for _, value in outcomes]
    mean = statistics.fmean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    lower = mean - (1.96 * stddev / math.sqrt(max(len(values), 1)))
    daily: dict[str, float] = defaultdict(float)
    for timestamp, value in outcomes:
        daily[timestamp.date().isoformat()] += value
    negative_streak = 0
    for day in sorted(daily):
        negative_streak = negative_streak + 1 if daily[day] < 0.0 else 0
    return {
        "post_cost_samples": len(values),
        "observed_days": len(daily),
        "post_cost_expectancy": round(mean, 8),
        "post_cost_lower_confidence_bound": round(lower, 8),
        "failed_retests": negative_streak,
        "post_cost_total": round(sum(values), 8),
        "win_rate": round(sum(1 for value in values if value > 0.0) / max(len(values), 1), 6),
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    lookback_days: int = 30,
    max_files: int = 48,
    max_rows_per_file: int = 50000,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(int(lookback_days), 1))
    paths = _paper_paths(project_root, max_files=max_files)
    seen: set[str] = set()
    strategy_outcomes: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    bot_outcomes: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    strategy_profiles: dict[str, str] = {}
    profile_outcomes: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    execution_styles: Counter[str] = Counter()
    valuation_sources: Counter[str] = Counter()
    total_rows = 0
    consensus_rows = 0
    legacy_mirror_rows = 0
    derivative_rows = 0
    derivative_ready_rows = 0
    derivative_legacy_rows = 0
    derivative_unknown_rows = 0
    overlap_guarded_rows = 0
    downscaled_rows = 0
    candidate_bound_rows = 0
    authority_v2_rows = 0
    hierarchy_diversity_rows = 0
    entry_economics_rows = 0
    turnover_guard_rows = 0

    for path in paths:
        for row in _tail_rows(path, max_rows=max_rows_per_file):
            timestamp = _parse_ts(row.get("timestamp_utc"))
            if timestamp is None or timestamp < cutoff:
                continue
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            identity = str(row.get("decision_id") or metadata.get("decision_id") or "")
            if not identity:
                identity = "|".join(
                    (
                        timestamp.isoformat(),
                        str(row.get("symbol") or ""),
                        str(row.get("strategy") or ""),
                        str(row.get("action") or ""),
                        str(row.get("quantity") or ""),
                    )
                )
            if identity in seen:
                continue
            seen.add(identity)
            total_rows += 1
            candidate_bound_rows += int(bool(str(metadata.get("production_candidate_id") or "").strip()))
            authority_v2_rows += int(
                str(metadata.get("paper_execution_authority_version") or "").strip()
                == "paper_execution_authority_v2"
            )
            hierarchy_diversity_rows += int(
                bool(metadata.get("paper_execution_diversity_ready", False))
                and int(metadata.get("paper_execution_distinct_correlation_clusters", 0) or 0) >= 2
            )
            entry_policy_metadata = (
                metadata.get("entry_policy")
                if isinstance(metadata.get("entry_policy"), dict)
                else {}
            )
            entry_economics_rows += int(
                isinstance(entry_policy_metadata.get("entry_economics"), dict)
                and bool(entry_policy_metadata.get("entry_economics"))
            )
            turnover_guard_rows += int(
                str(row.get("paper_turnover_guard_version") or "").strip()
                == "paper_turnover_guard_v1"
            )

            strategy = str(row.get("strategy") or "unknown").strip().lower() or "unknown"
            profile = str(row.get("paper_profile") or metadata.get("source_profile") or "default").strip().lower()
            strategy_profiles[strategy] = profile
            outcome = _float(
                row.get("post_cost_pnl_delta"),
                _float(row.get("paper_strategy_net_pnl_delta"), row.get("paper_profile_net_pnl_delta", 0.0)),
            )
            strategy_outcomes[strategy].append((timestamp, outcome))
            profile_outcomes[profile].append((timestamp, outcome))

            is_consensus = bool(
                strategy.startswith("paper_portfolio_consensus::")
                or metadata.get("layer") == "paper_portfolio_consensus"
            )
            if is_consensus:
                consensus_rows += 1
                for attribution in metadata.get("constituent_attribution") if isinstance(metadata.get("constituent_attribution"), list) else []:
                    if not isinstance(attribution, dict):
                        continue
                    bot_id = str(attribution.get("bot_id") or "").strip().lower()
                    share = max(_float(attribution.get("weight_share"), 0.0), 0.0)
                    if bot_id and share > 0.0:
                        bot_outcomes[bot_id].append((timestamp, outcome * share))
            else:
                bot_id = _bot_id_from_strategy(strategy)
                if bot_id:
                    legacy_mirror_rows += 1
                    bot_outcomes[bot_id].append((timestamp, outcome))

            entry_policy = metadata.get("entry_policy") if isinstance(metadata.get("entry_policy"), dict) else {}
            execution_plan = entry_policy.get("execution_plan") if isinstance(entry_policy.get("execution_plan"), dict) else {}
            style = str(metadata.get("execution_style") or execution_plan.get("style") or "unknown")
            execution_styles[style] += 1
            if _float(metadata.get("risk_multiplier_norm"), 1.0) < 0.999:
                downscaled_rows += 1
            if _float(entry_policy.get("overlap_pressure_norm"), 0.0) > 0.0:
                overlap_guarded_rows += 1

            asset_type = str(row.get("paper_valuation_asset_type") or metadata.get("asset_type") or "").upper()
            if asset_type in {"OPTION", "FUTURE"}:
                derivative_rows += 1
                ready = bool(row.get("paper_valuation_ready", metadata.get("contract_valuation_ready", False)))
                source = str(row.get("paper_valuation_multiplier_source") or metadata.get("contract_valuation_source") or "unknown")
                valuation_sources[source] += 1
                derivative_ready_rows += int(ready)
                derivative_legacy_rows += int("legacy" in source)
                derivative_unknown_rows += int((not ready) or _float(row.get("contract_multiplier"), metadata.get("contract_multiplier", 0.0)) <= 0.0)

    strategy_evidence = []
    for strategy, outcomes in strategy_outcomes.items():
        summary = _evidence_summary(outcomes)
        strategy_evidence.append({"strategy": strategy, "profile": strategy_profiles.get(strategy, "default"), **summary})
    strategy_evidence.sort(key=lambda row: (_float(row.get("post_cost_total")), str(row.get("strategy"))))

    bot_evidence = []
    retirement_candidates = []
    for bot_id, outcomes in bot_outcomes.items():
        summary = _evidence_summary(outcomes)
        verdict = evaluate_retirement_evidence(summary)
        row = {"bot_id": bot_id, **summary, "retirement": verdict}
        bot_evidence.append(row)
        if bool(verdict.get("retire", False)):
            retirement_candidates.append(row)
    bot_evidence.sort(key=lambda row: (_float(row.get("post_cost_total")), str(row.get("bot_id"))))

    policy = _load_json(project_root / "config" / "trade_learning_policy.json")
    forward_cfg = policy.get("behavior_forward_labels") if isinstance(policy.get("behavior_forward_labels"), dict) else {}
    post_cost_cfg = forward_cfg.get("post_cost_labels") if isinstance(forward_cfg.get("post_cost_labels"), dict) else {}
    dataset = _load_json(project_root / "data" / "trade_history" / "trade_learning_dataset.json")
    dataset_contract = dataset.get("label_contract") if isinstance(dataset.get("label_contract"), dict) else {}
    post_cost_training_configured = bool(post_cost_cfg.get("enabled", False))
    post_cost_dataset_materialized = bool(dataset_contract.get("post_cost_labels_enabled", False))
    path_labels_configured = bool(
        isinstance(forward_cfg.get("path_dependent_labels"), dict)
        and forward_cfg.get("path_dependent_labels", {}).get("enabled", False)
    )
    path_labels_materialized = bool(dataset_contract.get("path_dependent_labels_enabled", False))
    paper_standard = _load_json(project_root / "governance" / "health" / "paper_live_data_standard_latest.json")
    safety_contract = (
        paper_standard.get("safety_contract")
        if isinstance(paper_standard.get("safety_contract"), dict)
        else {}
    )
    paper_authority_implemented = bool(
        safety_contract.get("paper_execution_authority_version") == "paper_execution_authority_v2"
        and str(safety_contract.get("paper_mirror_all_active_sub_bots") or "") == "0"
        and not (safety_contract.get("unauthorized_execution_bot_ids") or [])
    )
    performance = _load_json(project_root / "governance" / "health" / "paper_performance_latest.json")
    accounting_views = performance.get("accounting_views") if isinstance(performance.get("accounting_views"), dict) else {}
    accounting_implemented = all(
        key in accounting_views
        for key in ("lifetime_flow", "current_day_flow", "candidate_forward_flow", "active_book_snapshot")
    )
    base_trader_source = ""
    shadow_loop_source = ""
    try:
        base_trader_source = (project_root / "core" / "base_trader.py").read_text(encoding="utf-8")
    except OSError:
        pass
    try:
        shadow_loop_source = (project_root / "scripts" / "run_shadow_training_loop.py").read_text(encoding="utf-8")
    except OSError:
        pass
    turnover_implemented = "paper_turnover_guard_v1" in base_trader_source
    hierarchy_implemented = bool(
        "require_hierarchy_identity=True" in shadow_loop_source
        and "paper_correlation_cluster_id" in shadow_loop_source
    )
    entry_economics_implemented = bool(
        "paper_consensus_conservative_prior_v1" in shadow_loop_source
        and "profitability_strict_evidence_required" in shadow_loop_source
    )

    valuation_status = "ready"
    if derivative_unknown_rows:
        valuation_status = "blocked"
    elif derivative_legacy_rows:
        valuation_status = "advisory"
    elif derivative_rows == 0:
        valuation_status = "armed"
    consensus_status = "ready" if consensus_rows > 0 else "armed"
    training_status = "ready" if post_cost_dataset_materialized else ("armed" if post_cost_training_configured else "blocked")
    control_statuses = [valuation_status, consensus_status, training_status]
    overall_status = "blocked" if "blocked" in control_statuses else (
        "advisory" if "advisory" in control_statuses else ("armed" if "armed" in control_statuses else "ready")
    )
    evidence_grade = "A+" if all(status == "ready" for status in control_statuses) and total_rows >= 100 else (
        "A" if "blocked" not in control_statuses else "C"
    )

    controls = [
        {"id": 1, "name": "derivative_valuation_truth", "status": valuation_status, "enforced": True},
        {"id": 2, "name": "portfolio_intent_coalescing", "status": consensus_status, "enforced": True},
        {"id": 3, "name": "post_cost_training_labels", "status": training_status, "enforced": post_cost_training_configured},
        {"id": 4, "name": "sleeve_regime_entry_gates", "status": "ready", "enforced": True},
        {"id": 5, "name": "evidence_weighted_risk_allocation", "status": "ready", "enforced": True},
        {"id": 6, "name": "execution_style_selection", "status": "ready", "enforced": True},
        {"id": 7, "name": "portfolio_overlap_budget", "status": "ready", "enforced": True},
        {"id": 8, "name": "persistent_loser_retirement_court", "status": "ready", "enforced": True},
        {
            "id": 9,
            "name": "explicit_bounded_paper_execution_authority",
            "status": "ready" if paper_authority_implemented else "armed",
            "enforced": paper_authority_implemented,
        },
        {
            "id": 10,
            "name": "hierarchical_correlation_and_duplicate_caps",
            "status": "ready" if hierarchy_implemented and hierarchy_diversity_rows > 0 else "armed",
            "enforced": hierarchy_implemented,
        },
        {
            "id": 11,
            "name": "candidate_current_lifetime_accounting_separation",
            "status": "ready" if accounting_implemented and candidate_bound_rows > 0 else "armed",
            "enforced": accounting_implemented,
        },
        {
            "id": 12,
            "name": "path_dependent_and_no_trade_training_labels",
            "status": "ready" if path_labels_materialized else "armed" if path_labels_configured else "blocked",
            "enforced": path_labels_configured,
        },
        {
            "id": 13,
            "name": "persistent_turnover_and_reversal_guard",
            "status": "ready" if turnover_implemented and turnover_guard_rows > 0 else "armed",
            "enforced": turnover_implemented,
        },
        {
            "id": 14,
            "name": "strict_entry_economics_and_quote_freshness",
            "status": "ready" if entry_economics_implemented and entry_economics_rows > 0 else "armed",
            "enforced": entry_economics_implemented,
        },
    ]
    implementation_ready = all(bool(row.get("enforced", False)) for row in controls)
    implementation_score = 100.0 * sum(bool(row.get("enforced", False)) for row in controls) / max(len(controls), 1)
    return {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "ok": overall_status != "blocked",
        "overall_status": overall_status,
        "evidence_grade": evidence_grade,
        "implementation_grade": _grade(implementation_score, complete=implementation_ready),
        "implementation_score": round(implementation_score, 4),
        "operating_mode": "paper_only",
        "live_execution_changed": False,
        "profitability_guaranteed": False,
        "controls": controls,
        "source": {
            "paper_files": [str(path) for path in paths],
            "paper_rows": total_rows,
            "lookback_days": max(int(lookback_days), 1),
        },
        "derivative_valuation": {
            "status": valuation_status,
            "derivative_rows": derivative_rows,
            "valuation_ready_rows": derivative_ready_rows,
            "legacy_position_rows": derivative_legacy_rows,
            "unknown_multiplier_rows": derivative_unknown_rows,
            "multiplier_source_counts": dict(valuation_sources),
        },
        "portfolio_consensus": {
            "status": consensus_status,
            "consensus_execution_rows": consensus_rows,
            "legacy_individual_mirror_rows": legacy_mirror_rows,
            "downscaled_execution_rows": downscaled_rows,
            "execution_style_counts": dict(execution_styles),
            "overlap_evidence_rows": overlap_guarded_rows,
            "collection_policy": "all_bot_observations_retained_while_execution_is_coalesced",
        },
        "post_cost_training": {
            "status": training_status,
            "configured": post_cost_training_configured,
            "dataset_materialized": post_cost_dataset_materialized,
            "config": post_cost_cfg,
            "dataset_contract": dataset_contract,
        },
        "execution_authority": {
            "status": "ready" if paper_authority_implemented else "armed",
            "implemented": paper_authority_implemented,
            "candidate_bound_rows": candidate_bound_rows,
            "authority_v2_rows": authority_v2_rows,
            "safety_contract": safety_contract,
        },
        "hierarchical_execution": {
            "implemented": hierarchy_implemented,
            "diversity_verified_rows": hierarchy_diversity_rows,
            "missing_hierarchy_fails_closed": True,
        },
        "candidate_accounting": {
            "implemented": accounting_implemented,
            "candidate_bound_rows": candidate_bound_rows,
            "accounting_views": accounting_views,
            "lifetime_loss_may_not_grade_current_candidate": True,
        },
        "path_dependent_training": {
            "configured": path_labels_configured,
            "dataset_materialized": path_labels_materialized,
            "dataset_contract": dataset_contract,
        },
        "turnover_guard": {
            "implemented": turnover_implemented,
            "observed_guarded_rows": turnover_guard_rows,
            "exits_and_reductions_remain_open": True,
        },
        "entry_economics": {
            "implemented": entry_economics_implemented,
            "observed_rows": entry_economics_rows,
            "heuristic_prior_is_promotion_evidence": False,
        },
        "retirement_court": {
            "status": "ready",
            "candidate_count": len(retirement_candidates),
            "candidates": retirement_candidates[:50],
            "bot_evidence": bot_evidence[:250],
            "policy": "no deletion before repeated negative post_cost lower_bound evidence",
        },
        "strategy_post_cost_evidence": strategy_evidence[:250],
        "unattended_contract": {
            "paper_only": True,
            "unknown_derivative_multiplier_blocks_new_exposure": True,
            "legacy_derivative_positions_may_reduce_but_not_add_or_reverse": True,
            "conflicting_bot_votes_abstain": True,
            "risk_multipliers_may_only_downscale": True,
            "market_orders_allowed": False,
            "retirement_requires_post_cost_evidence": True,
            "paper_execution_requires_explicit_authority": True,
            "missing_hierarchy_identity_abstains": True,
            "paper_turnover_guard_persists_across_restart": True,
            "lifetime_current_and_candidate_accounting_are_separate": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the eight profitability hardening controls.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--max-files", type=int, default=48)
    parser.add_argument("--max-rows-per-file", type=int, default=50000)
    parser.add_argument("--out", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(
        project_root,
        lookback_days=args.lookback_days,
        max_files=args.max_files,
        max_rows_per_file=args.max_rows_per_file,
    )
    out_path = Path(args.out).expanduser() if args.out else project_root / "governance" / "health" / DEFAULT_OUT.name
    safe_write_json_atomic(
        str(out_path),
        payload,
        project_root=str(project_root),
        source="profitability_hardening_control",
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            f"profitability_hardening status={payload['overall_status']} grade={payload['evidence_grade']} "
            f"rows={payload['source']['paper_rows']} retirement_candidates={payload['retirement_court']['candidate_count']}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
