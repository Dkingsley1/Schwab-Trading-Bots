#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, load_json, write_payload


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "alpha_measurement_materialization_v1.json"
)
EXPECTED_DATASET_SCHEMA = (
    "behavior_dataset_v8_candidate_bound_multi_horizon_counterfactuals"
)
MEASUREMENT_IDS = (
    "information_coefficient_term_structure",
    "effective_breadth_transfer_coefficient",
    "hierarchical_bayesian_skill",
    "subsample_stability_selection",
    "economic_alpha_decomposition",
    "factor_neutral_residualization",
    "cross_fitted_causal_transportability",
    "capacity_impact_surface",
    "execution_alpha_attribution",
    "point_in_time_security_master_audit",
    "split_conformal_residual_calibration",
    "sequential_change_point_stability",
    "residual_redundancy_graph",
    "regime_conditional_robustness",
    "cost_stress_survival",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _utc(value: Any) -> datetime | None:
    raw = str(value or "").strip().replace("Z", "+00:00")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def _rank_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _correlation(_ranks(left), _ranks(right))


def _candidate_context(
    root: Path, policy: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidate_path = root / str(
        policy.get("candidate_state_path")
        or "governance/runtime/production_candidate_state.json"
    )
    performance_path = root / str(
        policy.get("paper_performance_path")
        or "governance/health/paper_performance_latest.json"
    )
    candidate = load_json(candidate_path)
    performance = load_json(performance_path)
    window = _as_dict(performance.get("profitability_evidence_window"))
    state_id = str(candidate.get("candidate_id") or "").strip()
    window_id = str(window.get("candidate_id") or "").strip()
    cutoff = _utc(
        window.get("candidate_cutoff_utc") or candidate.get("accepted_at_utc")
    )
    blockers: list[str] = []
    if not state_id:
        blockers.append("candidate_state_id_missing")
    if not window_id:
        blockers.append("paper_performance_candidate_id_missing")
    if state_id and window_id and state_id != window_id:
        blockers.append("candidate_identity_mismatch")
    if cutoff is None:
        blockers.append("candidate_cutoff_missing")
    if not bool(window.get("candidate_filter_active", False)):
        blockers.append("paper_performance_candidate_filter_inactive")
    if not bool(window.get("candidate_binding_required", False)):
        blockers.append("paper_performance_candidate_binding_not_required")
    mismatch_count = max(
        int(window.get("candidate_binding_mismatch_rows_excluded") or 0), 0
    )
    if mismatch_count:
        blockers.append("paper_performance_candidate_mismatch_rows_present")
    context = {
        "candidate_id": state_id or window_id,
        "candidate_generation": int(
            candidate.get("generation") or window.get("candidate_generation") or 0
        ),
        "candidate_cutoff_utc": cutoff.isoformat() if cutoff else "",
        "paper_evidence_through_utc": str(window.get("evidence_through_utc") or ""),
        "bound": not blockers,
        "blockers": blockers,
        "historical_fallback_used": False,
        "cross_candidate_pooling_used": False,
        "candidate_identity_rewritten": False,
    }
    receipts = {
        "candidate_state_path": str(candidate_path),
        "candidate_state_sha256": _sha256(candidate_path),
        "paper_performance_path": str(performance_path),
        "paper_performance_sha256": _sha256(performance_path),
    }
    return context, performance, receipts


def _row_context(
    row: Mapping[str, Any], feature_names: Sequence[str]
) -> dict[str, float]:
    context = {
        str(key): value
        for key, raw in _as_dict(row.get("measurement_context")).items()
        if (value := _number(raw)) is not None
    }
    vector = _as_list(row.get("features"))
    if len(vector) == len(feature_names):
        for index, name in enumerate(feature_names):
            if name in context:
                continue
            value = _number(vector[index])
            if value is not None:
                context[str(name)] = value
    return context


def _filter_dataset_rows(
    payload: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    binding_policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_schema = str(
        binding_policy.get("require_dataset_schema") or EXPECTED_DATASET_SCHEMA
    )
    observed_schema = str(payload.get("schema") or "")
    feature_names = [str(item) for item in _as_list(payload.get("feature_names"))]
    candidate_id = str(candidate.get("candidate_id") or "")
    cutoff = _utc(candidate.get("candidate_cutoff_utc"))
    minimum_schema = max(int(binding_policy.get("minimum_log_schema_version") or 2), 1)
    require_schema_valid = bool(binding_policy.get("require_row_schema_valid", True))
    require_binding = bool(binding_policy.get("require_row_candidate_binding", True))
    exclusions: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    for raw in _as_list(payload.get("data")):
        if not isinstance(raw, Mapping):
            exclusions["malformed_row"] += 1
            continue
        row = dict(raw)
        timestamp = _utc(row.get("timestamp_utc"))
        observed_id = str(row.get("production_candidate_id") or "").strip()
        binding = _as_dict(row.get("candidate_binding"))
        if observed_id != candidate_id:
            exclusions["candidate_identity_mismatch"] += 1
            continue
        if cutoff is None or timestamp is None or timestamp < cutoff:
            exclusions["pre_cutoff_or_invalid_timestamp"] += 1
            continue
        if int(row.get("log_schema_version") or 0) < minimum_schema:
            exclusions["log_schema_below_floor"] += 1
            continue
        if require_schema_valid and not bool(row.get("schema_valid", False)):
            exclusions["row_schema_invalid"] += 1
            continue
        if require_binding and not (
            bool(binding.get("candidate_bound", False))
            and str(binding.get("observed_candidate_id") or observed_id) == candidate_id
        ):
            exclusions["candidate_binding_receipt_missing_or_invalid"] += 1
            continue
        row["_timestamp"] = timestamp
        row["_context"] = _row_context(row, feature_names)
        kept.append(row)
    kept.sort(key=lambda row: row["_timestamp"])
    validation = {
        "dataset_schema_expected": expected_schema,
        "dataset_schema_observed": observed_schema,
        "dataset_schema_valid": observed_schema == expected_schema,
        "input_row_count": len(_as_list(payload.get("data"))),
        "candidate_row_count": len(kept),
        "excluded_row_count": sum(exclusions.values()),
        "exclusions": dict(sorted(exclusions.items())),
        "minimum_log_schema_version": minimum_schema,
        "all_kept_rows_schema_v2_or_newer": all(
            int(row.get("log_schema_version") or 0) >= minimum_schema for row in kept
        ),
        "historical_rows_relabelled": 0,
        "cross_candidate_rows_pooled": 0,
    }
    if observed_schema != expected_schema:
        kept = []
        validation["candidate_row_count"] = 0
        validation["dataset_blocker"] = "behavior_dataset_schema_upgrade_required"
    return kept, validation


def _horizon_label(seconds: int) -> str:
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def _forecast_observations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for row in rows:
        forecast = _number(row.get("signed_forecast_score"))
        timestamp = row.get("_timestamp")
        if forecast is None or not isinstance(timestamp, datetime):
            continue
        period = timestamp.strftime("%Y-%m-%dT%H:00:00Z")
        common = {
            "decision_id": str(row.get("decision_id") or row.get("id") or ""),
            "forecast": forecast,
            "period": period,
            "regime": str(row.get("regime") or "unspecified"),
            "sleeve": str(row.get("sleeve_id") or row.get("profile") or "unspecified"),
            "strategy_id": str(
                row.get("selected_strategy_id")
                or row.get("source_strategy")
                or "unspecified"
            ),
            "symbol": str(row.get("symbol") or ""),
            "evidence_class": "candidate_bound_counterfactual_forecast",
        }
        horizon_outcomes: dict[str, float] = {}
        primary = _number(row.get("forward_return_primary"))
        primary_seconds = max(int(row.get("horizon_seconds") or 0), 0)
        if primary is not None and primary_seconds:
            horizon_outcomes[_horizon_label(primary_seconds)] = primary
        auxiliary = _number(row.get("forward_return_aux"))
        auxiliary_seconds = max(int(row.get("aux_horizon_seconds") or 0), 0)
        if auxiliary is not None and auxiliary_seconds:
            horizon_outcomes[_horizon_label(auxiliary_seconds)] = auxiliary
        for horizon, outcome in sorted(
            _as_dict(row.get("counterfactual_action_outcomes")).items()
        ):
            realized = _number(_as_dict(outcome).get("raw_market_return"))
            if realized is not None:
                horizon_outcomes.setdefault(str(horizon), realized)
        observations.extend(
            {**common, "realized": realized, "horizon": horizon}
            for horizon, realized in sorted(horizon_outcomes.items())
        )
    return observations


def _trade_evidence_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = 0
    valid: list[dict[str, Any]] = []
    invalid: Counter[str] = Counter()
    for raw in rows:
        action = str(raw.get("action") or "HOLD").upper()
        if action not in {"BUY", "SELL"}:
            continue
        eligible += 1
        if not bool(raw.get("post_cost_label", False)):
            invalid["post_cost_label_missing"] += 1
            continue
        gross = _number(raw.get("gross_directional_forward_return"))
        cost_bps = _number(raw.get("round_trip_cost_bps"))
        post_cost = _number(raw.get("post_cost_forward_return"))
        if gross is None or cost_bps is None or post_cost is None:
            invalid["gross_cost_or_post_cost_missing"] += 1
            continue
        expected = gross - max(cost_bps, 0.0) / 10_000.0
        if abs(post_cost - expected) > 1e-7:
            invalid["post_cost_additive_identity_failed"] += 1
            continue
        row = dict(raw)
        row["_gross_directional_return"] = gross
        row["_execution_cost_return"] = max(cost_bps, 0.0) / 10_000.0
        row["_post_cost_return"] = post_cost
        valid.append(row)
    return valid, {
        "eligible_trade_decision_count": eligible,
        "valid_schema_v2_post_cost_trade_delta_count": len(valid),
        "invalid_post_cost_trade_delta_count": sum(invalid.values()),
        "invalid_reasons": dict(sorted(invalid.items())),
        "additive_identity": "post_cost_return=gross_directional_return-round_trip_cost_return",
        "hold_rows_count_as_realized_trade_pnl": False,
    }


def _period_breadth_matrices(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cells: dict[str, dict[str, list[tuple[float, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    coverage: Counter[str] = Counter()
    for row in rows:
        forecast = _number(row.get("signed_forecast_score"))
        realized = _number(row.get("forward_return_primary"))
        timestamp = row.get("_timestamp")
        symbol = str(row.get("symbol") or "").strip().upper()
        if (
            forecast is None
            or realized is None
            or not symbol
            or not isinstance(timestamp, datetime)
        ):
            continue
        period = timestamp.strftime("%Y-%m-%dT%H:00:00Z")
        action = str(row.get("action") or "HOLD").upper()
        weight = 1.0 if action == "BUY" else -1.0 if action == "SELL" else 0.0
        cells[period][symbol].append((forecast, realized, weight))
    for period_cells in cells.values():
        coverage.update(period_cells.keys())
    symbols = [symbol for symbol, _count in coverage.most_common(16)]
    if len(symbols) < 2:
        return {
            "forecast_matrix": [],
            "realized_matrix": [],
            "implemented_weight_matrix": [],
        }
    forecasts: list[list[float]] = []
    realized: list[list[float]] = []
    weights: list[list[float]] = []
    for period in sorted(cells):
        if not all(symbol in cells[period] for symbol in symbols):
            continue
        forecasts.append(
            [
                statistics.fmean(item[0] for item in cells[period][symbol])
                for symbol in symbols
            ]
        )
        realized.append(
            [
                statistics.fmean(item[1] for item in cells[period][symbol])
                for symbol in symbols
            ]
        )
        weights.append(
            [
                statistics.fmean(item[2] for item in cells[period][symbol])
                for symbol in symbols
            ]
        )
    return {
        "forecast_matrix": forecasts,
        "realized_matrix": realized,
        "implemented_weight_matrix": weights,
    }


def _aligned_context(
    rows: Sequence[Mapping[str, Any]], factor_keys: Sequence[str], *, outcome_key: str
) -> tuple[list[float], dict[str, list[float]], list[Mapping[str, Any]]]:
    selected: list[Mapping[str, Any]] = []
    outcomes: list[float] = []
    factors = {str(key): [] for key in factor_keys}
    for row in rows:
        outcome = _number(row.get(outcome_key))
        context = _as_dict(row.get("_context"))
        values = {str(key): _number(context.get(str(key))) for key in factor_keys}
        if (
            outcome is None
            or not values
            or any(value is None for value in values.values())
        ):
            continue
        selected.append(row)
        outcomes.append(outcome)
        for key, value in values.items():
            factors[key].append(float(value))
    return outcomes, factors, selected


def _prequential_predictions(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[float], list[float]]:
    history_forecasts: list[float] = []
    history_outcomes: list[float] = []
    predictions: list[float] = []
    outcomes: list[float] = []
    for row in rows:
        forecast = _number(row.get("signed_forecast_score"))
        outcome = _number(row.get("forward_return_primary"))
        if forecast is None or outcome is None:
            continue
        if len(history_forecasts) >= 10:
            denominator = sum(value * value for value in history_forecasts)
            beta = (
                sum(
                    left * right
                    for left, right in zip(history_forecasts, history_outcomes)
                )
                / denominator
                if denominator > 1e-12
                else 0.0
            )
            predictions.append(beta * forecast)
            outcomes.append(outcome)
        history_forecasts.append(forecast)
        history_outcomes.append(outcome)
    return predictions, outcomes


def _mlx_available() -> bool:
    try:
        import mlx.core as _mx  # noqa: F401
    except Exception:
        return False
    return True


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    penalty: float,
    backend: str,
) -> tuple[np.ndarray, str]:
    train_design = np.column_stack([np.ones(len(train_x)), train_x])
    test_design = np.column_stack([np.ones(len(test_x)), test_x])
    regularizer = np.eye(train_design.shape[1], dtype=float) * max(penalty, 1e-9)
    regularizer[0, 0] = 0.0
    if backend == "mlx":
        try:
            import mlx.core as mx

            design = mx.array(train_design, dtype=mx.float32)
            target = mx.array(train_y, dtype=mx.float32)
            test = mx.array(test_design, dtype=mx.float32)
            reg = mx.array(regularizer, dtype=mx.float32)
            # MLX currently requires solve on its CPU stream. Keeping the
            # operation explicit avoids a hidden NumPy fallback while the
            # surrounding fold preparation remains batched and deterministic.
            coefficients = mx.linalg.solve(
                design.T @ design + reg,
                design.T @ target,
                stream=mx.cpu,
            )
            prediction = test @ coefficients
            mx.eval(prediction)
            return np.asarray(prediction, dtype=float), "mlx_cpu"
        except Exception:
            pass
    coefficients = (
        np.linalg.pinv(train_design.T @ train_design + regularizer)
        @ train_design.T
        @ train_y
    )
    return test_design @ coefficients, "numpy"


def purged_walk_forward_diagnostic(
    rows: Sequence[Mapping[str, Any]],
    *,
    factor_keys: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    minimum_rows = max(int(config.get("minimum_rows") or 30), 8)
    maximum_features = max(int(config.get("maximum_features") or 16), 1)
    available_keys = [
        key
        for key in factor_keys
        if sum(
            _number(_as_dict(row.get("_context")).get(key)) is not None for row in rows
        )
        >= minimum_rows
    ][:maximum_features]
    prepared: list[tuple[datetime, list[float], float]] = []
    for row in rows:
        timestamp = row.get("_timestamp")
        outcome = _number(row.get("forward_return_primary"))
        context = _as_dict(row.get("_context"))
        values = [_number(context.get(key)) for key in available_keys]
        if (
            isinstance(timestamp, datetime)
            and outcome is not None
            and values
            and all(value is not None for value in values)
        ):
            prepared.append(
                (
                    timestamp,
                    [float(value) for value in values if value is not None],
                    outcome,
                )
            )
    prepared.sort(key=lambda item: item[0])
    requested_backend = str(config.get("backend") or "auto").lower()
    backend = (
        "mlx" if requested_backend in {"auto", "mlx"} and _mlx_available() else "numpy"
    )
    if len(prepared) < minimum_rows or not available_keys:
        return {
            "status": "insufficient_evidence",
            "available": False,
            "row_count": len(prepared),
            "minimum_rows": minimum_rows,
            "feature_count": len(available_keys),
            "requested_backend": requested_backend,
            "resolved_backend": backend,
            "mlx_available": _mlx_available(),
            "purged": True,
            "embargoed": True,
            "changes_runtime_decisions": False,
        }
    timestamps = [item[0] for item in prepared]
    x = np.asarray([item[1] for item in prepared], dtype=float)
    y = np.asarray([item[2] for item in prepared], dtype=float)
    minimum_train = max(
        int(config.get("minimum_train_rows") or 20), len(available_keys) + 3
    )
    minimum_test = max(int(config.get("minimum_test_rows") or 5), 3)
    fold_count = max(int(config.get("fold_count") or 5), 1)
    purge_seconds = max(int(config.get("purge_seconds") or 86400), 0)
    embargo_seconds = max(int(config.get("embargo_seconds") or 3600), 0)
    penalty = max(float(config.get("ridge_penalty") or 0.001), 1e-9)
    candidate_indices = np.arange(minimum_train, len(prepared))
    folds: list[dict[str, Any]] = []
    previous_test_end: datetime | None = None
    for raw_indices in np.array_split(candidate_indices, fold_count):
        if len(raw_indices) < minimum_test:
            continue
        test_indices = [int(index) for index in raw_indices]
        test_start = timestamps[test_indices[0]]
        if (
            previous_test_end is not None
            and test_start < previous_test_end + timedelta(seconds=embargo_seconds)
        ):
            test_indices = [
                index
                for index in test_indices
                if timestamps[index]
                >= previous_test_end + timedelta(seconds=embargo_seconds)
            ]
        if len(test_indices) < minimum_test:
            continue
        test_start = timestamps[test_indices[0]]
        train_cutoff = test_start - timedelta(seconds=purge_seconds)
        train_indices = [
            index
            for index in range(test_indices[0])
            if timestamps[index] <= train_cutoff
        ]
        if len(train_indices) < minimum_train:
            continue
        train_x = x[train_indices]
        train_y = y[train_indices]
        test_x = x[test_indices]
        test_y = y[test_indices]
        mean = np.mean(train_x, axis=0)
        scale = np.std(train_x, axis=0)
        scale[scale <= 1e-12] = 1.0
        prediction, used_backend = _ridge_predict(
            (train_x - mean) / scale,
            train_y,
            (test_x - mean) / scale,
            penalty=penalty,
            backend=backend,
        )
        pearson = _correlation(prediction.tolist(), test_y.tolist())
        rank_ic = _rank_correlation(prediction.tolist(), test_y.tolist())
        folds.append(
            {
                "fold": len(folds) + 1,
                "train_rows": len(train_indices),
                "test_rows": len(test_indices),
                "train_through_utc": timestamps[train_indices[-1]].isoformat(),
                "test_from_utc": timestamps[test_indices[0]].isoformat(),
                "test_through_utc": timestamps[test_indices[-1]].isoformat(),
                "purge_gap_seconds": (
                    timestamps[test_indices[0]] - timestamps[train_indices[-1]]
                ).total_seconds(),
                "pearson_ic": round(pearson, 8) if pearson is not None else None,
                "rank_ic": round(rank_ic, 8) if rank_ic is not None else None,
                "mean_squared_error": round(
                    float(np.mean((prediction - test_y) ** 2)), 12
                ),
                "backend": used_backend,
            }
        )
        previous_test_end = timestamps[test_indices[-1]]
    rank_values = [float(row["rank_ic"]) for row in folds if row["rank_ic"] is not None]
    available = bool(folds)
    used_backends = sorted({str(row.get("backend") or "unknown") for row in folds})
    resolved_backend = (
        used_backends[0]
        if len(used_backends) == 1
        else "+".join(used_backends) if used_backends else backend
    )
    return {
        "status": "measured" if available else "insufficient_evidence",
        "available": available,
        "passes": bool(rank_values and statistics.fmean(rank_values) > 0.0),
        "row_count": len(prepared),
        "feature_names": available_keys,
        "feature_count": len(available_keys),
        "fold_count": len(folds),
        "requested_backend": requested_backend,
        "resolved_backend": resolved_backend,
        "mlx_available": _mlx_available(),
        "purge_seconds": purge_seconds,
        "embargo_seconds": embargo_seconds,
        "purged": True,
        "embargoed": True,
        "mean_out_of_sample_rank_ic": (
            round(statistics.fmean(rank_values), 8) if rank_values else None
        ),
        "folds": folds,
        "candidate_bound": True,
        "promotion_authority": False,
        "changes_runtime_decisions": False,
    }


def _iter_jsonl(
    paths: Iterable[Path], *, tail_bytes: int = 16 * 1024 * 1024
) -> Iterable[dict[str, Any]]:
    for path in paths:
        try:
            with path.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(size - tail_bytes, 0))
                if size > tail_bytes:
                    handle.readline()
                for raw in handle:
                    try:
                        row = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if isinstance(row, dict):
                        yield row
        except OSError:
            continue


def _candidate_fills(
    root: Path, policy: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths: dict[str, Path] = {}
    for pattern in _as_list(policy.get("paper_fill_globs")):
        for raw in glob.glob(str(root / str(pattern)), recursive=True):
            path = Path(raw)
            if path.is_file():
                paths[str(path.resolve())] = path.resolve()
    candidate_id = str(candidate.get("candidate_id") or "")
    cutoff = _utc(candidate.get("candidate_cutoff_utc"))
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for row in _iter_jsonl(paths.values()):
        metadata = _as_dict(row.get("metadata"))
        observed_id = str(
            row.get("production_candidate_id")
            or metadata.get("production_candidate_id")
            or ""
        )
        timestamp = _utc(row.get("timestamp_utc"))
        if observed_id != candidate_id:
            rejected["candidate_mismatch"] += 1
            continue
        if cutoff is None or timestamp is None or timestamp < cutoff:
            rejected["pre_cutoff_or_invalid_timestamp"] += 1
            continue
        if int(row.get("log_schema_version") or 0) < 2 or not bool(
            row.get("schema_valid", True)
        ):
            rejected["schema_v2_invalid"] += 1
            continue
        execution = _as_dict(row.get("execution"))
        prices = _as_dict(row.get("prices"))
        normalized = {
            "side": row.get("side") or row.get("action") or row.get("instruction"),
            "quantity": row.get("quantity")
            or row.get("filled_quantity")
            or execution.get("filled_quantity"),
            "decision_price": row.get("decision_price")
            or prices.get("decision")
            or execution.get("decision_price"),
            "arrival_price": row.get("arrival_price")
            or prices.get("arrival")
            or execution.get("arrival_price"),
            "fill_price": row.get("fill_price")
            or row.get("price")
            or prices.get("fill")
            or execution.get("fill_price"),
            "fill_mid_price": row.get("fill_mid_price")
            or prices.get("mid")
            or execution.get("fill_mid_price"),
            "markout_price": row.get("markout_price")
            or prices.get("markout")
            or execution.get("markout_price"),
            "fee_bps": row.get("fee_bps") or execution.get("fee_bps"),
            "fee_amount": row.get("fee_amount") or execution.get("fee_amount"),
        }
        required = ("quantity", "decision_price", "arrival_price", "fill_price")
        if not str(normalized["side"] or "").strip() or any(
            _number(normalized[key]) is None for key in required
        ):
            rejected["execution_attribution_fields_missing"] += 1
            continue
        accepted.append(normalized)
    return accepted, {
        "source_file_count": len(paths),
        "candidate_schema_v2_fill_count": len(accepted),
        "rejected_fill_row_count": sum(rejected.values()),
        "rejected_reasons": dict(sorted(rejected.items())),
    }


def _measurement(
    inputs: Mapping[str, Any],
    *,
    evidence_class: str,
    economic_grade_eligible: bool,
    source: str,
) -> dict[str, Any]:
    return {
        "inputs": dict(inputs),
        "evidence_class": evidence_class,
        "economic_grade_eligible": bool(economic_grade_eligible),
        "source": source,
        "candidate_bound": True,
        "historical_fallback_used": False,
        "cross_candidate_pooling_used": False,
    }


def _research_routing(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        forecast = _number(row.get("signed_forecast_score"))
        realized = _number(row.get("forward_return_primary"))
        if forecast is None or realized is None:
            continue
        sleeve = str(row.get("sleeve_id") or row.get("profile") or "unspecified")
        regime = str(row.get("regime") or "unspecified")
        grouped[(sleeve, regime)].append((forecast, realized))
    floor = max(int(policy.get("minimum_observations_per_sleeve_regime") or 20), 4)
    minimum_ic = float(policy.get("minimum_rank_ic") or 0.0)
    routes: list[dict[str, Any]] = []
    for (sleeve, regime), values in sorted(grouped.items()):
        ic = _rank_correlation(
            [value[0] for value in values], [value[1] for value in values]
        )
        ready = len(values) >= floor and ic is not None
        routes.append(
            {
                "sleeve": sleeve,
                "regime": regime,
                "observation_count": len(values),
                "rank_ic": round(ic, 8) if ic is not None else None,
                "research_route": (
                    "retain_for_candidate_research"
                    if ready and ic >= minimum_ic
                    else "reduce_research_priority" if ready else "collect"
                ),
                "runtime_weight_change": 0.0,
            }
        )
    return {
        "routes": routes,
        "minimum_observations_per_sleeve_regime": floor,
        "eligible_research_route_count": sum(
            row["research_route"] == "retain_for_candidate_research" for row in routes
        ),
        "reduced_research_priority_count": sum(
            row["research_route"] == "reduce_research_priority" for row in routes
        ),
        "runtime_weight_mutation_allowed": False,
        "automatic_retirement_allowed": False,
        "live_execution_authority": False,
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    policy_path = (
        config_path or root / "config" / "alpha_measurement_materialization_v1.json"
    )
    policy = load_json(policy_path)
    authority = _as_dict(policy.get("authority"))
    if not authority or any(bool(value) for value in authority.values()):
        raise ValueError(
            "alpha materializer must have a complete zero-authority contract"
        )
    binding_policy = _as_dict(policy.get("candidate_binding"))
    if bool(binding_policy.get("allow_historical_fallback", True)) or bool(
        binding_policy.get("allow_cross_candidate_pooling", True)
    ):
        raise ValueError(
            "historical fallback and cross-candidate pooling are forbidden"
        )
    candidate, performance, receipts = _candidate_context(root, policy)
    dataset_path = root / str(
        policy.get("behavior_dataset_path")
        or "data/trade_history/trade_learning_dataset.json"
    )
    dataset = load_json(dataset_path)
    rows, schema_validation = _filter_dataset_rows(
        dataset,
        candidate=candidate,
        binding_policy=binding_policy,
    )
    forecast_observations = _forecast_observations(rows)
    trade_rows, post_cost_validation = _trade_evidence_rows(rows)
    factor_keys = [str(item) for item in _as_list(policy.get("factor_context_keys"))]
    stability_outcomes, stability_features, _stability_rows = _aligned_context(
        rows, factor_keys, outcome_key="forward_return_primary"
    )
    trade_outcomes, trade_factors, aligned_trade_rows = _aligned_context(
        trade_rows, factor_keys, outcome_key="_post_cost_return"
    )
    gross_returns = [
        float(row["_gross_directional_return"]) for row in aligned_trade_rows
    ]
    execution_costs = [
        float(row["_execution_cost_return"]) for row in aligned_trade_rows
    ]
    returns_by_group: dict[str, list[float]] = defaultdict(list)
    for row in trade_rows:
        group = str(
            row.get("selected_strategy_id")
            or row.get("sleeve_id")
            or row.get("profile")
            or "unspecified"
        )
        returns_by_group[group].append(float(row["_post_cost_return"]))
    predictions, conformal_outcomes = _prequential_predictions(rows)
    fills, fill_validation = _candidate_fills(root, policy, candidate)
    security_path = root / str(
        policy.get("security_master_path")
        or "governance/research/point_in_time_security_master_latest.json"
    )
    security_payload = load_json(security_path)
    security_records = _as_list(
        security_payload.get("security_records") or security_payload.get("records")
    )
    security_observations = (
        [
            {"timestamp_utc": row.get("timestamp_utc"), "symbol": row.get("symbol")}
            for row in rows
        ]
        if security_records
        else []
    )

    capacity_contexts = [_as_dict(row.get("_context")) for row in trade_rows]
    adv_values = [
        value
        for context in capacity_contexts
        if (value := _number(context.get("daily_dollar_volume"))) is not None
        and value > 0.0
    ]
    notional_values = [
        price * quantity
        for row in trade_rows
        if (price := _number(_as_dict(row.get("_context")).get("last_price")))
        is not None
        and (quantity := _number(row.get("quantity"))) is not None
        and price > 0.0
        and quantity > 0.0
    ]
    gross_bps = (
        statistics.fmean(float(row["_gross_directional_return"]) for row in trade_rows)
        * 10_000.0
        if trade_rows
        else None
    )
    cost_bps = (
        statistics.fmean(
            float(row.get("round_trip_cost_bps") or 0.0) for row in trade_rows
        )
        if trade_rows
        else None
    )
    spread_values = [
        value
        for context in capacity_contexts
        if (value := _number(context.get("spread_bps"))) is not None
    ]
    fee_values = [
        value
        for context in capacity_contexts
        if (value := _number(context.get("lag_fee_bps"))) is not None
    ]
    slippage_values = [
        value
        for context in capacity_contexts
        if (
            value := _number(
                context.get("expected_slippage_bps")
                if context.get("expected_slippage_bps") is not None
                else context.get("lag_slippage_bps")
            )
        )
        is not None
    ]
    impact_coefficients = [
        value
        for context in capacity_contexts
        if (value := _number(context.get("market_impact_coefficient"))) is not None
    ]
    volatility_values = [
        abs(value) * 10_000.0
        for context in capacity_contexts
        if (value := _number(context.get("vol_30m"))) is not None
    ]
    median_notional = statistics.median(notional_values) if notional_values else None
    notional_multipliers = [
        value
        for raw in _as_list(
            _as_dict(policy.get("capacity")).get("tested_notional_multipliers")
        )
        if (value := _number(raw)) is not None and value > 0.0
    ]
    capacity_inputs = {
        "expected_gross_alpha_bps": gross_bps,
        "half_spread_bps": (
            statistics.fmean(spread_values) / 2.0 if spread_values else None
        ),
        "fees_bps": statistics.fmean(fee_values) if fee_values else None,
        "baseline_slippage_bps": (
            statistics.fmean(slippage_values) if slippage_values else None
        ),
        "daily_dollar_volume": statistics.fmean(adv_values) if adv_values else None,
        "volatility_bps": (
            statistics.fmean(volatility_values) if volatility_values else None
        ),
        "impact_coefficient": (
            statistics.fmean(impact_coefficients) if impact_coefficients else None
        ),
        "notionals": (
            [median_notional * value for value in notional_multipliers]
            if median_notional is not None
            else []
        ),
    }
    uncertainty_bps = (
        statistics.stdev(
            [float(row["_gross_directional_return"]) * 10_000.0 for row in trade_rows]
        )
        / math.sqrt(len(trade_rows))
        if len(trade_rows) >= 2
        else None
    )
    causal_enabled = bool(
        _as_dict(policy.get("causal_transport")).get("enabled", False)
    )
    measurements = {
        "information_coefficient_term_structure": _measurement(
            {"observations": forecast_observations},
            evidence_class="candidate_bound_counterfactual_forecast",
            economic_grade_eligible=False,
            source="behavior_dataset_forward_price_paths",
        ),
        "effective_breadth_transfer_coefficient": _measurement(
            _period_breadth_matrices(rows),
            evidence_class="candidate_bound_counterfactual_forecast",
            economic_grade_eligible=False,
            source="behavior_dataset_period_symbol_panel",
        ),
        "hierarchical_bayesian_skill": _measurement(
            {"returns_by_group": dict(returns_by_group)},
            evidence_class="candidate_bound_post_cost_trade_delta",
            economic_grade_eligible=True,
            source="schema_v2_candidate_trade_deltas",
        ),
        "subsample_stability_selection": _measurement(
            {"features": stability_features, "outcomes": stability_outcomes},
            evidence_class="candidate_bound_counterfactual_forecast",
            economic_grade_eligible=False,
            source="point_in_time_decision_context",
        ),
        "economic_alpha_decomposition": _measurement(
            {
                "gross_active_returns": gross_returns,
                "factors": trade_factors,
                "execution_costs": execution_costs,
            },
            evidence_class="candidate_bound_post_cost_trade_delta",
            economic_grade_eligible=True,
            source="schema_v2_candidate_trade_deltas",
        ),
        "factor_neutral_residualization": _measurement(
            {"target_returns": trade_outcomes, "factors": trade_factors},
            evidence_class="candidate_bound_post_cost_trade_delta",
            economic_grade_eligible=True,
            source="schema_v2_candidate_trade_deltas",
        ),
        "cross_fitted_causal_transportability": _measurement(
            {"outcomes": [], "treatments": [], "controls": {}, "environments": []},
            evidence_class="identification_design_required",
            economic_grade_eligible=False,
            source="disabled_until_explicit_identification_design",
        ),
        "capacity_impact_surface": _measurement(
            capacity_inputs,
            evidence_class="candidate_bound_direct_liquidity_and_trade_evidence",
            economic_grade_eligible=bool(adv_values and trade_rows),
            source="direct_daily_dollar_volume_only_no_normalized_volume_proxy",
        ),
        "execution_alpha_attribution": _measurement(
            {"fills": fills},
            evidence_class="candidate_bound_schema_v2_fill",
            economic_grade_eligible=True,
            source="paper_execution_fill_logs",
        ),
        "point_in_time_security_master_audit": _measurement(
            {
                "security_records": security_records,
                "observations": security_observations,
            },
            evidence_class="candidate_bound_point_in_time_identity",
            economic_grade_eligible=False,
            source="effective_dated_security_master",
        ),
        "split_conformal_residual_calibration": _measurement(
            {"predictions": predictions, "outcomes": conformal_outcomes},
            evidence_class="candidate_bound_prequential_forecast",
            economic_grade_eligible=False,
            source="prequential_score_to_return_calibration",
        ),
        "sequential_change_point_stability": _measurement(
            {"values_by_group": dict(returns_by_group)},
            evidence_class="candidate_bound_post_cost_trade_delta",
            economic_grade_eligible=True,
            source="schema_v2_candidate_trade_deltas",
        ),
        "residual_redundancy_graph": _measurement(
            {"returns_by_group": dict(returns_by_group)},
            evidence_class="candidate_bound_post_cost_trade_delta",
            economic_grade_eligible=True,
            source="schema_v2_candidate_trade_deltas",
        ),
        "regime_conditional_robustness": _measurement(
            {
                "outcomes": [float(row["_post_cost_return"]) for row in trade_rows],
                "regimes": [
                    str(row.get("regime") or "unspecified") for row in trade_rows
                ],
            },
            evidence_class="candidate_bound_post_cost_trade_delta",
            economic_grade_eligible=True,
            source="schema_v2_candidate_trade_deltas",
        ),
        "cost_stress_survival": _measurement(
            {
                "expected_gross_alpha_bps": gross_bps,
                "base_cost_bps": cost_bps,
                "uncertainty_buffer_bps": uncertainty_bps,
            },
            evidence_class="candidate_bound_post_cost_trade_delta",
            economic_grade_eligible=True,
            source="schema_v2_candidate_trade_deltas",
        ),
    }
    if causal_enabled:
        raise ValueError(
            "causal transport materialization requires a separately reviewed identification design"
        )
    missing_measurements = sorted(set(MEASUREMENT_IDS) - set(measurements))
    if missing_measurements:
        raise ValueError(f"measurement inputs missing: {missing_measurements}")
    walk_forward = purged_walk_forward_diagnostic(
        rows,
        factor_keys=factor_keys,
        config=_as_dict(policy.get("walk_forward")),
    )
    counterfactual_counts: Counter[str] = Counter()
    hold_rows = 0
    rejected_rows = 0
    for row in rows:
        if str(row.get("action") or "HOLD").upper() == "HOLD":
            hold_rows += 1
        if str(row.get("decision") or "").upper() in {
            "BLOCK",
            "BLOCKED",
            "REJECT",
            "REJECTED",
        }:
            rejected_rows += 1
        counterfactual_counts.update(
            _as_dict(row.get("counterfactual_action_outcomes")).keys()
        )
    multiple_testing_path = (
        root / "governance/research/multiple_testing_guard_latest.json"
    )
    alpha_generation_path = (
        root / "governance/health/alpha_generation_control_latest.json"
    )
    multiple_testing = load_json(multiple_testing_path)
    alpha_generation = load_json(alpha_generation_path)
    latest_timestamp = max(
        (
            row["_timestamp"]
            for row in rows
            if isinstance(row.get("_timestamp"), datetime)
        ),
        default=_utc(candidate.get("paper_evidence_through_utc")),
    )
    now = _utc(generated_at_utc) or datetime.now(timezone.utc)
    receipts.update(
        {
            "policy_path": str(policy_path),
            "policy_sha256": _sha256(policy_path),
            "behavior_dataset_path": str(dataset_path),
            "behavior_dataset_sha256": _sha256(dataset_path),
            "security_master_path": str(security_path),
            "security_master_sha256": _sha256(security_path),
            "multiple_testing_path": str(multiple_testing_path),
            "multiple_testing_sha256": _sha256(multiple_testing_path),
            "alpha_generation_path": str(alpha_generation_path),
            "alpha_generation_sha256": _sha256(alpha_generation_path),
        }
    )
    blockers = list(candidate.get("blockers") or [])
    if not bool(schema_validation.get("dataset_schema_valid")):
        blockers.append("behavior_dataset_v8_required")
    if not rows:
        blockers.append("candidate_bound_forward_outcomes_not_materialized")
    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "policy_id": str(policy.get("policy_id") or ""),
        "status": "ready" if not blockers else "collecting_or_blocked",
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "candidate_binding": {
            **candidate,
            "evidence_through_utc": (
                latest_timestamp.isoformat() if latest_timestamp else ""
            ),
        },
        "measurements": measurements,
        "diagnostics": {
            "schema_v2_candidate_validation": schema_validation,
            "post_cost_trade_delta_validation": post_cost_validation,
            "fill_validation": fill_validation,
            "counterfactual_capture": {
                "candidate_decision_count": len(rows),
                "hold_decision_count": hold_rows,
                "rejected_decision_count": rejected_rows,
                "horizon_observation_counts": dict(
                    sorted(counterfactual_counts.items())
                ),
                "stored_in_behavior_dataset": True,
                "counts_as_realized_trade_profitability": False,
            },
            "purged_embargoed_walk_forward": walk_forward,
            "regime_and_decay_research_routing": _research_routing(
                rows, _as_dict(policy.get("research_routing"))
            ),
            "capacity_truth": {
                "direct_daily_dollar_volume_observation_count": len(adv_values),
                "normalized_relative_volume_used_as_adv": False,
                "unknown_cost_defaults_used": False,
                "capacity_remains_blocked_without_direct_adv": not bool(adv_values),
            },
            "redundancy_and_multiple_testing_gate": {
                "candidate_id": str(candidate.get("candidate_id") or ""),
                "multiple_testing_present": bool(multiple_testing),
                "multiple_testing_evidence_ready": bool(
                    multiple_testing.get("statistical_evidence_ready", False)
                ),
                "residual_groups_with_trade_evidence": len(returns_by_group),
                "automatic_strategy_retirement_allowed": False,
            },
            "cross_sleeve_allocation_gate": {
                "existing_control_present": bool(alpha_generation),
                "existing_evidence_ready": bool(
                    _as_dict(alpha_generation.get("cross_sleeve_alpha")).get(
                        "evidence_ready", False
                    )
                ),
                "factor_neutral_trade_rows": len(trade_outcomes),
                "allocation_requires_positive_residual_multiple_testing_regime_cost_and_capacity_evidence": True,
                "automatic_allocation_allowed": False,
                "live_execution_authority": False,
            },
            "causal_transport": {
                "enabled": False,
                "reason": str(
                    _as_dict(policy.get("causal_transport")).get("reason") or ""
                ),
                "causal_claim_proven": False,
            },
        },
        "source_receipts": receipts,
        "authority_contract": authority,
        "blockers": sorted(set(blockers)),
        "interpretation": {
            "candidate_forecast_measurements_are_realized_market_paths_not_realized_trades": True,
            "hold_counterfactuals_are_research_only": True,
            "economic_grade_requires_candidate_bound_post_cost_trade_or_fill_evidence": True,
            "implementation_does_not_guarantee_profitability": True,
            "live_execution_authority": False,
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize strict current-candidate alpha measurement inputs."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--config", default="config/alpha_measurement_materialization_v1.json"
    )
    parser.add_argument(
        "--out-file", default="governance/research/alpha_concept_inputs_latest.json"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    config_path = Path(args.config).expanduser()
    out_path = Path(args.out_file).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    if not out_path.is_absolute():
        out_path = root / out_path
    payload = build_payload(root, config_path=config_path)
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        validation = _as_dict(
            _as_dict(payload.get("diagnostics")).get("schema_v2_candidate_validation")
        )
        post_cost = _as_dict(
            _as_dict(payload.get("diagnostics")).get("post_cost_trade_delta_validation")
        )
        print(
            "alpha_measurement_inputs "
            f"status={payload.get('status', '')} "
            f"candidate={payload.get('candidate_id') or 'missing'} "
            f"decisions={validation.get('candidate_row_count', 0)} "
            f"post_cost_trades={post_cost.get('valid_schema_v2_post_cost_trade_delta_count', 0)}"
        )
    return 0 if not payload.get("blockers") else 2


if __name__ == "__main__":
    raise SystemExit(main())
