#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.profitability_statistics import risk_of_ruin_statistics
    from scripts.ops.long_runtime_common import iso_now, load_json, parse_iso_utc, write_payload
else:
    from core.profitability_statistics import risk_of_ruin_statistics
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, parse_iso_utc, write_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "profitability_evidence_firewall_v1.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "profitability_independent_validator_latest.json"


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _resolve(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or "")).expanduser()
    return path if path.is_absolute() else project_root / path


def _open_text(path: Path):
    if ".jsonl.gz" in path.name:
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _record_key(row: dict[str, Any], raw: str) -> str:
    metadata = _as_dict(row.get("metadata"))
    intent = _as_dict(row.get("order_intent_evidence"))
    semantic_order = _as_dict(intent.get("semantic_order"))
    execution_id = str(
        row.get("execution_id")
        or row.get("fill_id")
        or metadata.get("execution_id")
        or metadata.get("fill_id")
        or ""
    ).strip()
    if execution_id:
        return f"execution:{execution_id}"
    decision_id = str(
        row.get("decision_id")
        or metadata.get("decision_id")
        or semantic_order.get("decision_id")
        or ""
    ).strip()
    if decision_id:
        book_id = str(row.get("paper_book_id") or metadata.get("paper_book_id") or "unbound-book").strip()
        return f"paper-decision:{book_id}:{decision_id}"
    stable = json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return f"content:{hashlib.sha1(stable.encode('utf-8')).hexdigest()}"


def _calibration_only_reason(row: dict[str, Any]) -> str:
    metadata = _as_dict(row.get("metadata"))
    provenance = _as_dict(row.get("provenance"))
    if bool(row.get("independent_fill_evidence", False)) or bool(
        metadata.get("independent_fill_evidence", False)
    ):
        return "independent_fill_evidence"
    fill_source = str(row.get("paper_fill_source") or metadata.get("paper_fill_source") or "").strip().lower()
    if fill_source in {"market_replay_fill", "independent_fill_evidence"}:
        return fill_source
    account_mode = str(
        row.get("account_mode")
        or metadata.get("account_mode")
        or provenance.get("account_mode")
        or ""
    ).strip().lower()
    return "replay_account_mode" if account_mode == "replay" else ""


def _trade_log_roots(project_root: Path) -> list[Path]:
    roots = [
        project_root / "exports" / "trade_logs",
        project_root / "local_fallback_storage" / "exports" / "trade_logs",
    ]
    configured_external = str(os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "") or "").strip()
    if configured_external:
        roots.append(Path(configured_external).expanduser() / "exports" / "trade_logs")
    elif project_root.resolve(strict=False) == PROJECT_ROOT.resolve(strict=False):
        external_mount = Path(os.getenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/BOT_LOGS")).expanduser()
        external_project = str(
            os.getenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", "schwab_trading_bot") or "schwab_trading_bot"
        ).strip()
        roots.append(external_mount / external_project / "exports" / "trade_logs")
    return roots


def _source_files(project_root: Path, policy: dict[str, Any]) -> list[Path]:
    candidates = [_resolve(project_root, item) for item in _as_list(policy.get("source_paths"))]
    for raw_pattern in _as_list(policy.get("source_globs")):
        pattern = str(raw_pattern or "").strip()
        if pattern:
            candidates.extend(project_root.glob(pattern))
    seen_roots: set[str] = set()
    for root in _trade_log_roots(project_root):
        root_key = str(root.resolve(strict=False))
        if root_key in seen_roots or not root.exists():
            continue
        seen_roots.add(root_key)
        candidates.extend(root.rglob("paper_trades_*.jsonl"))
        candidates.extend(root.rglob("paper_trades_*.jsonl.gz"))
        candidates.extend(root.rglob("paper_trades_*.jsonl.local_fallback*"))
        candidates.extend(root.rglob("paper_trades_*.jsonl.gz.local_fallback*"))
    files: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            identity = str(path.resolve())
        except OSError:
            identity = str(path)
        if identity in seen or not path.is_file() or "independent_fills" in path.parts:
            continue
        seen.add(identity)
        files.append(path)

    def _priority(path: Path) -> tuple[int, str]:
        text = str(path)
        if "/exports/trade_logs/" in text:
            return 0, text
        if "/exports/paper_broker_bridge/" in text:
            return 2, text
        return 1, text

    return sorted(files, key=_priority)


def _iter_rows(
    paths: Iterable[Path],
    *,
    cutoff: Any,
    evidence_through: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    files_scanned = 0
    malformed_rows = 0
    duplicate_rows = 0
    pre_candidate_rows = 0
    post_snapshot_rows = 0
    calibration_rows_excluded = 0
    legacy_schema_rows_excluded = 0
    for path in paths:
        if not path.is_file():
            continue
        files_scanned += 1
        try:
            handle = _open_text(path)
        except OSError:
            continue
        with handle:
            for raw in handle:
                text = raw.strip()
                if not text or "post_cost_pnl_delta" not in text:
                    continue
                try:
                    row = json.loads(text)
                except Exception:
                    malformed_rows += 1
                    continue
                if not isinstance(row, dict):
                    malformed_rows += 1
                    continue
                if _calibration_only_reason(row):
                    calibration_rows_excluded += 1
                    continue
                if int(_safe_float(row.get("paper_pnl_schema_version"), 0.0)) < 2:
                    legacy_schema_rows_excluded += 1
                    continue
                timestamp = parse_iso_utc(row.get("timestamp_utc") or row.get("timestamp"))
                if cutoff is not None and (timestamp is None or timestamp < cutoff):
                    pre_candidate_rows += 1
                    continue
                if evidence_through is not None and (timestamp is None or timestamp > evidence_through):
                    post_snapshot_rows += 1
                    continue
                pnl = _safe_float(row.get("post_cost_pnl_delta"), float("nan"))
                returns = _safe_float(row.get("post_cost_return_bps"), float("nan"))
                if not math.isfinite(pnl) or not math.isfinite(returns):
                    malformed_rows += 1
                    continue
                key = _record_key(row, text)
                if key in seen:
                    duplicate_rows += 1
                    continue
                seen.add(key)
                rows.append(row)
    rows.sort(key=lambda row: str(row.get("timestamp_utc") or row.get("timestamp") or ""))
    return rows, {
        "configured_file_count": len(list(paths)) if not isinstance(paths, list) else len(paths),
        "files_scanned": files_scanned,
        "malformed_rows": malformed_rows,
        "duplicate_rows": duplicate_rows,
        "pre_candidate_rows": pre_candidate_rows,
        "post_snapshot_rows": post_snapshot_rows,
        "calibration_rows_excluded": calibration_rows_excluded,
        "legacy_schema_rows_excluded": legacy_schema_rows_excluded,
    }


def _close(left: Any, right: Any, *, absolute: float, relative: float) -> bool:
    left_value = _safe_float(left, float("nan"))
    right_value = _safe_float(right, float("nan"))
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return False
    return abs(left_value - right_value) <= max(absolute, relative * max(abs(left_value), abs(right_value), 1.0))


def _drawdown(values: list[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    maximum = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        maximum = max(maximum, peak - cumulative)
    return maximum


def build_payload(project_root: Path = PROJECT_ROOT, *, config_path: Path | None = None) -> dict[str, Any]:
    config = load_json(config_path or project_root / "config" / DEFAULT_CONFIG_PATH.name)
    validator_policy = _as_dict(config.get("independent_validator"))
    risk_policy = _as_dict(config.get("risk_of_ruin"))
    performance = load_json(project_root / "governance" / "health" / "paper_performance_latest.json")
    expectancy = _as_dict(performance.get("post_cost_expectancy"))
    window = _as_dict(performance.get("profitability_evidence_window"))
    cutoff = parse_iso_utc(window.get("candidate_cutoff_utc"))
    evidence_through = parse_iso_utc(window.get("evidence_through_utc") or performance.get("timestamp_utc"))
    state = load_json(project_root / "governance" / "runtime" / "production_candidate_state.json")
    if cutoff is None:
        windows = _as_dict(state.get("scope_windows_started_utc"))
        candidates = [parse_iso_utc(windows.get(key)) for key in ("execution", "data", "dependencies")]
        cutoff = max((value for value in candidates if value is not None), default=None)
    source_paths = _source_files(project_root, validator_policy)
    rows, scan = _iter_rows(source_paths, cutoff=cutoff, evidence_through=evidence_through)
    pnl_values = [_safe_float(row.get("post_cost_pnl_delta"), 0.0) for row in rows]
    return_values = [_safe_float(row.get("post_cost_return_bps"), 0.0) for row in rows]
    notional_values = [_safe_float(row.get("execution_notional"), 0.0) for row in rows]
    cost_values = [_safe_float(row.get("expected_execution_cost_amount"), 0.0) for row in rows]
    by_day: dict[str, dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "notional": 0.0})
    for row, pnl, notional in zip(rows, pnl_values, notional_values):
        timestamp = parse_iso_utc(row.get("timestamp_utc") or row.get("timestamp"))
        day = timestamp.date().isoformat() if timestamp is not None else "unknown"
        by_day[day]["pnl"] += pnl
        by_day[day]["notional"] += notional
    daily_rows = [
        {
            "day_utc": day,
            "post_cost_pnl": round(values["pnl"], 8),
            "execution_notional": round(values["notional"], 8),
            "active_return_bps": round(values["pnl"] / values["notional"] * 10_000.0, 8)
            if values["notional"] > 0.0
            else None,
        }
        for day, values in sorted(by_day.items())
        if day != "unknown"
    ]
    recomputed = {
        "sample_count": len(rows),
        "total_post_cost_pnl_delta": round(sum(pnl_values), 8),
        "mean_post_cost_pnl_delta": round(sum(pnl_values) / max(len(pnl_values), 1), 8),
        "mean_post_cost_return_bps": round(sum(return_values) / max(len(return_values), 1), 8),
        "execution_notional_total": round(sum(notional_values), 8),
        "expected_execution_cost_total": round(sum(cost_values), 8),
        "max_cumulative_drawdown_post_cost_pnl": round(_drawdown(pnl_values), 8),
        "first_sample_timestamp_utc": str(rows[0].get("timestamp_utc") or "") if rows else "",
        "last_sample_timestamp_utc": str(rows[-1].get("timestamp_utc") or "") if rows else "",
        "daily": daily_rows,
    }
    absolute_tolerance = _safe_float(validator_policy.get("absolute_tolerance"), 1e-6)
    relative_tolerance = _safe_float(validator_policy.get("relative_tolerance"), 1e-6)
    reported_sample_count = int(_safe_float(expectancy.get("sample_count"), -1))
    empty_report = bool(
        reported_sample_count == 0
        and str(expectancy.get("status") or "").strip().lower() == "no_schema_v2_trade_deltas"
    )

    def reported_total(key: str) -> Any:
        value = expectancy.get(key)
        return 0.0 if value is None and empty_report else value

    comparisons = {
        "sample_count": len(rows) == reported_sample_count,
        "total_post_cost_pnl_delta": _close(
            recomputed["total_post_cost_pnl_delta"],
            reported_total("total_post_cost_pnl_delta"),
            absolute=absolute_tolerance,
            relative=relative_tolerance,
        ),
        "execution_notional_total": _close(
            recomputed["execution_notional_total"],
            reported_total("execution_notional_total"),
            absolute=absolute_tolerance,
            relative=relative_tolerance,
        ),
        "max_cumulative_drawdown_post_cost_pnl": _close(
            recomputed["max_cumulative_drawdown_post_cost_pnl"],
            reported_total("max_cumulative_drawdown_post_cost_pnl"),
            absolute=absolute_tolerance,
            relative=relative_tolerance,
        ),
    }
    accounting_ready = bool(
        rows
        and evidence_through is not None
        and all(comparisons.values())
        and scan["malformed_rows"] == 0
    )
    risk = risk_of_ruin_statistics(
        [float(row["post_cost_pnl"]) for row in daily_rows],
        initial_capital=_safe_float(risk_policy.get("initial_capital"), 10_000.0),
        ruin_equity_fraction=_safe_float(risk_policy.get("ruin_equity_fraction"), 0.5),
        drawdown_budget_fraction=_safe_float(risk_policy.get("drawdown_budget_fraction"), 0.1),
        horizon_days=int(_safe_float(risk_policy.get("horizon_days"), 252)),
        iterations=int(_safe_float(risk_policy.get("iterations"), 2_000)),
        block_days=int(_safe_float(risk_policy.get("block_days"), 5)),
        minimum_days=int(_safe_float(risk_policy.get("minimum_days"), 30)),
        maximum_ruin_probability=_safe_float(risk_policy.get("maximum_ruin_probability"), 0.01),
        maximum_drawdown_breach_probability=_safe_float(
            risk_policy.get("maximum_drawdown_breach_probability"), 0.05
        ),
        seed_material=str(state.get("candidate_id") or "unbound-candidate"),
    )
    blockers = []
    if not source_paths:
        blockers.append("validator_source_paths_not_configured")
    if not rows:
        blockers.append("candidate_bound_post_cost_rows_pending")
    if evidence_through is None:
        blockers.append("paper_report_snapshot_watermark_missing")
    blockers.extend(key for key, passed in comparisons.items() if not passed)
    if scan["malformed_rows"]:
        blockers.append("malformed_schema_v2_rows")
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "ready" if accounting_ready else "evidence_pending",
        "implementation_ready": bool(validator_policy and source_paths),
        "evidence_ready": accounting_ready,
        "candidate_binding": {
            "candidate_id": str(state.get("candidate_id") or ""),
            "generation": int(_safe_float(state.get("generation"), 0.0)),
            "cutoff_utc": cutoff.isoformat() if cutoff is not None else "",
            "evidence_through_utc": evidence_through.isoformat() if evidence_through is not None else "",
            "bound": bool(state.get("candidate_id") and cutoff is not None and evidence_through is not None),
        },
        "source_paths": [str(path) for path in source_paths],
        "scan": scan,
        "recomputed": recomputed,
        "reported": {
            key: expectancy.get(key) if key == "sample_count" else reported_total(key)
            for key in (
                "sample_count",
                "total_post_cost_pnl_delta",
                "execution_notional_total",
                "max_cumulative_drawdown_post_cost_pnl",
            )
        },
        "reported_empty_window_normalized": empty_report,
        "comparisons": comparisons,
        "risk_of_ruin": risk,
        "blockers": sorted(set(blockers)),
        "control_contract": {
            "independent_accounting_implementation": True,
            "does_not_import_paper_performance_accounting": True,
            "candidate_cutoff_enforced": True,
            "paper_report_snapshot_watermark_enforced": True,
            "rows_after_snapshot_are_deferred_not_mismatched": True,
            "duplicate_rows_do_not_count_twice": True,
            "bridge_and_canonical_trade_mirrors_share_one_decision_identity": True,
            "active_local_fallback_and_external_trade_roots_reconciled": True,
            "calibration_only_rows_excluded_from_realized_pnl": True,
            "schema_v2_metric_scope_enforced": True,
            "empty_candidate_window_uses_explicit_zero_accounting": True,
            "missing_or_mismatched_evidence_fails_closed": True,
            "live_execution_authority": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently recompute candidate-bound post-cost paper results.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=Path("config") / DEFAULT_CONFIG_PATH.name)
    parser.add_argument("--out-file", type=Path, default=Path("governance/health") / DEFAULT_OUT_PATH.name)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    out_path = args.out_file if args.out_file.is_absolute() else project_root / args.out_file
    payload = build_payload(project_root, config_path=config_path)
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "profitability_independent_validator "
            f"status={payload['overall_status']} samples={payload['recomputed']['sample_count']} "
            f"evidence_ready={int(bool(payload['evidence_ready']))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
