#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_DIAGNOSTICS_DIR = PROJECT_ROOT / "governance" / "training_diagnostics"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "governance" / "health" / "training_label_audit_latest.json"
DEFAULT_MAX_DIAGNOSTIC_AGE_HOURS = 72.0


def _load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _registry_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _bot_diagnostic(diag_dir: Path, bot_id: str) -> dict[str, Any]:
    if not bot_id:
        return {}
    return _load_json(diag_dir / f"{bot_id}_latest.json")


def _float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _recommendation(row: dict[str, Any]) -> str:
    if not bool(row.get("diagnostic_present", False)):
        return "refresh_training_diagnostics"
    if not bool(row.get("diagnostic_fresh", True)):
        return "refresh_training_diagnostics"
    sample_count = _int(row.get("sample_count"))
    sequence_count = _int(row.get("sequence_count"))
    skipped_filtered = _int(row.get("skipped_filtered"))
    skipped_low_confidence = _int(row.get("skipped_low_confidence"))
    skipped_labels = _int(row.get("skipped_labels"))
    positive_rate = _float(row.get("positive_rate"))
    acted_coverage = _float(row.get("acted_coverage"), -1.0)
    acted_accuracy = _float(row.get("acted_accuracy"), -1.0)
    accuracy_lift = _float(row.get("accuracy_lift_over_majority"), 0.0)
    long_precision = _float(row.get("long_precision"), 0.0)
    short_precision = _float(row.get("short_precision"), 0.0)
    label_balance = _float(row.get("label_balance_score"), 0.0)
    if sample_count == 0 and sequence_count == 0:
        return "fix_shared_runtime_input"
    if sample_count == 0 and skipped_filtered > max(skipped_low_confidence, skipped_labels):
        return "relax_sample_filter"
    if sample_count == 0 and skipped_low_confidence > max(skipped_filtered, skipped_labels):
        return "relax_confidence_gate"
    if sample_count == 0 and skipped_labels > 0:
        return "rebalance_label_builder"
    if label_balance < 0.18 or positive_rate <= 0.03 or positive_rate >= 0.97:
        return "rebalance_label_builder"
    if acted_coverage >= 0.50 and accuracy_lift < 0.0:
        return "tighten_abstention_thresholds"
    if 0.0 <= acted_coverage <= 0.02 and sample_count > 0:
        return "loosen_abstention_thresholds"
    if acted_accuracy >= 0.0 and acted_accuracy < 0.53 and accuracy_lift < 0.0:
        return "tighten_or_relabel_for_quality"
    if long_precision > 0.0 and short_precision > 0.0 and abs(long_precision - short_precision) >= 0.18:
        return "use_side_specific_thresholds"
    return "monitor"


def _audit_row(registry_row: dict[str, Any], diag_dir: Path, *, max_diagnostic_age_hours: float) -> dict[str, Any]:
    bot_id = str(registry_row.get("bot_id") or "").strip().lower()
    diag_path = diag_dir / f"{bot_id}_latest.json" if bot_id else Path()
    diag = _bot_diagnostic(diag_dir, bot_id)
    metrics = diag.get("metrics") if isinstance(diag.get("metrics"), dict) else {}
    runtime_meta = diag.get("runtime_meta") if isinstance(diag.get("runtime_meta"), dict) else {}
    label_audit = runtime_meta.get("label_audit") if isinstance(runtime_meta.get("label_audit"), dict) else {}
    diagnostic_age_hours = None
    if diag_path and diag_path.exists():
        try:
            modified = datetime.fromtimestamp(diag_path.stat().st_mtime, tz=timezone.utc)
            diagnostic_age_hours = max((datetime.now(timezone.utc) - modified).total_seconds() / 3600.0, 0.0)
        except Exception:
            diagnostic_age_hours = None
    sample_count = _int(diag.get("sample_count", runtime_meta.get("sample_count", 0)))
    skipped_filtered = _int(diag.get("skipped_filtered", runtime_meta.get("skipped_filtered", 0)))
    skipped_low_confidence = _int(diag.get("skipped_low_confidence", runtime_meta.get("skipped_low_confidence", 0)))
    skipped_labels = _int(diag.get("skipped_labels", runtime_meta.get("skipped_labels", 0)))
    attempted = max(sample_count + skipped_filtered + skipped_low_confidence + skipped_labels, 0)
    out = {
        "bot_id": bot_id,
        "bot_role": str(registry_row.get("bot_role") or ""),
        "active": bool(registry_row.get("active", False)),
        "status": str(diag.get("status") or "missing_diagnostic"),
        "diagnostic_present": bool(diag_path and diag_path.exists()),
        "diagnostic_age_hours": round(float(diagnostic_age_hours), 3) if diagnostic_age_hours is not None else None,
        "diagnostic_fresh": bool(
            diagnostic_age_hours is not None and float(diagnostic_age_hours) <= max(float(max_diagnostic_age_hours), 0.0)
        ),
        "sample_count": sample_count,
        "eligible_sequences": _int(diag.get("eligible_sequences", runtime_meta.get("eligible_sequences", 0))),
        "sequence_count": _int(diag.get("sequence_count", runtime_meta.get("sequence_count", 0))),
        "observation_count": _int(diag.get("observation_count", runtime_meta.get("observation_count", 0))),
        "positive_rate": _float(diag.get("positive_rate", runtime_meta.get("positive_rate", 0.0))),
        "acted_coverage": _float(metrics.get("acted_coverage"), -1.0),
        "acted_accuracy": _float(metrics.get("acted_accuracy"), -1.0),
        "accuracy_lift_over_majority": _float(metrics.get("accuracy_lift_over_majority"), 0.0),
        "long_precision": _float(metrics.get("long_precision"), 0.0),
        "short_precision": _float(metrics.get("short_precision"), 0.0),
        "label_balance_score": _float(metrics.get("label_balance_score"), 0.0),
        "precision_balance_score": _float(metrics.get("precision_balance_score"), 0.0),
        "long_acted_count": _int(metrics.get("long_acted_count", 0)),
        "short_acted_count": _int(metrics.get("short_acted_count", 0)),
        "skipped_filtered": skipped_filtered,
        "skipped_low_confidence": skipped_low_confidence,
        "skipped_labels": skipped_labels,
        "acceptance_rate": round((sample_count / attempted), 6) if attempted > 0 else 0.0,
        "attempted_candidate_count": attempted,
        "label_audit": label_audit,
        "diagnostics_path": str(diag_path) if bot_id else "",
    }
    out["recommendation"] = _recommendation(out)
    return out


def build_label_audit_payload(
    *,
    registry_path: Path,
    diagnostics_dir: Path,
    max_diagnostic_age_hours: float = DEFAULT_MAX_DIAGNOSTIC_AGE_HOURS,
) -> dict[str, Any]:
    rows = [
        _audit_row(row, diagnostics_dir, max_diagnostic_age_hours=max_diagnostic_age_hours)
        for row in _registry_rows(registry_path)
    ]
    active_rows = [row for row in rows if bool(row.get("active"))]
    recommendation_counts = Counter(str(row.get("recommendation") or "") for row in active_rows)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(registry_path),
        "diagnostics_dir": str(diagnostics_dir),
        "active_rows": len(active_rows),
        "recommendation_counts": dict(sorted(recommendation_counts.items())),
        "active_zero_sample": [row for row in active_rows if _int(row.get("sample_count")) == 0][:25],
        "active_overacting": [
            row for row in active_rows
            if _float(row.get("acted_coverage"), -1.0) >= 0.5
        ][:25],
        "active_underacting": [
            row for row in active_rows
            if 0.0 <= _float(row.get("acted_coverage"), -1.0) <= 0.02 and _int(row.get("sample_count")) > 0
        ][:25],
        "active_unbalanced_labels": [
            row for row in active_rows
            if _float(row.get("label_balance_score"), 1.0) < 0.18 or _float(row.get("positive_rate"), 0.5) <= 0.03 or _float(row.get("positive_rate"), 0.5) >= 0.97
        ][:25],
        "top_actions": [],
    }
    top_actions: list[str] = []
    for name in [
        "refresh_training_diagnostics",
        "fix_shared_runtime_input",
        "relax_sample_filter",
        "relax_confidence_gate",
        "rebalance_label_builder",
        "tighten_abstention_thresholds",
        "use_side_specific_thresholds",
    ]:
        if recommendation_counts.get(name, 0) > 0:
            top_actions.append(name)
    payload["top_actions"] = top_actions
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit runtime label quality and abstention behavior across the registry.")
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--diagnostics-dir", default=str(DEFAULT_DIAGNOSTICS_DIR))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--max-diagnostic-age-hours", type=float, default=DEFAULT_MAX_DIAGNOSTIC_AGE_HOURS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_label_audit_payload(
        registry_path=Path(args.registry_path).expanduser(),
        diagnostics_dir=Path(args.diagnostics_dir).expanduser(),
        max_diagnostic_age_hours=float(args.max_diagnostic_age_hours),
    )
    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "training_label_audit "
            f"active_rows={int(payload['active_rows'])} "
            f"zero_sample={len(payload['active_zero_sample'])} "
            f"overacting={len(payload['active_overacting'])} "
            f"underacting={len(payload['active_underacting'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
