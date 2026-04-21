#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "research" / "multiple_testing_guard_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    health_root = project_root / "governance" / "health"
    walk_root = project_root / "governance" / "walk_forward"

    ablation = _load_json(health_root / "replay_feature_ablation_latest.json")
    counterfactual = _load_json(health_root / "counterfactual_replay_latest.json")
    promotion_readiness = _load_json(walk_root / "promotion_readiness_latest.json")

    ablation_block = ablation.get("ablation") if isinstance(ablation.get("ablation"), dict) else {}
    strict_checks = ablation.get("strict_checks") if isinstance(ablation.get("strict_checks"), dict) else {}
    profiles_reviewed = counterfactual.get("profiles_reviewed") if isinstance(counterfactual.get("profiles_reviewed"), list) else []

    feature_hypotheses = 0
    for key, value in ablation_block.items():
        if key == "baseline":
            continue
        if isinstance(value, dict):
            feature_hypotheses += 1
    if feature_hypotheses <= 0:
        feature_hypotheses = max(_safe_int(ablation_block.get("e2e_feature_count"), 0) + _safe_int(ablation_block.get("paper_feature_count"), 0), 0)

    counterfactual_candidates = _safe_int(counterfactual.get("candidate_count"), 0)
    considered_bots = _safe_int(promotion_readiness.get("considered_bots"), 0)
    family_size = max(feature_hypotheses + counterfactual_candidates + considered_bots, 0)
    method = "benjamini_hochberg_fdr" if family_size >= 10 else "bonferroni" if family_size > 0 else "not_applicable"
    base_alpha = 0.05
    corrected_alpha = base_alpha if method == "benjamini_hochberg_fdr" else round(base_alpha / max(family_size, 1), 6) if family_size > 0 else 0.0
    regime_segments = sorted({str(profile).strip().lower() for profile in profiles_reviewed if str(profile).strip()})
    if not regime_segments:
        regime_segments = ["global"]

    hypotheses = [
        {
            "family": "feature_ablation",
            "hypothesis_count": feature_hypotheses,
            "evidence_path": str(health_root / "replay_feature_ablation_latest.json"),
        },
        {
            "family": "counterfactual_threshold_search",
            "hypothesis_count": counterfactual_candidates,
            "evidence_path": str(health_root / "counterfactual_replay_latest.json"),
        },
        {
            "family": "promotion_candidates",
            "hypothesis_count": considered_bots,
            "evidence_path": str(walk_root / "promotion_readiness_latest.json"),
        },
    ]

    ok = bool(ablation.get("ok", False) and counterfactual.get("ok", False) and family_size > 0)
    overall_status = "ready" if ok else "needs_work"
    if family_size <= 0 or not ablation or not counterfactual:
        overall_status = "blocked"

    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": overall_status,
        "base_alpha": base_alpha,
        "correction_method": method,
        "corrected_alpha": corrected_alpha,
        "family_size": family_size,
        "hypotheses": hypotheses,
        "regime_segments": regime_segments,
        "strict_checks": strict_checks,
        "baseline_metrics": ablation_block.get("baseline") if isinstance(ablation_block.get("baseline"), dict) else {},
        "delta_metrics": ablation.get("delta") if isinstance(ablation.get("delta"), dict) else {},
        "failed_checks": ablation.get("failed_checks") if isinstance(ablation.get("failed_checks"), list) else [],
        "recommendations": [
            "Keep correction families stable across feature ablation, counterfactual threshold search, and promotion review batches.",
            "Segment research verdicts by lane or regime when profiles_reviewed spans materially different sleeves.",
        ],
        "source_files": {
            "replay_feature_ablation": str(health_root / "replay_feature_ablation_latest.json"),
            "counterfactual_replay": str(health_root / "counterfactual_replay_latest.json"),
            "promotion_readiness": str(walk_root / "promotion_readiness_latest.json"),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a multiple-testing control artifact for replay and promotion research.")
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
            "multiple_testing_guard "
            f"status={payload['overall_status']} "
            f"family_size={int(payload.get('family_size', 0) or 0)} "
            f"method={payload.get('correction_method', '')}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
