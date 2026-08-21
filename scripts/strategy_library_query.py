#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY_PATH = (
    PROJECT_ROOT / "governance" / "research" / "sleeve_strategy_library_latest.json"
)
DEFAULT_FAMILIES_PATH = (
    PROJECT_ROOT / "governance" / "research" / "sleeve_strategy_families_latest.json"
)
GOOD_VERDICTS = {"validated_good", "promising_unconfirmed"}
BAD_VERDICTS = {"weak", "retirement_candidate"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _quality(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("quality_assessment")
    return dict(value) if isinstance(value, Mapping) else {}


def _regime(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("regime_assessment")
    return dict(value) if isinstance(value, Mapping) else {}


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    sleeve: str = "",
    verdict: str = "",
    tier: str = "",
    relevance: str = "",
    good_only: bool = False,
    bad_only: bool = False,
) -> list[dict[str, Any]]:
    sleeve_key = _normalize(sleeve)
    verdict_key = _normalize(verdict)
    tier_key = _normalize(tier)
    relevance_key = _normalize(relevance)
    result: list[dict[str, Any]] = []
    for row in rows:
        quality = _quality(row)
        regime = _regime(row)
        row_verdict = _normalize(quality.get("verdict"))
        if sleeve_key and _normalize(row.get("sleeve_id")) != sleeve_key:
            continue
        if verdict_key and row_verdict != verdict_key:
            continue
        if tier_key and _normalize(row.get("library_tier")) != tier_key:
            continue
        if relevance_key and _normalize(regime.get("relevance")) != relevance_key:
            continue
        if good_only and row_verdict not in GOOD_VERDICTS:
            continue
        if bad_only and row_verdict not in BAD_VERDICTS:
            continue
        result.append(row)
    result.sort(
        key=lambda row: (
            -float(_quality(row).get("quality_score") or -1.0),
            -float(_quality(row).get("evidence_maturity_percent") or 0.0),
            str(row.get("sleeve_id") or ""),
            str(row.get("strategy_name") or ""),
        )
    )
    return result


def filter_families(
    rows: list[dict[str, Any]],
    *,
    sleeve: str = "",
    objective: str = "",
    family: str = "",
) -> list[dict[str, Any]]:
    sleeve_key = _normalize(sleeve)
    objective_key = _normalize(objective)
    family_key = _normalize(family)
    selected: list[dict[str, Any]] = []
    for row in rows:
        searchable = " ".join(
            str(row.get(key) or "")
            for key in ("family_id", "family_name", "archetype")
        )
        if sleeve_key and _normalize(row.get("sleeve_id")) != sleeve_key:
            continue
        if objective_key and _normalize(row.get("objective_class")) != objective_key:
            continue
        if family_key and family_key not in _normalize(searchable):
            continue
        selected.append(row)
    return sorted(
        selected,
        key=lambda row: (
            str(row.get("sleeve_id") or ""),
            str(row.get("objective_class") or ""),
            str(row.get("family_id") or ""),
        ),
    )


def _display_row(row: Mapping[str, Any]) -> str:
    quality = _quality(row)
    regime = _regime(row)
    score = quality.get("quality_score")
    score_text = "n/a" if score is None else f"{float(score):.1f}"
    maturity = float(quality.get("evidence_maturity_percent") or 0.0)
    return (
        f"{str(row.get('sleeve_id') or ''):<34} "
        f"{str(row.get('strategy_name') or ''):<64} "
        f"tier={str(row.get('library_tier') or ''):<13} "
        f"regime={str(regime.get('relevance') or 'unknown'):<8} "
        f"verdict={str(quality.get('verdict') or 'unknown'):<25} "
        f"score={score_text:<5} evidence={maturity:.1f}%"
    )


def _display_family(row: Mapping[str, Any]) -> str:
    evidence = row.get("family_evidence")
    evidence_row = dict(evidence) if isinstance(evidence, Mapping) else {}
    verdicts = evidence_row.get("verdict_counts")
    verdict_row = dict(verdicts) if isinstance(verdicts, Mapping) else {}
    materialized = list(row.get("materialized_conditions") or [])
    supported = list(row.get("supported_conditions") or [])
    return (
        f"{str(row.get('sleeve_id') or ''):<34} "
        f"{str(row.get('family_name') or ''):<50} "
        f"kind={str(row.get('family_kind') or ''):<22} "
        f"objective={str(row.get('objective_class') or ''):<29} "
        f"variants={int(row.get('variant_count') or 0):>2} "
        f"conditions_supported={len(supported)} materialized={len(materialized)} "
        f"verdicts={json.dumps(verdict_row, sort_keys=True, separators=(',', ':'))}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search the 12,000-strategy sleeve library and evidence scorecards."
    )
    parser.add_argument("--library", default=str(DEFAULT_LIBRARY_PATH))
    parser.add_argument("--families-path", default=str(DEFAULT_FAMILIES_PATH))
    parser.add_argument("--families", action="store_true")
    parser.add_argument("--family", default="")
    parser.add_argument("--objective", default="")
    parser.add_argument("--sleeve", default="")
    parser.add_argument("--verdict", default="")
    parser.add_argument("--tier", default="")
    parser.add_argument("--regime-relevance", default="")
    parser.add_argument("--good", action="store_true")
    parser.add_argument("--bad", action="store_true")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.families:
        payload = _read_json(Path(args.families_path).resolve())
        rows = [
            dict(row)
            for row in payload.get("families") or []
            if isinstance(row, Mapping)
        ]
        selected = filter_families(
            rows,
            sleeve=args.sleeve,
            objective=args.objective,
            family=args.family,
        )
        limit = max(int(args.limit), 0)
        displayed = selected[:limit] if limit else selected
        if args.json:
            print(
                json.dumps(
                    {
                        "status": str(payload.get("status") or "missing"),
                        "consolidation_contract": payload.get("consolidation_contract") or {},
                        "condition_coverage": payload.get("condition_coverage") or {},
                        "matching_count": len(selected),
                        "displayed_count": len(displayed),
                        "families": displayed,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
        else:
            contract = payload.get("consolidation_contract") or {}
            print(
                "[strategy-families] "
                f"status={payload.get('status', 'missing')} "
                f"canonical={contract.get('canonical_record_count', 0)}/1989 "
                f"lineage={contract.get('lineage_covered_strategy_count', 0)}/12000 "
                f"hot={contract.get('native_hot_family_count', 0)} "
                f"cold_parents={contract.get('cold_parent_family_count', 0)} "
                f"cold_variants={contract.get('cold_child_variant_count', 0)} "
                f"matches={len(selected)} shown={len(displayed)}"
            )
            for row in displayed:
                print(_display_family(row))
        return 0 if payload.get("ok", False) else 2

    payload = _read_json(Path(args.library).resolve())
    rows = [dict(row) for row in payload.get("strategies") or [] if isinstance(row, Mapping)]
    selected = filter_rows(
        rows,
        sleeve=args.sleeve,
        verdict=args.verdict,
        tier=args.tier,
        relevance=args.regime_relevance,
        good_only=args.good,
        bad_only=args.bad,
    )
    limit = max(int(args.limit), 0)
    displayed = selected[:limit] if limit else selected
    if args.json:
        print(
            json.dumps(
                {
                    "status": str(payload.get("status") or "missing"),
                    "current_regime": payload.get("current_regime") or {},
                    "library_contract": payload.get("library_contract") or {},
                    "matching_count": len(selected),
                    "displayed_count": len(displayed),
                    "strategies": displayed,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    else:
        library = payload.get("library_contract") or {}
        current_regime = payload.get("current_regime") or {}
        print(
            "[strategy-library] "
            f"status={payload.get('status', 'missing')} "
            f"strategies={library.get('strategy_count', 0)}/12000 "
            f"sleeves={library.get('sleeve_count', 0)} "
            f"hot={library.get('hot_strategy_count', 0)} "
            f"cold={library.get('cold_strategy_count', 0)} "
            f"regime={current_regime.get('current_regime', 'unknown')} "
            f"regime_ready={current_regime.get('activation_ready', False)} "
            f"matches={len(selected)} shown={len(displayed)}"
        )
        for row in displayed:
            print(_display_row(row))
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
