#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.advanced_quant_models import (
    QUANT_MODEL_FEATURE_KEYS,
    quant_model_inventory,
    summarize_quant_model_features,
)


HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "quant_model_control_latest.json"
EXTERNAL_CONTEXT_PATH = PROJECT_ROOT / "exports" / "external_context" / "quant_model_control_latest.json"
REPORT_DIR = PROJECT_ROOT / "exports" / "reports" / "quant_model_control"
MD_PATH = REPORT_DIR / "quant_model_control_latest.md"
PDF_PATH = REPORT_DIR / "quant_model_control_latest.pdf"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _sample_features() -> dict[str, float]:
    one_numbers = _load_json(PROJECT_ROOT / "governance" / "health" / "one_numbers_latest.json")
    resource = _load_json(PROJECT_ROOT / "governance" / "health" / "resource_guard_latest.json")
    return {
        "last_price": 100.0,
        "atm_strike": 100.0,
        "expiry_days": 30.0,
        "risk_free_rate": 0.045,
        "implied_volatility": 0.22,
        "mom_1m": float(one_numbers.get("combined_pnl_proxy", 0.0) or 0.0),
        "mom_5m": float(one_numbers.get("combined_pnl_proxy", 0.0) or 0.0) * 0.5,
        "pct_from_close": float(one_numbers.get("combined_blocked_rate", 0.0) or 0.0) * -0.01,
        "ctx_SPY_mom_5m": 0.0,
        "ctx_QQQ_mom_5m": 0.0,
        "ctx_TLT_mom_5m": 0.0,
        "ctx_UUP_mom_5m": 0.0,
        "ctx_VIX_X_pct_from_close": float(one_numbers.get("decision_stale_windows_4h", 0) or 0) * 0.001,
        "market_micro_tradeability_score_norm": max(0.0, min(1.0, float(resource.get("cpu_free_pct", 50.0) or 50.0) / 100.0)),
        "flow_direction_signed": 0.0,
        "options_surface_change_norm": 0.25,
        "order_book_imbalance_norm": 0.15,
        "book_depth_health_norm": 0.60,
        "bid_ask_spread_stress_norm": 0.20,
        "quote_fade_risk_norm": 0.20,
        "source_confidence_norm": 0.60,
        "decision_provenance_coverage_norm": 0.55,
        "golden_replay_pass_rate_norm": 0.70,
        "critic_correction_success_norm": 0.55,
        "macro_event_density_norm": 0.20,
        "volatility_cluster_norm": 0.25,
        "causal_treatment_signal_norm": 0.45,
        "confounder_balance_norm": 0.30,
        "rule_consistency_norm": 0.60,
        "cross_asset_invariance_norm": 0.58,
        "reward_stability_norm": 0.52,
        "simulator_gradient_stability_norm": 0.48,
        "symmetry_consistency_norm": 0.55,
    }


def _status_from_features(features: dict[str, float]) -> tuple[str, list[str]]:
    resource_pressure = float(features.get("quant_model_resource_pressure_norm", 0.0) or 0.0)
    tail_pressure = max(
        float(features.get("quant_cvar_tail_risk_norm", 0.0) or 0.0),
        float(features.get("quant_copula_dependency_norm", 0.0) or 0.0),
        float(features.get("quant_heston_vol_risk_norm", 0.0) or 0.0),
        float(features.get("quant_merton_jump_risk_norm", 0.0) or 0.0),
    )
    data_confidence = float(features.get("quant_model_data_confidence_norm", 0.0) or 0.0)
    actions: list[str] = []
    status = "ready"
    if resource_pressure >= 0.80:
        status = "degraded"
        actions.append("lower QUANT_MODEL_* caps through memory-efficiency before running broad quant sweeps")
    if tail_pressure >= 0.75:
        status = "watch"
        actions.append("keep quant tail-risk signals as dampeners; do not promote them into execution without review")
    if data_confidence < 0.45:
        status = "needs_data"
        actions.append("let quant-model bots collect more live paper/context observations before training")
    if not actions:
        actions.append("keep the quant-model layer in research-only collection mode")
    return status, actions


def build_payload() -> dict[str, Any]:
    inventory = quant_model_inventory()
    features = summarize_quant_model_features(
        _sample_features(),
        external_snapshots={
            "one_numbers": _load_json(PROJECT_ROOT / "governance" / "health" / "one_numbers_latest.json"),
            "resource_guard": _load_json(PROJECT_ROOT / "governance" / "health" / "resource_guard_latest.json"),
            "memory_efficiency": _load_json(PROJECT_ROOT / "governance" / "health" / "memory_efficiency_control_latest.json"),
        },
    )
    status, actions = _status_from_features(features)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": status in {"ready", "watch"},
        "overall_status": status,
        "features": features,
        "derived": {
            "global_features": {key: float(features.get(key, 0.0) or 0.0) for key in QUANT_MODEL_FEATURE_KEYS},
            "symbol_features": {},
        },
        "inventory": inventory,
        "execution_policy": inventory.get("execution_policy", {}),
        "infrastructure_bots": [
            "brain_refinery_v452_quant_model_resource_guard_bot",
            "brain_refinery_v453_quant_model_cache_schema_guard_bot",
            "brain_refinery_v455_quant_engine_regression_guard_bot",
            "brain_refinery_v488_signature_hawkes_games_regression_guard_bot",
            "brain_refinery_v496_order_book_transformer_resource_guard_bot",
            "brain_refinery_v497_agentic_quant_memory_guard_bot",
            "brain_refinery_v504_causal_omni_symbolic_regression_guard_bot",
            "brain_refinery_v505_rlbf_dms_equivariant_resource_guard_bot",
        ],
        "recommended_actions": actions,
        "artifact_paths": {
            "json": str(HEALTH_PATH),
            "external_context": str(EXTERNAL_CONTEXT_PATH),
            "markdown": str(MD_PATH),
            "pdf": str(PDF_PATH),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    inventory = payload.get("inventory") if isinstance(payload.get("inventory"), dict) else {}
    resource = inventory.get("resource_profile") if isinstance(inventory.get("resource_profile"), dict) else {}
    features = payload.get("features") if isinstance(payload.get("features"), dict) else {}
    lines = [
        "# Quant Model Control",
        "",
        f"Generated UTC: {payload.get('timestamp_utc')}",
        "",
        "## Summary",
        "",
        f"- Overall status: {payload.get('overall_status')}",
        f"- Implemented models: {len(list(inventory.get('implemented_models') or []))}",
        f"- Feature keys: {len(list(inventory.get('feature_keys') or []))}",
        f"- Resource pressure: {float(features.get('quant_model_resource_pressure_norm', 0.0) or 0.0):.3f}",
        f"- Data confidence: {float(features.get('quant_model_data_confidence_norm', 0.0) or 0.0):.3f}",
        "",
        "## Implemented Model Layer",
        "",
    ]
    for model in list(inventory.get("implemented_models") or []):
        lines.append(f"- {model}")
    lines.extend(["", "## Runtime Caps", ""])
    for key, value in sorted(resource.items()):
        lines.append(f"- {key}: {value}")
    hooks = inventory.get("mlx_hooks") if isinstance(inventory.get("mlx_hooks"), dict) else {}
    if hooks:
        lines.extend(["", "## MLX Hooks", ""])
        for key, value in sorted(hooks.items()):
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Operating Policy", ""])
    lines.append("- Direct execution: blocked")
    lines.append("- Paper trading: blocked for quant research bots until data floors clear")
    lines.append("- Master/grand-master usage: risk context, confidence dampening, and research feature enrichment")
    lines.extend(["", "## Recommended Actions", ""])
    for action in list(payload.get("recommended_actions") or []):
        lines.append(f"- {action}")
    return "\n".join(lines)


def write_report(*, render_pdf: bool = True) -> dict[str, Any]:
    payload = build_payload()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(payload)
    MD_PATH.write_text(markdown, encoding="utf-8")
    _write_json(HEALTH_PATH, payload)
    _write_json(EXTERNAL_CONTEXT_PATH, payload)
    if render_pdf:
        from scripts.ops.sendout_pdf_refresh import render_text_pdf

        payload["pdf"] = render_text_pdf("Quant Model Control", MD_PATH, PDF_PATH)
        _write_json(HEALTH_PATH, payload)
        _write_json(EXTERNAL_CONTEXT_PATH, payload)
    else:
        payload["pdf"] = {"ok": PDF_PATH.exists(), "pdf_path": str(PDF_PATH)}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the advanced quant-model control, resource, and report artifact.")
    parser.add_argument("--no-render-pdf", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = write_report(render_pdf=not args.no_render_pdf)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "quant_model_control "
            f"status={payload.get('overall_status')} "
            f"resource_pressure={float(((payload.get('features') or {}).get('quant_model_resource_pressure_norm', 0.0) or 0.0)):.3f}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
