import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ops" / "decision_intelligence_layer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("decision_intelligence_layer", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load decision_intelligence_layer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_decision_intelligence_layer_integrates_all_six_sections(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    walk = tmp_path / "governance" / "walk_forward"

    shadow_dir = tmp_path / "governance" / "shadow_crypto"
    shadow_dir.mkdir(parents=True)
    rows = [
        {
            "timestamp_utc": "2026-05-19T14:00:00+00:00",
            "symbol": "BTC-USD",
            "action": "SELL",
            "score": 0.38,
            "threshold": 0.58,
            "pnl_proxy": 0.012,
            "return_1m": -0.004,
            "reasons": ["quant_strategy_conviction=0.740", "flow_dir=-0.520"],
            "grand_master_meta": {
                "quant_strategy_conviction": 0.74,
                "quant_strategy_fit": 0.80,
                "quant_strategy_execution_alignment": 0.82,
            },
            "features": {
                "mom_1m": -0.003,
                "mom_5m": -0.006,
                "flow_direction_signed": -0.52,
                "lead_lag_signal_signed": -0.40,
                "market_micro_order_flow_imbalance_norm": 0.31,
                "market_micro_tradeability_score_norm": 0.78,
                "quant_kelly_fraction_norm": 0.66,
                "quant_strategy_risk_adjusted_conviction_norm": 0.74,
                "quant_strategy_portfolio_fit_norm": 0.80,
                "quant_strategy_execution_alignment_norm": 0.82,
                "quant_strategy_allocation_bias_norm": 0.76,
                "quant_strategy_crypto_basis_edge_norm": 0.69,
                "quant_cvar_tail_risk_norm": 0.44,
                "allocation_confidence_norm": 0.72,
            },
        }
    ]
    (shadow_dir / "shadow_pnl_attribution_20260519.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    _write_json(
        health / "strategy_attribution_latest.json",
        {
            "ok": True,
            "day": "20260519",
            "row_count": 20,
            "total_pnl_proxy": 0.25,
            "top_lane": "shadow_crypto",
            "top_layer": "grand_master",
        },
    )
    _write_json(
        health / "paper_performance_latest.json",
        {
            "ok": True,
            "history_daily_series": [
                {"ending_net_pnl_total": 1.0, "change_vs_previous_day": 0.10},
                {"ending_net_pnl_total": 1.4, "change_vs_previous_day": 0.40},
            ],
        },
    )
    _write_json(
        walk / "promotion_readiness_latest.json",
        {
            "promote_ok": False,
            "coverage_ok": True,
            "considered_bots": 20,
            "fail_share": 0.30,
            "readiness_margin": -0.05,
            "blocking_reasons": ["fail_share_above_limit"],
            "near_pass_examples": [
                {
                    "bot_id": "brain_refinery_v10",
                    "runs": 11,
                    "forward_mean": 0.60,
                    "delta": 0.02,
                    "failed_gates": {"runs": True},
                }
            ],
        },
    )
    _write_json(walk / "promotion_gate_latest.json", {"promote_ok": False, "coverage_ok": True})
    _write_json(walk / "promotion_bottleneck_latest.json", {"overall_status": "needs_work"})
    _write_json(
        tmp_path / "governance" / "research" / "decay_monitor_latest.json",
        {
            "ok": True,
            "overall_status": "needs_work",
            "weak_sleeve_count": 1,
            "weak_sleeves": [{"profile": "crypto_futures", "ending_net_pnl_total": -1.2}],
            "latest_change_vs_previous_day": -0.1,
            "pnl_slope": -0.2,
        },
    )
    _write_json(
        tmp_path / "governance" / "platform_intelligence" / "duplicate_alpha_overlap_latest.json",
        {
            "overall_status": "needs_work",
            "overlap_cluster_count": 12,
            "high_overlap_cluster_count": 2,
            "overlap_clusters": [{"cluster_id": "c1", "members": ["a", "b"]}],
        },
    )
    _write_json(
        tmp_path / "governance" / "platform_stabilization_quality" / "duplicate_alpha_compression_latest.json",
        {"overall_status": "needs_work", "overlap_cluster_count": 12, "recommended_commands": ["compress"]},
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "blocked",
            "snapshot_ready": True,
            "snapshot_age_minutes": 10.0,
            "coverage_repair_ready": True,
            "resource_guard": {"ok": True, "memory_pressure_state": "green", "swap_used_gb": 0.2},
            "runtime_backend_parity": {"parity_state": "ready"},
            "training_quality": {"overall_status": "blocked"},
            "precompute_targets": [
                {
                    "bot_id": "brain_refinery_v10",
                    "family": "general",
                    "priority": 15.0,
                    "reasons": ["coverage_seed"],
                    "actions": ["reuse_shared_snapshot_for_walk_forward"],
                }
            ],
            "recommended_actions": ["wait for ingestion quality"],
        },
    )
    for name in (
        "market_micro_sync_latest.json",
        "crypto_market_context_sync_latest.json",
        "market_crypto_correlation_sync_latest.json",
        "fx_market_context_sync_latest.json",
        "source_verification_latest.json",
    ):
        _write_json(health / name, {"ok": True})

    payload = module.build_payload(tmp_path, symbol="BTC", max_files=4, max_rows=50)

    assert payload["overall_status"] == "needs_work"
    assert set(payload["sections"]) == {
        "quant_attribution_ledger",
        "promotion_gate_intelligence",
        "shadow_paper_feedback_loop",
        "strategy_decay_monitor",
        "duplicate_alpha_governor",
        "training_launch_preflight",
        "market_move_explainer",
    }
    assert payload["sections"]["quant_attribution_ledger"]["status"] == "ready"
    assert payload["sections"]["promotion_gate_intelligence"]["closest_candidates"][0]["bot_id"] == "brain_refinery_v10"
    assert payload["sections"]["shadow_paper_feedback_loop"]["agreement"] == "aligned"
    assert payload["sections"]["duplicate_alpha_governor"]["high_overlap_cluster_count"] == 2
    training = payload["sections"]["training_launch_preflight"]
    assert training["status"] == "needs_work"
    assert training["mode"] == "prep_only"
    assert training["launch_allowed"] is False
    assert training["prep_allowed"] is True
    assert training["safe_prep_targets"][0]["bot_id"] == "brain_refinery_v10"
    market = payload["sections"]["market_move_explainer"]
    assert market["overall_status"] == "ready"
    assert market["ranked_drivers"][0]["direction"] == "selling"
    assert any(action for action in payload["integrated_actions"])


def test_training_launch_preflight_allows_small_canary_when_gates_clear(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "ready",
            "snapshot_ready": True,
            "snapshot_age_minutes": 5.0,
            "coverage_repair_ready": True,
            "resource_guard": {"ok": True, "memory_pressure_state": "green", "swap_used_gb": 0.1},
            "runtime_backend_parity": {"parity_state": "ready"},
            "training_quality": {"overall_status": "ready"},
            "precompute_targets": [{"bot_id": "brain_refinery_v50", "priority": 18.0, "actions": ["reuse_shared_snapshot_for_walk_forward"]}],
        },
    )

    preflight = module.build_training_launch_preflight(tmp_path)

    assert preflight["status"] == "ready"
    assert preflight["mode"] == "canary_training_allowed"
    assert preflight["launch_allowed"] is True
    assert preflight["safe_prep_targets"][0]["launch_recommendation"] == "canary_ok"
    assert any(
        any("coverage-gap-closer" in part for part in command)
        for command in preflight["recommended_commands"]
    )


def test_training_launch_preflight_consumes_runtime_launch_contract(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "blocked",
            "snapshot_ready": True,
            "snapshot_age_minutes": 1.0,
            "coverage_repair_ready": True,
            "resource_guard": {"ok": True, "memory_pressure_state": "green", "swap_used_gb": 0.1},
            "runtime_backend_parity": {"parity_state": "ready"},
            "training_quality": {"overall_status": "blocked"},
            "training_launch_contract": {
                "mode": "prep_only",
                "launch_allowed": False,
                "prep_allowed": True,
                "launch_blockers": ["backpressure_overload_severe", "training_quality_blocked"],
                "backpressure_gate": {"pending_lines": 87605, "severe": True, "cooling_down": False},
                "canary_batch": [{"bot_id": "brain_refinery_v10"}],
                "repair_first_targets": [{"bot_id": "brain_refinery_v35"}],
                "prep_targets": [
                    {"bot_id": "brain_refinery_v10", "training_stage": "promotion_confirmation", "priority": 15.0},
                    {"bot_id": "brain_refinery_v35", "training_stage": "repair_first", "priority": 15.0, "needs_runtime_input_repair": True},
                ],
                "recommended_prep_commands": [["./scripts/ops/opsctl.sh", "training-runtime-control", "--json"]],
                "recommended_retrain_command": [],
            },
        },
    )

    preflight = module.build_training_launch_preflight(tmp_path)

    assert preflight["mode"] == "prep_only"
    assert preflight["backpressure_gate"]["pending_lines"] == 87605
    assert "backpressure_overload_severe" in preflight["blockers"]
    assert preflight["canary_batch_bot_ids"] == ["brain_refinery_v10"]
    assert preflight["repair_first_bot_ids"] == ["brain_refinery_v35"]
    assert preflight["safe_prep_targets"][1]["launch_recommendation"] == "repair_first"
