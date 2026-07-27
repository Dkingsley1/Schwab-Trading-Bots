from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import library_utilization_router as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_library_utilization_router_maps_all_non_mlx_packages_and_keeps_mlx_default(tmp_path: Path) -> None:
    lock = tmp_path / "config" / "requirements.lock.txt"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(
        "\n".join(
            [
                "mlx==0.31.1",
                "pandas==3.0.1",
                "duckdb==1.5.0",
                "QuantLib==1.42.1",
                "torch==2.10.0",
                "aiohttp==3.13.3",
                "schwab-py==1.5.1",
                "matplotlib==3.10.8",
                "pytest==9.0.2",
            ]
        ),
        encoding="utf-8",
    )
    health = tmp_path / "governance" / "health"
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "advisory", "throttle_profile": "soft_cap", "memory_pressure_level": "normal"})
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "advisory", "runtime_caps": {"max_concurrent_mlx_jobs": 2}})
    payload = src.build_payload(
        tmp_path,
        lock_file=lock,
        installed_versions={
            "mlx": "0.31.1",
            "pandas": "3.0.1",
            "duckdb": "1.5.0",
            "quantlib": "1.42.1",
            "torch": "2.10.0",
            "aiohttp": "3.13.3",
            "schwab-py": "1.5.1",
            "matplotlib": "3.10.8",
            "pytest": "9.0.2",
        },
    )

    lanes = payload["library_utilization_matrix"]["package_to_lane"]

    assert payload["overall_status"] == "ready"
    assert payload["coverage"]["coverage_ratio"] == 1.0
    assert payload["coverage"]["locked_non_mlx_package_count"] == 8
    assert "mlx" not in lanes
    assert lanes["pandas"] == "dataframe_feature_engine"
    assert lanes["duckdb"] == "storage_sql"
    assert lanes["quantlib"] == "quant_derivatives_risk"
    assert lanes["torch"] == "portable_ml_replay"
    assert payload["control_contract"]["uses_all_managed_non_mlx_libraries"] is True
    assert payload["control_contract"]["default_ml_backend"] == "mlx"
    assert payload["recommended_runtime_env"]["PRIMARY_ML_RUNTIME_BACKEND"] == "mlx"
    assert payload["recommended_runtime_env"]["PORTABLE_MODEL_REPLAY_POLICY"] == "canary_or_off_hours_only"


def test_library_utilization_router_apply_writes_env_caps(tmp_path: Path) -> None:
    lock = tmp_path / "config" / "requirements.lock.txt"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("pandas==3.0.1\nonnxruntime==1.24.3\n", encoding="utf-8")
    _write_json(tmp_path / "governance" / "health" / "runtime_throttle_control_latest.json", {"overall_status": "blocked", "throttle_profile": "protect_live", "memory_pressure_level": "high"})
    payload = src.build_payload(
        tmp_path,
        lock_file=lock,
        installed_versions={"pandas": "3.0.1", "onnxruntime": "1.24.3"},
    )
    override_path = tmp_path / "config" / ".env.library_utilization_router_override"
    result = src.write_outputs(
        payload,
        out_path=tmp_path / "governance" / "health" / "library_utilization_router_latest.json",
        external_context_path=tmp_path / "exports" / "external_context" / "library_utilization_router_latest.json",
        markdown_path=tmp_path / "exports" / "reports" / "operator" / "library_utilization_router_latest.md",
        override_path=override_path,
        apply=True,
    )
    override = override_path.read_text(encoding="utf-8")

    assert payload["runtime_caps"]["profile"] == "protect_live"
    assert payload["runtime_caps"]["max_portable_model_replay_jobs"] == 0
    assert result["applied"] is True
    assert "LIBRARY_UTILIZATION_ROUTER_ENABLED='1'" in override
    assert "PRIMARY_ML_RUNTIME_BACKEND='mlx'" in override


def test_library_utilization_router_routes_runtime_ahead_and_optional_fallback_as_advisory(tmp_path: Path) -> None:
    lock = tmp_path / "config" / "requirements.lock.txt"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(
        "\n".join(
            [
                "pandas==3.0.1",
                "pandas-ta==0.4.71b0",
                "ta==0.11.0",
                "requests==2.32.5",
            ]
        ),
        encoding="utf-8",
    )

    payload = src.build_payload(
        tmp_path,
        lock_file=lock,
        installed_versions={
            "pandas": "3.0.3",
            "ta": "0.11.0",
            "requests": "2.34.2",
        },
    )
    coverage = payload["coverage"]
    rows = {row["package"]: row for row in payload["package_inventory"]}

    assert payload["overall_status"] == "advisory"
    assert payload["ok"] is True
    assert coverage["missing_runtime_count"] == 0
    assert coverage["runtime_ahead_of_lock_count"] == 2
    assert coverage["optional_fallback_active_count"] == 1
    assert rows["pandas"]["status"] == "runtime_ahead_of_lock"
    assert rows["pandas-ta"]["status"] == "optional_fallback_active"
    assert rows["pandas-ta"]["available_fallback_packages"] == ["ta", "pandas", "numpy"][:2]
    assert "adopt newer runtime packages into the lock after canary evidence instead of marking active routes failed" in payload["recommended_actions"]


def test_library_utilization_router_stages_candidate_libraries_without_missing_runtime_debt(tmp_path: Path) -> None:
    lock = tmp_path / "config" / "requirements.lock.txt"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("pandas==3.0.1\n", encoding="utf-8")
    candidate_routes = tmp_path / "config" / "library_candidate_routes_v1.json"
    candidate_routes.write_text(
        json.dumps(
            {
                "candidate_libraries": [
                    {
                        "package": "pandera",
                        "lane": "dataframe_feature_engine",
                        "runtime_family": "python",
                        "priority": "high",
                        "reason": "schema checks",
                        "install_window": "maintenance",
                        "target_surfaces": ["feature_store"],
                        "target_functions": ["schema_validation"],
                        "compatibility_notes": ["import smoke first"],
                    },
                    {
                        "package": "mlxvm",
                        "lane": "language_reasoning",
                        "runtime_family": "mlx",
                        "priority": "high",
                        "reason": "model pinning",
                        "install_window": "maintenance",
                        "target_surfaces": ["research_pipeline"],
                        "target_functions": ["mlx_model_revision_pinning"],
                        "promotion_gate": "sandbox_model_cache_smoke",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = src.build_payload(
        tmp_path,
        lock_file=lock,
        installed_versions={"pandas": "3.0.1"},
    )
    candidates = {row["package"]: row for row in payload["candidate_library_routes"]}

    assert payload["overall_status"] == "ready"
    assert payload["coverage"]["missing_runtime_count"] == 0
    assert payload["candidate_library_matrix"]["candidate_package_count"] == 2
    assert payload["candidate_library_matrix"]["mapped_candidate_ratio"] == 1.0
    assert candidates["pandera"]["status"] == "candidate_only"
    assert candidates["pandera"]["runtime_family"] == "python"
    assert candidates["pandera"]["target_surfaces"] == ["feature_store"]
    assert candidates["pandera"]["target_functions"] == ["schema_validation"]
    assert candidates["pandera"]["compatibility_notes"] == ["import smoke first"]
    assert candidates["pandera"]["activation_profiles"] == ["live", "research"]
    assert candidates["pandera"]["activation_state"] == "profile_eligible_pending_install"
    assert candidates["mlxvm"]["runtime_family"] == "mlx"
    assert candidates["mlxvm"]["promotion_gate"] == "sandbox_model_cache_smoke"
    assert candidates["mlxvm"]["activation_profiles"] == ["research"]
    assert payload["candidate_library_matrix"]["runtime_family_counts"] == {"mlx": 1, "python": 1}
    assert payload["candidate_library_matrix"]["activation_state_counts"] == {"profile_eligible_pending_install": 2}
    assert payload["candidate_library_matrix"]["activation_profile_to_packages"]["live"] == ["pandera"]
    assert payload["candidate_library_matrix"]["activation_profile_to_packages"]["research"] == ["mlxvm", "pandera"]
    assert payload["candidate_library_matrix"]["target_function_to_packages"]["mlx_model_revision_pinning"] == ["mlxvm"]
    assert candidates["pandera"]["soak_policy"] == "do_not_count_candidate_only_as_missing_runtime"
    assert payload["control_contract"]["candidate_add_policy"] == "stage_candidates_without_dependency_mutation_then_install_only_in_maintenance_after_smoke"
    assert payload["control_contract"]["candidate_activation_state"] == "profile_eligible_is_not_live_enabled_until_installed_smoked_and_feature_gated"


def test_library_utilization_router_maps_new_candidate_lanes() -> None:
    assert src._infer_lane("statsforecast") == "time_series_forecasting"
    assert src._infer_lane("pyod") == "anomaly_drift_detection"
    assert src._infer_lane("lancedb") == "vector_memory_retrieval"
    assert src._infer_lane("aiolimiter") == "provider_rate_limit_cache"
    assert src._infer_lane("pandas-datareader") == "financial_filings_macro"
    assert src._infer_lane("sqlglot") == "sql_lineage_contracts"
    assert src._infer_lane("networkx") == "graph_network_analysis"
    assert src._infer_lane("lifelines") == "causal_survival_research"
    assert src._infer_lane("simpy") == "simulation_sensitivity"
    assert src._infer_lane("deepdiff") == "data_contract_validation"
    assert src._infer_lane("opentelemetry-sdk") == "telemetry_tracing"
    assert src._infer_lane("scalene") == "runtime_performance_profiling"
    assert src._infer_lane("ruff") == "production_quality_gates"
    assert src._infer_lane("bandit") == "security_supply_chain"
    assert src._infer_lane("locust") == "load_resilience_testing"
    assert src._infer_lane("arq") == "queue_job_orchestration"
    assert src._infer_lane("dynaconf") == "config_release_controls"
    assert src._infer_lane("aiometer") == "async_flow_control"
    assert src._infer_lane("rapidfuzz") == "nlp_tokenization_research"
    assert src._infer_lane("feedparser") == "broker_market_data"
    assert src._is_mlx_routed("mlxvm") is True
