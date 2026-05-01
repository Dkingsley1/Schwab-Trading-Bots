import importlib.util
import json
import subprocess
from pathlib import Path


MODULE_PATH = Path("/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/system_summary_report.py")
spec = importlib.util.spec_from_file_location("system_summary_report", MODULE_PATH)
system_summary_report = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(system_summary_report)


def test_system_summary_pdf_renderer_uses_app_bundle_for_headless_policy(monkeypatch, tmp_path):
    browser_bin = tmp_path / "Google Chrome.app" / "Contents" / "MacOS" / "Google Chrome"
    browser_bin.parent.mkdir(parents=True, exist_ok=True)
    browser_bin.write_text("", encoding="utf-8")

    monkeypatch.setattr(system_summary_report, "APP_BROWSER_CANDIDATES", (browser_bin,))
    monkeypatch.setattr(system_summary_report.shutil, "which", lambda _name: None)

    renderer, kind = system_summary_report._pdf_renderer_binary(allow_gui_renderer=False)

    assert renderer == str(browser_bin)
    assert kind == "browser_app"


def test_system_summary_run_returns_timeout_payload(monkeypatch):
    def _timeout(*_args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["chrome"], timeout=kwargs.get("timeout", 2.0))

    monkeypatch.setattr(system_summary_report.subprocess, "run", _timeout)

    rc, out, err = system_summary_report._run(["chrome"], timeout_seconds=2.0)

    assert rc == 124
    assert out == ""
    assert "timeout_after_seconds=2.0" in err


def test_system_summary_report_builds_compiled_payload(tmp_path):
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    reports_root = project_root / "exports" / "reports"
    one_numbers_root = project_root / "exports" / "one_numbers" / "latest"
    showcase_root = project_root / "docs" / "showcase" / "generated"

    (health_root).mkdir(parents=True, exist_ok=True)
    (reports_root / "project_timeline").mkdir(parents=True, exist_ok=True)
    (reports_root / "system_explainers").mkdir(parents=True, exist_ok=True)
    (reports_root / "showcase").mkdir(parents=True, exist_ok=True)
    (reports_root / "system_summary").mkdir(parents=True, exist_ok=True)
    one_numbers_root.mkdir(parents=True, exist_ok=True)
    showcase_root.mkdir(parents=True, exist_ok=True)
    (project_root / "governance" / "champion_challenger").mkdir(parents=True, exist_ok=True)
    (project_root / "governance" / "experiments").mkdir(parents=True, exist_ok=True)

    (health_root / "section_grade_guard_latest.json").write_text(
        json.dumps(
            {
                "overall_letter_grade": "A+",
                "raw_overall_letter_grade": "A+",
                "sections": [
                    {"section": "architecture_and_modularity", "letter_grade": "A+", "raw_letter_grade": "A+", "score": 100.0, "state": "at_floor"},
                    {"section": "training_and_model_quality", "letter_grade": "A+", "raw_letter_grade": "A+", "score": 98.0, "state": "at_floor"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (health_root / "platform_control_plane_latest.json").write_text(
        json.dumps(
            {
                "institutional_readiness": {"overall_score": 98.67, "overall_status": "industry_leaning", "weakest_domains": []},
                "institutional_domains_by_slug": {
                    "formal_model_governance": {"score": 100.0},
                    "observability_and_slo": {"score": 100.0},
                },
            }
        ),
        encoding="utf-8",
    )
    (showcase_root / "highlights_latest.json").write_text(
        json.dumps(
            {
                "bot_summary": {
                    "active_count": 30,
                    "total_registered": 107,
                    "active_roles": {"core": 18, "infrastructure_sub_bot": 4},
                    "top_active_bots": [
                        {
                            "bot_id": "brain_refinery_v4_simple",
                            "bot_role": "core",
                            "test_accuracy": 0.64,
                            "quality_score": 93.2,
                        }
                    ],
                },
                "architecture_summary": {"upgrade_count": 12, "ready_count": 12, "portable_host_profile": "max_throughput"},
                "special_features": {
                    "adaptive_apple_silicon_brain": "Host-aware tuning keeps MLX-native performance first-class.",
                    "self_healing_ops_plane": "Bounded autopilots keep the stack readable under pressure.",
                },
            }
        ),
        encoding="utf-8",
    )
    (one_numbers_root / "one_numbers_summary.json").write_text(
        json.dumps(
            {
                "combined_decision_total_rows": 84766,
                "combined_governance_total_rows": 121039,
                "combined_blocked_rate": 0.050881,
                "paper_executed_total": 128,
            }
        ),
        encoding="utf-8",
    )
    (health_root / "cost_telemetry_latest.json").write_text(
        json.dumps({"storage_cost_proxy": {"tracked_sqlite_gb": 165.581, "pressure_index": 0.62}}),
        encoding="utf-8",
    )
    (health_root / "live_readiness_smoke_latest.json").write_text(
        json.dumps({"readiness_score": 100.0, "mode": "validate_only"}),
        encoding="utf-8",
    )
    (health_root / "incident_closeout_autopilot_latest.json").write_text(
        json.dumps({"closeout_score": 92.0}),
        encoding="utf-8",
    )
    (health_root / "training_quality_control_latest.json").write_text(
        json.dumps({"training_quality_index": 109.38, "training_quality_base_score": 100.0, "training_quality_bonus_score": 9.38, "training_quality_score": 100.0}),
        encoding="utf-8",
    )
    (health_root / "training_lineage_manifest_latest.json").write_text(
        json.dumps({"lineage_score": 100.0}),
        encoding="utf-8",
    )
    (health_root / "chaos_drill_coordinator_latest.json").write_text(
        json.dumps({"overall_status": "ready", "restore_discipline": {"restore_proof_ready": True}}),
        encoding="utf-8",
    )
    (health_root / "cross_host_parity_report_latest.json").write_text(
        json.dumps({"overall_status": "ready", "proof_written_count": 3, "nightly_proof_ready": True}),
        encoding="utf-8",
    )
    (health_root / "incident_timeline_latest.json").write_text(
        json.dumps(
            {
                "overall_status": "degraded",
                "watch_surfaces": [
                    {"surface": "release_window", "status": "scheduled"},
                    {"surface": "portable_sidecar", "status": "monitoring"},
                ],
                "recommended_actions": ["Review release packet before supervised canary."],
            }
        ),
        encoding="utf-8",
    )
    (health_root / "architecture_upgrade_scoreboard_latest.json").write_text(
        json.dumps(
            {
                "upgrade_count": 3,
                "ready_count": 3,
                "rows": [
                    {
                        "title": "Adaptive Apple Silicon Brain",
                        "status": "ready",
                        "proof": "host_profile=max_throughput memory_architecture=unified",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (reports_root / "project_timeline" / "project_timeline_latest.md").write_text(
        "# Timeline\n\n- Added a signed immutable experiment ledger.\n- Added nightly cross-host parity proofs.\n",
        encoding="utf-8",
    )
    (project_root / "governance" / "experiments" / "immutable_experiment_ledger_latest.json").write_text(
        json.dumps(
            {
                "overall_status": "ready",
                "append_only_ready": True,
                "latest_exact_replay_ready": True,
                "latest_signature_ready": True,
                "latest_attestation_ready": True,
                "ledger_row_count": 2,
            }
        ),
        encoding="utf-8",
    )
    (project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json").write_text(
        json.dumps(
            {
                "packet_complete": True,
                "ready_for_committee": True,
                "committee_packet_seed_ready": True,
                "trained_models_complete": True,
                "signature": {"verified": True},
                "replayability_contract": {"exact_replay_ready": True},
            }
        ),
        encoding="utf-8",
    )
    (reports_root / "system_explainers" / "framework_map_v2_latest.html").write_text("<html></html>", encoding="utf-8")
    (showcase_root / "special_features_latest.html").write_text("<html></html>", encoding="utf-8")
    (reports_root / "report_pdf_bundle_latest.html").write_text("<html></html>", encoding="utf-8")

    payload = system_summary_report.build_payload(project_root, refresh_supporting_artifacts=False)
    rendered = system_summary_report._render_html(payload)

    assert payload["section_grade_board"]["overall_letter_grade"] == "A+"
    assert payload["active_bots"]["active_count"] == 30
    assert payload["proof_stack"]["immutable_ledger"]["latest_exact_replay_ready"] is True
    assert payload["institutional_readiness"]["strongest_domains"][0]["score"] >= payload["institutional_readiness"]["frontier_domains"][0]["score"]
    assert "release_window" in payload["watchlist"][0]
    assert any(row["slug"] == "framework_map" and row["available"] for row in payload["report_catalog"])
    assert "Trading Platform Executive Packet" in rendered
    assert "Proof Stack" in rendered
    assert "Institutional Domain Map" in rendered
    assert "Live Watchlist" in rendered
    assert "brain_refinery_v4_simple" in rendered
    assert "Adaptive Apple Silicon Brain" in rendered
    assert "Added a signed immutable experiment ledger." in rendered
