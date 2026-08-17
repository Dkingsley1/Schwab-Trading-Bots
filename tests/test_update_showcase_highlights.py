import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import update_showcase_highlights as highlights


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_showcase_snapshot_surfaces_real_world_readiness(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    docs_root = project_root / "docs" / "showcase"
    generated_root = docs_root / "generated"
    health_root = project_root / "governance" / "health"
    reports_root = project_root / "exports" / "reports"
    readme_path = project_root / "README.md"

    monkeypatch.setattr(highlights, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(highlights, "DOCS_ROOT", docs_root)
    monkeypatch.setattr(highlights, "GENERATED_ROOT", generated_root)
    monkeypatch.setattr(highlights, "README_PATH", readme_path)
    monkeypatch.setattr(highlights, "HIGHLIGHTS_JSON", generated_root / "highlights_latest.json")
    monkeypatch.setattr(highlights, "HIGHLIGHTS_MD", generated_root / "highlights_latest.md")
    monkeypatch.setattr(highlights, "SPECIAL_FEATURES_HTML", generated_root / "special_features_latest.html")

    project_root.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(
        f"{highlights.README_START}\nold\n{highlights.README_END}\n",
        encoding="utf-8",
    )
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": "bot_a", "bot_role": "signal_sub_bot", "active": True, "test_accuracy": 0.62, "quality_score": 0.51},
                {"bot_id": "bot_b", "bot_role": "infrastructure_sub_bot", "active": True, "test_accuracy": 0.58, "quality_score": 0.49},
            ],
            "master_policy": {"protected_collection_lane_floors": {"infrastructure_sub_bot": 6}},
        },
    )
    _write_json(health_root / "data_ingress_latest_alpha.json", {"loop_state": "running", "iter": 4, "api_error": 0})
    _write_json(health_root / "crypto_market_context_sync_latest.json", {"ok_source_count": 12, "source_count": 14, "news_ok_source_count": 4, "news_source_count": 5})
    _write_json(health_root / "data_source_divergence_latest.json", {"ok": True, "worst_relative_spread": 0.0125})
    _write_json(health_root / "market_crypto_correlation_sync_latest.json", {"mode": "exact", "aligned_pairs": 9, "cache_hits": 20, "cache_misses": 1})
    _write_json(
        health_root / "training_success_latest.json",
        {
            "timestamp_utc": "2026-04-21T14:00:00+00:00",
            "confirmed_training_success": False,
            "trained_count": 0,
            "failure_count": 1,
            "reason": "training_failures_present:1",
            "failure_details": [{"reason": "ModuleNotFoundError: No module named 'mlx'"}],
        },
    )
    _write_json(health_root / "shadow_watchdog_tripwire_latest.json", {"active": False})
    _write_json(
        health_root / "process_watchdog_latest.json",
        {
            "timestamp_utc": "2026-04-21T14:20:00+00:00",
            "status": [
                {"name": "paper", "running": 1, "heartbeat_ok": True},
                {"name": "futures", "running": 1, "heartbeat_ok": True},
            ],
            "restart_storms": [],
            "alerts": [],
        },
    )
    _write_json(
        health_root / "live_readiness_smoke_latest.json",
        {
            "overall_status": "ready",
            "readiness_score": 94.2,
            "broker_ready": True,
            "session_ready": True,
            "process_watchdog": {"healthy": True},
        },
    )
    _write_json(
        health_root / "live_money_readiness_contract_latest.json",
        {
            "overall_status": "blocked",
            "faithful_live_money_ready": False,
            "live_money_locked": True,
            "grade_summary": {
                "required_section_count": 14,
                "ready_required_section_count": 13,
            },
        },
    )
    _write_json(
        health_root / "live_runtime_separation_control_latest.json",
        {
            "overall_status": "degraded",
            "shared_host_pressure": {"contention_score": 2},
            "release_contract": {"live_lane_should_be_read_only": True},
        },
    )
    _write_json(
        health_root / "platform_control_plane_latest.json",
        {
            "institutional_readiness": {
                "overall_status": "advancing",
                "overall_score": 81.25,
                "domain_count": 12,
                "weakest_domains": [
                    {"slug": "immutable_experiment_tracking", "title": "Immutable experiment tracking", "score": 40.0}
                ],
            }
        },
    )
    _write_json(
        health_root / "pytorch_replay_canary_latest.json",
        {
            "recommendations": ["keep_mlx_live_default_backend"],
            "mlx_shadow_assist": {
                "status": "active_candidates",
                "eligible_source_profiles": [{"source_profile": "intraday_aggressive"}],
            },
            "scoreboard": {"runs_tracked": 3, "positive_calibrated_runs": 1, "active_assist_candidate_runs": 2},
        },
    )
    _write_json(
        health_root / "autonomy_control_plane_latest.json",
        {
            "overall_status": "degraded",
            "autonomy_score": 78.3,
            "lane_recovery_playbooks": {"triggered_playbook_count": 2},
        },
    )
    _write_json(
        health_root / "portable_brain_contract_latest.json",
        {
            "host_contract": {
                "host_profile": "max_throughput",
                "chip": "Apple M5 Max",
                "memory_architecture": "unified",
                "shared_cpu_gpu_memory_pool": True,
                "memory_competitive_advantage": "Apple Silicon unified memory keeps CPU, GPU, and MLX tensors in one pool.",
            },
            "cross_platform_proof_node": {"status": "ready"},
        },
    )
    _write_json(health_root / "mode_switchboard_mission_control_latest.json", {"mode_counts": {"active": 2}})
    _write_json(health_root / "decision_provenance_cards_latest.json", {"card_count": 4})
    _write_json(health_root / "notification_escalation_ladder_latest.json", {"remote_pager_ready": True})
    _write_json(health_root / "incident_review_packet_latest.json", {"review_required": True})
    _write_json(health_root / "chaos_drill_coordinator_latest.json", {"drill_program": {"program_score": 82.0}})
    _write_json(
        health_root / "macro_event_intelligence_latest.json",
        {
            "overall_status": "ready",
            "source": "C-SPAN",
            "speaker": "Supreme Court / C-SPAN legal coverage",
            "transcript_quality": "aligned_transcript",
            "transcript_quality_score": 0.8645,
            "cue_match_score": 1.0,
            "stance": "neutral",
            "sentiment_hint": -0.1209,
            "market_relevance": "high",
        },
    )
    _write_json(
        health_root / "architecture_upgrade_scoreboard_latest.json",
        {
            "upgrade_count": 12,
            "ready_count": 8,
            "special_features_map": {
                "adaptive_apple_silicon_brain": "Adaptive Apple Silicon Brain: host-aware tuning now recognizes `Apple M5 Max`, sees memory architecture `unified`, and lands on `max_throughput` before the stack starts.",
                "three_mode_switchboard": "Three-Mode Switchboard: mission control now tracks shadow/paper/live with `2` active modes and runtime clearance `awaiting_coverage_cycles`.",
                "event_to_trade_intelligence": "Event-to-Trade Intelligence: the macro lane now surfaces live-detection and media ingest proof as `degraded` with `live_detected=0 media_status=missing`.",
                "self_healing_ops_plane": "Self-Healing Ops Plane: autonomy currently sits at `78.30/100` with `2` triggered playbooks.",
                "portable_brain_contract": "Portable Brain Contract: the host contract now recommends `native` mode with proof-node status `ready` and backend `onnx` while keeping the broker/runtime seam portable.",
            },
        },
    )
    _write_json(health_root / "incident_timeline_latest.json", {"open_incident_count": 1})
    _write_json(
        project_root / "governance" / "champion_challenger" / "promotion_autopilot_packet_latest.json",
        {
            "autopilot_state": "awaiting_approval",
            "approval_record": {"approval_state": "awaiting_operator_signoff"},
        },
    )
    _write_json(reports_root / "daily_ops_report_latest.json", {"quality": {"data_quality_score": 0.93}})

    rc = highlights.main()

    assert rc == 0
    snapshot = json.loads((generated_root / "highlights_latest.json").read_text(encoding="utf-8"))
    markdown = (generated_root / "highlights_latest.md").read_text(encoding="utf-8")
    special_features_html = (generated_root / "special_features_latest.html").read_text(encoding="utf-8")
    updated_readme = readme_path.read_text(encoding="utf-8")

    assert snapshot["readiness_summary"]["institutional_score"] == 81.25
    assert snapshot["readiness_summary"]["live_status"] == "blocked"
    assert snapshot["readiness_summary"]["live_ready_section_count"] == 13
    assert snapshot["readiness_summary"]["live_required_section_count"] == 14
    assert snapshot["readiness_summary"]["live_money_locked"] is True
    assert snapshot["readiness_summary"]["runtime_smoke_status"] == "ready"
    assert set(snapshot["artifacts"]) == {"crypto_context", "correlation", "training"}
    assert snapshot["pytorch_summary"]["assist_candidate_count"] == 1
    assert snapshot["autonomy_summary"]["autonomy_score"] == 78.3
    assert snapshot["architecture_summary"]["ready_count"] == 8
    assert "Institutional-readiness score is `81.25/100`" in markdown
    assert "PyTorch research lane: `1` assist candidates over `3` tracked runs." in markdown
    assert "Autonomy posture: `degraded` at `78.30/100`" in markdown
    assert "Architecture posture: `8/12` proof surfaces ready" in markdown
    assert "Adaptive Apple Silicon Brain: host-aware tuning now recognizes `Apple M5 Max`" in markdown
    assert "unified-memory-aware runtime tuning on Apple Silicon" in markdown
    assert "## Special Feature Proof Notes" in markdown
    assert "### Adaptive Apple Silicon Brain" in markdown
    assert "Recognized host `Apple M5 Max`" in markdown
    assert "Memory architecture is `unified`" in markdown
    assert "Special Features And Highlights" in special_features_html
    assert "Feature Proof Surface" in special_features_html
    assert "Executive Feature Report" in special_features_html
    assert "Why it matters" in special_features_html
    assert "Interpretation Notes" in special_features_html
    assert "Recommendations" in special_features_html
    assert "Adaptive Apple Silicon Brain" in special_features_html
    assert "Recognized host" in special_features_html
    assert "Apple Silicon unified memory gives the live stack one shared CPU and GPU pool" in special_features_html
    assert "broker-specific news, options, and calendar context now sit behind adapter seams" in special_features_html.lower()
    assert "Latest macro event status is" in special_features_html
    assert "Institutional readiness: `81.25/100` with status `advancing`." in updated_readme
    assert "live-money gate `blocked` at `13/14` required sections with live locked `True`" in updated_readme
    assert "runtime smoke `ready` at `94.20/100`" in updated_readme
    assert "Autonomy posture: `78.30/100` with status `degraded`" in updated_readme
    assert "Architecture upgrades: `8/12` ready proof surfaces" in updated_readme
