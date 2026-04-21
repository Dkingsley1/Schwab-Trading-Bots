import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.strategy_research_lane as lane


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_strategy_research_lane_aggregates_research_and_promotion_views(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    governance = project_root / "governance"
    (governance / "shadow_intraday_aggressive_equities").mkdir(parents=True, exist_ok=True)
    (governance / "champion_challenger").mkdir(parents=True, exist_ok=True)
    (project_root / "exports" / "paper_broker_bridge" / "paper").mkdir(parents=True, exist_ok=True)

    (governance / "shadow_intraday_aggressive_equities" / "shadow_pnl_attribution_20260401.jsonl").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-01T14:00:00+00:00",
                "symbol": "AAPL",
                "bot_id": "brain_refinery_v4_simple",
                "action": "BUY",
                "layer": "grand_master",
                "pnl_proxy": 1.25,
                "return_1m": 0.01,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (project_root / "exports" / "paper_broker_bridge" / "paper" / "paper_bridge_orders_20260401.jsonl").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-01T14:00:00+00:00",
                "symbol": "AAPL",
                "action": "BUY",
                "strategy": "paper_mirror::alpha",
                "metadata": {"source_profile": "intraday_aggressive"},
                "model_score": 0.71,
                "threshold": 0.60,
                "tradeability_score": 0.55,
                "allocation_conflict_norm": 0.10,
                "realized_pnl_total": 1.0,
                "unrealized_pnl_total": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        governance / "champion_challenger" / "registry.json",
        {
            "champion": {
                "name": "current_alpha",
                "stage": "paper",
                "since_utc": "2026-04-01T13:00:00+00:00",
            },
            "history": [{"name": "previous_beta"}],
            "last_event": {"action": "hold"},
        },
    )
    _write_json(
        governance / "health" / "derived_state_latest.json",
        {
            "timestamp_utc": "2026-04-01T14:10:00+00:00",
            "ok": True,
            "risk_level": "medium",
            "gross_exposure_cap": 0.72,
            "max_total_actions_per_hour": 88,
            "sleeves": {"core": {"target_weight": 0.45}},
            "source_paths": {"allocator": "allocator.json"},
        },
    )

    def fake_command_runner(name: str, cmd: list[str]) -> tuple[dict, dict]:
        if name == "research_sandbox":
            payload = {
                "ok": True,
                "steps": [
                    {"cmd": ["build_trade_learning_dataset.py"], "rc": 0},
                    {"cmd": ["walk_forward_validate.py"], "rc": 0},
                    {"cmd": ["walk_forward_promotion_gate.py"], "rc": 0},
                ],
            }
            out_file = project_root / "exports" / "research_sandbox" / "latest.json"
        else:
            payload = {
                "promote_ok": False,
                "coverage_ok": True,
                "considered_bots": 5,
                "failed_bots": 1,
                "readiness_margin": 0.12,
                "recommended_retrain": {
                    "include_bot_ids": ["brain_refinery_v12_news_shocks"],
                    "regime_focus": "shock",
                },
            }
            out_file = project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json"
        _write_json(out_file, payload)
        return payload, {
            "name": name,
            "mode": "subprocess",
            "ok": True,
            "rc": 0,
            "duration_ms": 1.0,
            "cmd": cmd,
            "out_file": str(out_file),
        }

    payload = lane.build_strategy_research_payload(
        project_root,
        day="20260401",
        max_rows=100,
        command_runner=fake_command_runner,
    )

    assert payload["ok"] is True
    assert payload["strategy_attribution"]["top_lane"] == "shadow_intraday_aggressive_equities"
    assert payload["counterfactual_replay"]["top_candidate"]["profile"] == "intraday_aggressive"
    assert payload["research_sandbox"]["failed_steps"] == []
    assert payload["promotion_readiness"]["recommended_retrain"]["regime_focus"] == "shock"
    assert payload["summary"]["recommended_action"] == "run_targeted_retrain"
    assert payload["champion"]["name"] == "current_alpha"
    assert payload["derived_state"]["risk_level"] == "medium"
    assert (project_root / "governance" / "health" / "strategy_attribution_latest.json").exists()
    assert (project_root / "exports" / "reports" / "strategy_attribution_latest.md").exists()


def test_strategy_research_lane_reuses_fresh_artifacts(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk_forward = project_root / "governance" / "walk_forward"
    (project_root / "governance" / "champion_challenger").mkdir(parents=True, exist_ok=True)

    _write_json(
        health / "strategy_attribution_latest.json",
        {
            "timestamp_utc": "2099-04-01T14:00:00+00:00",
            "ok": True,
            "row_count": 1,
            "file_count": 1,
            "top_lane": "shadow_intraday_aggressive_equities",
            "top_layer": "grand_master",
            "total_pnl_proxy": 1.2,
        },
    )
    _write_json(
        health / "counterfactual_replay_latest.json",
        {
            "timestamp_utc": "2099-04-01T14:01:00+00:00",
            "ok": True,
            "profiles_reviewed": ["intraday_aggressive"],
            "candidate_count": 3,
            "top_candidates": [{"profile": "intraday_aggressive", "aggregate_net_pnl_total": 2.5}],
        },
    )
    _write_json(
        walk_forward / "promotion_readiness_latest.json",
        {
            "timestamp_utc": "2099-04-01T14:02:00+00:00",
            "promote_ok": False,
            "coverage_ok": True,
            "considered_bots": 2,
            "failed_bots": 0,
            "readiness_margin": 0.2,
            "recommended_retrain": {},
        },
    )
    _write_json(
        project_root / "exports" / "research_sandbox" / "latest.json",
        {
            "timestamp_utc": "2099-04-01T14:03:00+00:00",
            "ok": True,
            "steps": [{"cmd": ["walk_forward_validate.py"], "rc": 0}],
        },
    )
    _write_json(project_root / "governance" / "champion_challenger" / "registry.json", {"champion": {"name": "alpha"}})

    def _failing_runner(_name: str, _cmd: list[str]) -> tuple[dict, dict]:
        raise AssertionError("fresh artifact should have been reused")

    payload = lane.build_strategy_research_payload(
        project_root,
        day="20260401",
        max_rows=100,
        skip_sandbox=False,
        max_age_minutes=60,
        sandbox_max_age_minutes=120,
        command_runner=_failing_runner,
    )

    assert payload["ok"] is True
    assert all(step["mode"] == "artifact" for step in payload["steps"])


def test_strategy_research_lane_skip_sandbox_without_existing_artifact_stays_green(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk_forward = project_root / "governance" / "walk_forward"
    (project_root / "governance" / "champion_challenger").mkdir(parents=True, exist_ok=True)

    _write_json(
        health / "strategy_attribution_latest.json",
        {
            "timestamp_utc": "2099-04-01T14:00:00+00:00",
            "ok": True,
            "row_count": 1,
            "file_count": 1,
            "top_lane": "shadow_intraday_aggressive_equities",
            "top_layer": "grand_master",
            "total_pnl_proxy": 1.2,
        },
    )
    _write_json(
        health / "counterfactual_replay_latest.json",
        {
            "timestamp_utc": "2099-04-01T14:01:00+00:00",
            "ok": True,
            "profiles_reviewed": ["intraday_aggressive"],
            "candidate_count": 1,
            "top_candidates": [{"profile": "intraday_aggressive", "aggregate_net_pnl_total": 2.5}],
        },
    )
    _write_json(
        walk_forward / "promotion_readiness_latest.json",
        {
            "timestamp_utc": "2099-04-01T14:02:00+00:00",
            "promote_ok": False,
            "coverage_ok": True,
            "considered_bots": 2,
            "failed_bots": 0,
            "readiness_margin": 0.2,
            "recommended_retrain": {},
        },
    )
    _write_json(project_root / "governance" / "champion_challenger" / "registry.json", {"champion": {"name": "alpha"}})

    def _failing_runner(_name: str, _cmd: list[str]) -> tuple[dict, dict]:
        raise AssertionError("skip-sandbox lane should not call subprocesses when fresh artifacts exist")

    payload = lane.build_strategy_research_payload(
        project_root,
        day="20260401",
        max_rows=100,
        skip_sandbox=True,
        max_age_minutes=60,
        sandbox_max_age_minutes=120,
        command_runner=_failing_runner,
    )

    assert payload["ok"] is True
    assert payload["research_sandbox"]["ok"] is True
    assert payload["research_sandbox"]["step_count"] == 0
    sandbox_step = next(step for step in payload["steps"] if step["name"] == "research_sandbox")
    assert sandbox_step["skipped"] is True
    assert sandbox_step["reason"] == "skip_sandbox_without_existing_artifact"
