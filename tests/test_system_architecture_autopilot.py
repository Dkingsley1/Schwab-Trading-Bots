import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import system_architecture_autopilot as src


def _synthetic_graph() -> dict:
    return {
        "overall_status": "blocked",
        "blocked_node_count": 2,
        "degraded_node_count": 2,
        "stale_node_count": 0,
        "blocked_edge_count": 1,
        "authority_violation_count": 1,
        "authority_violations": [{"node_id": "unsafe_live", "path": "ALLOW_ORDER_EXECUTION"}],
        "nodes": [
            {
                "node_id": "health_fast",
                "class": "readiness",
                "status": "ready",
                "depends_on": [],
                "commands": [["./scripts/ops/opsctl.sh", "health-fast", "--json"]],
                "authority_violations": [],
            },
            {
                "node_id": "runtime_throttle",
                "class": "runtime",
                "status": "degraded",
                "depends_on": ["health_fast"],
                "commands": [["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"]],
                "authority_violations": [],
            },
            {
                "node_id": "paper_ramp",
                "class": "paper_execution",
                "status": "ready",
                "depends_on": ["health_fast", "runtime_throttle"],
                "commands": [["./scripts/ops/opsctl.sh", "paper-400-ramp", "--json"]],
                "authority_violations": [],
            },
            {
                "node_id": "all_sleeves_launcher",
                "class": "sleeve_runtime",
                "status": "ready",
                "depends_on": ["paper_ramp"],
                "commands": [["./scripts/ops/opsctl.sh", "health-fast", "--json"]],
                "authority_violations": [],
            },
            {
                "node_id": "runtime_paper_guard",
                "class": "governance",
                "status": "ready",
                "depends_on": ["paper_ramp", "runtime_throttle"],
                "commands": [["./scripts/ops/opsctl.sh", "runtime-paper-regression-guard", "--json"]],
                "authority_violations": [],
            },
            {
                "node_id": "adaptive_regression_guard",
                "class": "adaptive_governance",
                "status": "blocked",
                "depends_on": ["runtime_throttle"],
                "commands": [["./scripts/ops/opsctl.sh", "adaptive-regression-guard", "--apply", "--json"]],
                "authority_violations": [],
            },
            {
                "node_id": "system_self_model",
                "class": "self_model",
                "status": "degraded",
                "depends_on": ["adaptive_regression_guard"],
                "commands": [["./scripts/ops/opsctl.sh", "big-platform-brain", "--json"]],
                "authority_violations": [],
            },
            {
                "node_id": "unsafe_live",
                "class": "architecture",
                "status": "blocked",
                "depends_on": [],
                "commands": [["./scripts/ops/opsctl.sh", "start-live"]],
                "authority_violations": [{"path": "ALLOW_ORDER_EXECUTION"}],
            },
        ],
        "blocked_nodes": ["adaptive_regression_guard", "unsafe_live"],
        "degraded_nodes": ["runtime_throttle", "system_self_model"],
        "stale_nodes": [],
    }


def _graph_builder(_project_root: Path, _apply: bool) -> dict:
    return _synthetic_graph()


def test_architecture_autopilot_plans_dependency_ordered_phases(tmp_path: Path) -> None:
    payload = src.build_payload(tmp_path, graph_builder=_graph_builder)
    node_order = [step["node_id"] for step in payload["repair_plan"]]

    assert payload["overall_status"] == "blocked"
    assert node_order == [
        "runtime_throttle",
        "adaptive_regression_guard",
        "unsafe_live",
        "system_self_model",
    ]
    assert payload["phase_rollup"][0]["phase_name"] == "foundation_truth"
    assert payload["phase_rollup"][-1]["phase_name"] == "self_model_and_operator_view"
    assert payload["repair_step_count"] == 4
    assert payload["safe_repair_step_count"] == 3


def test_architecture_autopilot_apply_writes_plan_without_executing(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str], _project_root: Path, _timeout_sec: int) -> dict:
        calls.append(cmd)
        return {"cmd": cmd, "rc": 0, "payload": {"overall_status": "ready"}}

    payload = src.build_payload(tmp_path, apply=True, runner=runner, graph_builder=_graph_builder)
    plan_path = tmp_path / "governance" / "architecture_contracts" / "system_architecture_autopilot_plan_latest.json"
    written = json.loads(plan_path.read_text(encoding="utf-8"))

    assert calls == []
    assert plan_path.exists()
    assert written["repair_step_count"] == payload["repair_step_count"]
    assert written["execute_safe_repairs"] is False


def test_architecture_autopilot_executes_only_safe_steps(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str], _project_root: Path, timeout_sec: int) -> dict:
        calls.append(cmd)
        return {
            "cmd": cmd,
            "rc": 0,
            "payload": {"overall_status": "ready"},
            "stdout_tail": "",
            "stderr_tail": "",
            "timeout_sec": timeout_sec,
        }

    payload = src.build_payload(
        tmp_path,
        apply=True,
        execute_safe_repairs=True,
        runner=runner,
        graph_builder=_graph_builder,
    )

    assert [call[1] for call in calls] == [
        "runtime-throttle",
        "adaptive-regression-guard",
        "big-platform-brain",
    ]
    assert "start-live" not in {part for call in calls for part in call}
    assert payload["attempt_count"] == 3
    unsafe_step = next(step for step in payload["repair_plan"] if step["node_id"] == "unsafe_live")
    assert unsafe_step["safe_to_execute"] is False


def test_architecture_autopilot_builds_seven_expansion_layers(tmp_path: Path) -> None:
    health_path = tmp_path / "governance" / "health" / "health_fast_latest.json"
    health_path.parent.mkdir(parents=True, exist_ok=True)
    health_path.write_text(
        json.dumps(
            {
                "overall_status": "guarded_ready",
                "operational_readiness": {
                    "guarded_paper": {"status": "ready", "blockers": []},
                    "live_execution": {"status": "blocked_read_only"},
                },
                "process_watchdog": {
                    "all_sleeves_effective_runtime": {
                        "status": "ready",
                        "child_process_count": 108,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    adaptive_path = tmp_path / "governance" / "health" / "adaptive_regression_guard_latest.json"
    adaptive_path.write_text(
        json.dumps(
            {
                "overall_status": "blocked",
                "active_regression_count": 2,
                "persistent_regression_count": 2,
                "critical_regression_count": 1,
                "surfaces": [
                    {
                        "surface": "incident_closeout",
                        "state": "blocked",
                        "adaptive_severity": "critical",
                        "summary": "open incidents remain",
                        "memory": {"consecutive_non_ready_count": 4, "consecutive_blocked_count": 4},
                    },
                    {
                        "surface": "autonomy_control",
                        "state": "degraded",
                        "summary": "autonomy score below self-clear",
                        "memory": {"consecutive_non_ready_count": 4},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = src.build_payload(tmp_path, graph_builder=_graph_builder)
    layer_ids = [row["layer_id"] for row in payload["architecture_expansion_layers"]]

    assert payload["architecture_expansion_summary"]["layer_count"] == 7
    assert layer_ids == [
        "dependency_map",
        "repair_ordering",
        "live_execution_firewall",
        "paper_governance_split",
        "degradation_explainability",
        "adaptive_memory_pressure",
        "operator_visibility",
    ]
    assert payload["live_execution_firewall"]["live_execution_authority"] is False
    assert payload["live_execution_firewall"]["unsafe_nodes"] == ["unsafe_live"]
    assert payload["paper_governance_split"]["guarded_paper_ready"] is True
    assert payload["paper_governance_split"]["paper_governance_split_active"] is True
    assert payload["paper_governance_split"]["all_sleeves_child_process_count"] == 108
    assert payload["adaptive_memory_pressure"]["critical_surfaces"][0]["surface"] == "incident_closeout"
    assert payload["operator_visibility"]["operator_commands"]["safe_refresh"][1] == "system-architecture-autopilot"


def test_architecture_autopilot_ranks_runtime_load_shedder_when_paper_cpu_blocks_guarded_paper(tmp_path: Path) -> None:
    health_path = tmp_path / "governance" / "health" / "health_fast_latest.json"
    health_path.parent.mkdir(parents=True, exist_ok=True)
    health_path.write_text(
        json.dumps(
            {
                "overall_status": "degraded",
                "operational_readiness": {
                    "guarded_paper": {"status": "blocked", "blockers": ["runtime_status=degraded"]},
                    "live_execution": {"status": "blocked_read_only"},
                },
                "process_watchdog": {
                    "all_sleeves_effective_runtime": {"status": "ready", "child_process_count": 108}
                },
            }
        ),
        encoding="utf-8",
    )
    runtime_path = tmp_path / "governance" / "health" / "runtime_throttle_control_latest.json"
    runtime_path.write_text(
        json.dumps(
            {
                "overall_status": "degraded",
                "host_saturation_score": 42.45,
                "compute_pressure_level": "elevated",
                "memory_pressure_level": "normal",
                "soft_cap_advisory_reclassification": {
                    "measurements": {
                        "paper_execution_hot": True,
                        "paper_execution_cpu_percent": 85.9,
                        "paper_execution_paused": False,
                        "paper_ramp_armed": True,
                        "storage_writer_hot": False,
                        "storage_writer_cpu_percent": 0.0,
                        "bot_owned_cpu_percent": 219.1,
                        "bot_owned_pressure_dominant": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    adaptive_path = tmp_path / "governance" / "health" / "adaptive_regression_guard_latest.json"
    adaptive_path.write_text(
        json.dumps({"overall_status": "ready", "active_regression_count": 0, "surfaces": []}),
        encoding="utf-8",
    )

    payload = src.build_payload(tmp_path, apply=True, graph_builder=_graph_builder)
    benefit_path = tmp_path / "governance" / "architecture_contracts" / "system_architecture_benefit_backlog_latest.json"
    benefit = json.loads(benefit_path.read_text(encoding="utf-8"))
    top = benefit["active_candidates"][0]

    assert payload["architecture_benefit_summary"]["top_candidate_id"] == "paper_runtime_load_shedder"
    assert top["candidate_id"] == "paper_runtime_load_shedder"
    assert {"guarded_paper_blocked_runtime", "paper_execution_hot", "runtime_degraded"} <= set(top["matched_signals"])
    assert top["authority_contract"]["does_not_enable_live_execution"] is True
    assert ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--max-renice-processes", "30", "--json"] in top["safe_commands"]
