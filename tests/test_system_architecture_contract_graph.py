import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import system_architecture_contract_graph as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_contract_artifacts(project_root: Path, *, status: str = "ready") -> None:
    for contract in src.CONTRACT_NODES:
        _write_json(
            project_root / contract["artifact"],
            {
                "timestamp_utc": src.iso_now(),
                "overall_status": status,
                "ok": status == "ready",
            },
        )


def test_architecture_contract_graph_ready_when_dependencies_are_fresh(tmp_path: Path) -> None:
    _seed_contract_artifacts(tmp_path)

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["node_count"] == len(src.CONTRACT_NODES)
    assert payload["blocked_edge_count"] == 0
    assert payload["authority_violation_count"] == 0
    assert payload["architecture_contract_graph"]["live_execution_authority"] is False
    paper_edges = [row for row in payload["edges"] if row["to"] == "paper_ramp"]
    assert {row["from"] for row in paper_edges} == {"health_fast", "runtime_throttle", "storage_control"}


def test_architecture_contract_graph_treats_guarded_ready_as_ready(tmp_path: Path) -> None:
    _seed_contract_artifacts(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "health_fast_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "guarded_ready",
            "ok": True,
        },
    )

    payload = src.build_payload(tmp_path)
    health_node = next(row for row in payload["nodes"] if row["node_id"] == "health_fast")

    assert payload["overall_status"] == "ready"
    assert health_node["status"] == "ready"
    assert health_node["raw_status"] == "ready"


def test_architecture_contract_graph_blocks_live_authority_violation(tmp_path: Path) -> None:
    _seed_contract_artifacts(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "ready",
            "ALLOW_ORDER_EXECUTION": True,
        },
    )

    payload = src.build_payload(tmp_path)
    runtime_node = next(row for row in payload["nodes"] if row["node_id"] == "runtime_throttle")

    assert payload["overall_status"] == "blocked"
    assert payload["authority_violation_count"] == 1
    assert runtime_node["status"] == "blocked"
    assert payload["authority_violations"][0]["path"] == "ALLOW_ORDER_EXECUTION"


def test_architecture_contract_graph_uses_health_fast_all_sleeves_reconciliation(tmp_path: Path) -> None:
    _seed_contract_artifacts(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "health_fast_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "ready",
            "ok": True,
            "operational_readiness": {"guarded_paper": {"ok": True, "status": "ready"}},
            "process_watchdog": {
                "all_sleeves_effective_runtime": {
                    "ok": True,
                    "status": "ready",
                    "launcher_live": True,
                    "child_process_live": True,
                    "child_process_count": 99,
                    "child_fanout_ok": True,
                    "heartbeat_ok": True,
                    "launcher_artifact_reason": "launcher_artifact_jobs_not_all_running",
                }
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "all_sleeves_launcher_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "blocked",
            "ok": False,
            "reason": "launcher_artifact_jobs_not_all_running",
        },
    )

    payload = src.build_payload(tmp_path)
    all_sleeves = next(row for row in payload["nodes"] if row["node_id"] == "all_sleeves_launcher")

    assert payload["overall_status"] == "ready"
    assert all_sleeves["status"] == "ready"
    assert all_sleeves["raw_status"] == "blocked"
    assert all_sleeves["reconciliations"][0]["active"] is True


def test_architecture_contract_graph_apply_writes_config_and_graph(tmp_path: Path) -> None:
    _seed_contract_artifacts(tmp_path)

    payload = src.build_payload(tmp_path, apply=True)

    assert payload["overall_status"] == "ready"
    config_path = tmp_path / "config" / "system_architecture_contract_graph_v1.json"
    graph_path = tmp_path / "governance" / "architecture_contracts" / "system_architecture_contract_graph_latest.json"
    assert config_path.exists()
    assert graph_path.exists()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert len(config["nodes"]) == len(src.CONTRACT_NODES)
