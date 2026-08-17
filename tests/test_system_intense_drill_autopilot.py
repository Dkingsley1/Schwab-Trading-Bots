import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import system_intense_drill_autopilot as src


def _spec(drill_id: str, family: str = "runtime") -> dict:
    return {
        "drill_id": drill_id,
        "family": family,
        "title": drill_id,
        "cmd": ["./scripts/ops/opsctl.sh", drill_id, "--json"],
        "timeout_sec": 30,
        "intensity": "test",
    }


def test_intense_drill_plans_runtime_improvement_without_execution(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str], _project_root: Path, _timeout_sec: int) -> dict:
        calls.append(cmd)
        return {
            "cmd": cmd,
            "rc": 2,
            "payload": {
                "overall_status": "degraded",
                "operational_readiness": {
                    "guarded_paper": {
                        "status": "blocked",
                        "blockers": ["runtime_status=degraded"],
                    }
                },
            },
            "stdout_tail": "",
            "stderr_tail": "",
        }

    payload = src.build_payload(
        tmp_path,
        apply=True,
        runner=runner,
        drill_specs=[_spec("fast_health_gate", "readiness")],
    )
    improvement_path = tmp_path / "governance" / "drills" / "system_intense_drill_improvement_plan_latest.json"
    written = json.loads(improvement_path.read_text(encoding="utf-8"))

    assert payload["overall_status"] == "blocked"
    assert payload["deficiency_count"] == 1
    assert payload["attempt_count"] == 0
    assert calls == [["./scripts/ops/opsctl.sh", "fast_health_gate", "--json"]]
    assert ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--max-renice-processes", "30", "--json"] in [
        row["cmd"] for row in payload["improvement_plan"]
    ]
    assert written["improvement_plan"][0]["cmd"][1] == "runtime-throttle"


def test_intense_drill_executes_only_safe_improvements(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str], _project_root: Path, _timeout_sec: int) -> dict:
        calls.append(cmd)
        if cmd[1] == "architecture-drill":
            return {
                "cmd": cmd,
                "rc": 2,
                "payload": {
                    "overall_status": "blocked",
                    "architecture_benefit_summary": {"top_candidate_id": "unsafe_live"},
                    "architecture_benefit_backlog": {
                        "active_candidates": [
                            {
                                "candidate_id": "unsafe_live",
                                "score": 99,
                                "safe_commands": [
                                    ["./scripts/ops/opsctl.sh", "start-live"],
                                    ["./scripts/ops/opsctl.sh", "system-architecture-autopilot", "--apply", "--json"],
                                ],
                            }
                        ]
                    },
                },
                "stdout_tail": "",
                "stderr_tail": "",
            }
        return {"cmd": cmd, "rc": 0, "payload": {"overall_status": "ready", "ok": True}, "stdout_tail": "", "stderr_tail": ""}

    payload = src.build_payload(
        tmp_path,
        apply=True,
        execute_safe_improvements=True,
        runner=runner,
        drill_specs=[_spec("architecture-drill", "architecture") | {"drill_id": "architecture_autopilot"}],
    )

    assert payload["attempt_count"] >= 1
    assert ["./scripts/ops/opsctl.sh", "start-live"] not in calls
    assert ["./scripts/ops/opsctl.sh", "system-architecture-autopilot", "--apply", "--json"] in calls
    assert payload["skipped_improvements"][0]["cmd"] == ["./scripts/ops/opsctl.sh", "start-live"]


def test_intense_drill_ready_when_all_drills_pass(tmp_path: Path) -> None:
    def runner(cmd: list[str], _project_root: Path, _timeout_sec: int) -> dict:
        return {"cmd": cmd, "rc": 0, "payload": {"overall_status": "ready", "ok": True}, "stdout_tail": "", "stderr_tail": ""}

    payload = src.build_payload(
        tmp_path,
        runner=runner,
        drill_specs=[_spec("runtime_pressure_gate"), _spec("paper_replay_integrity", "replay")],
    )

    assert payload["overall_status"] == "ready"
    assert payload["ready_drill_count"] == 2
    assert payload["deficiency_count"] == 0
