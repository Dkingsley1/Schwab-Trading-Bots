from pathlib import Path

from scripts.ops import grade_lift_hardening as src


def test_grade_lift_hardening_rolls_up_blocked_and_ready_steps(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str], project_root: Path, timeout_sec: int) -> dict:
        calls.append(list(cmd))
        joined = " ".join(cmd)
        if "training_lineage_manifest.py" in joined:
            return {"cmd": cmd, "rc": 2, "payload": {"overall_status": "blocked", "lineage_score": 15.0}, "stdout_tail": "", "stderr_tail": ""}
        if "security_evidence_autofix.py" in joined:
            return {"cmd": cmd, "rc": 0, "payload": {"overall_status": "ready", "ok": True}, "stdout_tail": "", "stderr_tail": ""}
        return {"cmd": cmd, "rc": 0, "payload": {"overall_status": "ready", "ok": True}, "stdout_tail": "", "stderr_tail": ""}

    payload = src.build_payload(tmp_path, apply_storage_remediations=False, runner=runner)

    assert payload["overall_status"] == "blocked"
    assert payload["blocked_step_count"] >= 1
    assert any(step["name"] == "training_lineage_manifest" and step["status"] == "blocked" for step in payload["steps"])
    assert any("security_evidence_autofix.py" in " ".join(cmd) for cmd in calls)
    assert any("security_hardening_audit.py" in " ".join(cmd) for cmd in calls)
    assert any("premarket_token_guard.py" in " ".join(cmd) for cmd in calls)
    assert any("session_ready_check.py" in " ".join(cmd) for cmd in calls)
    assert any("canary_rollout_guard.py" in " ".join(cmd) for cmd in calls)
    assert any("runtime_throttle_control.py" in " ".join(cmd) for cmd in calls)
