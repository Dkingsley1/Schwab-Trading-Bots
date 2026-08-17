from __future__ import annotations

from pathlib import Path

from scripts.ops import paper_truth_dependency_refresh as refresh


def _fresh(*, stale_operational: list[str] | None = None, stale_promotion: list[str] | None = None) -> dict:
    operational = list(stale_operational or [])
    promotion = list(stale_promotion or [])
    return {
        "assessment_performed": True,
        "operational_inputs_fresh": not operational,
        "promotion_evidence_inputs_fresh": not operational and not promotion,
        "stale_operational_inputs": operational,
        "stale_promotion_evidence_inputs": promotion,
        "inputs": {},
    }


def _runner(seen: list[str]):
    def run(name: str, cmd: list[str], **kwargs) -> dict:
        seen.append(name)
        payload = {}
        if name == "paper_truth_verify":
            payload = {
                "ok": True,
                "overall_status": "ready",
                "score": 100.0,
                "raw_metric_score": 91.5,
                "promotion_ready": False,
            }
        return {
            "name": name,
            "cmd": cmd,
            "rc": 0,
            "timed_out": False,
            "payload": payload,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    return run


def test_dependency_refresh_runs_only_final_verifier_when_inputs_are_fresh(tmp_path: Path, monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(refresh, "_freshness", lambda project_root: _fresh())

    payload = refresh.build_payload(tmp_path, runner=_runner(seen))

    assert seen == ["paper_truth_verify"]
    assert payload["overall_status"] == "ready"
    assert payload["operational_conformance_score"] == 100.0
    assert payload["promotion_ready"] is False
    assert payload["selected_repair_groups"] == []


def test_dependency_refresh_report_repair_is_json_only_and_browser_free(tmp_path: Path) -> None:
    groups = {group["id"]: group for group in refresh._repair_groups(tmp_path)}
    commands = {name: cmd for name, cmd, _ in groups["paper_performance_truth"]["commands"]}

    assert "--json-only" in commands["paper_performance"]
    assert "--allow-gui-pdf-renderer" not in commands["paper_performance"]


def test_dependency_refresh_repairs_only_stale_groups_then_regrades(tmp_path: Path, monkeypatch) -> None:
    seen: list[str] = []
    snapshots = iter(
        [
            _fresh(stale_operational=["broker_truth", "ingestion_storage"]),
            _fresh(),
        ]
    )
    monkeypatch.setattr(refresh, "_freshness", lambda project_root: next(snapshots))

    payload = refresh.build_payload(tmp_path, runner=_runner(seen))

    assert payload["selected_repair_groups"] == ["auth_account_truth", "ingestion_truth"]
    assert seen == [
        "schwab_auth_post_refresh",
        "ingestion_backpressure",
        "ingestion_storage",
        "paper_truth_verify",
    ]
    assert payload["stale_operational_inputs_after"] == []
    assert payload["overall_status"] == "ready"
    assert payload["safety_contract"]["refreshes_only_stale_dependency_groups"] is True
    assert payload["safety_contract"]["live_execution_allowed"] is False


def test_dependency_refresh_rebuilds_replay_when_counterfactual_source_advances(tmp_path: Path, monkeypatch) -> None:
    seen: list[str] = []
    snapshots = iter([_fresh(stale_promotion=["counterfactual"]), _fresh()])
    monkeypatch.setattr(refresh, "_freshness", lambda project_root: next(snapshots))

    payload = refresh.build_payload(tmp_path, runner=_runner(seen))

    assert payload["selected_repair_groups"] == ["replay_truth"]
    assert seen == ["counterfactual_replay", "paper_replay", "paper_truth_verify"]
    assert payload["stale_promotion_evidence_inputs_after"] == []
    assert payload["overall_status"] == "ready"


def test_dependency_refresh_rebuilds_performance_when_candidate_epoch_advances(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen: list[str] = []
    snapshots = iter(
        [
            _fresh(stale_operational=["paper_performance"], stale_promotion=["calibration"]),
            _fresh(),
        ]
    )
    monkeypatch.setattr(refresh, "_freshness", lambda project_root: next(snapshots))

    payload = refresh.build_payload(tmp_path, runner=_runner(seen))

    assert payload["selected_repair_groups"] == ["paper_performance_truth"]
    assert seen == ["paper_calibration", "paper_performance", "paper_truth_verify"]
    assert payload["overall_status"] == "ready"


def test_dependency_refresh_fails_closed_when_operational_input_remains_stale(tmp_path: Path, monkeypatch) -> None:
    seen: list[str] = []
    snapshots = iter(
        [
            _fresh(stale_operational=["broker_truth"]),
            _fresh(stale_operational=["broker_truth"]),
        ]
    )
    monkeypatch.setattr(refresh, "_freshness", lambda project_root: next(snapshots))

    payload = refresh.build_payload(tmp_path, runner=_runner(seen))

    assert payload["overall_status"] == "blocked"
    assert payload["ok"] is False
    assert payload["stale_operational_inputs_after"] == ["broker_truth"]
    assert payload["recovery_contract"]["stale_operational_truth_fails_closed"] is True
