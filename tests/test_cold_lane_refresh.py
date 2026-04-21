import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import cold_lane_refresh as cold_lane


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_cold_lane_refresh_skips_when_strategy_artifact_is_fresh(tmp_path, monkeypatch, capsys) -> None:
    project_root = tmp_path / "project"
    strategy_out = project_root / "governance" / "health" / "strategy_research_latest.json"
    _write_json(strategy_out, {"timestamp_utc": "2099-04-01T14:00:00+00:00", "ok": True})
    out_file = project_root / "governance" / "health" / "cold_lane_refresh_latest.json"
    lock_file = project_root / "governance" / "health" / "cold_lane_refresh.lock"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cold_lane_refresh.py",
            "--project-root",
            str(project_root),
            "--strategy-out-file",
            str(strategy_out),
            "--out-file",
            str(out_file),
            "--lock-file",
            str(lock_file),
            "--json",
        ],
    )

    rc = cold_lane.main()
    payload = json.loads(capsys.readouterr().out.strip())

    assert rc == 0
    assert payload["skipped"] is True
    assert payload["reason"] == "fresh_strategy_research_reused"


def test_cold_lane_refresh_runs_full_strategy_research_when_stale(tmp_path, monkeypatch, capsys) -> None:
    project_root = tmp_path / "project"
    strategy_out = project_root / "governance" / "health" / "strategy_research_latest.json"
    out_file = project_root / "governance" / "health" / "cold_lane_refresh_latest.json"
    lock_file = project_root / "governance" / "health" / "cold_lane_refresh.lock"

    class _Result:
        def __init__(self, stdout: str) -> None:
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def _fake_run(cmd, cwd=None, capture_output=None, text=None, check=None):  # noqa: ANN001
        assert "strategy_research_lane.py" in " ".join(str(part) for part in cmd)
        return _Result(
            json.dumps(
                {
                    "ok": True,
                    "promotable": False,
                    "research_sandbox_ok": True,
                    "summary": {"recommended_action": "monitor_and_refresh"},
                }
            )
        )

    monkeypatch.setattr(cold_lane, "_run_resource_guard", lambda profile: (True, {"ok": True, "rc": 0}))
    monkeypatch.setattr(cold_lane.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cold_lane_refresh.py",
            "--project-root",
            str(project_root),
            "--strategy-out-file",
            str(strategy_out),
            "--out-file",
            str(out_file),
            "--lock-file",
            str(lock_file),
            "--json",
            "--force",
        ],
    )

    rc = cold_lane.main()
    payload = json.loads(capsys.readouterr().out.strip())

    assert rc == 0
    assert payload["ran"] is True
    assert payload["ok"] is True
    assert payload["strategy_summary"]["recommended_action"] == "monitor_and_refresh"
