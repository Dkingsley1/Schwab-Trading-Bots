import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import promotion_pipeline


def test_candidate_scoped_admission_command_uses_promotion_candidates(tmp_path: Path, monkeypatch) -> None:
    gate_path = tmp_path / "promotion_gate_latest.json"
    out_path = tmp_path / "new_bot_admission_guard_promotion_pipeline_latest.json"
    gate_path.write_text(
        json.dumps(
            {
                "considered_bot_ids": ["brain_refinery_v10_seasonal"],
                "pass_examples": [{"bot_id": "brain_refinery_v10_seasonal"}],
                "near_pass_examples": [{"bot_id": "brain_refinery_v42"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(promotion_pipeline, "PROMOTION_GATE_PATH", gate_path)
    monkeypatch.setattr(promotion_pipeline, "CANDIDATE_ADMISSION_PATH", out_path)

    base_cmd = ["python", "scripts/new_bot_admission_guard.py", "--json"]
    cmd, candidate_ids = promotion_pipeline._candidate_scoped_admission_cmd(base_cmd)

    assert candidate_ids == ["brain_refinery_v10_seasonal", "brain_refinery_v42"]
    assert "--include-bot-ids" in cmd
    assert "brain_refinery_v10_seasonal,brain_refinery_v42" in cmd
    assert "--out-file" in cmd
    assert str(out_path) in cmd


def test_candidate_scoped_admission_command_falls_back_without_candidates(tmp_path: Path, monkeypatch) -> None:
    gate_path = tmp_path / "promotion_gate_latest.json"
    gate_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(promotion_pipeline, "PROMOTION_GATE_PATH", gate_path)

    base_cmd = ["python", "scripts/new_bot_admission_guard.py", "--json"]
    cmd, candidate_ids = promotion_pipeline._candidate_scoped_admission_cmd(base_cmd)

    assert cmd == base_cmd
    assert candidate_ids == []
