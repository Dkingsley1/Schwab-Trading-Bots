import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.new_bot_graduation_gate as graduation_gate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_graduation_gate_excludes_coverage_and_restore_rows_from_scope(tmp_path: Path, monkeypatch, capsys) -> None:
    registry_path = tmp_path / "master_bot_registry.json"
    walk_forward_path = tmp_path / "walk_forward_latest.json"
    out_path = tmp_path / "new_bot_graduation_latest.json"

    _write_json(
        registry_path,
        {
            "sub_bots": [
                {"bot_id": "brain_refinery_v35_dmi_state_machine", "active": True},
                {
                    "bot_id": "brain_refinery_v77_support_control",
                    "active": True,
                    "reason": "support_control",
                },
                {
                    "bot_id": "brain_refinery_v91_restore_canary",
                    "active": True,
                    "reason": "manual_canary_restore_signal",
                },
            ]
        },
    )
    _write_json(
        walk_forward_path,
        {
            "bots": {
                "brain_refinery_v35_dmi_state_machine": {
                    "runs": 24,
                    "forward_mean": 0.72,
                    "delta": 0.04,
                    "status": "pass",
                }
            }
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "new_bot_graduation_gate.py",
            "--registry",
            str(registry_path),
            "--walk-forward-file",
            str(walk_forward_path),
            "--out-file",
            str(out_path),
            "--min-mature-bots-floor",
            "1",
            "--json",
        ],
    )

    rc = graduation_gate.main()
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["ok"] is True
    assert payload["graduation_scope_active_count"] == 1
    assert payload["coverage_exempt_active_count"] == 2
    assert payload["immature_active_count"] == 0
    assert payload["thresholds"]["effective_min_mature_bots"] == 1
