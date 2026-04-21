import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.derived_state_snapshot as derived_state


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_build_derived_state_snapshot_rolls_up_allocator_risk_and_execution(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    allocator_path = project_root / "governance" / "allocator" / "sleeve_allocator_latest.json"
    risk_path = project_root / "governance" / "risk" / "portfolio_risk_latest.json"
    execution_path = project_root / "governance" / "risk" / "execution_budget_latest.json"

    _write_json(
        allocator_path,
        {
            "gross_risk_budget": 0.82,
            "target_weights": {"core": 0.45, "bond": 0.15},
            "policy": {"reasons": ["dq_low:72.00"]},
        },
    )
    _write_json(
        risk_path,
        {
            "risk_level": "medium",
            "risk_score": 32.5,
            "limits": {
                "gross_exposure_cap": 0.7,
                "sleeve_exposure_caps": {"core": 0.315, "bond": 0.105},
                "max_single_symbol_share": 0.15,
                "max_intraday_turnover": 0.9,
            },
        },
    )
    _write_json(
        execution_path,
        {
            "global": {
                "max_total_actions_per_hour": 88,
                "max_total_open_orders": 11,
                "multiplier": 0.75,
            },
            "sleeves": {
                "core": {"max_actions_per_hour": 54, "max_open_orders": 6},
                "bond": {"max_actions_per_hour": 8, "max_open_orders": 2},
            },
        },
    )

    payload = derived_state.build_derived_state_snapshot(
        project_root,
        allocator_path=allocator_path,
        risk_path=risk_path,
        execution_budget_path=execution_path,
    )

    assert payload["ok"] is True
    assert payload["risk_level"] == "medium"
    assert payload["gross_exposure_cap"] == 0.7
    assert payload["max_total_actions_per_hour"] == 88
    assert payload["sleeves"]["core"]["target_weight"] == 0.45
    assert payload["sleeves"]["core"]["exposure_cap"] == 0.315
