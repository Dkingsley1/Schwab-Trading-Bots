import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from core.global_central_bank_context import (
    CENTRAL_BANK_CROSS_SOURCE_FEATURE_KEYS,
    GLOBAL_CENTRAL_BANK_FEATURE_KEYS,
)
from scripts.build_behavior_dataset_from_decisions import FEATURE_NAMES, _decision_feature_vector


def test_behavior_vector_prefers_symbol_scoped_central_bank_context() -> None:
    global_context = {
        **{key: 0.25 for key in GLOBAL_CENTRAL_BANK_FEATURE_KEYS},
        **{key: 0.25 for key in CENTRAL_BANK_CROSS_SOURCE_FEATURE_KEYS},
    }
    symbol_context = {
        "central_bank_sync_available_norm": 1.0,
        "central_bank_sync_fx_coverage_norm": 1.0,
        "central_bank_policy_fx_confirmation_norm": 0.9,
    }

    vector, _, _ = _decision_feature_vector(
        row={
            "ts_utc": datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc),
            "symbol": "FXE",
            "action": "HOLD",
            "role_idx": 0.5,
            "quantity": 0,
            "features": {},
        },
        gov={},
        lag_exec=(0.0, 0.0, 0.0),
        paper_snapshot={},
        lag_paper=(0.0, 0.0, 0.0),
        snapshot_context={},
        external_context=global_context,
        external_meta={
            "central_bank_cross_source": {
                "symbol_features": {"FXE": symbol_context},
            }
        },
        event_windows=[],
    )

    assert len(vector) == len(FEATURE_NAMES)
    assert vector[FEATURE_NAMES.index("central_bank_policy_fx_confirmation_norm")] == 0.9
    assert vector[FEATURE_NAMES.index("central_bank_sync_fx_coverage_norm")] == 1.0
    assert vector[FEATURE_NAMES.index("global_central_bank_context_available_norm")] == 0.25
