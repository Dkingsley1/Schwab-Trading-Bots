import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.golden_replay_regression_guard as src


def test_golden_replay_guard_allows_seed_ready_registry_when_pack_is_missing() -> None:
    payload = src.build_payload(
        golden_pack={},
        replay_hash_registry={
            "ok": True,
            "details": {
                "paper": {"current_hash": "paper-hash"},
                "e2e": {"current_hash": "e2e-hash"},
            },
        },
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "degraded"
    assert payload["seed_ready"] is True
    assert payload["case_count"] == 0
