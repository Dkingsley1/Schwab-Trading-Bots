import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.provider_mesh_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_provider_mesh_control_tracks_required_collectors_and_cooldowns(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "collector_contracts_latest.json",
        {
            "average_quality_score": 0.82,
            "required_failure_count": 0,
            "soft_failure_count": 1,
            "required_failures": [],
            "soft_failures": ["fx_market_context"],
            "rows": [
                {
                    "name": "official_macro_context",
                    "required": True,
                    "contract_ok": True,
                    "payload_present": True,
                    "payload_size_bytes": 512,
                    "quality_score": 0.91,
                },
                {
                    "name": "fx_market_context",
                    "required": True,
                    "contract_ok": True,
                    "payload_present": True,
                    "payload_size_bytes": 256,
                    "quality_score": 0.66,
                },
                {
                    "name": "sec_edgar_context",
                    "required": False,
                    "contract_ok": True,
                    "payload_present": True,
                    "payload_size_bytes": 128,
                    "quality_score": 0.84,
                },
            ],
        },
    )
    _write_json(
        health / "source_verification_latest.json",
        {
            "overall": {
                "all_verified": True,
                "all_cross_verified": False,
                "counts": {
                    "cross_verified": 2,
                    "single_verified": 1,
                    "single_unverified": 0,
                },
            }
        },
    )
    _write_json(
        health / "fx_twelve_data_guard_latest.json",
        {
            "kind": "daily_quota",
            "symbol": "EURUSD",
            "cooldown_until_utc": "2099-01-01T00:05:00+00:00",
            "failure_count": 3,
        },
    )

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert payload["provider_groups"]["required_context"]["status"] == "ready"
    assert payload["provider_groups"]["verification_mesh"]["status"] == "ready"
    assert payload["provider_groups"]["verification_mesh"]["depth_status"] == "single_source_verified"
    assert payload["provider_groups"]["quota_limited_providers"]["status"] == "degraded"
    assert payload["cooldowns"][0]["active"] is True
    assert "treat provider cooldowns as mesh-level state and serve last-good snapshots until the provider recovers" in payload["recommended_actions"]


def test_provider_mesh_control_ready_when_all_sources_verified_without_cooldowns(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "collector_contracts_latest.json",
        {
            "average_quality_score": 0.92,
            "required_failure_count": 0,
            "soft_failure_count": 0,
            "required_failures": [],
            "soft_failures": [],
            "rows": [
                {
                    "name": "official_macro_context",
                    "required": True,
                    "contract_ok": True,
                    "payload_present": True,
                    "payload_size_bytes": 512,
                    "quality_score": 0.97,
                },
                {
                    "name": "market_micro_context",
                    "required": True,
                    "contract_ok": True,
                    "payload_present": True,
                    "payload_size_bytes": 256,
                    "quality_score": 0.91,
                },
                {
                    "name": "sec_edgar_context",
                    "required": False,
                    "contract_ok": True,
                    "payload_present": True,
                    "payload_size_bytes": 128,
                    "quality_score": 0.84,
                },
            ],
        },
    )
    _write_json(
        health / "source_verification_latest.json",
        {
            "overall": {
                "all_verified": True,
                "all_cross_verified": False,
                "counts": {
                    "cross_verified": 2,
                    "single_source_verified": 1,
                    "single_source_unverified": 0,
                },
            }
        },
    )
    _write_json(health / "fx_twelve_data_guard_latest.json", {})

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["provider_groups"]["verification_mesh"]["status"] == "ready"
    assert payload["provider_groups"]["verification_mesh"]["depth_status"] == "single_source_verified"
    assert "single_verified=1" in payload["provider_groups"]["verification_mesh"]["summary"]
    assert "cross-verify more sources to raise optional verification depth from ready to A+" in payload["recommended_actions"]
