import json
from pathlib import Path

from scripts.ops import control_surface_ownership as ownership


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed(tmp_path: Path) -> Path:
    owner = tmp_path / "scripts" / "owner.py"
    owner.parent.mkdir(parents=True, exist_ok=True)
    owner.write_text("RESOURCE = 'state.json'\n", encoding="utf-8")
    config = {
        "policy_id": "test",
        "controls": [
            {
                "control_id": "state",
                "resource_path": "governance/runtime/state.json",
                "owner_source": "scripts/owner.py",
                "owner_command": ["scripts/owner.py"],
                "owner_marker": "state.json",
                "mutation_mode": "mutable_state",
                "coordination": "file_lock",
                "lock_path": "governance/locks/state.lock",
            }
        ],
    }
    path = tmp_path / "config" / "ownership.json"
    _write(path, config)
    return path


def test_valid_registry_proves_exclusive_source_backed_owner(tmp_path: Path) -> None:
    config = _seed(tmp_path)

    payload = ownership.build_payload(tmp_path, config_path=config)

    assert payload["ok"] is True
    assert payload["grade"] == "A+"
    assert payload["control_contract"]["one_declared_writer_per_resource"] is True
    assert payload["evidence_epoch"]["receipt_sha256"]


def test_duplicate_resource_owners_fail_closed(tmp_path: Path) -> None:
    config = _seed(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    duplicate = dict(payload["controls"][0])
    duplicate["control_id"] = "second_owner"
    payload["controls"].append(duplicate)
    _write(config, payload)

    result = ownership.build_payload(tmp_path, config_path=config)

    assert result["ok"] is False
    assert result["duplicate_resource_paths"] == ["governance/runtime/state.json"]


def test_mutable_owner_requires_explicit_coordination(tmp_path: Path) -> None:
    config = _seed(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["controls"][0]["coordination"] = ""
    _write(config, payload)

    result = ownership.build_payload(tmp_path, config_path=config)

    assert result["ok"] is False
    assert "state:coordination_contract_missing" in result["blockers"]


def test_repository_registry_is_complete_and_routable() -> None:
    payload = ownership.build_payload(ownership.PROJECT_ROOT)

    assert payload["ok"] is True
    assert payload["ready_control_count"] == payload["control_count"] >= 10
