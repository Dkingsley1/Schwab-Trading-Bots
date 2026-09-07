import json
from pathlib import Path

import pytest

from core.storage_router import inspect_storage_path
from scripts.ops import ingestion_data_contract as src
from scripts.ops import ingestion_storage_control as control

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def configured_root(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    for name in (
        "sleeve_ingestion_routing_v2.json",
        "tiered_ingestion_lifecycle_v1.json",
    ):
        (root / "config" / name).write_bytes(
            (PROJECT_ROOT / "config" / name).read_bytes()
        )
    external = tmp_path / "external"
    external.mkdir()
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", str(external))
    monkeypatch.setenv(
        "BOT_LOGS_LOCAL_FALLBACK_ROOT", str(root / "local_fallback_storage")
    )
    monkeypatch.setenv("BOT_LOGS_ACTIVE_MODE", "external")
    monkeypatch.setenv("BOT_LOGS_PREFER_EXTERNAL", "1")
    return root, external


def test_mixed_routes_report_physical_paths_not_global_intent(configured_root):
    root, external = configured_root
    (root / "data").mkdir()
    local_data = root / "local_fallback_storage" / "data"
    local_data.mkdir(parents=True)
    database = local_data / "jsonl_link.sqlite3"
    database.write_bytes(b"metadata only; not a real database")
    (root / "data" / database.name).symlink_to(database)
    (external / "exports").mkdir()
    (root / "exports").symlink_to(external / "exports")

    result = src.build_data_plane_definition(root)
    routes = {row["relative_path"]: row for row in result["route_observations"]}
    assert result["definition_status"] == "defined"
    assert routes["data"]["observed_location"] == "project_local"
    assert routes["data/jsonl_link.sqlite3"]["observed_location"] == "local_fallback"
    assert routes["exports"]["observed_location"] == "configured_external"
    assert routes["data/jsonl_link.sqlite3"]["size_bytes"] == database.stat().st_size
    assert not routes["data/jsonl_link.sqlite3"]["integrity_verified"]
    assert not any(result["authority"].values())
    assert not result["durability_contract"]["exactly_once_end_to_end_claimed"]


def test_broken_parent_link_does_not_create_target(configured_root):
    root, external = configured_root
    (root / "data").symlink_to(external / "missing")
    result = src.build_data_plane_definition(root)
    assert {"relative_path": "data", "status": "missing"} in result[
        "route_observation_findings"
    ]
    assert not (external / "missing").exists()


def test_missing_sidecars_are_not_broken_route_findings(configured_root):
    root, external = configured_root
    (external / "data").mkdir()
    (root / "data").symlink_to(external / "data")
    result = src.build_data_plane_definition(root)
    findings = {row["relative_path"] for row in result["route_observation_findings"]}
    assert "data/jsonl_link.sqlite3-wal" not in findings
    assert "data/jsonl_link.sqlite3" in findings


@pytest.mark.parametrize("colocated", [False, True])
def test_sidecar_location_is_checked_against_database(configured_root, colocated):
    root, external = configured_root
    (root / "data").mkdir()
    database = external / "jsonl_link.sqlite3"
    database.write_bytes(b"metadata only")
    (root / "data" / database.name).symlink_to(database)
    sidecar = root / "data" / (database.name + "-wal")
    if colocated:
        target = external / sidecar.name
        target.write_bytes(b"wal")
        sidecar.symlink_to(target)
    else:
        sidecar.write_bytes(b"wal on wrong route")
    result = src.build_data_plane_definition(root)
    mismatches = [
        row
        for row in result["route_observation_findings"]
        if row["status"] == "sqlite_sidecar_route_mismatch"
    ]
    assert bool(mismatches) is not colocated


def test_nested_relative_symlink_and_parent_resolution(tmp_path):
    (tmp_path / "real" / "child").mkdir(parents=True)
    (tmp_path / "real" / "target").write_text("evidence")
    (tmp_path / "alias").symlink_to("real/child")
    result = inspect_storage_path(tmp_path / "alias" / ".." / "target")
    assert result["status"] == "present"
    assert result["resolved_path"] == str(tmp_path / "real" / "target")


@pytest.mark.parametrize("indirect", [False, True])
def test_protected_route_is_rejected_before_target_metadata(
    tmp_path, monkeypatch, indirect
):
    path = Path("/Volumes/VIDEO/private/data")
    if indirect:
        (tmp_path / "alias").symlink_to("/Volumes/VIDEO/private")
        path = tmp_path / "alias" / "data"
    real_lstat = Path.lstat

    def checked_lstat(candidate, *args, **kwargs):
        assert not candidate.is_relative_to(
            "/Volumes/VIDEO"
        ), "protected metadata access"
        return real_lstat(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", checked_lstat)
    result = inspect_storage_path(path)
    assert result["status"] == "protected_path"
    assert not result["integrity_verified"]


def test_symlink_cycle_is_bounded(tmp_path):
    (tmp_path / "a").symlink_to("b")
    (tmp_path / "b").symlink_to("a")
    result = inspect_storage_path(tmp_path / "a")
    assert result["status"] == "symlink_loop"
    assert len(result["symlinks"]) == 40


def test_inspection_errors_are_not_reported_present(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "lstat", fail)
    result = inspect_storage_path(tmp_path)
    assert result["status"] == "inspection_error"
    assert result["error_type"] == "PermissionError"


def test_file_cannot_be_traversed_as_a_directory(tmp_path):
    (tmp_path / "file").write_text("data")
    result = inspect_storage_path(tmp_path / "file" / "..")
    assert result["status"] == "inspection_error"
    assert result["error_type"] == "NotADirectoryError"


def test_case_variant_and_double_slash_protected_target_are_rejected(monkeypatch):
    import stat
    from types import SimpleNamespace

    def fake_lstat(candidate):
        assert str(candidate).casefold() != "/volumes/video"
        return SimpleNamespace(st_mode=stat.S_IFDIR)

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    assert inspect_storage_path("//volumes/video/private")["status"] == "protected_path"


def test_policy_values_come_from_owners(configured_root):
    root, _ = configured_root
    route_path = root / "config" / "sleeve_ingestion_routing_v2.json"
    policy = json.loads(route_path.read_text())
    policy["lane_contracts"]["core"]["priority"] = 97
    route_path.write_text(json.dumps(policy))
    result = src.build_data_plane_definition(root)
    assert result["lane_contracts"]["core"]["priority"] == 97
    assert set(result["family_routes"]) == set(policy["family_routes"])
    assert len(result["routing_policy_sha256"]) == 64
    assert result["lifecycle_thresholds"]["cold_after_days"] == 14
    assert not any(result["lifecycle_authority"].values())


@pytest.mark.parametrize("lane", ["unregistered", ["core"], None])
def test_unknown_or_malformed_family_lane_never_looks_defined(configured_root, lane):
    root, _ = configured_root
    path = root / "config" / "sleeve_ingestion_routing_v2.json"
    policy = json.loads(path.read_text())
    policy["family_routes"]["balanced_directional"]["lane"] = lane
    path.write_text(json.dumps(policy))
    result = src.build_data_plane_definition(root)
    assert result["definition_status"] == "needs_attention"
    assert "undefined_family_lane:balanced_directional" in result["definition_errors"]


def test_invalid_policy_does_not_fall_back_to_other_checkout(configured_root):
    root, _ = configured_root
    (root / "config" / "sleeve_ingestion_routing_v2.json").write_text("invalid")
    result = src.build_data_plane_definition(root)
    assert result["definition_status"] == "needs_attention"
    assert result["lane_contracts"] == {}


def test_lifecycle_authority_drift_is_reported(configured_root):
    root, _ = configured_root
    path = root / "config" / "tiered_ingestion_lifecycle_v1.json"
    policy = json.loads(path.read_text())
    policy["authority"]["source_delete_authority"] = True
    path.write_text(json.dumps(policy))
    result = src.build_data_plane_definition(root)
    assert result["definition_status"] == "needs_attention"
    assert result["lifecycle_authority"] == {}


def test_protected_policy_is_not_read(configured_root, monkeypatch):
    root, _ = configured_root
    path = root / "config" / "sleeve_ingestion_routing_v2.json"
    path.unlink()
    path.symlink_to("/Volumes/VIDEO/private/policy.json")
    monkeypatch.setattr(
        src, "load_ingestion_routing_policy", lambda *_: pytest.fail("protected read")
    )
    result = src.build_data_plane_definition(root)
    assert result["definition_status"] == "needs_attention"


def test_definitions_cli_does_not_build_full_health_or_write_artifacts(
    configured_root, monkeypatch, capsys
):
    root, _ = configured_root
    output = root / "must-not-exist.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "ingestion_storage_control.py",
            "--project-root",
            str(root),
            "--definitions-only",
            "--json",
            "--out-file",
            str(output),
        ],
    )
    monkeypatch.setattr(
        control, "build_payload", lambda *_: pytest.fail("full health invoked")
    )
    assert control.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["definition_status"] == "defined"
    assert not output.exists()
    assert not (root / "governance").exists()


def test_full_control_report_includes_definition(configured_root):
    root, _ = configured_root
    payload = control.build_payload(root)
    assert payload["data_plane_definition"]["definition_status"] == "defined"
    assert (
        payload["data_plane_definition"]["ingestion_stages"][1]["stage"]
        == "transport_fetched"
    )
