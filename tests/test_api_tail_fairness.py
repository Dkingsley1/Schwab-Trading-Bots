import pytest

from scripts.ops import backpressure_drainer_fleet as fleet


def row(name, age=10, pending=100):
    return dict(
        source_rel=f"governance/channels/api/{name}/api_20260915.jsonl",
        pending_lines=pending,
        oldest_pending_age_seconds=age,
    )


@pytest.mark.parametrize(
    "crypto,critical", [(False, False), (True, False), (True, True)]
)
def test_api_tail_stays_inside_existing_lane_and_focus_budget(crypto, critical):
    prefix = "default_crypto_coinbase" if crypto else "regular"
    rows = [row(f"{prefix}_{index}") for index in range(12)]
    tail = row(f"{prefix}_old", 7200, 1)
    shard = "CRYPTO_API_INGRESS" if crypto else "API_INGRESS"
    before_shards, before = fleet._api_ingress_drainer_env({}, rows, critical=critical)
    shards, after = fleet._api_ingress_drainer_env({}, [*rows, tail], critical=critical)
    focus_key = f"SQL_LINK_SERVICE_SHARD_{shard}_PATH_CONTAINS"
    paths = after[focus_key].split(",")
    assert len(paths) == (8 if crypto else 12)
    assert paths[0] == rows[0]["source_rel"]
    assert tail["source_rel"] in paths
    assert shards == before_shards
    assert {k: v for k, v in before.items() if k != focus_key} == {
        k: v for k, v in after.items() if k != focus_key
    }


def test_api_tail_does_not_add_unselected_lane_or_displace_single_slot():
    rows = [row(f"regular_{index}") for index in range(12)]
    tail = row("default_crypto_coinbase_old", 7200, 1)
    before = fleet._api_ingress_drainer_env({}, rows, critical=False)
    assert fleet._api_ingress_drainer_env({}, [*rows, tail], critical=False) == before
    rows = [row(f"regular_{index}") for index in range(11)] + [
        row("default_crypto_coinbase_current")
    ]
    before = fleet._api_ingress_drainer_env({}, rows, critical=False)
    assert fleet._api_ingress_drainer_env({}, [*rows, tail], critical=False) == before


def test_api_tail_ignores_idle_and_recent_sources():
    rows = [row(f"regular_{index}") for index in range(12)]
    before = fleet._api_ingress_drainer_env({}, rows, critical=False)
    assert (
        fleet._api_ingress_drainer_env(
            {}, [*rows, row("idle", 7200, 0), row("recent", 5, 1)], critical=False
        )
        == before
    )


def test_new_append_does_not_hide_overdue_checkpoint_from_api_focus():
    rows = [row(f"regular_{index}") for index in range(12)]
    tail = {
        **row("continuously_appending", 1, 1),
        "checkpoint_service_age_seconds": 7200,
    }
    collected = fleet._collect_sources(
        {"top_deferred_pending_files": [*rows, tail]},
        ("governance/channels/api/",),
        keys=("top_deferred_pending_files",),
    )
    _, env = fleet._api_ingress_drainer_env({}, collected, critical=False)
    paths = env["SQL_LINK_SERVICE_SHARD_API_INGRESS_PATH_CONTAINS"].split(",")
    assert paths[0] == rows[0]["source_rel"]
    assert tail["source_rel"] in paths
    assert len(paths) == 12
    assert collected[-1]["oldest_pending_age_seconds"] == 1
