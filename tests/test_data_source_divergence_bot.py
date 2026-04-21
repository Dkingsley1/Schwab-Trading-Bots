from scripts import data_source_divergence_bot


def test_within_profile_ignores_two_point_bucket() -> None:
    payload = data_source_divergence_bot._summarize_bucket_map(
        {
            data_source_divergence_bot._meta_key(
                shadow_dir="shadow_conservative_equities",
                symbol="RSP",
                minute="2026-04-16T12:47:00+00:00",
            ): [200.0001, 193.34]
        },
        max_relative_spread=0.03,
        timestamp_utc="2026-04-16T13:08:01+00:00",
        window_hours=2,
        comparison_mode="within_profile",
        scope="shadow_conservative_equities",
        profile_dirs=["shadow_conservative_equities"],
        min_price_count=3,
    )

    assert payload["ok"] is True
    assert payload["compared_buckets"] == 0
    assert payload["offenders"] == []
    assert payload["min_price_count"] == 3


def test_cross_profile_still_flags_two_source_spread() -> None:
    payload = data_source_divergence_bot._summarize_bucket_map(
        {
            data_source_divergence_bot._meta_key(
                symbol="RSP",
                minute="2026-04-16T12:47:00+00:00",
            ): [200.0001, 193.34]
        },
        max_relative_spread=0.03,
        timestamp_utc="2026-04-16T13:08:01+00:00",
        window_hours=2,
        comparison_mode="cross_profile",
        scope="all_profiles",
        profile_dirs=["shadow_aggressive_equities", "shadow_conservative_equities"],
        min_price_count=2,
    )

    assert payload["ok"] is False
    assert payload["compared_buckets"] == 1
    assert payload["offenders"][0]["symbol"] == "RSP"
    assert payload["min_price_count"] == 2
