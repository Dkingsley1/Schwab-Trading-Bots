from scripts.ops import sql_access_runtime_audit as src


def test_package_rows_detect_drift_states() -> None:
    rows, ok = src._package_rows(
        ("duckdb", "duckdb-engine", "SQLAlchemy", "adbc-driver-manager", "adbc-driver-sqlite"),
        {
            "duckdb": "1.5.0",
            "duckdb-engine": "0.17.0",
            "sqlalchemy": "2.0.48",
            "adbc-driver-manager": "1.10.0",
        },
        {
            "duckdb": "1.5.0",
            "duckdb-engine": "0.17.1",
            "sqlalchemy": "2.0.48",
            "adbc-driver-sqlite": "1.10.0",
        },
    )

    assert ok is False
    assert rows == [
        {
            "package": "duckdb",
            "locked_version": "1.5.0",
            "installed_version": "1.5.0",
            "status": "ok",
        },
        {
            "package": "duckdb-engine",
            "locked_version": "0.17.0",
            "installed_version": "0.17.1",
            "status": "version_mismatch",
        },
        {
            "package": "sqlalchemy",
            "locked_version": "2.0.48",
            "installed_version": "2.0.48",
            "status": "ok",
        },
        {
            "package": "adbc-driver-manager",
            "locked_version": "1.10.0",
            "installed_version": None,
            "status": "missing_runtime",
        },
        {
            "package": "adbc-driver-sqlite",
            "locked_version": None,
            "installed_version": "1.10.0",
            "status": "missing_lock",
        },
    ]


def test_recommendations_highlight_sql_access_paths() -> None:
    recommendations = src._recommendations(
        [
            {
                "package": "duckdb",
                "locked_version": "1.5.0",
                "installed_version": "1.5.0",
                "status": "ok",
            }
        ],
        [
            {"name": "adbc_sqlite_smoke", "ok": True},
            {"name": "sqlalchemy_duckdb_smoke", "ok": True},
        ],
    )

    assert "candidate_arrow_native_sqlite_reads_via_adbc" in recommendations
    assert "candidate_duckdb_sqlalchemy_analytics_bridge" in recommendations


def test_data_library_roles_define_hot_and_analytics_paths() -> None:
    roles = src._data_library_roles()

    assert roles["sqlite"].startswith("primary hot-path ingestion")
    assert "analytical read offload" in roles["duckdb"]
    assert "compatibility layer" in roles["pandas"]
