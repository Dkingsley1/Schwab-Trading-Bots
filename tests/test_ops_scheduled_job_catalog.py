from scripts.ops import ops_scheduled_job_catalog as catalog


def test_scheduled_job_catalog_has_unique_identity_and_artifacts() -> None:
    jobs = catalog.DEFAULT_JOB_SPECS

    assert len(jobs) == 45
    assert len({job.job_id for job in jobs}) == len(jobs)
    assert len({job.label for job in jobs}) == len(jobs)
    assert len({job.artifact for job in jobs}) == len(jobs)
    assert (
        sum(job.install_policy == catalog.ACTIVE_INSTALL_POLICY for job in jobs) == 42
    )
    assert (
        sum(job.install_policy == catalog.REMOVED_INSTALL_POLICY for job in jobs) == 3
    )


def test_scheduled_job_catalog_rows_have_bounded_runtime_contracts() -> None:
    allowed_policies = {
        catalog.ACTIVE_INSTALL_POLICY,
        catalog.REMOVED_INSTALL_POLICY,
    }

    for job in catalog.DEFAULT_JOB_SPECS:
        assert job.install_policy in allowed_policies
        assert job.label.startswith("com.dankingsley.ops.")
        assert job.runner.startswith("scripts/ops/")
        assert job.cadence_seconds >= (20 if job.job_id == "runtime_smooth_mode" else 60)
        assert job.freshness_slo_seconds >= job.cadence_seconds
        assert job.deadline_seconds >= 0
        assert job.owner
        assert job.authority_boundary
