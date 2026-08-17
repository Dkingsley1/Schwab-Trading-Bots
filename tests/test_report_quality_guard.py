import json
from pathlib import Path

from scripts.ops import report_quality_guard as guard


def _valid_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n" + (b"x" * 12000) + b"\n%%EOF\n")


def test_report_quality_guard_accepts_report_ready_pdfs(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    reports = project_root / "exports" / "reports"
    health.mkdir(parents=True)
    paper_pdf = reports / "paper_performance_latest.pdf"
    post_pdf = reports / "post_trade_analysis_latest.pdf"
    _valid_pdf(paper_pdf)
    _valid_pdf(post_pdf)

    (health / "report_pdf_bundle_latest.json").write_text(
        json.dumps(
            {
                "overall_status": "ready",
                "entries": [
                    {
                        "slug": "paper_performance",
                        "title": "Paper Performance",
                        "pdf_path": str(paper_pdf),
                        "detail": "report_ready_paper_performance_pdf",
                    },
                    {
                        "slug": "post_trade_analysis",
                        "title": "Post-Trade Analysis",
                        "pdf_path": str(post_pdf),
                        "detail": "report_ready_post_trade_pdf",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = guard.build_payload(project_root, repair=False)

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["entry_count"] == 2
    assert payload["metrics"]["report_ready_renderer_count"] == 2
    assert not payload["blockers"]


def test_report_quality_guard_blocks_tiny_or_non_ready_pdf(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    reports = project_root / "exports" / "reports"
    health.mkdir(parents=True)
    bad_pdf = reports / "paper_performance_latest.pdf"
    bad_pdf.parent.mkdir(parents=True)
    bad_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    (health / "report_pdf_bundle_latest.json").write_text(
        json.dumps(
            {
                "overall_status": "ready",
                "entries": [
                    {
                        "slug": "paper_performance",
                        "title": "Paper Performance",
                        "pdf_path": str(bad_pdf),
                        "detail": "deterministic_text_pdf",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = guard.build_payload(project_root, repair=False)

    assert payload["overall_status"] == "blocked"
    assert payload["metrics"]["small_pdf_count"] == 1
    assert payload["blockers"][0]["name"] == "pdf_integrity_failed"
    assert payload["degraded_checks"][0]["name"] == "report_ready_renderer_missing"
