from __future__ import annotations

from pathlib import Path

from scripts.ops import codex_project_guard as src


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_contract_files(root: Path, readme_extra: str = "") -> None:
    _write(
        root / "AGENTS.md",
        "\n".join(
            [
                "# Codex Project Guardrails",
                "## Source Of Truth",
                "## Scope Discipline",
                "## Current Separate Domains",
                "## Regression Guardrails",
                "Use per-surface retry budgets.",
                "Run codex-project-guard.",
            ]
        ),
    )
    _write(
        root / "docs" / "architecture" / "SOURCE_OF_TRUTH.md",
        "\n".join(
            [
                "Operator commands",
                "Report opening and PDF fallbacks",
                "Schwab auth handshake",
                "Sleeve performance metrics",
                "Decision and signal evidence",
                "Storage routing",
            ]
        ),
    )
    _write(
        root / "docs" / "architecture" / "ADR-0001-system-source-of-truth.md",
        "# ADR: System Source Of Truth And Signal Evidence\n",
    )
    _write(
        root / "README.md",
        "\n".join(
            [
                "docs/architecture/SOURCE_OF_TRUTH.md",
                "docs/architecture/ADR-0001-system-source-of-truth.md",
                "Sortino ratio",
                "Sharpe ratio",
                "signal_generation_*.jsonl",
                readme_extra,
            ]
        ),
    )


def test_codex_project_guard_ready_when_contract_markers_exist(tmp_path: Path) -> None:
    _write_contract_files(tmp_path)

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["blocked_guard_count"] == 0


def test_codex_project_guard_blocks_separate_domain_doc_drift(tmp_path: Path) -> None:
    _write_contract_files(tmp_path, readme_extra="Logic Pro 96 kHz sample rate")

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    boundary = next(row for row in payload["guards"] if row["name"] == "separate_domain_doc_boundary")
    assert boundary["status"] == "blocked"
    assert boundary["hits"]


def test_codex_project_guard_blocks_mixed_staged_scope(tmp_path: Path) -> None:
    _write_contract_files(tmp_path)

    payload = src.build_payload(
        tmp_path,
        include_staged=True,
        staged_paths=[
            "README.md",
            "scripts/ops/apple_silicon_profile.py",
        ],
    )

    assert payload["overall_status"] == "blocked"
    staged = next(row for row in payload["guards"] if row["name"] == "staged_scope_boundary")
    assert staged["status"] == "blocked"
