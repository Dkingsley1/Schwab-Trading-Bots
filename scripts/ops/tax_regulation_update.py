#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import load_json, write_payload
    from scripts.ops.trading_tax_estimator import validate_policy
else:
    from .long_runtime_common import PROJECT_ROOT, load_json, write_payload
    from .trading_tax_estimator import validate_policy


IRS_ROOT = "https://www.irs.gov"
IRS_INFLATION_INDEX_URL = f"{IRS_ROOT}/newsroom/inflation-adjusted-tax-items-by-tax-year"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "config" / "trading_tax_regulation_manifest_v1.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "tax_regulation_update_latest.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "tax" / "tax_regulation_update_state.json"
DEFAULT_TIMEOUT_SECONDS = 45
MAX_SOURCE_BYTES = 32 * 1024 * 1024


def _dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _number(raw: Any) -> float | None:
    if raw in {None, ""}:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _money(raw: str) -> int:
    return int(str(raw).replace("$", "").replace(",", "").strip())


def _trusted_url(url: str, trusted_hosts: set[str]) -> bool:
    parsed = urlparse(str(url or ""))
    return parsed.scheme == "https" and parsed.hostname in trusted_hosts


def _fetch(url: str, *, trusted_hosts: set[str]) -> tuple[bytes, dict[str, Any]]:
    if not _trusted_url(url, trusted_hosts):
        raise ValueError(f"untrusted_source_url:{url}")
    try:
        import requests
    except Exception as exc:
        raise RuntimeError("requests_dependency_missing") from exc
    response = requests.get(
        url,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        headers={"User-Agent": "schwab-trading-bot-tax-regulation-verifier/1.0"},
    )
    response.raise_for_status()
    content = bytes(response.content)
    if len(content) > MAX_SOURCE_BYTES:
        raise ValueError(f"official_source_too_large:{len(content)}")
    return content, {
        "url": str(response.url),
        "status_code": int(response.status_code),
        "content_type": str(response.headers.get("content-type") or ""),
        "content_length": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "etag": str(response.headers.get("etag") or ""),
        "last_modified": str(response.headers.get("last-modified") or ""),
    }


def _soup(content: bytes) -> Any:
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        raise RuntimeError("beautifulsoup4_dependency_missing") from exc
    return BeautifulSoup(content.decode("utf-8", errors="replace"), "html.parser")


def discover_revenue_procedure(tax_year: int, *, trusted_hosts: set[str]) -> dict[str, Any]:
    index_content, index_meta = _fetch(IRS_INFLATION_INDEX_URL, trusted_hosts=trusted_hosts)
    index = _soup(index_content)
    release_url = ""
    target_phrase = f"tax year {int(tax_year)}"
    for anchor in index.select("a[href]"):
        text = anchor.get_text(" ", strip=True).lower()
        href = str(anchor.get("href") or "")
        if target_phrase in text and "inflation adjustment" in text:
            candidate = urljoin(IRS_ROOT, href)
            if _trusted_url(candidate, trusted_hosts):
                release_url = candidate
                break
    if not release_url:
        return {
            "ok": False,
            "status": "awaiting_official_irs_release",
            "tax_year": int(tax_year),
            "index_source": index_meta,
        }

    release_content, release_meta = _fetch(release_url, trusted_hosts=trusted_hosts)
    release = _soup(release_content)
    revenue_procedure_url = ""
    revenue_procedure_title = ""
    for anchor in release.select("a[href]"):
        text = anchor.get_text(" ", strip=True)
        href = str(anchor.get("href") or "")
        if "revenue procedure" not in text.lower():
            continue
        candidate = urljoin(IRS_ROOT, href)
        if candidate.lower().endswith(".pdf") and _trusted_url(candidate, trusted_hosts):
            revenue_procedure_url = candidate
            revenue_procedure_title = text
            break
    if not revenue_procedure_url:
        return {
            "ok": False,
            "status": "official_release_missing_revenue_procedure",
            "tax_year": int(tax_year),
            "index_source": index_meta,
            "release_source": release_meta,
        }
    return {
        "ok": True,
        "status": "discovered",
        "tax_year": int(tax_year),
        "index_source": index_meta,
        "release_source": release_meta,
        "release_url": release_url,
        "revenue_procedure_url": revenue_procedure_url,
        "revenue_procedure_title": revenue_procedure_title,
    }


def _pdf_layout_text(content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("pypdf_dependency_missing") from exc
    reader = PdfReader(io.BytesIO(content))
    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except TypeError:
            text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def _table_segment(text: str, table_number: int) -> str:
    start_match = re.search(rf"TABLE\s+{int(table_number)}\s+-", text, flags=re.IGNORECASE)
    if not start_match:
        raise ValueError(f"ordinary_table_{table_number}_missing")
    next_match = re.search(rf"TABLE\s+{int(table_number) + 1}\s+-", text[start_match.end():], flags=re.IGNORECASE)
    end = start_match.end() + next_match.start() if next_match else len(text)
    return text[start_match.start():end]


def _parse_ordinary_table(segment: str) -> list[dict[str, Any]]:
    # IRS tables are two-column PDFs. Restrict threshold parsing to the left
    # column so tax-base amounts in the right column cannot look like brackets.
    left_column = " ".join(line[:72] for line in segment.splitlines())
    normalized_left = re.sub(r"\s+", " ", left_column)
    normalized = re.sub(r"\s+", " ", segment)
    thresholds = [_money(value) for value in re.findall(r"not over\s+\$([\d,]+)", normalized_left, flags=re.IGNORECASE)]
    rates = [int(value) / 100.0 for value in re.findall(r"(\d{1,2})%", normalized)]
    rates = rates[:7]
    if len(thresholds) != 6 or len(rates) != 7:
        raise ValueError(f"ordinary_table_parse_failed:thresholds={thresholds}:rates={rates}")
    return [
        {"up_to_usd": threshold, "rate": rates[index]}
        for index, threshold in enumerate(thresholds)
    ] + [{"up_to_usd": None, "rate": rates[-1]}]


def _capital_gain_pair(segment: str, label_pattern: str) -> tuple[int, int]:
    match = re.search(
        rf"{label_pattern}\s+\$([\d,]+)\s+\$([\d,]+)",
        segment,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"capital_gain_threshold_missing:{label_pattern}")
    return _money(match.group(1)), _money(match.group(2))


def _preferential_rows(zero_upper: int, fifteen_upper: int) -> list[dict[str, Any]]:
    return [
        {"up_to_taxable_income_usd": int(zero_upper), "rate": 0.0},
        {"up_to_taxable_income_usd": int(fifteen_upper), "rate": 0.15},
        {"up_to_taxable_income_usd": None, "rate": 0.20},
    ]


def parse_revenue_procedure(text: str, *, tax_year: int) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", text)
    if f"taxable years beginning in {int(tax_year)}" not in normalized.lower():
        raise ValueError("revenue_procedure_tax_year_not_found")

    table_text_match = re.search(
        rf"\.01\s+Tax Rate Tables\.\s+For taxable years beginning in {int(tax_year)}.*?\.03\s+Maximum Capital Gains Rate",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not table_text_match:
        raise ValueError("ordinary_rate_table_section_missing")
    table_text = table_text_match.group(0)
    mfj = _parse_ordinary_table(_table_segment(table_text, 1))
    hoh = _parse_ordinary_table(_table_segment(table_text, 2))
    single = _parse_ordinary_table(_table_segment(table_text, 3))
    mfs = _parse_ordinary_table(_table_segment(table_text, 4))

    capital_matches = list(
        re.finditer(r"\.03\s+Maximum Capital Gains Rate", normalized, flags=re.IGNORECASE)
    )
    if not capital_matches:
        raise ValueError("capital_gain_section_missing")
    capital_start = capital_matches[-1].start()
    capital_end_match = re.search(r"\.04\s+Adoption Credit", normalized[capital_start:], flags=re.IGNORECASE)
    capital_end = capital_start + capital_end_match.start() if capital_end_match else len(normalized)
    capital = normalized[capital_start:capital_end]
    mfj_cap = _capital_gain_pair(capital, r"Married Individuals Filing Joint Returns and(?: Surviving Spouse)?")
    mfs_cap = _capital_gain_pair(capital, r"Married Individuals Filing Separate Returns")
    hoh_cap = _capital_gain_pair(capital, r"Heads of Household")
    single_cap = _capital_gain_pair(capital, r"All Other Individuals")

    standard_matches = list(re.finditer(r"\.14\s+Standard Deduction", normalized, flags=re.IGNORECASE))
    if not standard_matches:
        raise ValueError("standard_deduction_section_missing")
    standard_start = standard_matches[-1].start()
    standard_end_match = re.search(r"\.15\s+Cafeteria Plans", normalized[standard_start:], flags=re.IGNORECASE)
    standard_end = standard_start + standard_end_match.start() if standard_end_match else len(normalized)
    standard = normalized[standard_start:standard_end]

    def deduction(label: str) -> int:
        match = re.search(rf"{label}.*?\$([\d,]+)", standard, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"standard_deduction_missing:{label}")
        return _money(match.group(1))

    standard_mfj = deduction(r"Married Individuals Filing Joint Returns and Surviving Spouses")
    standard_hoh = deduction(r"Heads of Households")
    standard_single = deduction(r"Unmarried Individuals")
    standard_mfs = deduction(r"Married Individuals Filing Separate Returns")

    return {
        "tax_year": int(tax_year),
        "ordinary_income_brackets": {
            "single": single,
            "married_filing_jointly": mfj,
            "married_filing_separately": mfs,
            "head_of_household": hoh,
            "qualifying_surviving_spouse": deepcopy(mfj),
        },
        "preferential_capital_gain_brackets": {
            "single": _preferential_rows(*single_cap),
            "married_filing_jointly": _preferential_rows(*mfj_cap),
            "married_filing_separately": _preferential_rows(*mfs_cap),
            "head_of_household": _preferential_rows(*hoh_cap),
            "qualifying_surviving_spouse": _preferential_rows(*mfj_cap),
        },
        "standard_deduction_usd": {
            "single": standard_single,
            "married_filing_jointly": standard_mfj,
            "married_filing_separately": standard_mfs,
            "head_of_household": standard_hoh,
            "qualifying_surviving_spouse": standard_mfj,
        },
    }


def _rate_shape(policy: dict[str, Any]) -> dict[str, Any]:
    def rates(section: str) -> dict[str, list[float]]:
        return {
            status: [float(_number(_dict(row).get("rate")) or 0.0) for row in _list(rows)]
            for status, rows in _dict(policy.get(section)).items()
        }

    section_1256 = _dict(policy.get("section_1256"))
    return {
        "ordinary_rates": rates("ordinary_income_brackets"),
        "preferential_rates": rates("preferential_capital_gain_brackets"),
        "section_1256_split": [
            _number(section_1256.get("long_term_fraction")),
            _number(section_1256.get("short_term_fraction")),
        ],
        "niit_rate": _number(_dict(policy.get("net_investment_income_tax")).get("rate")),
    }


def _semantic_source_checks(*, trusted_hosts: set[str]) -> dict[str, Any]:
    sources = {
        "capital_rules": "https://www.irs.gov/taxtopics/tc409",
        "niit_rules": "https://www.irs.gov/taxtopics/tc559",
        "investment_rules": "https://www.irs.gov/publications/p550",
    }
    required = {
        "capital_rules": ["short-term", "long-term", "capital gain"],
        "niit_rules": ["3.8 percent", "$250,000", "$200,000", "$125,000"],
        "investment_rules": ["60/40 rule", "wash sale", "qualified dividends"],
    }
    rows: dict[str, Any] = {}
    ok = True
    for name, url in sources.items():
        try:
            content, meta = _fetch(url, trusted_hosts=trusted_hosts)
            text = _soup(content).get_text(" ", strip=True).lower()
            missing = [phrase for phrase in required[name] if phrase.lower() not in text]
            row = {**meta, "ok": not missing, "missing_required_phrases": missing}
        except Exception as exc:
            row = {"url": url, "ok": False, "error": f"{type(exc).__name__}:{exc}"}
        rows[name] = row
        ok = ok and bool(row.get("ok", False))
    return {"ok": ok, "sources": rows}


def _policy_path_for_year(manifest: dict[str, Any], tax_year: int) -> Path:
    configured = str(_dict(manifest.get("policy_paths_by_tax_year")).get(str(tax_year)) or "").strip()
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else PROJECT_ROOT / path
    generated_dir = str(_dict(manifest.get("annual_rollover")).get("active_generated_directory") or "governance/tax/regulations")
    return PROJECT_ROOT / generated_dir / f"us_federal_{int(tax_year)}.json"


def _previous_policy(manifest: dict[str, Any], tax_year: int) -> tuple[Path, dict[str, Any]]:
    current_path = _policy_path_for_year(manifest, tax_year)
    current = load_json(current_path)
    if current:
        return current_path, current
    for year in range(int(tax_year) - 1, int(tax_year) - 6, -1):
        path = _policy_path_for_year(manifest, year)
        payload = load_json(path)
        if payload:
            return path, payload
    return Path(), {}


def _build_candidate(
    previous: dict[str, Any],
    extracted: dict[str, Any],
    *,
    discovery: dict[str, Any],
    pdf_meta: dict[str, Any],
    semantic_checks: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    tax_year = int(extracted["tax_year"])
    candidate = deepcopy(previous)
    candidate["policy_id"] = f"us_federal_individual_trading_tax_{tax_year}"
    candidate["tax_year"] = tax_year
    candidate["effective_start"] = f"{tax_year}-01-01"
    candidate["effective_end"] = f"{tax_year}-12-31"
    candidate["verification_status"] = "verified_against_official_sources"
    candidate["verified_at_utc"] = now.isoformat()
    for key in ("ordinary_income_brackets", "preferential_capital_gain_brackets", "standard_deduction_usd"):
        candidate[key] = deepcopy(extracted[key])

    prior_sources = [
        row for row in _list(previous.get("source_references"))
        if "Revenue Procedure" not in str(_dict(row).get("title") or "")
    ]
    candidate["source_references"] = [
        {
            "title": str(discovery.get("revenue_procedure_title") or f"IRS Revenue Procedure for {tax_year}"),
            "url": str(discovery.get("revenue_procedure_url") or ""),
            "supports": [f"{tax_year} ordinary brackets", f"{tax_year} capital gain thresholds", f"{tax_year} standard deduction"],
        },
        {
            "title": f"IRS tax inflation adjustment release for {tax_year}",
            "url": str(discovery.get("release_url") or ""),
            "supports": ["official tax-year release and Revenue Procedure lineage"],
        },
        *prior_sources,
    ]
    candidate["automatic_rollover_evidence"] = {
        "generated_at_utc": now.isoformat(),
        "generator": "tax_regulation_update.py",
        "source_sha256": pdf_meta.get("sha256"),
        "source_content_length": pdf_meta.get("content_length"),
        "semantic_rule_sources_verified": bool(semantic_checks.get("ok", False)),
        "inherited_rule_sections": [
            "capital loss deduction limits",
            "NIIT structure and thresholds",
            "holding period rules",
            "wash-sale window",
            "Section 1256 split",
            "qualified-dividend treatment",
            "equity-option treatment",
        ],
    }
    return candidate


def refresh_tax_year(
    tax_year: int,
    *,
    manifest: dict[str, Any],
    approve_structural_change: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    timestamp = now or datetime.now(timezone.utc)
    trusted_hosts = {str(value) for value in _list(manifest.get("trusted_source_hosts")) if str(value)}
    discovery = discover_revenue_procedure(tax_year, trusted_hosts=trusted_hosts)
    if not discovery.get("ok"):
        return {
            "timestamp_utc": timestamp.isoformat(),
            "ok": int(tax_year) > timestamp.year,
            "status": discovery.get("status"),
            "tax_year": int(tax_year),
            "activated": False,
            "discovery": discovery,
        }

    try:
        pdf_content, pdf_meta = _fetch(str(discovery["revenue_procedure_url"]), trusted_hosts=trusted_hosts)
        extracted = parse_revenue_procedure(_pdf_layout_text(pdf_content), tax_year=tax_year)
        semantic_checks = _semantic_source_checks(trusted_hosts=trusted_hosts)
    except Exception as exc:
        return {
            "timestamp_utc": timestamp.isoformat(),
            "ok": False,
            "status": "official_source_parse_or_verification_failed",
            "tax_year": int(tax_year),
            "activated": False,
            "error": f"{type(exc).__name__}:{exc}",
            "discovery": discovery,
        }

    previous_path, previous = _previous_policy(manifest, tax_year)
    if not previous:
        return {
            "timestamp_utc": timestamp.isoformat(),
            "ok": False,
            "status": "prior_verified_policy_missing",
            "tax_year": int(tax_year),
            "activated": False,
            "discovery": discovery,
        }
    candidate = _build_candidate(
        previous,
        extracted,
        discovery=discovery,
        pdf_meta=pdf_meta,
        semantic_checks=semantic_checks,
        now=timestamp,
    )
    validation = validate_policy(candidate, requested_tax_year=tax_year)
    structural_change = _rate_shape(candidate) != _rate_shape(previous)
    candidate_dir = PROJECT_ROOT / str(
        _dict(manifest.get("annual_rollover")).get("candidate_directory")
        or "governance/tax/regulation_candidates"
    )
    candidate_path = candidate_dir / f"us_federal_{int(tax_year)}.json"
    write_payload(candidate_path, candidate)

    blockers: list[str] = []
    if not validation.get("ok"):
        blockers.extend(validation.get("issues") or [])
    if not semantic_checks.get("ok"):
        blockers.append("structural_rule_source_semantics_not_verified")
    if structural_change and not approve_structural_change:
        blockers.append("tax_rate_structure_changed_explicit_review_required")
    activated = not blockers
    active_path = _policy_path_for_year(manifest, tax_year)
    if activated:
        write_payload(active_path, candidate)
    return {
        "timestamp_utc": timestamp.isoformat(),
        "ok": activated,
        "status": "activated" if activated else "candidate_staged_review_required",
        "tax_year": int(tax_year),
        "activated": activated,
        "active_policy_path": str(active_path),
        "candidate_path": str(candidate_path),
        "previous_policy_path": str(previous_path),
        "structural_change": structural_change,
        "blockers": sorted(set(blockers)),
        "policy_validation": validation,
        "discovery": discovery,
        "revenue_procedure_source": pdf_meta,
        "semantic_source_checks": semantic_checks,
    }


def check_tax_year(tax_year: int, *, manifest: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    timestamp = now or datetime.now(timezone.utc)
    policy_path = _policy_path_for_year(manifest, tax_year)
    policy = load_json(policy_path)
    validation = validate_policy(policy, requested_tax_year=tax_year) if policy else {
        "ok": False,
        "status": "blocked",
        "issues": ["current_tax_year_policy_missing"],
        "tax_year": int(tax_year),
    }
    return {
        "timestamp_utc": timestamp.isoformat(),
        "ok": bool(validation.get("ok", False)),
        "status": "ready" if validation.get("ok") else "blocked",
        "tax_year": int(tax_year),
        "activated": False,
        "active_policy_path": str(policy_path),
        "policy_validation": validation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify and roll forward annual IRS trading-tax policy.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--tax-year", type=int, default=0)
    parser.add_argument("--refresh", action="store_true", help="Fetch official IRS sources and build a verified candidate.")
    parser.add_argument("--auto", action="store_true", help="Check current year and prepare next year during the configured release window.")
    parser.add_argument("--approve-structural-change", action="store_true")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    return parser


def _refresh_due(state: dict[str, Any], *, tax_year: int, now: datetime, interval_seconds: float) -> bool:
    raw = str(_dict(state.get("last_attempt_by_tax_year")).get(str(tax_year)) or "").strip()
    if not raw:
        return True
    try:
        attempted = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return True
    if attempted.tzinfo is None:
        attempted = attempted.replace(tzinfo=timezone.utc)
    return (now - attempted.astimezone(timezone.utc)).total_seconds() >= max(float(interval_seconds), 0.0)


def _record_attempt(state: dict[str, Any], *, tax_year: int, now: datetime, status: str) -> None:
    attempts = dict(_dict(state.get("last_attempt_by_tax_year")))
    statuses = dict(_dict(state.get("last_status_by_tax_year")))
    attempts[str(tax_year)] = now.isoformat()
    statuses[str(tax_year)] = str(status)
    write_payload(
        DEFAULT_STATE_PATH,
        {
            "timestamp_utc": now.isoformat(),
            "schema_version": 1,
            "last_attempt_by_tax_year": attempts,
            "last_status_by_tax_year": statuses,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_json(Path(args.manifest).expanduser())
    now = datetime.now(timezone.utc)
    target_year = int(args.tax_year or now.year)
    state = load_json(DEFAULT_STATE_PATH)
    if args.auto:
        rollover = _dict(manifest.get("annual_rollover"))
        release_month = int(_number(rollover.get("expected_release_window_start_month")) or 9)
        interval_seconds = float(_number(rollover.get("check_interval_seconds")) or 86400.0)
        current = check_tax_year(now.year, manifest=manifest, now=now)
        if not current.get("ok"):
            should_refresh = bool(args.refresh) or _refresh_due(
                state,
                tax_year=now.year,
                now=now,
                interval_seconds=interval_seconds,
            )
            payload = (
                refresh_tax_year(
                    now.year,
                    manifest=manifest,
                    approve_structural_change=bool(args.approve_structural_change),
                    now=now,
                )
                if should_refresh
                else current
            )
            if should_refresh:
                _record_attempt(state, tax_year=now.year, now=now, status=str(payload.get("status") or ""))
        elif now.month >= release_month:
            next_year = now.year + 1
            next_check = check_tax_year(next_year, manifest=manifest, now=now)
            should_refresh = (
                not next_check.get("ok")
                and (
                    bool(args.refresh)
                    or _refresh_due(state, tax_year=next_year, now=now, interval_seconds=interval_seconds)
                )
            )
            payload = (
                refresh_tax_year(
                    next_year,
                    manifest=manifest,
                    approve_structural_change=bool(args.approve_structural_change),
                    now=now,
                )
                if should_refresh
                else next_check
            )
            if should_refresh:
                _record_attempt(state, tax_year=next_year, now=now, status=str(payload.get("status") or ""))
            payload["current_year_policy"] = current
        else:
            payload = current
            payload["next_year_rollover_status"] = "before_expected_irs_release_window"
    elif args.refresh:
        payload = refresh_tax_year(
            target_year,
            manifest=manifest,
            approve_structural_change=bool(args.approve_structural_change),
            now=now,
        )
    else:
        payload = check_tax_year(target_year, manifest=manifest, now=now)

    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "tax_regulation_update "
            f"status={payload.get('status')} "
            f"tax_year={payload.get('tax_year')} "
            f"activated={int(bool(payload.get('activated', False)))}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
