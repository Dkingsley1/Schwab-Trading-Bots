from __future__ import annotations

import html
import re
from typing import Any


_VTT_INLINE_TAG_RE = re.compile(r"</?c(?:\.[^>]*)?>", re.IGNORECASE)
_VTT_TIMESTAMP_TAG_RE = re.compile(r"<\d{1,2}:\d{2}:\d{2}\.\d{3}>")
_HTMLISH_TAG_RE = re.compile(r"<[^>]+>")
_BRACKET_NOISE_RE = re.compile(r"\[(?:applause|music|laughter|captions?[^\]]*|captioning[^\]]*)\]", re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"\s+")
_SPEAKER_LABEL_RE = re.compile(
    r"^(?P<label>(?:[A-Z][A-Za-z.'-]*|[A-Z]{2,}|[A-Z]{1,4}\.)"
    r"(?:\s+(?:[A-Z][A-Za-z.'-]*|[A-Z]{2,}|[A-Z]{1,4}\.)){0,4})"
    r":\s*(?P<body>.*)$"
)
_SPEAKER_PREFIXES = {
    "MR",
    "MRS",
    "MS",
    "MISS",
    "DR",
    "SEN",
    "SENATOR",
    "REP",
    "REPRESENTATIVE",
    "CHAIR",
    "CHAIRMAN",
    "CHAIRWOMAN",
    "JUSTICE",
    "JUDGE",
    "GOV",
    "GOVERNOR",
    "PRESIDENT",
    "SECRETARY",
    "HOST",
    "MODERATOR",
    "VOICE",
}


def _norm_compare_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(token or "").lower())


def collapse_repeated_phrases(text: str) -> str:
    words = [chunk for chunk in str(text or "").split() if chunk]
    if len(words) < 4:
        return " ".join(words).strip()
    out: list[str] = []
    idx = 0
    while idx < len(words):
        best_window = 0
        max_candidate = min(12, (len(words) - idx) // 2)
        for window in range(max_candidate, 1, -1):
            left = [_norm_compare_token(token) for token in words[idx : idx + window]]
            right = [_norm_compare_token(token) for token in words[idx + window : idx + (2 * window)]]
            if left and left == right and any(left):
                best_window = window
                break
        if best_window:
            out.extend(words[idx : idx + best_window])
            idx += best_window * 2
            continue
        out.append(words[idx])
        idx += 1
    return " ".join(out).strip()


def clean_transcript_text(raw: str) -> str:
    text = html.unescape(str(raw or ""))
    text = _VTT_INLINE_TAG_RE.sub(" ", text)
    text = _VTT_TIMESTAMP_TAG_RE.sub(" ", text)
    text = _HTMLISH_TAG_RE.sub(" ", text)
    text = _BRACKET_NOISE_RE.sub(" ", text)
    text = collapse_repeated_phrases(text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def split_speaker_label(text: str) -> tuple[str, str]:
    cleaned = clean_transcript_text(text)
    match = _SPEAKER_LABEL_RE.match(cleaned)
    if not match:
        return "", cleaned
    label = " ".join(str(match.group("label") or "").split()).strip()
    prefix = re.sub(r"[^A-Za-z]", "", label.split()[0] if label else "").upper()
    if prefix not in _SPEAKER_PREFIXES and len(label.split()) < 2:
        return "", cleaned
    body = clean_transcript_text(match.group("body"))
    return label, body


def annotate_speaker_turns(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    out: list[dict[str, Any]] = []
    speakers: list[str] = []
    current_speaker = ""
    turn_index = -1
    for raw in rows:
        row = dict(raw or {})
        speaker, body = split_speaker_label(str(row.get("text") or ""))
        if speaker:
            if speaker != current_speaker:
                current_speaker = speaker
                speakers.append(speaker)
                turn_index += 1
            row["speaker"] = speaker
            row["speaker_turn_index"] = max(turn_index, 0)
            row["text"] = body
        elif current_speaker:
            row["speaker"] = current_speaker
            row["speaker_turn_index"] = max(turn_index, 0)
            row["text"] = clean_transcript_text(str(row.get("text") or ""))
        else:
            row["text"] = clean_transcript_text(str(row.get("text") or ""))
        out.append(row)
    return out, speakers
