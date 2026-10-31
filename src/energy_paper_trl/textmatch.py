"""Shared lexical matching, clause filtering and quote-span helpers.

Everything that decides "does this text contain this term" goes through
``term_pattern`` so the rule engine, the gate layer and the attribution layer
cannot disagree about tokenisation again. Quoting goes through ``quote_span`` so
a clipped quote still carries the exact source span it came from.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

# A sentence may mix a negated/background clause with a completed-result clause
# ("Compared with X, efficiency improved by 3%"). Filtering the whole sentence
# loses the result, so clauses are filtered instead.
CLAUSE_SPLIT = re.compile(
    r"(?<=[;:,])\s+|\s+(?:but|however|whereas|while|although|though|yet)\s+",
    re.I,
)
NEGATION_EXCEPTION = re.compile(
    r"\bno\s+doubt\b|\bnot\s+only\b|\bnothing\s+but\b|\bnot\s+surprisingly\b",
    re.I,
)


def term_pattern(term: str) -> re.Pattern[str]:
    """Word-boundary regex for ASCII terms, raw substring for CJK terms.

    The previous implementation used ``term in sentence`` for the rule backend,
    which made "test" hit "latest" and "design" hit "designated". ASCII terms also
    accept a trailing plural "s" so "field test" matches "field tests".
    """
    tokens = [re.escape(t) for t in str(term).split()]
    if not tokens:
        return re.compile(re.escape(str(term)), re.I)
    body = r"\s+".join(tokens)
    if any("\u4e00" <= ch <= "\u9fff" for ch in str(term)):
        return re.compile(body, re.I)
    if not str(term).rstrip().endswith(("s", "S")):
        body += r"s?"
    return re.compile(r"(?<!\w)" + body + r"(?!\w)", re.I)


@dataclass(frozen=True)
class Quote:
    """A quote plus the exact span it was taken from.

    ``quote`` may be shortened; ``quote_start``/``quote_end`` always delimit the
    full source span, so an auditor can recover the original with
    ``text[start:end]`` and verify ``quote`` is its prefix (plus the ellipsis).
    """

    quote: str
    start: int
    end: int
    truncated: bool

    def as_dict(self) -> dict:
        return {
            "quote": self.quote,
            "start": self.start,
            "end": self.end,
            "truncated": self.truncated,
        }


def clip_span(text: str, start: int, end: int, limit: int = 240) -> Quote:
    """Clip a span for display while keeping the exact source offsets."""
    start = max(0, int(start))
    end = min(len(text), int(end))
    raw = text[start:end]
    if len(raw) <= limit:
        return Quote(raw, start, end, False)
    return Quote(raw[: max(1, limit - 1)] + "…", start, end, True)


def clauses(text: str) -> list[tuple[int, int, str]]:
    """Split text into clauses with exact offsets (delimiters excluded)."""
    out: list[tuple[int, int, str]] = []
    cursor = 0
    for match in CLAUSE_SPLIT.finditer(text):
        lo, hi = cursor, match.start()
        while lo < hi and text[lo].isspace():
            lo += 1
        while hi > lo and text[hi - 1].isspace():
            hi -= 1
        if hi > lo:
            out.append((lo, hi, text[lo:hi]))
        cursor = match.end()
    lo = cursor
    while lo < len(text) and text[lo].isspace():
        lo += 1
    if len(text) > lo:
        out.append((lo, len(text), text[lo:]))
    return out or [(0, len(text), text)]


def clean_clauses(text: str, negative: re.Pattern[str], background: re.Pattern[str]) -> list[tuple[int, int, str]]:
    """Clauses that are neither negated/future nor background/other-study."""
    good = []
    for lo, hi, clause in clauses(text):
        probe = NEGATION_EXCEPTION.sub(" ", clause)
        if negative.search(probe) or background.search(probe):
            continue
        good.append((lo, hi, clause))
    return good


def iter_span_matches(text: str, terms: tuple[str, ...] | list[str]) -> Iterator[tuple[str, int, int]]:
    """Yield (term, start, end) for every term match, in text order."""
    hits = []
    for term in terms:
        for match in term_pattern(term).finditer(text):
            hits.append((match.start(), match.end(), term))
    hits.sort()
    for start, end, term in hits:
        yield term, start, end
