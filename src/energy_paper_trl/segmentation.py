"""Exact sentence spans shared by the rule and explanation backends.

`Sentence.start`/`Sentence.end` index into the ORIGINAL string, so a quote can be
recovered with `text[start:end]`. Rules and explanations must agree on these
offsets: an explanation that cannot be mapped back to the source text is not
auditable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SPLIT = re.compile(r"[。！？!?；;\n]+|(?<!\d)\.(?!\d)")


@dataclass(frozen=True)
class Sentence:
    start: int
    end: int
    text: str


def _emit(text: str, lo: int, hi: int) -> list[Sentence]:
    # Trim surrounding whitespace while keeping offsets exact.
    while lo < hi and text[lo].isspace():
        lo += 1
    while hi > lo and text[hi - 1].isspace():
        hi -= 1
    return [Sentence(lo, hi, text[lo:hi])] if hi > lo else []


def sentences(text: str) -> list[Sentence]:
    """Split into trimmed, non-empty sentences with exact source offsets."""
    text = str(text or "")
    out: list[Sentence] = []
    cursor = 0
    for match in SPLIT.finditer(text):
        out.extend(_emit(text, cursor, match.start()))
        cursor = match.end()
    out.extend(_emit(text, cursor, len(text)))
    return out
