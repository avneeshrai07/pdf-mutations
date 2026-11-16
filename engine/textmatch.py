"""Locate company names in the text layer from character positions.

page.search_for() can return only part of a wide or letter-spaced name (seen on CAD title
blocks), so the box is built from the real character boxes instead. Matching ignores case,
spaces and punctuation. Acronyms (e.g. IOCL) match only as a whole upper-case token.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pymupdf as fitz

_TOK_PUNCT = " \t.,;:()[]{}\"'`!?|*_"


@dataclass
class NameHit:
    rect: fitz.Rect
    size: float
    color: Tuple[float, float, float]
    origin: fitz.Point  # baseline start of the first character
    bold: bool


def _rgb(span) -> Tuple[float, float, float]:
    c = span.get("color", 0)
    return (((c >> 16) & 255) / 255.0, ((c >> 8) & 255) / 255.0, (c & 255) / 255.0)


def find_name_hits(page: fitz.Page, keys: Sequence[str], acronyms=frozenset()) -> List[NameHit]:
    """`keys` are normalised company names (see config.normalize), longest first.
    `acronyms` match only as a whole, upper-case token."""
    hits: List[NameHit] = []
    try:
        blocks = page.get_text("rawdict")["blocks"]
    except Exception:
        return hits
    for b in blocks:
        for line in b.get("lines", []):
            chars, owner = [], []  # alnum chars of the line + the span each came from
            for span in line["spans"]:
                for ch in span["chars"]:
                    if ch["c"].isalnum():
                        chars.append(ch)
                        owner.append(span)
            if not chars:
                continue
            text = "".join(ch["c"].lower() for ch in chars)
            used = [False] * len(text)
            for key in keys:
                start = 0
                while True:
                    i = text.find(key, start)
                    if i < 0:
                        break
                    j = i + len(key)
                    start = j
                    if any(used[i:j]):
                        continue
                    for k in range(i, j):
                        used[k] = True
                    rect = fitz.Rect(chars[i]["bbox"])
                    for ch in chars[i + 1:j]:
                        rect |= fitz.Rect(ch["bbox"])
                    sp = owner[i]
                    hits.append(NameHit(rect=rect, size=sp["size"], color=_rgb(sp),
                                        origin=fitz.Point(chars[i]["origin"]), bold=bool(sp.get("flags", 0) & 16)))
    if acronyms:
        hits += _acronym_hits(blocks, acronyms, list(hits))
    return hits


def _acronym_hits(blocks, acronyms, existing: List[NameHit]) -> List[NameHit]:
    out: List[NameHit] = []
    for b in blocks:
        for line in b.get("lines", []):
            tokens, cur = [], []
            for span in line["spans"]:
                for ch in span["chars"]:
                    if ch["c"].isspace():
                        if cur:
                            tokens.append(cur)
                        cur = []
                    else:
                        cur.append((ch, span))
            if cur:
                tokens.append(cur)
            for tok in tokens:
                word = "".join(c["c"] for c, _ in tok).strip(_TOK_PUNCT)
                if word not in acronyms:
                    continue
                chars = [(c, sp) for c, sp in tok if c["c"] not in _TOK_PUNCT]
                if not chars:
                    continue
                rect = fitz.Rect(chars[0][0]["bbox"])
                for c, _ in chars[1:]:
                    rect |= fitz.Rect(c["bbox"])
                if any((h.rect & rect).get_area() > 0.3 * rect.get_area() for h in existing + out):
                    continue  # already covered by a longer name match
                sp = chars[0][1]
                out.append(NameHit(rect=rect, size=sp["size"], color=_rgb(sp),
                                   origin=fitz.Point(chars[0][0]["origin"]), bold=bool(sp.get("flags", 0) & 16)))
    return out
