"""Find company names automatically (no list needed).

Signal: a company name in drawings/documents almost always ends in a legal suffix
("... LTD", "... LIMITED", "... PVT LTD", "... INC", "... GMBH"). From each text line we take
the legal suffix and walk backwards over the capitalised words in front of it, stopping at
ordinary words ("PROPERTY OF", "DEVELOPED BY", "OWNER:" ...). Example:

  "THIS DOCUMENT ... DEVELOPED BY MEGHA ENGINEERING & INFRASTRUCTURES LTD IS THE PROPERTY OF"
                                  -> "MEGHA ENGINEERING & INFRASTRUCTURES LTD"

Every detected name is then expanded into the other forms that appear in the same drawing:
LTD <-> LIMITED, PVT <-> PRIVATE, CORP <-> CORPORATION, the name without its suffix, and the
acronym (MEGHA ENGINEERING INFRASTRUCTURES LTD -> "MEIL"; matched only as a whole token).
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pymupdf as fitz

from .config import normalize

# a name ENDS with one of these
END_SUFFIX = {
    "LTD", "LIMITED", "INC", "INCORPORATED", "LLC", "LLP", "CORP", "CORPORATION", "PLC",
    "GMBH", "BHD", "PTE", "LTDA",
}
# interchangeable spellings, used to generate variants
ALTERNATES: Dict[str, List[str]] = {
    "LTD": ["LIMITED"], "LIMITED": ["LTD"],
    "PVT": ["PRIVATE"], "PRIVATE": ["PVT"],
    "CORP": ["CORPORATION"], "CORPORATION": ["CORP"],
    "INC": ["INCORPORATED"], "INCORPORATED": ["INC"],
}
# ordinary words that end the walk backwards (the name starts after them)
STOP = {
    "THE", "OF", "BY", "TO", "FOR", "IS", "ARE", "WAS", "AND", "OR", "AT", "IN", "ON", "FROM",
    "WITH", "AS", "BE", "THIS", "THAT", "THESE", "UNDER", "WITHOUT", "ITS", "BETWEEN",
    "M/S", "MS", "MESSRS", "PROPERTY", "OWNER", "CLIENT", "CONTRACTOR", "PMC", "PROJECT",
    "DRAWING", "DOCUMENT", "DEVELOPED", "PREPARED", "ISSUED", "APPROVED", "CHECKED", "DRAWN",
    "CONSULTANT", "ENGINEER", "ARCHITECT", "VENDOR", "SUPPLIER", "FABRICATOR", "COPYRIGHT",
    "DESIGNED", "SUBMITTED", "REVIEWED", "CERTIFIED", "COURTESY", "REF", "NOTE", "NOTES",
    "WWW", "HTTP", "HTTPS", "TEL", "FAX", "EMAIL", "PH", "NO", "SHEET", "SCALE", "REV",
}
_PUNCT = " \t.,;:()[]{}\"'`!?|*_"
MAX_WORDS = 8


def _clean(tok: str) -> str:
    return tok.strip(_PUNCT)


def detect_in_line(line: str) -> List[str]:
    """Company names found in one text line (in their original spelling)."""
    toks = line.split()
    found: List[str] = []
    for i, tok in enumerate(toks):
        c = _clean(tok)
        if c.upper() not in END_SUFFIX:
            continue
        if i + 1 < len(toks) and _clean(toks[i + 1]).upper() in END_SUFFIX:
            continue  # "PVT LTD": take the name at the last suffix
        words = [c]
        k = i - 1
        while k >= 0 and len(words) < MAX_WORDS:
            raw = toks[k]
            w = _clean(raw)
            if not w or raw.endswith(":") or raw == ":":
                break
            up = w.upper()
            if up in STOP or w.isdigit():
                break
            if w != "&" and not (w.isupper() or w[0].isupper()):
                break  # lower-case word: ordinary prose
            if w != "&" and not any(ch.isalnum() for ch in w):
                break
            words.insert(0, w)
            k -= 1
        while words and words[0] == "&":
            words.pop(0)
        core = [w for w in words if w != "&" and w.upper() not in END_SUFFIX and w.upper() not in ("PVT", "PRIVATE")]
        if core and len("".join(core)) >= 3:
            found.append(" ".join(words))
    return found


def detect_in_page(page: fitz.Page) -> List[str]:
    names: List[str] = []
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        return names
    for b in blocks:
        for line in b.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if len(text) >= 5:
                names += detect_in_line(text)
    return names


def detect_in_doc(doc: fitz.Document) -> List[str]:
    seen: Dict[str, str] = {}
    for page in doc:
        for n in detect_in_page(page):
            seen.setdefault(normalize(n), n)
    return list(seen.values())


@dataclass
class NameSet:
    """Everything to look for in one document."""
    keys: Tuple[str, ...] = ()            # normalised names, longest first
    acronyms: frozenset = frozenset()     # whole-token, case-sensitive
    variants: Tuple[str, ...] = ()        # readable spellings (for metadata / OCR rewrite)
    detected: Tuple[str, ...] = ()        # names found automatically (for the report)


def _alternatives(words: Sequence[str]) -> Iterable[List[str]]:
    options = []
    for w in words:
        opts = [w]
        for alt in ALTERNATES.get(w.upper(), []):
            opts.append(alt if w.isupper() else alt.title())
        options.append(opts[:2])
    for combo in itertools.islice(itertools.product(*options), 8):
        yield list(combo)


def expand(names: Sequence[str], keep: Set[str] = frozenset(), exclude: Set[str] = frozenset(),
           detected: Sequence[str] = ()) -> NameSet:
    """Turn names into search keys: spelling variants, without-suffix form and acronym.
    `keep`/`exclude` hold normalised names that must never be replaced (e.g. the project name)."""
    keys: Set[str] = set()
    acr: Set[str] = set()
    readable: Set[str] = set()
    for name in names:
        words = [w for w in (_clean(t) for t in name.split()) if w]
        if not words:
            continue
        for combo in _alternatives(words):
            readable.add(" ".join(combo))
            keys.add(normalize("".join(combo)))
            # name without its trailing legal-suffix words
            core = list(combo)
            while core and core[-1].upper() in END_SUFFIX | {"PVT", "PRIVATE"}:
                core.pop()
            if core and (len(core) >= 2 or len("".join(core)) >= 6):
                readable.add(" ".join(core))
                keys.add(normalize("".join(core)))
        letters = [w[0].upper() for w in words if w != "&" and w[0].isalpha()]
        if 3 <= len(letters) <= 6:
            acr.add("".join(letters))
    bad = {normalize(k) for k in keep} | set(exclude)
    keys = {k for k in keys if len(k) >= 3 and k not in bad and not any(b and b in k for b in exclude)}
    acr = {a for a in acr if normalize(a) not in bad and normalize(a) not in exclude}
    return NameSet(
        keys=tuple(sorted(keys, key=len, reverse=True)),
        acronyms=frozenset(acr),
        variants=tuple(sorted(readable, key=len, reverse=True)),
        detected=tuple(detected),
    )
