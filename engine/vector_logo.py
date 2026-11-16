"""Find a vector-drawn logo next to a company name (CAD / title-block style PDFs).

The company NAME is searchable text, so it is the anchor:
  1. Take the name's text line (search_for can return only part of a wide, letter-spaced name).
  2. Look at the table cells (rectangles) drawn around it. The logo cell is the rectangle
     just LEFT of the name: touching/near the name, overlapping it vertically.
  3. The logo = union of the vector shapes inside that cell (the cell border itself and
     long border lines are ignored).
Fallback when no cell rectangle exists: the vector shapes in a band just left of the name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import pymupdf as fitz

from .config import normalize

LINE_TOL = 2.5  # a path thinner than this (pt) is a straight line, not artwork


@dataclass
class LogoMatch:
    logo: fitz.Rect          # bounding box of the logo shapes
    cell: Optional[fitz.Rect]  # table cell holding the logo (None in fallback mode)
    n_shapes: int
    method: str              # "cell" or "band"


def expand_to_line(page: fitz.Page, hit: fitz.Rect, keys: Sequence[str]) -> fitz.Rect:
    """If the whole text line is just the company name, return the line's box instead of the
    (possibly partial) search hit. Otherwise return the hit unchanged."""
    try:
        blocks = page.get_text("dict")["blocks"]  # no clip: a clip would truncate the line text
    except Exception:
        return hit
    for b in blocks:
        for line in b.get("lines", []):
            text = normalize("".join(s["text"] for s in line["spans"]))
            lb = fitz.Rect(line["bbox"])
            if text in keys and lb.intersects(hit):
                return lb
    return hit


def _cells(drawings: list, near: fitz.Rect) -> List[fitz.Rect]:
    """Rectangles (table cells) drawn near `near`."""
    cells: List[fitz.Rect] = []
    for d in drawings:
        if not d["rect"].intersects(near):
            continue
        for it in d["items"]:
            if it[0] == "re":
                r = fitz.Rect(it[1])
                if r.intersects(near) and r.width > LINE_TOL and r.height > LINE_TOL:
                    cells.append(r)
    return cells


def find_logo(page: fitz.Page, name: fitz.Rect, drawings: Optional[list] = None) -> Optional[LogoMatch]:
    h = max(name.height, 4.0)
    if drawings is None:
        drawings = page.get_drawings()
    win = fitz.Rect(name.x0 - 8 * h, name.y0 - 3 * h, name.x1 + h, name.y1 + 3 * h) & page.rect
    near = [d["rect"] for d in drawings if d["rect"].intersects(win)]

    def shapes_in(box: fitz.Rect) -> List[fitz.Rect]:
        out = []
        for r in near:
            c = fitz.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
            if not box.contains(c) or r.width > 1.05 * box.width or r.height > 1.05 * box.height:
                continue  # shape belongs to another cell (centre test keeps shapes poking past the border)
            if abs(r.width - box.width) < 1 and abs(r.height - box.height) < 1:
                continue  # the cell border itself
            if (r.width < LINE_TOL and r.height > 0.9 * box.height) or \
               (r.height < LINE_TOL and r.width > 0.9 * box.width):
                continue  # a border line along the cell
            out.append(r)
        return out

    # ---- 1) cell rectangle immediately left of the name
    cy0, cy1 = name.y0, name.y1
    cand = []
    for c in _cells(drawings, win):
        overlap = min(c.y1, cy1) - max(c.y0, cy0)
        if (c.x1 <= name.x0 + 1 and name.x0 - c.x1 <= 3 * h and overlap >= 0.5 * h
                and c.height >= 0.8 * h and c.width <= 8 * h):
            cand.append(c)
    for c in sorted(cand, key=lambda c: (-c.x1, c.get_area())):
        sh = shapes_in(c)
        if sh:
            logo = fitz.Rect(sh[0])
            for r in sh[1:]:
                logo |= r
            return LogoMatch(logo=logo, cell=c, n_shapes=len(sh), method="cell")

    # ---- 2) fallback: shapes in a band just left of the name
    band = fitz.Rect(name.x0 - 5 * h, name.y0 - 1.2 * h, name.x0 - 0.3 * h, name.y1 + 1.2 * h)
    sh = [r for r in near if band.contains(fitz.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
          and r.width >= LINE_TOL and r.height >= LINE_TOL and r.height < 0.95 * band.height
          and r.width < 0.95 * band.width]
    # group touching shapes into clusters and keep the heaviest one (the logo is the biggest mass;
    # stray table text/lines in the band form small clusters)
    clusters: List[list] = []
    for r in sh:
        grown = r + (-0.1 * h, -0.1 * h, 0.1 * h, 0.1 * h)
        hit = [c for c in clusters if any(grown.intersects(m) for m in c)]
        merged = [r]
        for c in hit:
            merged += c
            clusters.remove(c)
        clusters.append(merged)
    if clusters:
        best = max(clusters, key=lambda c: sum(m.get_area() for m in c))
        logo = fitz.Rect(best[0])
        for r in best[1:]:
            logo |= r
        if logo.width >= 0.5 * h and logo.height >= 0.5 * h:
            return LogoMatch(logo=logo, cell=None, n_shapes=len(best), method="band")
    return None
