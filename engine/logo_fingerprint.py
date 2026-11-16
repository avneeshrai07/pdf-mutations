"""Template matching for vector logos.

A vector logo is a fixed set of drawing paths. Learn it once from one page (shape geometry,
normalised so it does not depend on position or scale), then find it on any page by geometry
alone - no company name, no table layout needed.

A match = the page has the same shapes, at the same relative positions, as the template.
score = share of template shapes found (1.0 = perfect). Anything below `min_score` is reported
as "not found" so it can be reviewed instead of being silently skipped.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pymupdf as fitz

_MIN = 1e-3


def _points(item) -> list:
    kind = item[0]
    if kind == "l":
        return [item[1], item[2]]
    if kind == "c":
        return [item[1], item[2], item[3], item[4]]
    if kind == "re":
        r = item[1]
        return [fitz.Point(r.x0, r.y0), fitz.Point(r.x1, r.y1)]
    if kind == "qu":
        q = item[1]
        return [q.ul, q.ur, q.ll, q.lr]
    return []


def shape_sig(d: dict) -> str:
    """Position- and scale-independent fingerprint of one drawing path."""
    r = d["rect"]
    w, h = max(r.width, _MIN), max(r.height, _MIN)
    parts = [str(len(d["items"])), str(d.get("fill")), str(d.get("even_odd"))]
    for it in d["items"]:
        parts.append(it[0])
        for p in _points(it):
            parts.append(f"{(p.x - r.x0) / w:.2f},{(p.y - r.y0) / h:.2f}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def _cheap_key(d: dict):
    return (len(d["items"]), "".join(it[0] for it in d["items"]))


@dataclass
class Template:
    shapes: List[dict]  # {sig, rx, ry, rw, rh, key}  relative to the logo box
    aspect: float       # logo width / height

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"aspect": self.aspect, "shapes": self.shapes}), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "Template":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return Template(shapes=raw["shapes"], aspect=raw["aspect"])


@dataclass
class TemplateMatch:
    rect: fitz.Rect
    score: float


def learn(page: fitz.Page, logo: fitz.Rect, drawings: Optional[list] = None) -> Template:
    drawings = drawings if drawings is not None else page.get_drawings()
    L = fitz.Rect(logo)
    shapes = []
    for d in drawings:
        r = d["rect"]
        c = fitz.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
        if not L.contains(c) or r.width > L.width * 1.05 or r.height > L.height * 1.05:
            continue
        shapes.append(dict(sig=shape_sig(d), key=[len(d["items"]), "".join(it[0] for it in d["items"])],
                           rx=(r.x0 - L.x0) / L.width, ry=(r.y0 - L.y0) / L.height,
                           rw=r.width / L.width, rh=r.height / L.height))
    if not shapes:
        raise ValueError("no shapes inside the given logo box")
    return Template(shapes=shapes, aspect=L.width / L.height)


def find(page: fitz.Page, tpl: Template, min_score: float = 0.9,
         drawings: Optional[list] = None) -> List[TemplateMatch]:
    drawings = drawings if drawings is not None else page.get_drawings()
    # anchors = the biggest template shapes (most distinctive)
    anchors = sorted(tpl.shapes, key=lambda s: s["rw"] * s["rh"], reverse=True)[:3]
    keys = {tuple(a["key"]) for a in anchors}
    cands = [d for d in drawings if _cheap_key(d) in keys]
    matches: List[TemplateMatch] = []
    for a in anchors:
        for d in cands:
            if _cheap_key(d) != tuple(a["key"]) or shape_sig(d) != a["sig"]:
                continue
            r = d["rect"]
            # derive the logo box from this one shape
            Lw = r.width / a["rw"] if a["rw"] > 0.02 else None
            Lh = r.height / a["rh"] if a["rh"] > 0.02 else None
            if Lw is None and Lh is None:
                continue
            Lw = Lw or Lh * tpl.aspect
            Lh = Lh or Lw / tpl.aspect
            L = fitz.Rect(r.x0 - a["rx"] * Lw, r.y0 - a["ry"] * Lh, 0, 0)
            L.x1, L.y1 = L.x0 + Lw, L.y0 + Lh
            if any(abs(L.x0 - m.rect.x0) < 0.05 * Lw and abs(L.y0 - m.rect.y0) < 0.05 * Lh for m in matches):
                continue
            score = _score(drawings, tpl, L)
            if score >= min_score:
                matches.append(TemplateMatch(rect=L, score=score))
    return matches


def _score(drawings: list, tpl: Template, L: fitz.Rect) -> float:
    zone = L + (-0.05 * L.width, -0.05 * L.height, 0.05 * L.width, 0.05 * L.height)
    local = [d for d in drawings if d["rect"].intersects(zone)]
    by_key: Dict[tuple, list] = {}
    for d in local:
        by_key.setdefault(_cheap_key(d), []).append(d)
    found = 0
    tol_x, tol_y = 0.02 * L.width + 0.05, 0.02 * L.height + 0.05
    for s in tpl.shapes:
        ex, ey = L.x0 + s["rx"] * L.width, L.y0 + s["ry"] * L.height
        for d in by_key.get(tuple(s["key"]), ()):
            r = d["rect"]
            if abs(r.x0 - ex) <= tol_x and abs(r.y0 - ey) <= tol_y and shape_sig(d) == s["sig"]:
                found += 1
                break
    return found / len(tpl.shapes)
