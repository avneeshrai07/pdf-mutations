"""Rebrand a single PDF: swap company names / logos for the project name."""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pymupdf as fitz

from . import ocr
from .config import Config, normalize
from .logos import LogoMatcher
from . import logo_fingerprint as fp
from .name_detect import NameSet, detect_in_doc, detect_in_page, expand
from .textmatch import NameHit, find_name_hits
from .vector_logo import find_logo

_matcher: Optional[LogoMatcher] = None
_matcher_cfg_id: Optional[int] = None


def _get_matcher(cfg: Config) -> LogoMatcher:
    """One matcher per process, so OCR results and reference hashes are reused across PDFs."""
    global _matcher, _matcher_cfg_id
    if _matcher is None or _matcher_cfg_id != id(cfg):
        _matcher, _matcher_cfg_id = LogoMatcher(cfg), id(cfg)
    return _matcher


@dataclass
class Result:
    src: str
    dst: str = ""
    status: str = "ok"  # ok | unchanged | skipped | error
    text_hits: int = 0
    image_hits: int = 0
    ocr_hits: int = 0
    vector_hits: int = 0   # vector logos removed
    residual: int = 0      # company names / logos still present after editing (should be 0)
    detected: list = field(default_factory=list)  # company names found automatically
    notes: str = ""
    error: str = ""


@dataclass
class _Op:
    rect: fitz.Rect
    fill: Tuple[float, float, float]
    new_text: str
    fontsize: float = 0.0
    color: Tuple[float, float, float] = (0, 0, 0)
    origin: Optional[fitz.Point] = None  # baseline start for text-layer hits


_templates: Optional[list] = None
_templates_key: Optional[tuple] = None


def _get_templates(cfg: Config) -> list:
    """Learned vector-logo templates (loaded once per process)."""
    global _templates, _templates_key
    key = (str(cfg.vector_templates_dir), cfg.vector_logo_enabled)
    if _templates is None or _templates_key != key:
        _templates, _templates_key = [], key
        d = cfg.vector_templates_dir
        if cfg.vector_logo_enabled and d and Path(d).is_dir():
            _templates = [fp.Template.load(f) for f in sorted(Path(d).glob("logo_template_*.json"))]
    return _templates


def _names_for(doc: fitz.Document, cfg: Config) -> NameSet:
    """Names to remove from this document: the configured ones plus the ones found automatically."""
    proj = normalize(cfg.project_name)
    detected: list = []
    if cfg.auto_detect_names:
        for n in detect_in_doc(doc):
            k = normalize(n)
            if proj in k or k in proj or k in cfg.keep_keys:  # not the project name, not a protected name
                continue
            detected.append(n)
    return expand(list(cfg.companies) + detected, keep=set(cfg.keep_keys), exclude={proj}, detected=detected)


# ---------------------------------------------------------------- helpers
def _luma(c: Tuple[float, float, float]) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _contrast(color: Tuple[float, float, float], fill: Tuple[float, float, float]):
    if abs(_luma(color) - _luma(fill)) >= 0.4:
        return color
    return (0, 0, 0) if _luma(fill) > 0.5 else (1, 1, 1)


def _background(page: fitz.Page, rect: fitz.Rect, cfg: Config) -> Tuple[float, float, float]:
    if cfg.fill_color != "auto":
        return cfg.fill_color  # type: ignore[return-value]
    clip = (fitz.Rect(rect) + (-2, -2, 2, 2)) & page.rect
    if clip.is_empty:
        return (1, 1, 1)
    try:
        pix = page.get_pixmap(clip=clip, colorspace=fitz.csRGB, alpha=False)
    except Exception:
        return (1, 1, 1)
    w, h, s = pix.width, pix.height, pix.samples
    pts = [(x, y) for x in range(w) for y in (0, h - 1)] + [(x, y) for y in range(h) for x in (0, w - 1)]
    chans = [[], [], []]
    for x, y in pts:
        i = (y * w + x) * 3
        for c in range(3):
            chans[c].append(s[i + c])
    med = [sorted(ch)[len(ch) // 2] / 255.0 for ch in chans]
    return (med[0], med[1], med[2])


def _text_style(page: fitz.Page, rect: fitz.Rect):
    """Font size, colour and baseline of the first text span inside rect."""
    try:
        d = page.get_text("dict", clip=rect)
    except Exception:
        return None
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    c = span.get("color", 0)
                    rgb = (((c >> 16) & 255) / 255.0, ((c >> 8) & 255) / 255.0, (c & 255) / 255.0)
                    return span["size"], rgb, fitz.Point(span["origin"]), bool(span.get("flags", 0) & 16)
    return None


def _dedupe(rects: List[fitz.Rect]) -> List[fitz.Rect]:
    kept: List[fitz.Rect] = []
    for r in sorted(rects, key=lambda r: r.get_area(), reverse=True):
        if not any(k.contains(r) or (k & r).get_area() > 0.6 * r.get_area() for k in kept):
            kept.append(r)
    return kept


def _apply(page: fitz.Page, ops: List[_Op], images: int, text: int) -> None:
    if not ops:
        return
    for op in ops:
        page.add_redact_annot(op.rect, fill=op.fill)
    page.apply_redactions(images=images, graphics=fitz.PDF_REDACT_LINE_ART_NONE, text=text)


def _write(page: fitz.Page, op: _Op, bold: bool = False) -> None:
    """Draw the replacement text. Shrinks the font until it fits the box."""
    font = "hebo" if bold else "helv"
    color = _contrast(op.color, op.fill)
    if op.origin is not None:  # text-layer hit: keep baseline, shrink to the original width
        size = op.fontsize or 10
        avail = max(op.rect.width, 1)
        width = fitz.get_text_length(op.new_text, fontname=font, fontsize=size)
        if width > avail:
            size = max(size * avail / width, 1.5)  # always fit the original width, never overlap neighbours
        page.insert_text(op.origin, op.new_text, fontname=font, fontsize=size, color=color)
        return
    size = op.fontsize or min(op.rect.height * 0.6, 24)
    box = fitz.Rect(op.rect)
    while size >= 3:
        if page.insert_textbox(box, op.new_text, fontname=font, fontsize=size,
                               color=color, align=fitz.TEXT_ALIGN_CENTER) >= 0:
            return
        size -= 0.5
    page.insert_text(fitz.Point(box.x0, box.y1 - 1), op.new_text, fontname=font, fontsize=3, color=color)


# ---------------------------------------------------------------- per page
def _text_pass(page: fitz.Page, cfg: Config, hits: List[NameHit]) -> int:
    if not hits:
        return 0
    ops: List[_Op] = []
    for h in hits:
        op = _Op(rect=h.rect, fill=_background(page, h.rect, cfg), new_text=cfg.project_name,
                 fontsize=h.size, color=h.color, origin=h.origin)
        op.bold = h.bold  # type: ignore[attr-defined]
        ops.append(op)
    _apply(page, ops, images=fitz.PDF_REDACT_IMAGE_NONE, text=fitz.PDF_REDACT_TEXT_REMOVE)
    for op in ops:
        _write(page, op, bold=getattr(op, "bold", False))
    return len(ops)


def _vector_detect(page: fitz.Page, cfg: Config, hits: List[NameHit], drawings: list) -> List[fitz.Rect]:
    """Where are the vector logos? Learned templates first, name-anchored search as fallback."""
    rects: List[fitz.Rect] = []
    for tpl in _get_templates(cfg):
        rects += [m.rect for m in fp.find(page, tpl, cfg.vector_min_score, drawings)]
    if not rects and cfg.vector_fallback_name_anchor and hits:
        for h in sorted(hits, key=lambda h: h.rect.height, reverse=True):
            m = find_logo(page, h.rect, drawings)
            if m:
                rects.append(m.logo)
                break
    return _dedupe(rects)


def _vector_remove(page: fitz.Page, cfg: Config, rects: List[fitz.Rect]) -> None:
    for r in rects:
        page.add_redact_annot(r + (-0.5, -0.5, 0.5, 0.5), fill=False)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, text=fitz.PDF_REDACT_TEXT_NONE,
                          graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED)
    if cfg.vector_logo_text:
        for r in rects:
            _write(page, _Op(rect=r, fill=_background(page, r, cfg), new_text=cfg.project_name))


def _residual(page: fitz.Page, cfg: Config, names: NameSet) -> int:
    """Company names / logos still on the page after editing."""
    proj = normalize(cfg.project_name)
    keys = tuple(k for k in names.keys if k not in proj)  # project name may contain a company word
    n = len(find_name_hits(page, keys, names.acronyms)) if keys else 0
    if cfg.auto_detect_names:  # any other company-looking name still there?
        for nm in detect_in_page(page):
            k = normalize(nm)
            if proj not in k and k not in proj and k not in cfg.keep_keys:
                n += 1
    if cfg.vector_logo_enabled:
        tpls = _get_templates(cfg)
        if tpls:
            d = page.get_drawings()
            for tpl in tpls:
                n += len(fp.find(page, tpl, cfg.vector_min_score, d))
    return n


def _image_pass(doc: fitz.Document, page: fitz.Page, cfg: Config, matcher: LogoMatcher,
                verdicts: dict, names: NameSet) -> int:
    page_area = page.rect.get_area()
    ops: List[_Op] = []
    for img in page.get_images(full=True):
        xref = img[0]
        if xref not in verdicts:
            try:
                info = doc.extract_image(xref)
            except Exception:
                info = None
            ok = bool(info) and info["ext"].lower() in cfg.logo_image_extensions
            verdicts[xref] = bool(ok and matcher.is_logo(info["image"], names.keys))
        if not verdicts[xref]:
            continue
        for r in page.get_image_rects(xref):
            if r.is_empty or r.get_area() > cfg.max_logo_page_fraction * page_area:
                continue
            ops.append(_Op(rect=r, fill=_background(page, r, cfg), new_text=cfg.project_name))
    if not ops:
        return 0
    mode = fitz.PDF_REDACT_IMAGE_REMOVE if cfg.image_removal == "remove" else fitz.PDF_REDACT_IMAGE_PIXELS
    _apply(page, ops, images=mode, text=fitz.PDF_REDACT_TEXT_NONE)
    for op in ops:
        _write(page, op)
    return len(ops)


def _scanned_pass(page: fitz.Page, cfg: Config, names: NameSet) -> int:
    zoom = 2.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
    ops: List[_Op] = []
    for txt, (x0, y0, x1, y1) in ocr.read_lines(pix.tobytes("png")):
        if not any(k in normalize(txt) for k in names.keys):
            continue
        new = txt
        for name in names.variants:
            new = re.sub(re.escape(name), cfg.project_name, new, flags=re.IGNORECASE)
        if new == txt:  # matched only after normalisation -> replace the whole line
            new = cfg.project_name
        r = fitz.Rect(x0 / zoom, y0 / zoom, x1 / zoom, y1 / zoom)
        ops.append(_Op(rect=r + (-1, -1, 1, 1), fill=_background(page, r, cfg), new_text=new,
                       fontsize=r.height * 0.7))
    if not ops:
        return 0
    _apply(page, ops, images=fitz.PDF_REDACT_IMAGE_PIXELS, text=fitz.PDF_REDACT_TEXT_NONE)
    for op in ops:
        _write(page, op)
    return len(ops)


def _scrub_metadata(doc: fitz.Document, cfg: Config, names: NameSet) -> bool:
    meta = dict(doc.metadata or {})
    changed = False
    for k, v in meta.items():
        if not v or k in ("format", "encryption"):
            continue
        new = v
        for name in names.variants:
            new = re.sub(re.escape(name), cfg.project_name, new, flags=re.IGNORECASE)
        if new != v:
            meta[k], changed = new, True
    if changed:
        doc.set_metadata(meta)
    return changed


# ---------------------------------------------------------------- public
def process_pdf(src: Path, dst: Path, cfg: Config, dry_run: bool = False) -> Result:
    res = Result(src=str(src), dst=str(dst))
    try:
        if cfg.skip_existing and dst.exists() and not dry_run:
            res.status = "skipped"
            return res
        matcher = _get_matcher(cfg) if (cfg.ocr_images or cfg.logo_reference_dir) else None
        doc = fitz.open(src)
        try:
            if doc.needs_pass:
                raise RuntimeError("PDF is password protected")
            verdicts: dict = {}
            names = _names_for(doc, cfg)
            res.detected = list(names.detected)
            vector_on = cfg.vector_logo_enabled and (_get_templates(cfg) or cfg.vector_fallback_name_anchor)
            no_logo_pages = []
            for page in doc:
                # 1) detect first (the name is the anchor for the fallback logo search) ...
                hits = find_name_hits(page, names.keys, names.acronyms) if (cfg.replace_text or vector_on) else []
                vrects: List[fitz.Rect] = []
                if vector_on:
                    vrects = _vector_detect(page, cfg, hits, page.get_drawings())
                # 2) ... then edit
                if cfg.replace_text:
                    res.text_hits += _text_pass(page, cfg, hits)
                if vrects:
                    _vector_remove(page, cfg, vrects)
                    res.vector_hits += len(vrects)
                page_images = 0
                if matcher is not None:
                    page_images = _image_pass(doc, page, cfg, matcher, verdicts, names)
                    res.image_hits += page_images
                if vector_on and hits and not vrects and not page_images:
                    no_logo_pages.append(page.number + 1)  # name present, but no logo of any kind removed
                if cfg.ocr_scanned_pages and not page.get_text().strip():
                    res.ocr_hits += _scanned_pass(page, cfg, names)
                res.residual += _residual(page, cfg, names)
            if no_logo_pages:
                res.notes = f"company name found but no logo on page(s) {no_logo_pages[:10]}"
            meta_changed = _scrub_metadata(doc, cfg, names) if cfg.scrub_metadata else False
            changed = bool(res.text_hits or res.image_hits or res.ocr_hits or res.vector_hits or meta_changed)
            flagged = res.residual > 0 or bool(res.notes)
            if dry_run:
                res.status = "review" if flagged else ("ok" if changed else "unchanged")
                return res
            dst.parent.mkdir(parents=True, exist_ok=True)
            if changed:
                tmp = dst.with_suffix(".tmp")
                doc.save(tmp, garbage=3, deflate=True)
                tmp.replace(dst)
                if flagged:
                    res.status = "review"  # saved, but a human should look at it
            else:
                res.status = "unchanged"
                if cfg.copy_unchanged:
                    shutil.copy2(src, dst)
        finally:
            doc.close()
    except Exception as exc:  # one bad file must not stop a 1000-file run
        res.status, res.error = "error", f"{type(exc).__name__}: {exc}"
    return res
