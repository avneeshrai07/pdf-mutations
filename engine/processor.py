"""Rebrand a single PDF: swap company names / logos for the project name."""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import fitz

from . import ocr
from .config import Config, normalize
from .logos import LogoMatcher

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
    error: str = ""


@dataclass
class _Op:
    rect: fitz.Rect
    fill: Tuple[float, float, float]
    new_text: str
    fontsize: float = 0.0
    color: Tuple[float, float, float] = (0, 0, 0)
    origin: Optional[fitz.Point] = None  # baseline start for text-layer hits


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
            size = max(size * avail / width, 4)
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
def _text_pass(page: fitz.Page, cfg: Config) -> int:
    hits: List[fitz.Rect] = []
    for key_src in cfg.companies:
        hits += page.search_for(key_src)
    rects = _dedupe(hits)
    if not rects:
        return 0
    ops: List[_Op] = []
    for r in rects:
        fill = _background(page, r, cfg)
        style = _text_style(page, r)
        op = _Op(rect=r, fill=fill, new_text=cfg.project_name)
        bold = False
        if style:
            op.fontsize, op.color, op.origin, bold = style
        else:
            op.fontsize, op.origin = r.height / 1.25, fitz.Point(r.x0, r.y1 - r.height * 0.22)
        op.bold = bold  # type: ignore[attr-defined]
        ops.append(op)
    _apply(page, ops, images=fitz.PDF_REDACT_IMAGE_NONE, text=fitz.PDF_REDACT_TEXT_REMOVE)
    for op in ops:
        _write(page, op, bold=getattr(op, "bold", False))
    return len(ops)


def _image_pass(doc: fitz.Document, page: fitz.Page, cfg: Config, matcher: LogoMatcher,
                verdicts: dict) -> int:
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
            verdicts[xref] = bool(ok and matcher.is_logo(info["image"]))
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


def _scanned_pass(page: fitz.Page, cfg: Config) -> int:
    zoom = 2.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
    ops: List[_Op] = []
    for txt, (x0, y0, x1, y1) in ocr.read_lines(pix.tobytes("png")):
        if not any(k in normalize(txt) for k in cfg.company_keys):
            continue
        new = txt
        for name in cfg.companies:
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


def _scrub_metadata(doc: fitz.Document, cfg: Config) -> bool:
    meta = dict(doc.metadata or {})
    changed = False
    for k, v in meta.items():
        if not v or k in ("format", "encryption"):
            continue
        new = v
        for name in cfg.companies:
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
            for page in doc:
                if cfg.replace_text:
                    res.text_hits += _text_pass(page, cfg)
                if matcher is not None:
                    res.image_hits += _image_pass(doc, page, cfg, matcher, verdicts)
                if cfg.ocr_scanned_pages and not page.get_text().strip():
                    res.ocr_hits += _scanned_pass(page, cfg)
            meta_changed = _scrub_metadata(doc, cfg) if cfg.scrub_metadata else False
            changed = bool(res.text_hits or res.image_hits or res.ocr_hits or meta_changed)
            if dry_run:
                res.status = "ok" if changed else "unchanged"
                return res
            dst.parent.mkdir(parents=True, exist_ok=True)
            if changed:
                tmp = dst.with_suffix(".tmp")
                doc.save(tmp, garbage=3, deflate=True)
                tmp.replace(dst)
            else:
                res.status = "unchanged"
                if cfg.copy_unchanged:
                    shutil.copy2(src, dst)
        finally:
            doc.close()
    except Exception as exc:  # one bad file must not stop a 1000-file run
        res.status, res.error = "error", f"{type(exc).__name__}: {exc}"
    return res
