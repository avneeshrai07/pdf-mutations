"""Thin lazy wrapper around RapidOCR (model is loaded once per process)."""
from __future__ import annotations

from typing import List, Tuple

_engine = None

# (text, (x0, y0, x1, y1)) in the pixel space of the image that was scanned
OcrLine = Tuple[str, Tuple[float, float, float, float]]


def _get_engine():
    global _engine
    if _engine is None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "OCR needs rapidocr + onnxruntime: pip install -r requirements.txt "
                "(or set options.ocr_images / ocr_scanned_pages to false)"
            ) from exc
        _engine = RapidOCR()
    return _engine


def read_lines(image: "bytes | object") -> List[OcrLine]:
    """OCR an image (encoded bytes or numpy array). Returns text lines with bounding boxes."""
    result = _get_engine()(image)
    txts = getattr(result, "txts", None)
    boxes = getattr(result, "boxes", None)
    if not txts or boxes is None:
        return []
    lines: List[OcrLine] = []
    for txt, box in zip(txts, boxes):
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        lines.append((str(txt), (min(xs), min(ys), max(xs), max(ys))))
    return lines
