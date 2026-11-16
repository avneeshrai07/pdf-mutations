"""Recognise logo images: by OCR text and by similarity to reference logos."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import fitz

from .config import Config, normalize
from . import ocr

_REF_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _decode(data: bytes):
    """Decode image bytes to a BGR numpy array, flattening transparency onto white."""
    import cv2
    import numpy as np

    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:  # exotic colourspace (CMYK jpeg, ...) -> let MuPDF convert it
        try:
            pix = fitz.Pixmap(data)
            if pix.colorspace is None or pix.colorspace.n != 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            arr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception:
            return None
        if arr is None:
            return None
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        alpha = arr[:, :, 3:4].astype("float32") / 255.0
        rgb = arr[:, :, :3].astype("float32")
        arr = (rgb * alpha + 255.0 * (1.0 - alpha)).astype("uint8")
    return arr


def _dhash(arr) -> int:
    """256-bit difference hash of a decoded image."""
    import cv2

    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


class LogoMatcher:
    """Decides whether an embedded image is one of the company logos."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._seen: Dict[str, bool] = {}  # md5(image bytes) -> verdict, shared across PDFs
        self._refs: List[int] = []
        if cfg.logo_reference_dir:
            for f in sorted(Path(cfg.logo_reference_dir).iterdir()):
                if f.suffix.lower() in _REF_EXTS:
                    arr = _decode(f.read_bytes())
                    if arr is not None:
                        self._refs.append(_dhash(arr))

    def is_logo(self, data: bytes) -> bool:
        key = hashlib.md5(data).hexdigest()
        if key in self._seen:
            return self._seen[key]
        verdict = self._check(data)
        self._seen[key] = verdict
        return verdict

    def _check(self, data: bytes) -> bool:
        arr = _decode(data)
        if arr is None:
            return False
        h, w = arr.shape[:2]
        if min(h, w) < self.cfg.min_logo_pixels:
            return False
        if self._refs:
            hv = _dhash(arr)
            if any(bin(hv ^ r).count("1") <= self.cfg.reference_max_distance for r in self._refs):
                return True
        if self.cfg.ocr_images:
            # tiny logos OCR badly: upscale so small text stays readable
            scale = 2.0 if max(h, w) < 400 else 1.0
            if scale != 1.0:
                import cv2

                arr = cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            text = normalize(" ".join(t for t, _ in ocr.read_lines(arr)))
            if text and any(k in text for k in self.cfg.company_keys):
                return True
        return False
