"""Zero-shot image classification with CLIP: is an embedded image a logo, signature, photo, ...?

Optional dependency (pip install torch open_clip_torch pillow). No training needed: the image
is compared with a text description of each class and the closest one wins. The model
(~350 MB) is downloaded on first use and cached.
"""
from __future__ import annotations

import io
from typing import Dict, List, Tuple

# class name -> several phrasings (averaged, which is more stable than one prompt)
CLASSES: Dict[str, List[str]] = {
    "logo": [
        "a company logo",
        "a brand logo on a white background",
        "a corporate emblem with the company name",
    ],
    "signature": [
        "a handwritten signature",
        "a scribbled signature in ink",
    ],
    "stamp": [
        "a document approval stamp with a table of text",
        "a rubber stamp or seal",
    ],
    "photo": [
        "a photograph",
        "a photo of a real-world scene",
    ],
    "drawing": [
        "a technical engineering drawing",
        "a CAD line drawing or diagram",
    ],
    "text": [
        "a scanned page of text",
        "a table of text and numbers",
    ],
    "icon": [
        "a small icon or symbol",
        "a simple pictogram",
    ],
}


class ImageClassifier:
    def __init__(self, model: str = "ViT-B-32", pretrained: str = "openai") -> None:
        try:
            import open_clip
            import torch
            from PIL import Image  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "CLIP scoring needs: pip install torch open_clip_torch pillow"
            ) from exc
        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(model)
        self.names = list(CLASSES)
        with torch.no_grad():
            feats = []
            for name in self.names:
                t = self.model.encode_text(tokenizer(CLASSES[name]))
                t = t / t.norm(dim=-1, keepdim=True)
                m = t.mean(dim=0)
                feats.append(m / m.norm())
            self.text = torch.stack(feats)

    def _load(self, data: bytes):
        from PIL import Image

        im = Image.open(io.BytesIO(data))
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))  # transparent logos on white
            bg.paste(im, mask=im.split()[-1])
            return bg
        return im.convert("RGB")

    def classify(self, data: bytes) -> Tuple[str, float, Dict[str, float]]:
        """Returns (best label, confidence 0-1, all scores)."""
        torch = self._torch
        img = self.preprocess(self._load(data)).unsqueeze(0)
        with torch.no_grad():
            f = self.model.encode_image(img)
            f = f / f.norm(dim=-1, keepdim=True)
            probs = (100.0 * f @ self.text.T).softmax(dim=-1)[0]
        scores = {n: float(p) for n, p in zip(self.names, probs)}
        best = max(scores, key=scores.get)
        return best, scores[best], scores
