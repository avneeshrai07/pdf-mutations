"""Phase 1 - EXTRACT ONLY. Nothing in input_dir is modified and no PDF is rewritten.

Reads every PDF in paths.input_dir and writes, into <output_dir>/phase1_extract/:
  images/<md5>.<ext>   every embedded image whose format is in logo_image_extensions
                       (each unique image saved once, even if used in 1000 PDFs)
  images.csv           one row per unique image: size, CLIP type (logo/signature/...), OCR text,
                       matches a company?, usage counts
  images_by_type/      copies of images CLIP calls logo / stamp / signature (pick your logos here)
  crops/               PNG crop around each company-name hit (shows the logo area, even
                       when the logo is vector artwork and not an embedded image)
  text_hits.csv        every place a company name appears in the text layer
  frequent_lines.csv   text lines that repeat across many PDFs (headers/footers/names)
                       -> use it to discover which company names to put in config.yaml

Run from the project root:  python phase1/extract.py [--config config.yaml] [--limit N]
"""
import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf as fitz  # noqa: E402

from engine import load_config  # noqa: E402
from engine import ocr  # noqa: E402
from engine.batch import collect_jobs  # noqa: E402
from engine.config import normalize  # noqa: E402
from engine.logos import _decode  # noqa: E402


MAX_CROPS = 200  # cap on crop images so a huge run doesn't flood the disk


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1: extract images/text hits from PDFs (read-only).")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    ap.add_argument("--limit", type=int, help="only scan the first N PDFs")
    ap.add_argument("--no-ocr", action="store_true", help="skip OCR of images (faster)")
    ap.add_argument("--no-clip", action="store_true",
                    help="skip CLIP classification (logo / signature / photo ...)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = cfg.output_dir / "phase1_extract"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(parents=True, exist_ok=True)

    jobs = [s for s, _ in collect_jobs(cfg)]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"Scanning {len(jobs)} PDF(s) in {cfg.input_dir}\nOutput: {out}\n")

    images = {}  # md5 -> info
    pdfs_per_image = defaultdict(set)
    line_pdfs = defaultdict(set)
    hit_rows = []
    errors = 0

    for n, src in enumerate(jobs, 1):
        try:
            doc = fitz.open(src)
        except Exception as exc:
            print(f"[{n}/{len(jobs)}] ERROR {src.name}: {exc}")
            errors += 1
            continue
        try:
            if doc.needs_pass:
                print(f"[{n}/{len(jobs)}] ERROR {src.name}: password protected")
                errors += 1
                continue
            seen_xref = {}
            n_img = n_hit = 0
            for pno, page in enumerate(doc, 1):
                # --- text layer
                for line in page.get_text().splitlines():
                    line = line.strip()
                    if 2 < len(line) <= 80:
                        line_pdfs[line].add(src)
                found = []  # (rect, company); overlapping hits ("YOGIJI" inside "YOGIJI DIGI LTD") counted once
                for name in sorted(cfg.companies, key=len, reverse=True):
                    for r in page.search_for(name):
                        if not any(k.contains(r) for k, _ in found):
                            found.append((r, name))
                if True:
                    for r, name in found:
                        ctx = " ".join(page.get_textbox(r + (-120, -2, 120, 2)).split())
                        crop = ""
                        if len(hit_rows) < MAX_CROPS:
                            h = max(r.height, 4)
                            clip = (fitz.Rect(r.x0 - 8 * h, r.y0 - 2.5 * h, r.x1 + 2 * h, r.y1 + 2.5 * h)
                                    & page.rect)
                            crop = f"crops/{src.stem[:60]}_p{pno}_{len(hit_rows)}.png"
                            zoom = min(4.0, 900 / max(clip.width, 1))
                            page.get_pixmap(clip=clip, matrix=fitz.Matrix(zoom, zoom)).save(out / crop)
                        hit_rows.append([str(src), pno, name, round(r.x0), round(r.y0), ctx, crop])
                        n_hit += 1
                # --- embedded images
                for img in page.get_images(full=True):
                    xref = img[0]
                    if xref not in seen_xref:
                        try:
                            info = doc.extract_image(xref)
                        except Exception:
                            info = None
                        seen_xref[xref] = info
                    info = seen_xref[xref]
                    if not info or info["ext"].lower() not in cfg.logo_image_extensions:
                        continue
                    data = info["image"]
                    key = hashlib.md5(data).hexdigest()
                    pdfs_per_image[key].add(src)
                    n_img += 1
                    if key not in images:
                        ext = info["ext"].lower()
                        (out / "images" / f"{key}.{ext}").write_bytes(data)
                        images[key] = dict(ext=ext, w=info["width"], h=info["height"], uses=0,
                                           ocr="", example=str(src), page=pno, data=data)
                    images[key]["uses"] += 1
            print(f"[{n}/{len(jobs)}] {src.name}: images={n_img} company_hits={n_hit}", flush=True)
        finally:
            doc.close()

    # --- OCR each UNIQUE image once
    if not args.no_ocr and images:
        print(f"\nOCR on {len(images)} unique image(s)...")
        for i, (key, im) in enumerate(images.items(), 1):
            if min(im["w"], im["h"]) < cfg.min_logo_pixels:
                continue
            arr = _decode(im["data"])
            if arr is None:
                continue
            try:
                im["ocr"] = " | ".join(t for t, _ in ocr.read_lines(arr))
            except Exception as exc:
                im["ocr"] = f"<ocr error: {exc}>"
    # --- CLIP: what kind of image is it? (logo / signature / stamp / photo / drawing / text / icon)
    if not args.no_clip and images:
        try:
            from engine.image_class import ImageClassifier

            print(f"\nCLIP classification of {len(images)} unique image(s) (first run downloads the model)...")
            clf = ImageClassifier()
        except ImportError as exc:
            print(f"\nSkipping CLIP: {exc}")
            clf = None
        if clf:
            for key, im in images.items():
                try:
                    label, conf, scores = clf.classify(im["data"])
                    im.update(kind=label, conf=conf, logo_score=scores["logo"])
                except Exception as exc:
                    im.update(kind=f"error: {exc}", conf=0.0, logo_score=0.0)
                if im.get("kind") in ("logo", "stamp", "signature"):  # copies grouped by type for easy picking
                    d = out / "images_by_type" / im["kind"]
                    d.mkdir(parents=True, exist_ok=True)
                    (d / f"{key}.{im['ext']}").write_bytes(im["data"])
    for im in images.values():
        im.pop("data", None)

    with (out / "images.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "width", "height", "pdf_count", "use_count", "looks_like", "confidence",
                    "logo_score", "matches_company", "ocr_text", "example_pdf", "example_page"])
        for key, im in sorted(images.items(), key=lambda kv: -len(pdfs_per_image[kv[0]])):
            match = any(k in normalize(im["ocr"]) for k in cfg.company_keys)
            w.writerow([f"images/{key}.{im['ext']}", im["w"], im["h"], len(pdfs_per_image[key]),
                        im["uses"], im.get("kind", ""), f"{im.get('conf', 0):.2f}" if "kind" in im else "",
                        f"{im.get('logo_score', 0):.2f}" if "kind" in im else "",
                        "YES" if match else "", im["ocr"], im["example"], im["page"]])

    with (out / "text_hits.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pdf", "page", "company", "x", "y", "context", "crop_image"])
        w.writerows(hit_rows)

    with (out / "frequent_lines.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pdf_count", "line"])
        for line, pset in sorted(line_pdfs.items(), key=lambda kv: -len(kv[1]))[:500]:
            if len(pset) > 1:
                w.writerow([len(pset), line])

    print(f"\nDone. PDFs scanned={len(jobs) - errors}, errors={errors}, "
          f"unique images={len(images)}, company text hits={len(hit_rows)}")
    print(f"Open: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
