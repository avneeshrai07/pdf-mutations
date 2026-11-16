"""Phase 1b - LOCATE company name + vector logo on each page (read-only).

For every PDF it finds the company-name text, then the logo drawn next to it, and writes into
<output_dir>/phase1_locate/:
  annotated/*.png   crop of the title block with the NAME boxed in blue and the LOGO boxed in red
  locations.csv     coordinates of name + logo per PDF/page (and whether a logo was found)

Run:  python phase1/locate.py [--config config.yaml] [--limit N]
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf as fitz  # noqa: E402

from engine import load_config  # noqa: E402
from engine.batch import collect_jobs  # noqa: E402
from engine.vector_logo import expand_to_line, find_logo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = cfg.output_dir / "phase1_locate"
    (out / "annotated").mkdir(parents=True, exist_ok=True)
    jobs = [s for s, _ in collect_jobs(cfg)]
    if args.limit:
        jobs = jobs[: args.limit]

    rows = []
    for n, src in enumerate(jobs, 1):
        try:
            doc = fitz.open(src)
        except Exception as exc:
            print(f"[{n}/{len(jobs)}] ERROR {src.name}: {exc}")
            continue
        with doc:
            for pno, page in enumerate(doc, 1):
                hits = []
                for name in sorted(cfg.companies, key=len, reverse=True):
                    for r in page.search_for(name):
                        if not any(k.contains(r) for k in hits):
                            hits.append(r)
                if not hits:
                    continue
                hits = [expand_to_line(page, r, cfg.company_keys) for r in hits]
                drawings = page.get_drawings()
                # the biggest name is the title-block one; small ones are disclaimers etc.
                anchor, match = None, None
                for r in sorted(hits, key=lambda r: r.height, reverse=True):
                    m = find_logo(page, r, drawings)
                    if m:
                        anchor, match = r, m
                        break
                if match:
                    box = (match.cell or match.logo) | anchor
                    clip = fitz.Rect(box.x0 - 20, box.y0 - 20, box.x1 + 20, box.y1 + 20) & page.rect
                    shape = page.new_shape()  # annotate a copy-free overlay then undo via fresh render
                    shape.draw_rect(match.logo)
                    shape.finish(color=(1, 0, 0), width=2)
                    shape.draw_rect(anchor)
                    shape.finish(color=(0, 0, 1), width=2)
                    shape.commit()
                    zoom = min(4.0, 1000 / max(clip.width, 1))
                    png = f"annotated/{src.stem[:60]}_p{pno}.png"
                    page.get_pixmap(clip=clip, matrix=fitz.Matrix(zoom, zoom)).save(out / png)
                    rows.append([str(src), pno, len(hits), *map(round, anchor), *map(round, match.logo),
                                 match.n_shapes, match.method, png])
                else:
                    rows.append([str(src), pno, len(hits), *map(round, hits[0]), "", "", "", "", 0, "NO LOGO", ""])
                print(f"[{n}/{len(jobs)}] {src.name} p{pno}: name hits={len(hits)} logo={match.method if match else 'NOT FOUND'}",
                      flush=True)

    with (out / "locations.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pdf", "page", "name_hits", "name_x0", "name_y0", "name_x1", "name_y1",
                    "logo_x0", "logo_y0", "logo_x1", "logo_y1", "logo_shapes", "method", "annotated_image"])
        w.writerows(rows)
    print(f"\nDone. Open {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
