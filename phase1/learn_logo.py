"""Phase 1c - LEARN the logo once, then CHECK it is found in every PDF (read-only).

1. Takes the first PDF where the company name + logo are located (name-anchored search).
2. Saves the logo's geometry to <output_dir>/phase1_template/logo_template_N.json (one per variant)
3. Scans ALL PDFs with that template (no company name needed) and writes template_check.csv:
   score per PDF (1.0 = identical logo found) - PDFs with no match are listed as MISSING
   so you can look at exactly those files.

Run:  python phase1/learn_logo.py [--config config.yaml] [--limit N] [--min-score 0.9]
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf as fitz  # noqa: E402

from engine import load_config  # noqa: E402
from engine import logo_fingerprint as fp  # noqa: E402
from engine.batch import collect_jobs  # noqa: E402
from engine.vector_logo import expand_to_line, find_logo  # noqa: E402


def learn_from_first(jobs, cfg):
    """Learn a template from the first PDF whose logo can be located next to the company name."""
    for src in jobs:
        try:
            doc = fitz.open(src)
        except Exception:
            continue
        with doc:
            for page in doc:
                hits = []
                for name in sorted(cfg.companies, key=len, reverse=True):
                    for r in page.search_for(name):
                        if not any(k.contains(r) for k in hits):
                            hits.append(r)
                drawings = page.get_drawings()
                for r in sorted((expand_to_line(page, h, cfg.company_keys) for h in hits),
                                key=lambda r: r.height, reverse=True):
                    m = find_logo(page, r, drawings)
                    if m:
                        return src, page.number + 1, fp.learn(page, m.logo, drawings), m.logo
    return None


def scan(src, templates, min_score):
    """Return [(template_no, page_no, TemplateMatch)] for every logo copy found in src."""
    found = []
    with fitz.open(src) as doc:
        for page in doc:
            drawings = page.get_drawings()
            for t_no, tpl in enumerate(templates, 1):
                for m in fp.find(page, tpl, min_score, drawings):
                    found.append((t_no, page.number + 1, m))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min-score", type=float, default=0.9)
    ap.add_argument("--max-variants", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = cfg.output_dir / "phase1_template"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [s for s, _ in collect_jobs(cfg)]
    if args.limit:
        jobs = jobs[: args.limit]

    templates, results, errors = [], {}, {}
    remaining = list(jobs)
    # Learn one template per logo "variant": learn from a PDF nobody matched yet, then
    # check which of the remaining PDFs it covers. Repeat until all covered / nothing learnable.
    while remaining and len(templates) < args.max_variants:
        learned = learn_from_first(remaining, cfg)
        if not learned:
            break
        src, pno, tpl, logo = learned
        templates.append(tpl)
        tpl.save(out / f"logo_template_{len(templates)}.json")
        print(f"Variant {len(templates)}: learned from {src.name} p{pno} "
              f"({len(tpl.shapes)} shapes, {logo.width:.0f}x{logo.height:.0f}pt)", flush=True)
        still = []
        for n, f in enumerate(remaining, 1):
            try:
                hits = scan(f, [tpl], args.min_score)
            except Exception as exc:
                errors[f] = str(exc)
                continue
            if hits:
                results.setdefault(f, []).extend((len(templates), p, m) for _, p, m in hits)
            else:
                still.append(f)
            print(f"   [{n}/{len(remaining)}] {'found  ' if hits else 'no match'} {f.name}", flush=True)
        remaining = still

    with (out / "template_check.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pdf", "status", "variant", "logo_copies", "best_score", "locations"])
        for f in jobs:
            if f in errors:
                w.writerow([str(f), "ERROR", "", 0, "", errors[f]])
            elif f in results:
                hs = results[f]
                best = max(hs, key=lambda h: h[2].score)
                w.writerow([str(f), "found", best[0], len(hs), f"{best[2].score:.3f}",
                            "; ".join(f"p{p}:{tuple(round(v) for v in m.rect)}" for _, p, m in hs)])
            else:
                w.writerow([str(f), "NEEDS REVIEW", "", 0, "", ""])
    n_ok = len(results)
    print(f"\n{n_ok}/{len(jobs)} PDFs covered by {len(templates)} logo variant(s); "
          f"{len(jobs) - n_ok} NEED REVIEW. Report: {out / 'template_check.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
