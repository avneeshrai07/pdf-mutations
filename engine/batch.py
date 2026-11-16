"""Run the processor over a whole folder of PDFs."""
from __future__ import annotations

import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import Config
from .processor import Result, process_pdf


def collect_jobs(cfg: Config) -> List[Tuple[Path, Path]]:
    pattern = "**/*" if cfg.recursive else "*"
    out_root = cfg.output_dir.resolve()
    sources = [
        p for p in sorted(cfg.input_dir.glob(pattern))
        if p.is_file() and p.suffix.lower() == ".pdf" and out_root not in p.resolve().parents
    ]
    jobs: List[Tuple[Path, Path]] = []
    used: Dict[Path, int] = {}
    for src in sources:
        if cfg.preserve_structure:
            dst = cfg.output_dir / src.relative_to(cfg.input_dir)
        else:
            dst = cfg.output_dir / src.name
            if dst in used:  # name clash when flattening -> add a counter
                used[dst] += 1
                dst = dst.with_name(f"{src.stem}_{used[dst]}{src.suffix}")
            else:
                used[dst] = 0
        jobs.append((src, dst))
    return jobs


def run(cfg: Config, dry_run: bool = False, limit: Optional[int] = None) -> List[Result]:
    if not dry_run:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)  # created if missing
    jobs = collect_jobs(cfg)
    if limit:
        jobs = jobs[:limit]
    total = len(jobs)
    print(f"Found {total} PDF(s) in {cfg.input_dir}" + ("  [dry run: nothing is written]" if dry_run else ""))
    results: List[Result] = []
    started = time.time()

    def show(i: int, r: Result) -> None:
        detail = f"text={r.text_hits} img={r.image_hits} vec={r.vector_hits} ocr={r.ocr_hits} left={r.residual}"
        extra = f"  {r.error or r.notes}" if (r.error or r.notes) else ""
        print(f"[{i}/{total}] {r.status:<9} {detail}  {Path(r.src).name}{extra}", flush=True)

    if cfg.workers > 1 and total > 1:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures = [pool.submit(process_pdf, s, d, cfg, dry_run) for s, d in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                show(i, r)
    else:
        for i, (s, d) in enumerate(jobs, 1):
            r = process_pdf(s, d, cfg, dry_run)
            results.append(r)
            show(i, r)

    counts: Dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print(f"\nDone in {time.time() - started:.1f}s: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # names that were found automatically (so you can check them)
    names: Dict[str, List[str]] = {}
    for r in results:
        for n in r.detected:
            names.setdefault(n, []).append(Path(r.src).name)
    if names:
        print("\nCompany names detected automatically (files):")
        for n, files in sorted(names.items(), key=lambda kv: -len(kv[1]))[:30]:
            print(f"  {len(files):>4}  {n}")
    if names and not dry_run:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        with (cfg.output_dir / "_detected_names.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "files", "example_file"])
            for n, files in sorted(names.items(), key=lambda kv: -len(kv[1])):
                w.writerow([n, len(files), files[0]])

    if results and not dry_run:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        report = cfg.output_dir / "_report.csv"
        with report.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["source", "output", "status", "text_hits", "image_hits", "vector_logos", "ocr_hits", "residual", "notes", "error"])
            for r in sorted(results, key=lambda r: r.src):
                w.writerow([r.src, r.dst, r.status, r.text_hits, r.image_hits, r.vector_hits, r.ocr_hits, r.residual, r.notes, r.error])
        print(f"Report: {report}")
    return results
