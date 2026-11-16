"""Entry point:  python run.py [--config config.yaml] [--dry-run] [--limit N]"""
import argparse
import sys
from pathlib import Path

from engine import load_config, run


def main() -> int:
    ap = argparse.ArgumentParser(description="Replace company names/logos in PDFs with a project name.")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")),
                    help="path to the YAML config (default: config.yaml next to run.py)")
    ap.add_argument("--dry-run", action="store_true", help="scan only, write nothing")
    ap.add_argument("--limit", type=int, help="only process the first N PDFs (good for a trial)")
    ap.add_argument("--workers", type=int, help="override options.workers")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.workers:
        cfg.workers = max(1, args.workers)
    results = run(cfg, dry_run=args.dry_run, limit=args.limit)
    return 1 if any(r.status == "error" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
