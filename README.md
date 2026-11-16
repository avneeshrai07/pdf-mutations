# pdf-mutations

Batch-edit thousands of PDFs: find company names and logos, remove them and put your project name in their place. Built on [PyMuPDF](https://github.com/pymupdf/PyMuPDF) (PDF editing) and [RapidOCR](https://github.com/RapidAI/RapidOCR) (reads text inside logo images).

## Setup

```bash
python -m venv myvenv
myvenv\Scripts\activate        # Windows  (Linux/macOS: source myvenv/bin/activate)
pip install -r requirements.txt
```

## Use

1. Set `paths.input_dir` (must exist, else the run stops with an error) and `paths.output_dir` (created if missing) to full paths.
2. Edit [config.yaml](config.yaml): `project_name`, `companies`, `logo_image_extensions`, `paths`.
3. Try a few files first, then run everything:

```bash
python run.py --dry-run --limit 5     # scan only, nothing written
python run.py --limit 5               # real run on 5 files
python run.py                         # all files
```

Results land in `output_dir` together with `_report.csv` (hits and errors per file). Re-running skips files already in `output_dir`, so an interrupted run can be resumed.

## Layout

```
config.yaml        what to remove, replacement name, input/output folders
run.py             command-line entry point
engine/
  config.py        loads + validates config.yaml
  batch.py         finds PDFs, runs them (optionally in parallel), writes the report
  processor.py     per-PDF logic: text replace, logo removal, scanned-page OCR, metadata
  logos.py         decides if an embedded image is a logo (OCR text / reference logos)
  ocr.py           lazy RapidOCR wrapper
```

## How logos are found

- **Text layer**: every company name is searched on every page (case-insensitive) and overwritten with the project name, keeping the original font size/colour where possible.
- **Embedded images** whose format is in `logo_image_extensions`: removed when OCR finds a company name in them, or when they look like a file in `logo_reference_dir`. The project name is drawn in the freed box.
- **Scanned pages** (no text layer): enable `ocr_scanned_pages` to OCR the page and cover the company names.

## Limitations

- Logos drawn as vector shapes, or images with no readable name, are not detected by OCR. Drop a sample of each logo in `logo_reference_dir` to catch those.
- A name split across two lines is not found in the text layer.
- The replacement text uses the built-in Helvetica font (Latin characters only).
- Always spot-check `output/` on a handful of files before processing the whole set.

## Phase 1 helpers (read-only, never modify your PDFs)

Run these first to learn what is in your PDFs; results go to `<output_dir>/phase1_*`.

| Script | Purpose |
|---|---|
| `python phase1/extract.py` | Saves every embedded PNG/JPG once (`images.csv` shows how many PDFs share each), company-name hits with crops, and repeated text lines (to discover names for `companies`). |
| `python phase1/locate.py` | Finds the company name + a vector logo next to it; writes annotated crops (name blue, logo red). |
| `python phase1/learn_logo.py` | Learns vector logos as templates (`logo_template_N.json`, one per variant) and lists PDFs that need review. |

### Two kinds of logos
- **Embedded image logos** (PNG/JPG): copy the logo files from `phase1_extract/images/` into a folder and set `logo_reference_dir`. Logos with readable text are also caught by OCR (`ocr_images`).
- **Vector logos** (drawn shapes, typical for CAD title blocks): run `learn_logo.py`, then `run.py` uses the templates (`vector_logo` section).

`_report.csv` marks a file `review` when a company name/logo is still present after editing, or a name was found on a page where no logo was removed. List every spelling of a company in `companies` (e.g. both "LTD" and "LIMITED"), otherwise part of the name can be left behind.

### Image type scoring (CLIP)
`phase1/extract.py` also classifies each unique image with CLIP (`engine/image_class.py`): `looks_like` = logo / signature / stamp / photo / drawing / text / icon, plus `confidence` and `logo_score` in `images.csv`. Copies of logo/stamp/signature images are grouped in `phase1_extract/images_by_type/`. Treat low confidence (< 0.5) as "check by eye". Needs `torch` + `open_clip_torch` (first run downloads a ~350 MB model); skip with `--no-clip`.
