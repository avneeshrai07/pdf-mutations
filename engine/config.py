"""Load and validate config.yaml."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import yaml


def normalize(text: str) -> str:
    """Lower-case and strip everything except letters/digits, for fuzzy matching."""
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


@dataclass
class Config:
    input_dir: Path
    output_dir: Path
    project_name: str
    companies: Tuple[str, ...]
    recursive: bool = True
    preserve_structure: bool = True
    logo_image_extensions: Tuple[str, ...] = ("png", "jpg", "jpeg")
    logo_reference_dir: Optional[Path] = None
    replace_text: bool = True
    ocr_images: bool = True
    ocr_scanned_pages: bool = False
    scrub_metadata: bool = True
    image_removal: str = "remove"
    max_logo_page_fraction: float = 0.25
    min_logo_pixels: int = 16
    reference_max_distance: int = 25
    fill_color: Union[str, Tuple[float, float, float]] = "auto"
    workers: int = 1
    skip_existing: bool = True
    copy_unchanged: bool = True
    # vector (drawn) logos, e.g. CAD title blocks
    vector_logo_enabled: bool = True
    vector_templates_dir: Optional[Path] = None
    vector_min_score: float = 0.9
    vector_fallback_name_anchor: bool = False
    vector_logo_text: bool = False  # also write the project name in the removed logo's box
    # automatic company-name detection (legal suffix: LTD, LIMITED, INC, ...)
    auto_detect_names: bool = True
    keep_names: Tuple[str, ...] = ()  # names that must never be replaced

    # normalised company names, longest first (so "Acme Corp" wins over "Acme")
    company_keys: Tuple[str, ...] = field(init=False, default=())
    keep_keys: Tuple[str, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        self.keep_keys = tuple(normalize(k) for k in self.keep_names if normalize(k))
        self.company_keys = tuple(
            sorted({normalize(c) for c in self.companies if normalize(c)}, key=len, reverse=True)
        )


def load_config(path: Union[str, Path]) -> Config:
    path = Path(path).resolve()
    if not path.is_file():
        raise SystemExit(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent

    def resolve(p: str) -> Path:
        pp = Path(str(p)).expanduser()
        return (pp if pp.is_absolute() else base / pp).resolve()

    paths = raw.get("paths") or {}
    opts = raw.get("options") or {}

    if not paths.get("input_dir") or not paths.get("output_dir"):
        raise SystemExit("config: paths.input_dir and paths.output_dir are required")
    project_name = str(raw.get("project_name") or "").strip()
    if not project_name:
        raise SystemExit("config: project_name is required")
    companies = tuple(str(c).strip() for c in (raw.get("companies") or []) if str(c).strip())
    auto = raw.get("auto_detect_names") or {}
    auto_on = bool(auto.get("enabled", True))
    if not companies and not auto_on:
        raise SystemExit("config: list at least one name in companies, or enable auto_detect_names")

    exts = tuple(
        str(e).lower().lstrip(".")
        for e in (raw.get("logo_image_extensions") or ["png", "jpg", "jpeg"])
    )
    if "jpg" in exts and "jpeg" not in exts:
        exts += ("jpeg",)

    fill = opts.get("fill_color", "auto")
    if isinstance(fill, (list, tuple)):
        if len(fill) != 3:
            raise SystemExit("config: options.fill_color must be 'auto' or [R, G, B]")
        fill = tuple(float(v) / 255.0 for v in fill)
    elif str(fill).lower() != "auto":
        raise SystemExit("config: options.fill_color must be 'auto' or [R, G, B]")
    else:
        fill = "auto"

    removal = str(opts.get("image_removal", "remove")).lower()
    if removal not in ("remove", "pixels"):
        raise SystemExit("config: options.image_removal must be 'remove' or 'pixels'")

    vec = raw.get("vector_logo") or {}
    vt = vec.get("templates_dir")
    ref = raw.get("logo_reference_dir")
    cfg = Config(
        input_dir=resolve(paths["input_dir"]),
        output_dir=resolve(paths["output_dir"]),
        project_name=project_name,
        companies=companies,
        recursive=bool(paths.get("recursive", True)),
        preserve_structure=bool(paths.get("preserve_structure", True)),
        logo_image_extensions=exts,
        logo_reference_dir=resolve(ref) if ref else None,
        replace_text=bool(opts.get("replace_text", True)),
        ocr_images=bool(opts.get("ocr_images", True)),
        ocr_scanned_pages=bool(opts.get("ocr_scanned_pages", False)),
        scrub_metadata=bool(opts.get("scrub_metadata", True)),
        image_removal=removal,
        max_logo_page_fraction=float(opts.get("max_logo_page_fraction", 0.25)),
        min_logo_pixels=int(opts.get("min_logo_pixels", 16)),
        reference_max_distance=int(opts.get("reference_max_distance", 25)),
        fill_color=fill,
        workers=max(1, int(opts.get("workers", 1))),
        skip_existing=bool(opts.get("skip_existing", True)),
        copy_unchanged=bool(opts.get("copy_unchanged", True)),
        vector_logo_enabled=bool(vec.get("enabled", True)),
        vector_templates_dir=resolve(vt) if vt else None,
        vector_min_score=float(vec.get("min_score", 0.9)),
        vector_fallback_name_anchor=bool(vec.get("fallback_to_name_anchor", False)),
        vector_logo_text=bool(vec.get("write_project_name_in_logo_box", False)),
        auto_detect_names=auto_on,
        keep_names=tuple(str(k).strip() for k in (auto.get("keep") or []) if str(k).strip()),
    )
    if cfg.vector_logo_enabled and cfg.vector_templates_dir is None:
        default = cfg.output_dir / "phase1_template"  # where phase1/learn_logo.py writes them
        cfg.vector_templates_dir = default

    if not cfg.input_dir.is_dir():
        raise SystemExit(f"ERROR: input folder does not exist: {cfg.input_dir}")
    if cfg.input_dir == cfg.output_dir:
        raise SystemExit("config: output_dir must differ from input_dir")
    if cfg.logo_reference_dir and not cfg.logo_reference_dir.is_dir():
        raise SystemExit(f"config: logo_reference_dir does not exist: {cfg.logo_reference_dir}")
    return cfg
