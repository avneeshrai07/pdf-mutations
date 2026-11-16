"""PDF rebranding engine: replace company names/logos with a project name."""
from .batch import run
from .config import Config, load_config
from .processor import process_pdf

__all__ = ["run", "load_config", "Config", "process_pdf"]
