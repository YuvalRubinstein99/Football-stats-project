"""Paths shared by scripts and notebooks, independent of the working directory."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'results'

# Optional local installation used by the desktop environment; never committed.
local_dependencies = ROOT / '.prediction_deps'
if local_dependencies.is_dir():
    sys.path.insert(0, str(local_dependencies))
