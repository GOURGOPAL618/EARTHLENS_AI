"""
EARTHLENS AI — Core Package
============================
Author  : Gouragopal Mohapatra
Purpose : Satellite Intelligence Platform
"""

__version__ = "1.0.0"
__author__  = "Gouragopal Mohapatra"

# ── Analysis Engine ────────────────────────────────────────────────────────────
from earthlens_core.analysis_engine   import preprocessing
from earthlens_core.analysis_engine   import ndvi
from earthlens_core.analysis_engine   import water_detection

# ── Visualization Hub ──────────────────────────────────────────────────────────
from earthlens_core.visualization_hub import map_view
from earthlens_core.visualization_hub import plots

# ── Data Pipeline ──────────────────────────────────────────────────────────────
from earthlens_core.data_pipeline     import sentinel_api
from earthlens_core.data_pipeline     import landsat_api

__all__ = [
    # Analysis
    "preprocessing",
    "ndvi",
    "water_detection",
    # Visualization
    "map_view",
    "plots",
    # Data Pipeline
    "sentinel_api",
    "landsat_api",
]