# ============================================================
#   EARTHLENS AI – CONFIGURATION SETTINGS
#   earthlens_config/settings.py
# ============================================================

import os
from pathlib import Path

# ------------------------------------------------------------
# 📁 BASE PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # project root

DATA_DIR            = BASE_DIR / "earthlens_data"
RAW_IMAGERY_DIR     = DATA_DIR / "raw_imagery"
PROCESSED_DIR       = DATA_DIR / "processed_insights"
ASSETS_DIR          = BASE_DIR / "earthlens_assets"

# Auto-create directories if they don't exist
for _dir in [RAW_IMAGERY_DIR, PROCESSED_DIR, ASSETS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 🛰️ SATELLITE API CREDENTIALS
# ------------------------------------------------------------

# --- Sentinel Hub (Copernicus) ---
SENTINEL_CLIENT_ID      = os.getenv("SENTINEL_CLIENT_ID", "your_sentinel_client_id_here")
SENTINEL_CLIENT_SECRET  = os.getenv("SENTINEL_CLIENT_SECRET", "your_sentinel_client_secret_here")
SENTINEL_INSTANCE_ID    = os.getenv("SENTINEL_INSTANCE_ID", "your_sentinel_instance_id_here")
SENTINEL_BASE_URL       = "https://services.sentinel-hub.com"

# --- Landsat (USGS Earth Explorer) ---
USGS_USERNAME   = os.getenv("USGS_USERNAME", "your_usgs_username_here")
USGS_PASSWORD   = os.getenv("USGS_PASSWORD", "your_usgs_password_here")
USGS_API_URL    = "https://m2m.cr.usgs.gov/api/api/json/stable"

# --- NASA Earthdata (optional fallback) ---
EARTHDATA_TOKEN = os.getenv("EARTHDATA_TOKEN", "your_earthdata_token_here")


# ------------------------------------------------------------
# 🌍 DEFAULT GEOGRAPHIC SETTINGS
# ------------------------------------------------------------

DEFAULT_LOCATION = {
    "name"      : "Delhi, India",
    "latitude"  : 28.6139,
    "longitude" : 77.2090,
}

DEFAULT_BBOX_BUFFER = 0.1   # degrees around selected point
DEFAULT_CRS         = "EPSG:4326"   # WGS84


# ------------------------------------------------------------
# 📅 DATA FETCH SETTINGS
# ------------------------------------------------------------

DEFAULT_DATE_RANGE_DAYS = 30        # look-back window for imagery
MAX_CLOUD_COVER_PERCENT = 20        # filter cloudy scenes
DEFAULT_IMAGE_RESOLUTION = 10       # metres per pixel (Sentinel-2 default)

SENTINEL_COLLECTIONS = {
    "L2A": "sentinel-2-l2a",        # surface reflectance (preferred)
    "L1C": "sentinel-2-l1c",        # top-of-atmosphere
}

LANDSAT_COLLECTIONS = {
    "L8": "LANDSAT_OT_C2_L2",      # Landsat 8 Collection 2 Level-2
    "L9": "LANDSAT_OT_C2_L2",      # Landsat 9 Collection 2 Level-2
}


# ------------------------------------------------------------
# 🌿 ANALYSIS ENGINE SETTINGS
# ------------------------------------------------------------

# NDVI thresholds
NDVI_THRESHOLDS = {
    "barren"     : (-1.0,  0.1),
    "sparse"     : ( 0.1,  0.3),
    "moderate"   : ( 0.3,  0.5),
    "dense"      : ( 0.5,  0.7),
    "very_dense" : ( 0.7,  1.0),
}

# Water detection (NDWI threshold)
NDWI_WATER_THRESHOLD = 0.0

# Preprocessing
PREPROCESSING = {
    "target_size"   : (256, 256),   # resize images to this before processing
    "normalize"     : True,
    "clip_percentile": (2, 98),     # stretch contrast
}


# ------------------------------------------------------------
# 🤖 AI / ML MODEL SETTINGS
# ------------------------------------------------------------

MODEL_DIR = BASE_DIR / "earthlens_core" / "intelligence_models"

CLASSIFIER_CONFIG = {
    "model_file"    : MODEL_DIR / "land_classifier.pkl",
    "classes"       : ["urban", "forest", "water", "agriculture", "barren"],
    "confidence_threshold": 0.60,
}


# ------------------------------------------------------------
# 🗺️ VISUALIZATION SETTINGS
# ------------------------------------------------------------

MAP_DEFAULTS = {
    "zoom_start"    : 10,
    "tile_layer"    : "CartoDB positron",   # clean basemap
    "attribution"   : "EarthLens AI | Satellite data © ESA / USGS",
}

NDVI_COLORMAP   = "RdYlGn"     # Red → Yellow → Green
WATER_COLOR     = "#1a78c2"
URBAN_COLOR     = "#e05c2a"
FOREST_COLOR    = "#2ca02c"

PLOT_THEME      = "plotly_dark"     # Streamlit dark theme compatible


# ------------------------------------------------------------
# 🖥️ STREAMLIT APP SETTINGS
# ------------------------------------------------------------

APP_CONFIG = {
    "title"         : "🛰️ EarthLens AI",
    "subtitle"      : "Satellite Intelligence Platform",
    "page_icon"     : "🌍",
    "layout"        : "wide",
    "sidebar_state" : "expanded",
}

SUPPORTED_ANALYSES = [
    "NDVI – Vegetation Health",
    "Water Body Detection",
    "Land Use Classification",
    "Change Detection (coming soon)",
]


# ------------------------------------------------------------
# ⚡ CACHING & PERFORMANCE
# ------------------------------------------------------------

CACHE_ENABLED       = True
CACHE_TTL_SECONDS   = 3600     # 1 hour
MAX_CONCURRENT_REQUESTS = 3


# ------------------------------------------------------------
# 📝 LOGGING
# ------------------------------------------------------------

LOG_LEVEL   = os.getenv("EARTHLENS_LOG_LEVEL", "INFO")
LOG_FILE    = BASE_DIR / "earthlens.log"


# ── SETTINGS object ────────────────────────────────────────────────────────────
SETTINGS = {
    "raw_imagery_dir"    : str(BASE_DIR / "earthlens_data" / "raw_imagery"),
    "processed_dir"      : str(BASE_DIR / "earthlens_data" / "processed_insights"),
    "default_lat"        : 28.57,
    "default_lon"        : 77.20,
    "default_zoom"       : 10,
    "app_title"          : "EarthLens AI",
    "app_icon"           : "🛰️",
}


# ------------------------------------------------------------
# 🔐 SECURITY NOTE
# ------------------------------------------------------------
# ⚠️  NEVER hardcode real API keys here!
# Use environment variables or a .env file (python-dotenv).
# Add .env to your .gitignore immediately.
#
# Quick setup:
#   pip install python-dotenv
#   from dotenv import load_dotenv; load_dotenv()
# ============================================================
