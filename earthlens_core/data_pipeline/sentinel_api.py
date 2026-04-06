"""
EARTHLENS AI — Sentinel-2 API Module
======================================
Author : Gouragopal Mohapatra
Purpose: Fetch real Sentinel-2 imagery from Copernicus
API    : Sentinel Hub (sentinelhub-py)
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

try:
    from sentinelhub import (
        SHConfig,
        SentinelHubRequest,
        DataCollection,
        MimeType,
        BBox,
        CRS as SH_CRS,
        bbox_to_dimensions,
    )
    SENTINELHUB_AVAILABLE = True
except ImportError:
    SENTINELHUB_AVAILABLE = False
    logger.warning("sentinelhub not available — using synthetic fallback")


# ── Config ─────────────────────────────────────────────────────────────────────
def get_sh_config() -> "SHConfig":
    """
    Load Sentinel Hub config from environment variables.
    Set these in your .env file:
        SH_CLIENT_ID     = your_client_id
        SH_CLIENT_SECRET = your_client_secret
    """
    if not SENTINELHUB_AVAILABLE:
        raise ImportError("sentinelhub package not installed!")

    config = SHConfig()
    config.sh_client_id     = os.getenv("SH_CLIENT_ID",     "")
    config.sh_client_secret = os.getenv("SH_CLIENT_SECRET", "")

    if not config.sh_client_id or not config.sh_client_secret:
        raise ValueError(
            "Sentinel Hub credentials missing!\n"
            "Set SH_CLIENT_ID and SH_CLIENT_SECRET in .env file.\n"
            "Free account: https://www.sentinel-hub.com/trial/"
        )

    logger.info("Sentinel Hub config loaded")
    return config


# ── Evalscript ─────────────────────────────────────────────────────────────────
EVALSCRIPT_ALL_BANDS = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B02","B03","B04","B08","B11","B12"],
            units: "DN"
        }],
        output: [
            { id: "B02", bands: 1, sampleType: "UINT16" },
            { id: "B03", bands: 1, sampleType: "UINT16" },
            { id: "B04", bands: 1, sampleType: "UINT16" },
            { id: "B08", bands: 1, sampleType: "UINT16" },
            { id: "B11", bands: 1, sampleType: "UINT16" },
            { id: "B12", bands: 1, sampleType: "UINT16" },
        ]
    };
}
function evaluatePixel(sample) {
    return {
        B02: [sample.B02],
        B03: [sample.B03],
        B04: [sample.B04],
        B08: [sample.B08],
        B11: [sample.B11],
        B12: [sample.B12],
    };
}
"""


# ── Fetch Bands ────────────────────────────────────────────────────────────────
def fetch_sentinel2_bands(
    bbox_coords : tuple,       # (west, south, east, north)
    date_from   : str,         # "YYYY-MM-DD"
    date_to     : str,         # "YYYY-MM-DD"
    resolution  : int  = 10,   # meters per pixel
    output_dir  : Path = None,
) -> dict:
    """
    Fetch Sentinel-2 bands from Sentinel Hub API.

    Returns: {
        "B02": array, "B03": array,
        "B04": array, "B08": array,
        "B11": array, "B12": array,
        "meta": rasterio_meta
    }
    """
    if not SENTINELHUB_AVAILABLE:
        logger.warning("Falling back to synthetic data...")
        return _synthetic_fallback(bbox_coords, output_dir)

    config = get_sh_config()

    west, south, east, north = bbox_coords
    bbox   = BBox(bbox=bbox_coords, crs=SH_CRS.WGS84)
    size   = bbox_to_dimensions(bbox, resolution=resolution)

    logger.info(
        f"Fetching Sentinel-2 | bbox={bbox_coords} | "
        f"date={date_from} to {date_to} | size={size}"
    )

    request = SentinelHubRequest(
        evalscript    = EVALSCRIPT_ALL_BANDS,
        input_data    = [
            SentinelHubRequest.input_data(
                data_collection = DataCollection.SENTINEL2_L2A,
                time_interval   = (date_from, date_to),
                mosaicking_order= "leastCC",   # least cloud cover
            )
        ],
        responses     = [
            SentinelHubRequest.output_response("B02", MimeType.TIFF),
            SentinelHubRequest.output_response("B03", MimeType.TIFF),
            SentinelHubRequest.output_response("B04", MimeType.TIFF),
            SentinelHubRequest.output_response("B08", MimeType.TIFF),
            SentinelHubRequest.output_response("B11", MimeType.TIFF),
            SentinelHubRequest.output_response("B12", MimeType.TIFF),
        ],
        bbox          = bbox,
        size          = size,
        config        = config,
    )

    data = request.get_data()[0]

    bands  = {}
    transform = from_bounds(west, south, east, north, size[0], size[1])
    meta = {
        "driver"   : "GTiff",
        "dtype"    : "uint16",
        "width"    : size[0],
        "height"   : size[1],
        "count"    : 1,
        "crs"      : CRS.from_epsg(4326),
        "transform": transform,
    }

    for band_name in ["B02","B03","B04","B08","B11","B12"]:
        bands[band_name] = data[band_name].squeeze().astype(np.uint16)

    bands["meta"] = meta

    # Save if output dir given
    if output_dir:
        _save_bands(bands, meta, output_dir)

    logger.success(f"Fetched {len(bands)-1} Sentinel-2 bands!")
    return bands


# ── Save Bands ─────────────────────────────────────────────────────────────────
def _save_bands(bands: dict, meta: dict, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for band_name, array in bands.items():
        if band_name == "meta":
            continue
        filepath = output_dir / f"sentinel2_live_{band_name}.tif"
        with rasterio.open(filepath, "w", **meta) as dst:
            dst.write(array, 1)
        logger.info(f"Saved: {filepath.name}")


# ── Synthetic Fallback ─────────────────────────────────────────────────────────
def _synthetic_fallback(bbox_coords: tuple,
                        output_dir: Path = None) -> dict:
    """
    Fallback — returns synthetic bands when no API key available.
    Same format as real fetch — drop-in replacement.
    """
    from scipy.ndimage import gaussian_filter

    west, south, east, north = bbox_coords
    WIDTH, HEIGHT = 256, 256

    transform = from_bounds(west, south, east, north, WIDTH, HEIGHT)
    meta = {
        "driver"   : "GTiff",
        "dtype"    : "uint16",
        "width"    : WIDTH,
        "height"   : HEIGHT,
        "count"    : 1,
        "crs"      : CRS.from_epsg(4326),
        "transform": transform,
    }

    np.random.seed(42)
    y, x   = np.mgrid[0:HEIGHT, 0:WIDTH]
    cx, cy = WIDTH // 2, HEIGHT // 2

    urban = np.exp(-((x-cx)**2 + (y-cy)**2) / (60**2))
    veg   = np.exp(-((x-80)**2 + (y-100)**2) / (30**2))
    water = np.clip(1 - np.abs((x - y*0.6) - 30) / 12, 0, 1)
    noise = gaussian_filter(np.random.rand(HEIGHT, WIDTH), sigma=8)

    band_configs = {
        "B02": urban*1800 + veg*500  + water*800  + noise*400 + 600,
        "B03": urban*1600 + veg*1000 + water*900  + noise*400 + 600,
        "B04": urban*1700 + veg*400  + water*600  + noise*350 + 500,
        "B08": urban*1200 + veg*5500 + water*400  + noise*500 + 800,
        "B11": urban*2200 + veg*700  + water*300  + noise*500 + 400,
        "B12": urban*2000 + veg*600  + water*200  + noise*450 + 300,
    }

    bands = {}
    for name, data in band_configs.items():
        bands[name] = np.clip(data, 0, 10000).astype(np.uint16)

    bands["meta"] = meta

    if output_dir:
        _save_bands(bands, meta, output_dir)
        logger.info("Synthetic bands saved to output_dir")

    logger.warning("Using SYNTHETIC data — add API keys for real data!")
    return bands


# ── Helper: date range ─────────────────────────────────────────────────────────
def last_30_days() -> tuple:
    """Returns (date_from, date_to) for last 30 days."""
    today     = datetime.utcnow()
    date_to   = today.strftime("%Y-%m-%d")
    date_from = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    return date_from, date_to