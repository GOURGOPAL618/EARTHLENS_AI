"""
EARTHLENS AI — Landsat API Module
===================================
Author : Gouragopal Mohapatra
Purpose: Fetch Landsat 8/9 imagery from USGS M2M REST API
API    : USGS Machine-to-Machine (M2M) REST API
Docs   : https://m2m.cr.usgs.gov/api/docs/json/
"""

import os
import json
import requests
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger


# ── Constants ──────────────────────────────────────────────────────────────────
M2M_URL        = "https://m2m.cr.usgs.gov/api/api/json/stable"
DATASET_NAME   = "landsat_ot_c2_l2"   # Landsat 8/9 OLI/TIRS Collection 2 L2


# ── Auth ───────────────────────────────────────────────────────────────────────
def login() -> str:
    """
    Login to USGS M2M API.
    Set in .env:
        USGS_USERNAME = your_username
        USGS_PASSWORD = your_password
    Free account: https://ers.cr.usgs.gov/register
    Returns: API token
    """
    username = os.getenv("USGS_USERNAME", "")
    password = os.getenv("USGS_PASSWORD", "")

    if not username or not password:
        raise ValueError(
            "USGS credentials missing!\n"
            "Set USGS_USERNAME and USGS_PASSWORD in .env\n"
            "Free account: https://ers.cr.usgs.gov/register"
        )

    payload = {"username": username, "password": password}
    response = _m2m_request("login", payload)
    token = response["data"]

    logger.success(f"USGS M2M login successful | user={username}")
    return token


def logout(token: str) -> None:
    """Logout from USGS M2M API."""
    _m2m_request("logout", {}, token)
    logger.info("USGS M2M logout successful")


# ── Search Scenes ──────────────────────────────────────────────────────────────
def search_scenes(
    token       : str,
    bbox_coords : tuple,       # (west, south, east, north)
    date_from   : str,         # "YYYY-MM-DD"
    date_to     : str,         # "YYYY-MM-DD"
    max_cloud   : int  = 20,   # max cloud cover %
    max_results : int  = 5,
) -> list:
    """
    Search for available Landsat scenes.
    Returns list of scene metadata.
    """
    west, south, east, north = bbox_coords

    payload = {
        "datasetName" : DATASET_NAME,
        "sceneFilter" : {
            "spatialFilter" : {
                "filterType" : "mbr",
                "lowerLeft"  : {"latitude": south, "longitude": west},
                "upperRight" : {"latitude": north, "longitude": east},
            },
            "acquisitionFilter" : {
                "start" : date_from,
                "end"   : date_to,
            },
            "cloudCoverFilter" : {
                "min" : 0,
                "max" : max_cloud,
            },
        },
        "maxResults"  : max_results,
        "startingNumber": 1,
    }

    response = _m2m_request("scene-search", payload, token)
    scenes   = response["data"]["results"]

    logger.info(f"Found {len(scenes)} Landsat scenes")
    for s in scenes:
        logger.info(
            f"  Scene: {s.get('entityId')} | "
            f"Cloud: {s.get('cloudCover')}% | "
            f"Date: {s.get('temporalCoverage', {}).get('startDate', 'N/A')}"
        )

    return scenes


# ── Download Scene ─────────────────────────────────────────────────────────────
def download_scene(
    token      : str,
    entity_id  : str,
    output_dir : Path,
) -> Path:
    """
    Download a Landsat scene.
    Returns path to downloaded file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get download options
    payload = {
        "datasetName" : DATASET_NAME,
        "entityIds"   : [entity_id],
    }
    response = _m2m_request("download-options", payload, token)
    options  = response["data"]

    if not options:
        raise ValueError(f"No download options for scene: {entity_id}")

    # Pick first available product
    product = options[0]
    download_id = product["id"]

    # Request download URL
    payload = {
        "downloads"         : [{"entityId": entity_id, "productId": download_id}],
        "downloadApplication": "EE",
    }
    response     = _m2m_request("download-request", payload, token)
    download_url = response["data"]["preparingDownloads"][0]["url"]

    # Download file
    filename = output_dir / f"{entity_id}.tar"
    logger.info(f"Downloading scene: {entity_id}")

    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0

        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Downloading... {pct:.1f}%", end="")

    print()
    logger.success(f"Scene downloaded: {filename}")
    return filename


# ── Extract Bands ──────────────────────────────────────────────────────────────
def extract_bands(tar_path: Path, output_dir: Path) -> dict:
    """
    Extract band GeoTIFF files from downloaded .tar archive.

    Landsat 8/9 Band Mapping:
        B2  → Blue
        B3  → Green
        B4  → Red        ← NDVI
        B5  → NIR        ← NDVI
        B6  → SWIR-1
        B7  → SWIR-2
    """
    import tarfile

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    band_map = {
        "B2": "B02",
        "B3": "B03",
        "B4": "B04",   # Red
        "B5": "B08",   # NIR — mapped to sentinel naming
        "B6": "B11",   # SWIR-1
        "B7": "B12",   # SWIR-2
    }

    extracted = {}

    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        for member in members:
            for landsat_band, sentinel_band in band_map.items():
                if f"_{landsat_band}.TIF" in member.name:
                    tar.extract(member, output_dir)
                    src_path  = output_dir / member.name
                    dest_path = output_dir / f"landsat_{sentinel_band}.tif"
                    src_path.rename(dest_path)
                    extracted[sentinel_band] = dest_path
                    logger.info(f"Extracted: {member.name} → {dest_path.name}")

    logger.success(f"Extracted {len(extracted)} bands")
    return extracted


# ── Full Pipeline ──────────────────────────────────────────────────────────────
def run_landsat_pipeline(
    bbox_coords : tuple,
    date_from   : str  = None,
    date_to     : str  = None,
    max_cloud   : int  = 20,
    output_dir  : Path = None,
) -> dict:
    """
    Complete Landsat pipeline:
    Login → Search → Download → Extract → Return band paths

    Returns: {
        "B02": path, "B03": path,
        "B04": path, "B08": path,
        "B11": path, "B12": path,
        "source": "landsat"
    }
    """
    # Default: last 30 days
    if not date_from or not date_to:
        today     = datetime.utcnow()
        date_to   = today.strftime("%Y-%m-%d")
        date_from = (today - timedelta(days=30)).strftime("%Y-%m-%d")

    output_dir = Path(output_dir) if output_dir else Path("earthlens_data/raw_imagery")

    logger.info("Starting Landsat pipeline...")

    try:
        # Login
        token = login()

        # Search
        scenes = search_scenes(
            token       = token,
            bbox_coords = bbox_coords,
            date_from   = date_from,
            date_to     = date_to,
            max_cloud   = max_cloud,
        )

        if not scenes:
            raise ValueError("No scenes found! Try different date range or location.")

        # Download first scene
        entity_id = scenes[0]["entityId"]
        tar_path  = download_scene(token, entity_id, output_dir / "temp")

        # Extract bands
        band_paths = extract_bands(tar_path, output_dir)
        band_paths["source"] = "landsat"

        # Logout
        logout(token)

        logger.success("Landsat pipeline complete!")
        return band_paths

    except Exception as e:
        logger.error(f"Landsat pipeline failed: {e}")
        logger.warning("Falling back to synthetic data...")
        return _synthetic_fallback(bbox_coords, output_dir)


# ── Synthetic Fallback ─────────────────────────────────────────────────────────
def _synthetic_fallback(bbox_coords: tuple,
                        output_dir: Path = None) -> dict:
    """
    Fallback — synthetic bands when no API credentials.
    Same band naming as real Landsat output.
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

    output_dir = Path(output_dir) if output_dir else Path("earthlens_data/raw_imagery")
    output_dir.mkdir(parents=True, exist_ok=True)

    band_paths = {}
    for band_name, data in band_configs.items():
        array    = np.clip(data, 0, 10000).astype(np.uint16)
        filepath = output_dir / f"landsat_{band_name}.tif"

        with rasterio.open(filepath, "w", **meta) as dst:
            dst.write(array, 1)

        band_paths[band_name] = filepath

    band_paths["source"] = "synthetic_landsat"
    logger.warning("Using SYNTHETIC Landsat data — add credentials for real data!")
    return band_paths


# ── M2M Request Helper ─────────────────────────────────────────────────────────
def _m2m_request(endpoint: str,
                 payload : dict,
                 token   : str = None) -> dict:
    """
    Generic M2M API request helper.
    """
    url     = f"{M2M_URL}/{endpoint}"
    headers = {"Content-Type": "application/json"}

    if token:
        headers["X-Auth-Token"] = token

    response = requests.post(
        url,
        headers = headers,
        data    = json.dumps(payload),
        timeout = 30,
    )
    response.raise_for_status()

    data = response.json()

    if data.get("errorCode"):
        raise ValueError(
            f"M2M API Error: {data['errorCode']} — {data.get('errorMessage')}"
        )

    return data
