"""
EARTHLENS AI — Google Earth Engine API Module
===============================================
Author  : Gouragopal Mohapatra
Purpose : Fetch real satellite data using Google Earth Engine
Data    : Sentinel-2, Landsat 8/9, MODIS
"""

import ee
import numpy as np
from pathlib import Path
from loguru import logger


# ── Constants ──────────────────────────────────────────────────────────────────
GEE_PROJECT = "alpine-guild-468014-e9"

COLLECTIONS = {
    "sentinel2" : "COPERNICUS/S2_SR_HARMONIZED",
    "landsat8"  : "LANDSAT/LC08/C02/T1_L2",
    "landsat9"  : "LANDSAT/LC09/C02/T1_L2",
    "modis"     : "MODIS/061/MOD09GA",
}

# Sentinel-2 band mapping
S2_BANDS = {
    "B02": "B2",    # Blue
    "B03": "B3",    # Green
    "B04": "B4",    # Red
    "B08": "B8",    # NIR
    "B11": "B11",   # SWIR-1
    "B12": "B12",   # SWIR-2
}

# Landsat 8/9 band mapping
LS_BANDS = {
    "B02": "SR_B2",   # Blue
    "B03": "SR_B3",   # Green
    "B04": "SR_B4",   # Red
    "B08": "SR_B5",   # NIR
    "B11": "SR_B6",   # SWIR-1
    "B12": "SR_B7",   # SWIR-2
}


# ── Initialize GEE ─────────────────────────────────────────────────────────────
def initialize_gee(project: str = GEE_PROJECT) -> bool:
    """
    Initialize Google Earth Engine.
    Returns True if successful.
    """
    try:
        ee.Initialize(project=project)
        logger.success(f"GEE initialized | project={project}")
        return True
    except Exception as e:
        logger.error(f"GEE initialization failed: {e}")
        return False


# ── Fetch Sentinel-2 ───────────────────────────────────────────────────────────
def fetch_sentinel2(bbox_coords : tuple,
                    date_from   : str,
                    date_to     : str,
                    cloud_pct   : int  = 20,
                    scale       : int  = 10) -> dict:
    """
    Fetch Sentinel-2 SR bands from GEE.

    bbox_coords : (west, south, east, north)
    date_from   : "YYYY-MM-DD"
    date_to     : "YYYY-MM-DD"
    cloud_pct   : max cloud cover %
    scale       : resolution in meters

    Returns: {"B02": array, "B03": array, ..., "meta": dict}
    """
    initialize_gee()

    west, south, east, north = bbox_coords
    region = ee.Geometry.Rectangle([west, south, east, north])

    logger.info(
        f"Fetching Sentinel-2 | "
        f"bbox={bbox_coords} | "
        f"date={date_from} to {date_to} | "
        f"cloud<{cloud_pct}%"
    )

    # Filter collection
    collection = (
        ee.ImageCollection(COLLECTIONS["sentinel2"])
        .filterBounds(region)
        .filterDate(date_from, date_to)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .first()
    )

    # Select bands
    band_list = list(S2_BANDS.values())
    image     = collection.select(band_list)

    # Get pixel data
    bands = _image_to_arrays(
        image      = image,
        region     = region,
        band_map   = S2_BANDS,
        scale      = scale,
        bbox_coords= bbox_coords,
    )

    logger.success(f"Sentinel-2 fetched! Bands: {list(bands.keys())}")
    return bands


# ── Fetch Landsat ──────────────────────────────────────────────────────────────
def fetch_landsat(bbox_coords : tuple,
                  date_from   : str,
                  date_to     : str,
                  cloud_pct   : int  = 20,
                  scale       : int  = 30) -> dict:
    """
    Fetch Landsat 8/9 SR bands from GEE.
    Returns: {"B02": array, "B03": array, ..., "meta": dict}
    """
    initialize_gee()

    west, south, east, north = bbox_coords
    region = ee.Geometry.Rectangle([west, south, east, north])

    logger.info(
        f"Fetching Landsat | "
        f"bbox={bbox_coords} | "
        f"date={date_from} to {date_to}"
    )

    # Try Landsat 9 first, fallback to 8
    for collection_name in ["landsat9", "landsat8"]:
        try:
            collection = (
                ee.ImageCollection(COLLECTIONS[collection_name])
                .filterBounds(region)
                .filterDate(date_from, date_to)
                .filter(ee.Filter.lt("CLOUD_COVER", cloud_pct))
                .sort("CLOUD_COVER")
                .first()
            )

            band_list = list(LS_BANDS.values())
            image     = collection.select(band_list)

            bands = _image_to_arrays(
                image      = image,
                region     = region,
                band_map   = LS_BANDS,
                scale      = scale,
                bbox_coords= bbox_coords,
            )

            logger.success(
                f"{collection_name.title()} fetched! "
                f"Bands: {list(bands.keys())}"
            )
            return bands

        except Exception as e:
            logger.warning(f"{collection_name} failed: {e} — trying next...")
            continue

    raise ValueError("No Landsat data available for given parameters!")


# ── Image to Arrays ────────────────────────────────────────────────────────────
def _image_to_arrays(image      : ee.Image,
                     region     : ee.Geometry,
                     band_map   : dict,
                     scale      : int,
                     bbox_coords: tuple) -> dict:
    """
    Convert GEE image to numpy arrays.
    Returns dict of {standard_band_name: numpy_array}
    """
    import requests as req
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    bands  = {}
    west, south, east, north = bbox_coords

    for std_name, gee_name in band_map.items():
        try:
            # Get download URL
            url = image.select(gee_name).getDownloadURL({
                "region" : region,
                "scale"  : scale,
                "format" : "NPY",
            })

            # Download array
            response = req.get(url, timeout=60)
            response.raise_for_status()

            arr = np.load(
                __import__("io").BytesIO(response.content),
                allow_pickle=True,
            )

            if arr.ndim == 3:
                arr = arr[:, :, 0]

            bands[std_name] = arr.astype(np.float32)
            logger.info(f"  ✅ {std_name} | shape={arr.shape}")

        except Exception as e:
            logger.warning(f"  ⚠️ {std_name} failed: {e}")
            continue

    # Add metadata
    if bands:
        sample = list(bands.values())[0]
        h, w   = sample.shape
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
        bands["meta"] = {
            "driver"   : "GTiff",
            "dtype"    : "float32",
            "width"    : w,
            "height"   : h,
            "count"    : 1,
            "crs"      : CRS.from_epsg(4326),
            "transform": from_bounds(west, south, east, north, w, h),
        }

    return bands


# ── NDVI from GEE (server-side) ────────────────────────────────────────────────
def compute_ndvi_gee(bbox_coords: tuple,
                     date_from  : str,
                     date_to    : str,
                     scale      : int = 10) -> dict:
    """
    Compute NDVI directly on GEE servers — faster!
    Returns: {"ndvi": array, "meta": dict}
    """
    initialize_gee()

    west, south, east, north = bbox_coords
    region = ee.Geometry.Rectangle([west, south, east, north])

    image = (
        ee.ImageCollection(COLLECTIONS["sentinel2"])
        .filterBounds(region)
        .filterDate(date_from, date_to)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .first()
    )

    # Server-side NDVI
    ndvi_image = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

    import requests as req
    url = ndvi_image.getDownloadURL({
        "region": region,
        "scale" : scale,
        "format": "NPY",
    })

    response = req.get(url, timeout=60)
    response.raise_for_status()

    ndvi_arr = np.load(
        __import__("io").BytesIO(response.content),
        allow_pickle=True,
    )

    if ndvi_arr.ndim == 3:
        ndvi_arr = ndvi_arr[:, :, 0]

    ndvi_arr = np.clip(ndvi_arr.astype(np.float32), -1.0, 1.0)

    h, w = ndvi_arr.shape
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    meta = {
        "driver"   : "GTiff",
        "dtype"    : "float32",
        "width"    : w,
        "height"   : h,
        "count"    : 1,
        "crs"      : CRS.from_epsg(4326),
        "transform": from_bounds(west, south, east, north, w, h),
    }

    logger.success(f"GEE NDVI computed | shape={ndvi_arr.shape}")
    return {"ndvi": ndvi_arr, "meta": meta}


# ── Full GEE Pipeline ──────────────────────────────────────────────────────────
def run_gee_pipeline(bbox_coords : tuple,
                     date_from   : str,
                     date_to     : str,
                     satellite   : str = "sentinel2",
                     cloud_pct   : int = 20) -> dict:
    """
    Complete GEE data fetch pipeline.

    satellite: "sentinel2" or "landsat"

    Returns: {
        "B02": array, ..., "B12": array,
        "meta": dict,
        "source": "gee_sentinel2" / "gee_landsat"
    }
    """
    logger.info(f"Starting GEE pipeline | satellite={satellite}")

    try:
        if satellite == "sentinel2":
            bands = fetch_sentinel2(
                bbox_coords = bbox_coords,
                date_from   = date_from,
                date_to     = date_to,
                cloud_pct   = cloud_pct,
            )
            bands["source"] = "gee_sentinel2"

        elif satellite == "landsat":
            bands = fetch_landsat(
                bbox_coords = bbox_coords,
                date_from   = date_from,
                date_to     = date_to,
                cloud_pct   = cloud_pct,
            )
            bands["source"] = "gee_landsat"

        else:
            raise ValueError(f"Unknown satellite: {satellite}")

        logger.success(f"GEE pipeline complete! Source={bands['source']}")
        return bands

    except Exception as e:
        logger.error(f"GEE pipeline failed: {e}")
        raise