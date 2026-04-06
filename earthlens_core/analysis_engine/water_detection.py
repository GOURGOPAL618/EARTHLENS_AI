"""
EARTHLENS AI — Water Detection Module
=======================================
Author : Gouragopal Mohapatra
Purpose: Detect water bodies using NDWI and MNDWI
Formula:
    NDWI  = (Green - NIR)   / (Green + NIR)
    MNDWI = (Green - SWIR1) / (Green + SWIR1)
Range  : -1.0 to +1.0
"""

import numpy as np
from pathlib import Path
from loguru import logger

from earthlens_core.analysis_engine.preprocessing import (
    preprocess_band,
    save_processed,
)


# ── Water Thresholds ───────────────────────────────────────────────────────────
WATER_THRESHOLD  = 0.0   # NDWI > 0 = water
MNDWI_THRESHOLD  = 0.0   # MNDWI > 0 = water (more accurate)

WATER_CLASSES = {
    "No Water" : (-1.0, 0.0),
    "Water"    : (0.0,  1.0),
}


# ── NDWI Calculation ───────────────────────────────────────────────────────────
def calculate_ndwi(green: np.ndarray,
                   nir: np.ndarray) -> np.ndarray:
    """
    NDWI = (Green - NIR) / (Green + NIR)
    Good for open water detection.
    """
    denominator = green + nir
    denominator[denominator == 0] = np.nan

    ndwi = (green - nir) / denominator
    ndwi = np.clip(ndwi, -1.0, 1.0)

    logger.info(
        f"NDWI calculated | "
        f"min={np.nanmin(ndwi):.3f} "
        f"max={np.nanmax(ndwi):.3f} "
        f"mean={np.nanmean(ndwi):.3f}"
    )
    return ndwi.astype(np.float32)


# ── MNDWI Calculation ──────────────────────────────────────────────────────────
def calculate_mndwi(green: np.ndarray,
                    swir1: np.ndarray) -> np.ndarray:
    """
    MNDWI = (Green - SWIR1) / (Green + SWIR1)
    Better for urban water detection (removes urban noise).
    """
    denominator = green + swir1
    denominator[denominator == 0] = np.nan

    mndwi = (green - swir1) / denominator
    mndwi = np.clip(mndwi, -1.0, 1.0)

    logger.info(
        f"MNDWI calculated | "
        f"min={np.nanmin(mndwi):.3f} "
        f"max={np.nanmax(mndwi):.3f} "
        f"mean={np.nanmean(mndwi):.3f}"
    )
    return mndwi.astype(np.float32)


# ── Water Mask ─────────────────────────────────────────────────────────────────
def create_water_mask(ndwi: np.ndarray,
                      mndwi: np.ndarray = None,
                      threshold: float = WATER_THRESHOLD) -> np.ndarray:
    """
    Create binary water mask.
    1 = Water, 0 = No Water
    If mndwi provided — combines both for better accuracy.
    """
    if mndwi is not None:
        # Water where BOTH indices agree
        water_mask = ((ndwi > threshold) & (mndwi > threshold)).astype(np.uint8)
        logger.info("Water mask created using NDWI + MNDWI combined")
    else:
        water_mask = (ndwi > threshold).astype(np.uint8)
        logger.info("Water mask created using NDWI only")

    water_pct = (water_mask.sum() / water_mask.size) * 100
    logger.info(f"Water coverage: {water_pct:.2f}%")

    return water_mask


# ── Water Stats ────────────────────────────────────────────────────────────────
def water_stats(ndwi: np.ndarray,
                water_mask: np.ndarray) -> dict:
    """
    Calculate water detection statistics.
    """
    valid = ndwi[~np.isnan(ndwi)]
    total = water_mask.size

    stats = {
        "ndwi_min"       : float(np.nanmin(ndwi)),
        "ndwi_max"       : float(np.nanmax(ndwi)),
        "ndwi_mean"      : float(np.nanmean(ndwi)),
        "water_pixels"   : int(water_mask.sum()),
        "total_pixels"   : int(total),
        "water_coverage" : round(float((water_mask.sum() / total) * 100), 2),
    }

    logger.info(f"Water Stats: {stats}")
    return stats


# ── Full Water Detection Pipeline ──────────────────────────────────────────────
def run_water_pipeline(green_path : str | Path,
                       nir_path   : str | Path,
                       swir1_path : str | Path = None,
                       output_dir : str | Path = None) -> dict:
    """
    Complete water detection pipeline:
    Load → Preprocess → NDWI → MNDWI → Mask → Stats → Save

    Returns: {
        "ndwi"       : ndwi array,
        "mndwi"      : mndwi array (if swir1 provided),
        "water_mask" : binary mask,
        "stats"      : statistics dict,
        "meta"       : rasterio metadata,
        "saved_to"   : output path (if saved)
    }
    """
    logger.info("Starting Water Detection pipeline...")

    # Load bands
    green, meta = preprocess_band(green_path)
    nir, _      = preprocess_band(nir_path)

    # NDWI
    ndwi  = calculate_ndwi(green, nir)
    mndwi = None

    # MNDWI (optional)
    if swir1_path:
        swir1, _ = preprocess_band(swir1_path)
        mndwi    = calculate_mndwi(green, swir1)

    # Water mask
    water_mask = create_water_mask(ndwi, mndwi)

    # Stats
    stats = water_stats(ndwi, water_mask)

    result = {
        "ndwi"       : ndwi,
        "mndwi"      : mndwi,
        "water_mask" : water_mask,
        "stats"      : stats,
        "meta"       : meta,
        "saved_to"   : None,
    }

    # Save if output dir given
    if output_dir:
        output_dir  = Path(output_dir)
        ndwi_path   = output_dir / "ndwi_delhi.tif"
        mask_path   = output_dir / "water_mask_delhi.tif"

        save_processed(ndwi, meta, ndwi_path)
        save_processed(water_mask.astype(np.float32), meta, mask_path)

        result["saved_to"] = str(ndwi_path)
        logger.success(f"Water detection saved to: {ndwi_path}")

    return result