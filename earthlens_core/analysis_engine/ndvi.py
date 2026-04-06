"""
EARTHLENS AI — NDVI Module
============================
Author : Gouragopal Mohapatra
Purpose: Calculate NDVI from Red (B04) and NIR (B08) bands
Formula: NDVI = (NIR - Red) / (NIR + Red)
Range  : -1.0 to +1.0
"""

import numpy as np
from pathlib import Path
from loguru import logger

from earthlens_core.analysis_engine.preprocessing import (
    preprocess_band,
    save_processed,
)
from earthlens_config.settings import SETTINGS


# ── NDVI Thresholds ────────────────────────────────────────────────────────────
NDVI_CLASSES = {
    "Water / Snow"      : (-1.0, 0.0),
    "Bare Soil / Urban" : (0.0,  0.2),
    "Sparse Vegetation" : (0.2,  0.4),
    "Moderate Vegetation": (0.4, 0.6),
    "Dense Vegetation"  : (0.6,  1.0),
}


# ── Core NDVI Calculation ──────────────────────────────────────────────────────
def calculate_ndvi(red: np.ndarray,
                   nir: np.ndarray) -> np.ndarray:
    """
    Calculate NDVI from Red and NIR arrays.
    Input arrays should be float32, already preprocessed.
    """
    # Avoid division by zero
    denominator = nir + red
    denominator[denominator == 0] = np.nan

    ndvi = (nir - red) / denominator

    # Clip to valid range
    ndvi = np.clip(ndvi, -1.0, 1.0)

    logger.info(
        f"NDVI calculated | "
        f"min={np.nanmin(ndvi):.3f} "
        f"max={np.nanmax(ndvi):.3f} "
        f"mean={np.nanmean(ndvi):.3f}"
    )
    return ndvi.astype(np.float32)


# ── NDVI from file paths ───────────────────────────────────────────────────────
def ndvi_from_files(red_path: str | Path,
                    nir_path: str | Path) -> tuple[np.ndarray, dict]:
    """
    Full pipeline:
    Load B04 + B08 → preprocess → calculate NDVI
    Returns: (ndvi_array, metadata)
    """
    logger.info("Starting NDVI pipeline...")

    red, meta = preprocess_band(red_path)
    nir, _    = preprocess_band(nir_path)

    # Shape check
    if red.shape != nir.shape:
        raise ValueError(
            f"Band shape mismatch: Red{red.shape} vs NIR{nir.shape}"
        )

    ndvi = calculate_ndvi(red, nir)
    logger.success("NDVI pipeline complete!")
    return ndvi, meta


# ── NDVI Statistics ────────────────────────────────────────────────────────────
def ndvi_stats(ndvi: np.ndarray) -> dict:
    """
    Calculate NDVI statistics.
    Returns dict with min, max, mean, std, coverage per class.
    """
    valid = ndvi[~np.isnan(ndvi)]
    total = valid.size

    stats = {
        "min"  : float(np.min(valid)),
        "max"  : float(np.max(valid)),
        "mean" : float(np.mean(valid)),
        "std"  : float(np.std(valid)),
    }

    # Class coverage percentage
    coverage = {}
    for class_name, (low, high) in NDVI_CLASSES.items():
        mask  = (valid >= low) & (valid < high)
        pct   = (mask.sum() / total) * 100
        coverage[class_name] = round(float(pct), 2)

    stats["coverage"] = coverage
    logger.info(f"NDVI Stats: {stats}")
    return stats


# ── Classify NDVI ──────────────────────────────────────────────────────────────
def classify_ndvi(ndvi: np.ndarray) -> np.ndarray:
    """
    Classify NDVI array into land cover classes.
    Returns integer class map:
    0 = Water/Snow
    1 = Bare Soil/Urban
    2 = Sparse Vegetation
    3 = Moderate Vegetation
    4 = Dense Vegetation
    """
    classified = np.zeros_like(ndvi, dtype=np.uint8)

    for idx, (_, (low, high)) in enumerate(NDVI_CLASSES.items()):
        mask = (ndvi >= low) & (ndvi < high)
        classified[mask] = idx

    logger.info("NDVI classification complete")
    return classified


# ── Run full NDVI pipeline ─────────────────────────────────────────────────────
def run_ndvi_pipeline(red_path: str | Path,
                      nir_path: str | Path,
                      output_dir: str | Path = None) -> dict:
    """
    Complete NDVI pipeline:
    Load → Preprocess → Calculate → Stats → Classify → Save

    Returns: {
        "ndvi"      : ndvi array,
        "classified": classified array,
        "stats"     : statistics dict,
        "meta"      : rasterio metadata,
        "saved_to"  : output path (if saved)
    }
    """
    # Calculate NDVI
    ndvi, meta = ndvi_from_files(red_path, nir_path)

    # Statistics
    stats = ndvi_stats(ndvi)

    # Classification
    classified = classify_ndvi(ndvi)

    result = {
        "ndvi"       : ndvi,
        "classified" : classified,
        "stats"      : stats,
        "meta"       : meta,
        "saved_to"   : None,
    }

    # Save if output dir given
    if output_dir:
        output_dir  = Path(output_dir)
        ndvi_path   = output_dir / "ndvi_delhi.tif"
        class_path  = output_dir / "ndvi_classified_delhi.tif"

        save_processed(ndvi, meta, ndvi_path)
        save_processed(classified.astype(np.float32), meta, class_path)

        result["saved_to"] = str(ndvi_path)
        logger.success(f"NDVI saved to: {ndvi_path}")

    return result